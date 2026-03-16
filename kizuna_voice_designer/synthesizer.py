"""Core synthesizer module for programmatic use."""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from kizuna_voice_designer.downloader import (
    setup_paths,
    ensure_pretrained_models,
    ensure_flow_model,
    CACHE_DIR,
)


class VoiceDesigner:
    """Text prompt-based voice synthesizer.

    Usage:
        from kizuna_voice_designer import VoiceDesigner

        vd = VoiceDesigner(device="cuda:0")
        audio, sr = vd.generate(
            prompt="30代前半の女性 落ち着きのある透明感ボイス",
            text="こんにちは、テスト音声です。",
        )
        vd.save("output.wav", audio, sr)
    """

    def __init__(
        self,
        device: str = "cuda:0",
        embedding_mode: Optional[str] = None,
        embedding_model: str = "Qwen/Qwen3-Embedding-4B",
        gguf_model: str = "Qwen/Qwen3-Embedding-4B-GGUF",
        gguf_file: str = "Qwen3-Embedding-4B-Q8_0.gguf",
        openrouter_key: Optional[str] = None,
        flow_model_path: Optional[str] = None,
        text_emb_dir: Optional[str] = None,
        cfg_scale: float = 3.5,
        text_scale: float = 1.3,
        noise_scale: float = 0.7,
        num_steps: int = 60,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        """Initialize VoiceDesigner.

        Args:
            device: CUDA device (e.g. "cuda:0", "cuda:1", "cpu")
            embedding_mode: "api", "local", or "local_lightweight". Auto-detected if None.
                - api: OpenRouter API (lightweight, needs API key)
                - local: transformers Qwen model (VRAM ~8GB, best quality)
                - local_lightweight: GGUF quantized model (VRAM ~4GB, slightly lower quality)
            embedding_model: HuggingFace model name for local mode.
            gguf_model: HuggingFace repo for GGUF model.
            gguf_file: GGUF filename within the repo.
            openrouter_key: OpenRouter API key for api mode.
            flow_model_path: Path to FlowMatching model. Auto-detected if None.
            text_emb_dir: Path to precomputed text embeddings.
            cfg_scale: CFG guidance scale.
            text_scale: Text conditioning scale.
            noise_scale: Noise scale for VITS decoder.
            num_steps: Number of FlowMatching sampling steps.
            top_k: Top-k for GPT sampling.
            top_p: Top-p for GPT sampling.
            temperature: Temperature for GPT sampling.
        """
        import torch

        # Setup paths for GPT-SoVITS (downloads if needed)
        self._gpt_sovits_dir = setup_paths()

        # Ensure pretrained models are available
        ensure_pretrained_models()

        self.openrouter_key = openrouter_key or ""

        # Embedding mode
        if embedding_mode is None:
            self.embedding_mode = "api" if self.openrouter_key else "local_lightweight"
        else:
            self.embedding_mode = embedding_mode

        # Paths
        if flow_model_path is None:
            flow_model_path = str(ensure_flow_model())
        self.flow_model_path = flow_model_path

        if text_emb_dir is None:
            if self.embedding_mode == "local":
                text_emb_dir = ""
            elif self.embedding_mode == "local_lightweight":
                # ダミーパスを設定してtransformersモデルのロードを回避
                # encode_textはFileNotFoundErrorになるが、_generate_geでGGUFにフォールバック
                text_emb_dir = str(CACHE_DIR / "models" / "text_embeddings")
            else:
                text_emb_dir = str(CACHE_DIR / "models" / "text_embeddings")
        self.text_emb_dir = text_emb_dir

        # Device
        self.device_str = self._normalize_device(device)

        # Params
        self.cfg_scale = cfg_scale
        self.text_scale = text_scale
        self.noise_scale = noise_scale
        self.num_steps = num_steps
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.embedding_model = embedding_model
        self.gguf_model = gguf_model
        self.gguf_file = gguf_file

        # Lazy-loaded models
        self._synthesizer = None
        self._tts = None
        self._llama_model = None

    def _normalize_device(self, device_str: str) -> str:
        import torch
        if device_str.startswith("cuda") and torch.cuda.is_available():
            try:
                req_idx = int(device_str.split(":")[1])
            except (IndexError, ValueError):
                req_idx = 0
            max_idx = torch.cuda.device_count() - 1
            if max_idx < 0:
                return "cpu"
            if req_idx > max_idx:
                return f"cuda:{max_idx}"
            return f"cuda:{req_idx}"
        return "cpu"

    def _get_synthesizer(self):
        if self._synthesizer is None:
            from kizuna_voice_designer.flowmatching import TextToVoiceFlowSynthesizer
            from kizuna_voice_designer.flowmatching_cfg import TextToVoiceFlowCFGSynthesizer

            is_cfg = "cfg" in self.flow_model_path
            cls = TextToVoiceFlowCFGSynthesizer if is_cfg else TextToVoiceFlowSynthesizer
            self._synthesizer = cls(
                flowmatching_model_path=self.flow_model_path,
                text_emb_dir=self.text_emb_dir,
                text_embedding_model=self.embedding_model,
                device=self.device_str,
            )
        return self._synthesizer

    def _get_tts(self):
        if self._tts is None:
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
            pretrained_root = self._gpt_sovits_dir / "GPT_SoVITS" / "pretrained_models"
            cfg = {
                "version": "v2Pro",
                "device": self.device_str,
                "is_half": True,
                "t2s_weights_path": str(pretrained_root / "s1v3.ckpt"),
                "vits_weights_path": str(pretrained_root / "v2Pro" / "s2Gv2Pro.pth"),
                "bert_base_path": str(pretrained_root / "chinese-roberta-wwm-ext-large"),
                "cnhuhbert_base_path": str(pretrained_root / "chinese-hubert-base"),
                "sv_model_path": str(pretrained_root / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"),
            }
            self._tts = TTS(TTS_Config({"custom": cfg}))
        return self._tts

    def _get_llama_model(self):
        if self._llama_model is None:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id=self.gguf_model,
                filename=self.gguf_file,
            )
            gpu_idx = 0
            if self.device_str.startswith("cuda:"):
                try:
                    gpu_idx = int(self.device_str.split(":")[1])
                except (IndexError, ValueError):
                    gpu_idx = 0

            self._llama_model = Llama(
                model_path=model_path,
                n_ctx=512,
                n_gpu_layers=-1,
                main_gpu=gpu_idx,
                embedding=True,
                verbose=False,
            )
        return self._llama_model

    def _fetch_lightweight_embedding(self, text: str) -> np.ndarray:
        model = self._get_llama_model()
        result = model.create_embedding(text)
        emb = result["data"][0]["embedding"]
        return np.array(emb, dtype=np.float32)

    def _fetch_openrouter_embedding(self, text: str) -> np.ndarray:
        import json
        import requests

        if not self.openrouter_key:
            raise RuntimeError("OPENROUTER_KEY not set; use embedding_mode='local' instead.")
        resp = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({"model": "qwen/qwen3-embedding-4b", "input": text}),
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter API error: {resp.status_code}")
        return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)

    def _generate_ge(self, prompt: str):
        """Generate GE (voice quality) embedding from prompt."""
        import torch
        from kizuna_voice_designer.flowmatching_cfg import TextToVoiceFlowCFGSynthesizer

        device = torch.device(self.device_str)
        synthesizer = self._get_synthesizer()
        is_cfg_model = isinstance(synthesizer, TextToVoiceFlowCFGSynthesizer)

        try:
            if is_cfg_model:
                ge = synthesizer.text_to_ge_embedding(
                    prompt, num_steps=self.num_steps,
                    guidance_scale=self.cfg_scale, text_scale=self.text_scale,
                    noise_scale=self.noise_scale,
                    seed=torch.randint(0, 2**31 - 1, ()).item(),
                )
            else:
                ge = synthesizer.text_to_ge_embedding(
                    prompt, num_steps=self.num_steps,
                    seed=torch.randint(0, 2**31 - 1, ()).item(),
                    cfg_scale=self.cfg_scale, text_scale=self.text_scale,
                )
        except FileNotFoundError:
            if self.embedding_mode == "api":
                emb_np = self._fetch_openrouter_embedding(prompt)
            elif self.embedding_mode == "local_lightweight":
                emb_np = self._fetch_lightweight_embedding(prompt)
            else:
                raise
            emb_t = torch.from_numpy(emb_np).to(device)
            if emb_t.dim() == 1:
                emb_t = emb_t.unsqueeze(0)
            if device.type == "cuda":
                emb_t = emb_t.half()

            original_encode = synthesizer.encode_text
            def _patched_encode(_text, max_length=512):
                if is_cfg_model:
                    target_dtype = synthesizer.flow_model.input_proj.weight.dtype
                    return emb_t.to(target_dtype)
                return emb_t
            synthesizer.encode_text = _patched_encode
            try:
                if is_cfg_model:
                    ge = synthesizer.text_to_ge_embedding(
                        prompt, num_steps=self.num_steps,
                        guidance_scale=self.cfg_scale, text_scale=self.text_scale,
                        noise_scale=self.noise_scale,
                    )
                else:
                    ge = synthesizer.text_to_ge_embedding(
                        prompt, num_steps=self.num_steps, seed=None,
                        cfg_scale=self.cfg_scale, text_scale=self.text_scale,
                    )
            finally:
                synthesizer.encode_text = original_encode

        return ge

    def _synthesize_with_ge(self, ge, text: str, lang: str = "all_ja"):
        """Synthesize audio using a pre-computed GE embedding."""
        import torch
        import torch.nn.functional as F
        from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import splits

        device = torch.device(self.device_str)
        tts = self._get_tts()
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        # Text preprocessing
        read_text = text.strip()
        if read_text and read_text[-1] not in splits:
            read_text += "\u3002" if lang != "en" else "."

        phones, bert_features, norm_text = tts.text_preprocessor.segment_and_extract_feature_for_text(read_text, lang)
        all_phoneme_ids = torch.LongTensor(phones).to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        bert = bert_features.to(device).to(dtype).unsqueeze(0)

        # GPT inference
        with torch.no_grad():
            pred_semantic, idx = tts.t2s_model.model.infer_panel(
                all_phoneme_ids, all_phoneme_len, None, bert,
                top_k=self.top_k, top_p=self.top_p,
                temperature=self.temperature, early_stop_num=50 * 30,
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

        # VITS decode
        vits_model = tts.vits_model
        ge_in = ge.unsqueeze(-1).to(device).to(dtype)

        with torch.no_grad():
            y_lengths = torch.LongTensor([pred_semantic.size(2) * 2]).to(device)
            text_lengths = torch.LongTensor([all_phoneme_ids.size(-1)]).to(device)
            quantized = vits_model.quantizer.decode(pred_semantic)
            if vits_model.semantic_frame_rate == "25hz":
                quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
            ge_512 = vits_model.ge_to512(ge_in.transpose(2, 1)).transpose(2, 1)
            x, m_p, logs_p, y_mask, _, _ = vits_model.enc_p(
                quantized, y_lengths, all_phoneme_ids, text_lengths, ge_512
            )
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.noise_scale
            z = vits_model.flow(z_p, y_mask, g=ge_in, reverse=True)
            o = vits_model.dec((z * y_mask)[:, :, :], g=ge_in)
            audio = o[0, 0].cpu().numpy()

        return audio

    def generate(
        self,
        prompt: str,
        text: str,
        lang: str = "all_ja",
        embedding: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Generate voice audio from text prompt.

        Args:
            prompt: Voice description prompt (e.g. "30代前半の女性 落ち着いた声").
                    Ignored if embedding is provided.
            text: Text to read aloud.
            lang: Language code ("all_ja", "ja", "en", "zh").
            embedding: Pre-computed voice embedding (from a previous generate() call).
                       If provided, prompt is ignored and this embedding is used directly.

        Returns:
            (audio, sample_rate, embedding):
                - audio: NumPy audio array (float32)
                - sample_rate: 32000
                - embedding: Voice embedding (np.ndarray). Save this to reuse the same voice.
        """
        import torch

        device = torch.device(self.device_str)

        if embedding is not None:
            ge = torch.from_numpy(embedding).to(device)
            if device.type == "cuda":
                ge = ge.half()
        else:
            ge = self._generate_ge(prompt)

        audio = self._synthesize_with_ge(ge, text, lang)
        embedding_out = ge.cpu().float().numpy()

        return audio, 32000, embedding_out

    def save(self, path: str, audio: np.ndarray, sr: int = 32000):
        """Save audio to WAV file."""
        sf.write(path, audio.astype(np.float32), sr)

    def save_embedding(self, path: str, embedding: np.ndarray):
        """Save voice embedding to .npy file.

        Args:
            path: Output path (e.g. "my_voice.npy").
            embedding: Embedding array from generate().
        """
        np.save(path, embedding)

    def load_embedding(self, path: str) -> np.ndarray:
        """Load voice embedding from .npy file.

        Args:
            path: Path to .npy file.

        Returns:
            embedding: NumPy array to pass to generate(embedding=...).
        """
        emb = np.load(path)
        if emb.ndim == 1:
            emb = emb[None, :]
        return emb
