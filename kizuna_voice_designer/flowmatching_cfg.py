#!/usr/bin/env python3
"""
テキストプロンプトから音声を生成するスクリプト（FlowMatching CFG版）
Classifier-Free Guidanceで条件付けを強化
"""

import math
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer

from kizuna_voice_designer.device_utils import resolve_device


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlockAttn(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(out_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        h = F.silu(self.norm1(x))
        h = self.linear1(h)
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.linear2(h)
        h = h + residual

        q = h.unsqueeze(1)
        kv = cond_vec.unsqueeze(1)
        attn_out, _ = self.attn(q, kv, kv)
        h = h + self.attn_norm(attn_out.squeeze(1))
        return h


class FlowMatchingVelocityNetCFG(nn.Module):
    def __init__(
        self,
        voice_dim: int = 1024,
        text_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 6,
        time_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.voice_dim = voice_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.input_proj = nn.Linear(voice_dim, hidden_dim)

        enc_depth = max(1, num_layers // 2)
        dec_depth = num_layers - enc_depth
        self.encoder = nn.ModuleList([ResidualBlockAttn(hidden_dim, hidden_dim, dropout) for _ in range(enc_depth)])
        self.bottleneck = ResidualBlockAttn(hidden_dim, hidden_dim, dropout, num_heads=8)
        self.decoder = nn.ModuleList([ResidualBlockAttn(hidden_dim, hidden_dim, dropout) for _ in range(dec_depth)])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, voice_dim),
        )

    def forward(self, x_t, t, text_emb):
        h = self.input_proj(x_t)
        time_emb = self.time_mlp(t)
        text_cond = self.text_proj(text_emb)
        h = h + time_emb + text_cond
        skips = []
        for layer in self.encoder:
            h = layer(h, text_cond)
            skips.append(h)
        h = self.bottleneck(h, text_cond)
        for layer in self.decoder[::-1]:
            if skips:
                h = h + skips.pop()
            h = layer(h, text_cond)
        return self.output_proj(h)


class TextToVoiceFlowCFGSynthesizer:
    """テキストプロンプトから音声を合成するクラス（FlowMatching CFG版）"""

    def __init__(
        self,
        flowmatching_model_path: str,
        text_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        text_emb_dir: str = "",
        device: str = "cuda",
    ):
        self.device = resolve_device(device)

        self.text_emb_dir = text_emb_dir
        if text_emb_dir:
            print(f"Using precomputed text embeddings from: {text_emb_dir}")
        else:
            print(f"Loading text embedding model: {text_embedding_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(text_embedding_model, trust_remote_code=True)
            self.text_model = AutoModel.from_pretrained(text_embedding_model, trust_remote_code=True, torch_dtype=torch.float16)
            self.text_model = self.text_model.to(self.device)
            self.text_model.eval()

        print(f"Loading FlowMatching CFG model: {flowmatching_model_path}")
        checkpoint = torch.load(flowmatching_model_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})

        self.flow_model = FlowMatchingVelocityNetCFG(
            voice_dim=config.get("voice_dim", 1024),
            text_dim=config.get("text_dim", 1024),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 6),
            time_dim=config.get("time_dim", 128),
            dropout=0.0,
        )
        self.flow_model.load_state_dict(checkpoint["model_state_dict"])
        self.flow_model = self.flow_model.to(self.device)
        self.flow_model.eval()

        self.num_sample_steps = config.get("num_sample_steps", 20)
        self.guidance_scale = config.get("guidance_scale", 5.0)

        print(f"Models loaded on {self.device}")
        print(f"Default guidance_scale: {self.guidance_scale}, num_sample_steps: {self.num_sample_steps}")

    @torch.inference_mode()
    def encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        if self.text_emb_dir:
            h = hashlib.md5(text.encode()).hexdigest()[:16]
            emb_path = Path(self.text_emb_dir) / f"{h}.npy"
            if not emb_path.exists():
                raise FileNotFoundError(f"Text embedding not found: {emb_path}")
            emb = np.load(emb_path)
            return torch.from_numpy(emb).float().to(self.device)

        inputs = self.tokenizer(
            text, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(self.device)

        outputs = self.text_model(**inputs)
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embedding = sum_embeddings / sum_mask

        return embedding.float()

    @torch.inference_mode()
    def sample_from_flow_cfg(
        self,
        text_emb: torch.Tensor,
        num_steps: int = None,
        guidance_scale: float = None,
        text_scale: float = 1.0,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """CFGを使ったサンプリング"""
        if num_steps is None:
            num_steps = self.num_sample_steps
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        # 統一dtype
        flow_dtype = self.flow_model.input_proj.weight.dtype
        text_emb = text_emb.to(flow_dtype)

        batch_size = text_emb.shape[0]
        voice_dim = self.flow_model.voice_dim

        # テキスト強調
        text_emb = text_emb * text_scale

        zero_emb = torch.zeros_like(text_emb, dtype=flow_dtype)
        x = noise_scale * torch.randn(batch_size, voice_dim, device=self.device, dtype=flow_dtype)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)

            # CFG: 条件なしと条件ありの両方を予測
            v_uncond = self.flow_model(x, t, zero_emb)
            v_cond = self.flow_model(x, t, text_emb)

            # CFGで合成
            v = v_uncond + guidance_scale * (v_cond - v_uncond)

            x = x + v * dt

        return x

    @torch.inference_mode()
    def text_to_ge_embedding(
        self,
        text_prompt: str,
        num_steps: int = None,
        guidance_scale: float = None,
        text_scale: float = 1.0,
        noise_scale: float = 1.0,
        seed: int = None,
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)

        text_emb = self.encode_text(text_prompt)
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
        ge_emb = self.sample_from_flow_cfg(
            text_emb,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            text_scale=text_scale,
            noise_scale=noise_scale,
        )

        return ge_emb
