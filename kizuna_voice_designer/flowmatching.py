#!/usr/bin/env python3
"""
テキストプロンプトから音声を生成するスクリプト（FlowMatching版）
参照音声なしで、テキストプロンプトからge埋め込みをFlow Matchingで生成して音声合成
"""

import math
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

from kizuna_voice_designer.device_utils import is_cuda_device, resolve_device


# Flow Matching用ネットワーク定義
class SinusoidalPositionEmbeddings(nn.Module):
    """時間ステップの正弦波位置埋め込み"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        dtype = time.dtype
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """残差ブロック"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.linear1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.linear2(h)

        return h + residual


class FlowMatchingVelocityNet(nn.Module):
    """Flow Matching用の速度場予測ネットワーク"""

    def __init__(
        self,
        voice_dim: int = 1024,
        text_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.voice_dim = voice_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # 時間埋め込み
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # テキスト条件の投影
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # 入力（x_t）の投影
        self.input_proj = nn.Linear(voice_dim, hidden_dim)

        # メインネットワーク（ResNet風）
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                ResidualBlock(hidden_dim, hidden_dim, dropout)
            )

        # 出力層
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, voice_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,      # [B, voice_dim] 現在の状態
        t: torch.Tensor,         # [B] 時間 (0~1)
        text_emb: torch.Tensor,  # [B, text_dim] テキスト条件
    ) -> torch.Tensor:
        # 各入力を投影
        h = self.input_proj(x_t)
        time_emb = self.time_mlp(t)
        text_cond = self.text_proj(text_emb)

        # 条件を加算
        h = h + time_emb + text_cond

        # ResNetブロック
        for layer in self.layers:
            h = layer(h)

        # 出力
        v = self.output_proj(h)
        return v


class TextToVoiceFlowSynthesizer:
    """テキストプロンプトから音声を合成するクラス（FlowMatching版）"""

    def __init__(
        self,
        flowmatching_model_path: str,
        text_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        text_emb_dir: str = "",
        device: str = "cuda",
    ):
        self.device = resolve_device(device)
        self.text_emb_dir = Path(text_emb_dir) if text_emb_dir else None

        # テキスト埋め込みモデルをロード
        if self.text_emb_dir:
            print(f"Using precomputed text embeddings from: {self.text_emb_dir}")
            self.tokenizer = None
            self.text_model = None
        else:
            print(f"Loading text embedding model: {text_embedding_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(text_embedding_model, trust_remote_code=True)
            model_dtype = torch.float16 if is_cuda_device(self.device) else torch.float32
            self.text_model = AutoModel.from_pretrained(
                text_embedding_model, trust_remote_code=True, torch_dtype=model_dtype
            )
            self.text_model = self.text_model.to(self.device)
            self.text_model.eval()

        # FlowMatching Networkをロード
        print(f"Loading FlowMatching model: {flowmatching_model_path}")
        map_location = self.device if is_cuda_device(self.device) else "cpu"
        checkpoint = torch.load(flowmatching_model_path, map_location=map_location, weights_only=False)
        config = checkpoint.get("config", {})

        self.flow_model = FlowMatchingVelocityNet(
            voice_dim=config.get("voice_dim", 1024),
            text_dim=config.get("text_dim", 1024),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 4),
            time_dim=config.get("time_dim", 128),
            dropout=0.0,  # 推論時はdropoff
        )
        self.flow_model.load_state_dict(checkpoint["model_state_dict"])
        self.flow_model = self.flow_model.to(self.device)
        if is_cuda_device(self.device):
            self.flow_model = self.flow_model.half()
        self.flow_model.eval()

        self.num_sample_steps = config.get("num_sample_steps", 10)

        print(f"Models loaded on {self.device}")

    @torch.inference_mode()
    def encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """テキストを埋め込みに変換（事前計算があればロード）"""
        if self.text_emb_dir:
            h = hashlib.md5(text.encode()).hexdigest()[:16]
            path = self.text_emb_dir / f"{h}.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            emb = torch.from_numpy(np.load(path)).to(self.device)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            return emb.half() if is_cuda_device(self.device) else emb.float()

        inputs = self.tokenizer(
            text, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(self.device)

        outputs = self.text_model(**inputs)
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embedding = sum_embeddings / sum_mask

        embedding = embedding.half() if is_cuda_device(self.device) else embedding.float()
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        return embedding

    @torch.inference_mode()
    def sample_from_flow(
        self,
        text_emb: torch.Tensor,
        num_steps: int = None,
        cfg_scale: float = 1.0,
        text_scale: float = 1.0,
    ) -> torch.Tensor:
        """Flow MatchingでODEを解いてge埋め込みを生成

        Args:
            text_emb: テキスト埋め込み [B, text_dim]
            num_steps: ODEソルバーのステップ数（Noneの場合はモデルのデフォルト）
        """
        if num_steps is None:
            num_steps = self.num_sample_steps

        batch_size = text_emb.shape[0]
        voice_dim = self.flow_model.voice_dim

        # dtype統一（flowモデルの重みdtypeに合わせる）
        flow_dtype = self.flow_model.input_proj.weight.dtype
        text_emb = text_emb.to(flow_dtype)

        # テキスト強調
        text_emb = text_emb * text_scale

        # x_0 ~ N(0, I) からスタート
        x = torch.randn(batch_size, voice_dim, device=self.device, dtype=flow_dtype)

        # Euler法でODEを解く
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device, dtype=flow_dtype)
            v_uncond = self.flow_model(x, t, torch.zeros_like(text_emb))
            v_cond = self.flow_model(x, t, text_emb)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            x = x + v * dt

        return x

    @torch.inference_mode()
    def text_to_ge_embedding(
        self,
        text_prompt: str,
        num_steps: int = None,
        seed: int = None,
        cfg_scale: float = 1.0,
        text_scale: float = 1.0,
    ) -> torch.Tensor:
        """テキストプロンプトをge埋め込み(1024次元)に変換

        Args:
            text_prompt: 声質を表すテキスト
            num_steps: ODEソルバーのステップ数
            seed: 乱数シード（再現性のため）
        """
        if seed is not None:
            torch.manual_seed(seed)

        text_emb = self.encode_text(text_prompt)
        ge_emb = self.sample_from_flow(
            text_emb,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            text_scale=text_scale,
        )

        return ge_emb
