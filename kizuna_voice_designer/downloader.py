"""初回実行時に必要なファイルを自動ダウンロード"""
import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

CACHE_DIR = Path.home() / ".cache" / "kizuna-voice-designer"


def _patch_sv_py(gpt_dir: Path):
    """sv.pyのパス解決を修正（元のRVC-Boss版は相対パスなので）"""
    sv_path = gpt_dir / "GPT_SoVITS" / "sv.py"
    if not sv_path.exists():
        return
    content = sv_path.read_text(encoding="utf-8")
    if "_SV_DIR" in content:
        return  # 既にパッチ済み
    patched = content.replace(
        'sys.path.append(f"{os.getcwd()}/GPT_SoVITS/eres2net")\n'
        'sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"',
        'from pathlib import Path as _Path\n'
        '_SV_DIR = _Path(__file__).resolve().parent\n'
        'sys.path.append(str(_SV_DIR / "eres2net"))\n'
        'sv_path = os.environ.get("SV_CKPT_PATH", str(_SV_DIR / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"))',
    )
    sv_path.write_text(patched, encoding="utf-8")


def ensure_gpt_sovits():
    """GPT-SoVITSの推論コードをダウンロード"""
    gpt_dir = CACHE_DIR / "GPT-SoVITS"
    if (gpt_dir / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py").exists():
        return gpt_dir

    gpt_dir.mkdir(parents=True, exist_ok=True)
    # 元のRVC-Boss版（パブリック）からダウンロード
    url = "https://github.com/RVC-Boss/GPT-SoVITS/archive/refs/heads/main.zip"
    zip_path = CACHE_DIR / "gpt-sovits.zip"
    print("Downloading GPT-SoVITS inference code...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CACHE_DIR)
    # 展開されたディレクトリをリネーム
    extracted = CACHE_DIR / "GPT-SoVITS-main"
    if extracted.exists():
        if gpt_dir.exists():
            shutil.rmtree(gpt_dir)
        extracted.rename(gpt_dir)
    zip_path.unlink(missing_ok=True)
    # sv.pyのパス解決を修正
    _patch_sv_py(gpt_dir)
    return gpt_dir


def ensure_pretrained_models():
    """GPT-SoVITS pretrained modelsをダウンロード"""
    gpt_dir = CACHE_DIR / "GPT-SoVITS"
    pretrained = gpt_dir / "GPT_SoVITS" / "pretrained_models"
    if (pretrained / "s1v3.ckpt").exists():
        return pretrained

    print("Downloading GPT-SoVITS pretrained models (first time only)...")
    files = [
        "s1v3.ckpt",
        "v2Pro/s2Gv2Pro.pth",
    ]
    for f in files:
        hf_hub_download("lj1995/GPT-SoVITS", f, local_dir=str(pretrained))

    # BERTモデル等はディレクトリ単位
    from huggingface_hub import snapshot_download
    for subdir in ["chinese-roberta-wwm-ext-large", "chinese-hubert-base"]:
        target = pretrained / subdir
        if not target.exists():
            snapshot_download(
                "lj1995/GPT-SoVITS",
                local_dir=str(pretrained),
                allow_patterns=[f"{subdir}/*"],
            )

    # sv model
    hf_hub_download(
        "lj1995/GPT-SoVITS",
        "sv/pretrained_eres2netv2w24s4ep4.ckpt",
        local_dir=str(pretrained),
    )

    return pretrained


def ensure_flow_model():
    """FlowMatchingモデルをダウンロード"""
    flow_dir = CACHE_DIR / "models" / "flow"
    model_path = flow_dir / "best_model.pt"
    if model_path.exists():
        return model_path

    flow_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading FlowMatching model...")

    # 方法1: パッケージ同梱のモデル（開発時）
    pkg_model = Path(__file__).resolve().parent.parent / "models" / "flow" / "best_model.pt"
    if pkg_model.exists():
        shutil.copy2(pkg_model, model_path)
        return model_path

    # 方法2: GitHub Releasesからダウンロード
    url = "https://github.com/kizuna-intelligence/kizuna-voice-designer/releases/download/v0.1.0-beta/models.zip"
    zip_path = CACHE_DIR / "models.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(flow_dir)
        zip_path.unlink(missing_ok=True)
    except Exception:
        print(f"Auto-download failed.")
        print(f"Please manually place the model at: {model_path}")
        print(f"Download: https://github.com/kizuna-intelligence/kizuna-voice-designer/releases")
        raise
    return model_path


def setup_paths():
    """sys.pathにGPT-SoVITSのパスを追加"""
    gpt_dir = ensure_gpt_sovits()
    gpt_dir_str = str(gpt_dir)
    gpt_sovits_str = str(gpt_dir / "GPT_SoVITS")
    eres2net_str = str(gpt_dir / "GPT_SoVITS" / "eres2net")

    # 重複追加を避ける
    for p in [gpt_dir_str, gpt_sovits_str, eres2net_str]:
        if p not in sys.path:
            sys.path.insert(0, p)

    return gpt_dir
