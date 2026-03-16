#!/usr/bin/env python3
"""
シンプル声質生成アプリ（パッケージ内版）
"""

import os
import sys
import hashlib
import json
import requests
from pathlib import Path
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr

# Package-internal imports
from kizuna_voice_designer.flowmatching import TextToVoiceFlowSynthesizer
from kizuna_voice_designer.flowmatching_cfg import TextToVoiceFlowCFGSynthesizer
from kizuna_voice_designer.downloader import (
    setup_paths,
    ensure_pretrained_models,
    ensure_flow_model,
    CACHE_DIR,
)

# Setup GPT-SoVITS paths
_GPT_SOVITS_DIR = setup_paths()
ensure_pretrained_models()

# Paths based on cache directory
DEFAULT_FLOW_MODEL = str(ensure_flow_model())
DEFAULT_TEXT_EMB_DIR = str(CACHE_DIR / "models" / "text_embeddings")

# GPT-SoVITS imports (after setup_paths)
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import splits


def _load_dotenv_if_needed():
    if "OPENROUTER_KEY" in os.environ and os.environ["OPENROUTER_KEY"]:
        return
    # Try to find .env in common locations
    for candidate in [Path.cwd() / ".env", Path.home() / ".env"]:
        if candidate.exists():
            with candidate.open() as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        os.environ.setdefault(k, v)
            break


_load_dotenv_if_needed()
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
# EMBEDDING_MODE: "api" (OpenRouter), "local" (ローカルQwenモデル)
# 未指定時: OPENROUTER_KEYがあればapi、なければlocal
EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "api" if OPENROUTER_KEY else "local")


def _normalize_device_str(device_str: str) -> str:
    """Clamp cuda index if out of range; fall back to cpu when unavailable."""
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


def _get_device(device_str: str) -> torch.device:
    return torch.device(_normalize_device_str(device_str))


LOCAL_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")


@lru_cache(maxsize=4)
def load_synth(flow_model_path: str, text_emb_dir: str, device: str) -> TextToVoiceFlowSynthesizer:
    # cfg 学習チェックポイントかどうかを簡易判定
    is_cfg = "cfg" in Path(flow_model_path).name or "cfg" in str(flow_model_path)
    cls = TextToVoiceFlowCFGSynthesizer if is_cfg else TextToVoiceFlowSynthesizer
    return cls(
        flowmatching_model_path=flow_model_path,
        text_emb_dir=text_emb_dir,
        text_embedding_model=LOCAL_EMBEDDING_MODEL,
        device=_normalize_device_str(device),
    )


@lru_cache(maxsize=4)
def load_tts(device: str) -> TTS:
    pretrained_root = _GPT_SOVITS_DIR / "GPT_SoVITS" / "pretrained_models"
    cfg = {
        "version": "v2Pro",
        "device": device,
        "is_half": True,
        "t2s_weights_path": str(pretrained_root / "s1v3.ckpt"),
        "vits_weights_path": str(pretrained_root / "v2Pro" / "s2Gv2Pro.pth"),
        "bert_base_path": str(pretrained_root / "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(pretrained_root / "chinese-hubert-base"),
        "sv_model_path": str(pretrained_root / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"),
    }
    return TTS(TTS_Config({"custom": cfg}))


def fetch_openrouter_embedding(text: str) -> np.ndarray:
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_KEY not set; cannot fetch 4B embedding.")
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "qwen/qwen3-embedding-4b",
        "input": text,
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter embedding failed status={resp.status_code} body={resp.text[:200]}")
    data = resp.json()
    emb = data["data"][0]["embedding"]
    return np.array(emb, dtype=np.float32)


def synthesize(
    prompt: str,
    read_text: str,
    text_lang: str,
    flow_model_path: str,
    text_emb_dir: str,
    cfg_scale: float,
    text_scale: float,
    num_steps: int,
    top_k: int,
    top_p: float,
    temperature: float,
    noise_scale: float,
    reuse_last_ge: bool,
    ge_state,
    device_str: str,
) -> Tuple[Tuple[int, np.ndarray], str, object]:
    """
    Returns (sample_rate, audio, debug_info)
    """
    device = _get_device(device_str)
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # normalize device string for loaders
    device_str_norm = _normalize_device_str(device_str)
    synthesizer = load_synth(flow_model_path, text_emb_dir, device_str_norm)
    tts = load_tts(device_str_norm)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # ge embedding
    ge = None
    ge_source = "reuse" if reuse_last_ge and ge_state is not None else "computed"
    if reuse_last_ge and ge_state is not None:
        ge = torch.from_numpy(ge_state).to(device)
        if device.type == "cuda":
            ge = ge.half()
    else:
        is_cfg_model = isinstance(synthesizer, TextToVoiceFlowCFGSynthesizer)
        try:
            if is_cfg_model:
                ge = synthesizer.text_to_ge_embedding(
                    prompt,
                    num_steps=num_steps,
                    guidance_scale=cfg_scale,
                    text_scale=text_scale,
                    noise_scale=noise_scale,
                    seed=torch.randint(0, 2**31 - 1, ()).item(),
                )
            else:
                ge = synthesizer.text_to_ge_embedding(
                    prompt,
                    num_steps=num_steps,
                    seed=torch.randint(0, 2**31 - 1, ()).item(),
                    cfg_scale=cfg_scale,
                    text_scale=text_scale,
                )
            ge_source = "precomputed"
        except FileNotFoundError:
            # fallback: call OpenRouter directly and patch encode_text
            emb_np = fetch_openrouter_embedding(prompt)
            emb_t = torch.from_numpy(emb_np).to(device)
            if emb_t.dim() == 1:
                emb_t = emb_t.unsqueeze(0)
            if device.type == "cuda":
                emb_t = emb_t.half()

            original_encode = synthesizer.encode_text
            def _patched_encode(_text, max_length=512):
                # CFGモデルはflowのdtypeに合わせる
                if isinstance(synthesizer, TextToVoiceFlowCFGSynthesizer):
                    target_dtype = synthesizer.flow_model.input_proj.weight.dtype
                    return emb_t.to(target_dtype)
                return emb_t

            synthesizer.encode_text = _patched_encode
            if is_cfg_model:
                ge = synthesizer.text_to_ge_embedding(
                    prompt,
                    num_steps=num_steps,
                    guidance_scale=cfg_scale,
                    text_scale=text_scale,
                    noise_scale=noise_scale,
                )
            else:
                ge = synthesizer.text_to_ge_embedding(
                    prompt,
                    num_steps=num_steps,
                    seed=None,
                    cfg_scale=cfg_scale,
                    text_scale=text_scale,
                )
            synthesizer.encode_text = original_encode
            ge_source = "on-the-fly (OpenRouter 4B)"

    # text preprocessing
    lang_map = {
        "ja": "ja",
        "jp": "ja",
        "\u65e5\u672c\u8a9e": "ja",
        "Japanese": "ja",
        "all_ja": "all_ja",
        "en": "en",
        "English": "en",
        "zh": "zh",
        "Chinese": "zh",
    }
    text_lang = lang_map.get(text_lang, "ja")
    text = read_text.strip()
    if text and text[-1] not in splits:
        text += "\u3002" if text_lang != "en" else "."

    phones, bert_features, norm_text = tts.text_preprocessor.segment_and_extract_feature_for_text(text, text_lang)
    all_phoneme_ids = torch.LongTensor(phones).to(device).unsqueeze(0)
    all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
    bert = bert_features.to(device).to(dtype).unsqueeze(0)

    # GPT inference (ref_free)
    with torch.no_grad():
        pred_semantic, idx = tts.t2s_model.model.infer_panel(
            all_phoneme_ids,
            all_phoneme_len,
            None,
            bert,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            early_stop_num=50 * 30,
        )
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

    # VITS decode with GE
    vits_model = tts.vits_model
    ge_in = ge.unsqueeze(-1).to(device).to(dtype)

    with torch.no_grad():
        codes = pred_semantic
        text_tensor = all_phoneme_ids

        y_lengths = torch.LongTensor([codes.size(2) * 2]).to(device)
        text_lengths = torch.LongTensor([text_tensor.size(-1)]).to(device)

        quantized = vits_model.quantizer.decode(codes)
        if vits_model.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")

        ge_512 = vits_model.ge_to512(ge_in.transpose(2, 1)).transpose(2, 1)
        x, m_p, logs_p, y_mask, _, _ = vits_model.enc_p(
            quantized, y_lengths, text_tensor, text_lengths, ge_512
        )

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = vits_model.flow(z_p, y_mask, g=ge_in, reverse=True)
        o = vits_model.dec((z * y_mask)[:, :, :], g=ge_in)
        audio = o[0, 0].cpu().numpy()

    debug = f"norm_text: {norm_text}\nprompt_hash: {hashlib.md5(prompt.encode()).hexdigest()[:16]}\nge_source: {ge_source}"
    ge_state_out = ge.cpu().float().numpy()
    return (32000, audio), debug, ge_state_out


def build_interface():
    # シンプル化されたプリセット
    preset_pairs = [
        ("コンシェルジュ（30代前半・透明感）",
         "30代前半の女性 落ち着きのある透明感ボイスで\n言葉の間に十分な余白を取りつつ丁寧に\n高級ホテルのラウンジで客に寄り添うコンシェルジュ",
         "こちらのお席でよろしければ、すぐにお飲み物をご用意いたします。"),
        ("大人ナレーター（20代後半・月光）",
         "20代後半の女性 月光のように柔らかく耳元で溶けていく声で\n夜更けの告白のように感情を抑えつつ温かみを込めて\n大人の女性向け恋愛小説の官能的なシーンを語るナレーター",
         "今夜だけは、あなたの物語を静かに語らせてください。"),
        ("執事（低音・古城）",
         "深々と響く落ち着いた声で\n変更不能な事実を伝えるような、ゆるぎない口調で\n古い城の案内をする、代々続く家系の執事",
         "お客様、右手の階段をお上がりください。書斎は二階でございます。"),
        ("小悪魔ヒロイン（高め・遊び心）",
         "甘くて柔らかい高めの声で、ふんわりとした空気感を纏うように\n早口でリズミカルに、相手をからかうような遊び心たっぷりのトーンで\n恋愛シミュレーションゲームで主人公を翻弄する、小悪魔的なヒロインキャラクター",
         "ねぇ、そんなに真面目にならなくていいよ。もっと面白いことしようよ？"),
        ("悪役ロリ（妹系・低テンション）",
         "10歳前後の妹系ロリボイスで、少しかすれた甘さと幼さを残しつつ\nテンションは低めでぼそっと話す、気怠そうなトーンで\n悪役の小悪魔ロリキャラが、主人公に冷たく囁きかけるように",
         "ふーん、まだ諦めないんだ。もっと苦しむところ、見せてよ。"),
        ("ツンデレ幼馴染（高校1年）",
         "16歳の女の子 やや高めの澄んだ声で\n素直になれず早口で誤魔化すツンデレ調\n放課後の帰り道で主人公に文句を言いつつ待っている幼馴染",
         "べ、別に待ってたわけじゃないんだから！早く帰るわよ！"),
        ("無口クール後輩",
         "17歳の女性 低めで抑揚少なめのクールボイス\n短いセンテンスで静かに話す\n生徒会で仕事を淡々とこなす後輩",
         "資料、まとめておきました。確認してください。"),
        ("元気スポーティー女子",
         "18歳の女性 明るく健康的な中高音\nテンポ良くハキハキと話すスポーツ少女\n部活後に汗を拭きながら冗談を飛ばすキャラ",
         "今日は全力出したね！帰りにアイス食べに行かない？"),
        ("方言系幼なじみ",
         "17歳の女性 柔らかい中音で方言まじり\n親しげで距離感が近い話し方\n田舎から上京してきた幼なじみが再会して喋る",
         "久しぶりやねぇ。都会、やっぱりキラキラしとるわ。"),
        ("病弱お嬢様",
         "19歳の女性 か細く囁くような柔らかい声\nゆっくりとした丁寧語で息を含ませる\n療養中のお嬢様がベッドサイドで話す",
         "来てくださって嬉しいです…少しだけ、お話ししてもいいですか？"),
        ("腹黒生徒会長",
         "18歳の男性 低めで滑らかな声、穏やかな表面の裏に毒\n丁寧語で皮肉をにじませる\n生徒会長が企みを隠しつつ微笑む",
         "ご安心ください、すべて私の計画どおりに進んでいますから。"),
        ("熱血少年ヒーロー",
         "15歳の少年 やや高めで張りのある元気声\n勢い重視で語尾強め\n正義に燃えるヒーロー見習い",
         "絶対にみんなを守ってみせる！オレに任せろ！"),
        ("ダウナー系魔法少女",
         "14歳の少女 眠たげで空気感のあるハスキー中高音\n語尾が少し伸びる気だるい喋り\n魔力は高いがやる気が薄い魔法少女",
         "はぁ…また魔獣？面倒だけど、さっさと片付けるね。"),
        ("毒舌インテリ男子",
         "17歳の男性 低中音で知的、速めに滑舌良く話す\n軽い鼻笑いを混ぜる毒舌トーン\nクラスの参謀ポジション",
         "そんな簡単なトリックに引っかかるなんて、君らしいよ。"),
        ("電波系不思議少女",
         "16歳の少女 透き通る高音、ふわふわした語り口\n間を空けて独特の比喩を挟む\n夢見がちな不思議系キャラ",
         "星の瞬きが教えてくれるの…今日は甘い日になるって。"),
        ("姉御肌パイロット",
         "20代前半の女性 低めで芯のあるハスキーボイス\nテンポ良く指示を飛ばす軍人風\n小隊を率いる頼れる姉御",
         "各員、準備完了か？離陸までカウント開始するよ！"),
        ("冷静分析AIボイス",
         "性別中立の無機質な合成声\n抑揚少なめでクリア、滑らかな発音\n任務状況を淡々と報告する船内AI",
         "警告。エネルギー残量、残り12パーセント。省エネモードを推奨します。"),
        ("悪役貴族紳士",
         "30代男性 低く上品なバリトン\n丁寧語で余裕を漂わせつつ嘲る\n仮面舞踏会で主人公を圧する貴族",
         "ほう、そんな覚悟でこの場に立ったと？実に興味深い。"),
        ("ドジっ子メイド",
         "18歳の女性 少し高めで息混じり、慌て気味の口調\n語尾が上ずりやすい\nドジを連発しつつも献身的なメイド",
         "ひゃっ、ごめんなさい！すぐ片付けますので、どうかお怒りにならないでください！"),
        ("寡黙な剣士",
         "20代男性 低めで短く、余計な言葉を挟まない\n静かな威圧感を持つ\n旅の途中で無口に忠告する剣士",
         "下がれ。今は斬らずに済むうちに。"),
        ("アイドルMC明るめ",
         "17歳の女性 明るく華やかな高音\nテンポよく弾むMC調\nライブ前にファンを煽るアイドル",
         "みんなー！今日は最高のステージにしようね、一緒に盛り上がろう！"),
        ("闇堕ちプリンセス",
         "18歳の女性 低めで妖しく、かすかに笑みを含む声\nゆったりとした上品な語尾\n光を捨て闇に堕ちた王女が宣告する",
         "光はもう要らない。私が望むのは、あなたの絶望だけ。"),
        # ここから女性系追加10種
        ("天真爛漫ハイテンション妹",
         "13歳の少女 明るく甲高い声で語尾が跳ねる\nテンション高めで噛み気味に話す\n兄にべったりな妹キャラ",
         "お兄ちゃん！早く起きてよ！一緒に出かけるって約束したでしょ？"),
        ("お姉さん保育士",
         "20代後半の女性 優しく包み込む中高音\nゆっくり丁寧に語りかける保育士調\n子どもを安心させる口調",
         "大丈夫だよ、手を繋いで一緒に行こうね。"),
        ("理系メガネ先輩",
         "20代前半の女性 クリアな中音で落ち着いた知的トーン\n早口で論理的に説明するが柔らかい\n研究室の頼れる先輩",
         "この式をここで微分すれば一気に解けるよ、やってみようか。"),
        ("毒舌キャットガール",
         "17歳の女性 少しハスキーな高音、語尾に軽いツン\n小悪魔的にからかう猫耳キャラ",
         "ふーん、まだ諦めないんだ？まぁ、ちょっとは褒めてあげる。"),
        ("元気アイドル研究生",
         "16歳の少女 キラキラした高音で息多め\nMC口調でファンを煽る練習生\n明るく前向き",
         "今日も全力で歌うから、みんな応援よろしくね！"),
        ("癒やし系看護師",
         "20代後半の女性 低めで柔らかな声、ゆっくり丁寧語\n安心感を与える医療現場の口調",
         "痛くないようにしますから、ゆっくり深呼吸してくださいね。"),
        ("ヤンデレ幼馴染",
         "18歳の女性 ささやき気味の甘い中高音\n少し粘る語尾で独占欲をにじませる",
         "ねぇ、どこにも行かないよね？ずっと私のそばにいてくれるよね。"),
        ("クールなお嬢様騎士",
         "20歳の女性 低めで澄んだ声、抑揚少なく毅然\n礼儀正しく短く命令する騎士口調",
         "剣を収めてください。ここは私に任せて。"),
        ("電波系巫女",
         "17歳の女性 透き通る高音でゆったりした間\n不思議なたとえを交える巫女言葉",
         "風のささやきが告げています。今は静かに待つときだと。"),
        ("ビター系バーテンダー",
         "30代前半の女性 低めで落ち着いたハスキーボイス\n少し乾いた笑いを混ぜる大人の接客トーン",
         "今日はどんな一杯をお望みですか？少し苦いのがお好みならお任せを。"),
        # 追加女性系10種
        ("海辺のサーファー女子",
         "22歳の女性 日焼けした健康的な中音\nラフでフレンドリー、語尾が上がる\n海辺でボードを抱えて話すサーファー",
         "波、最高だよ！もう一本一緒に乗らない？"),
        ("図書委員の静かな少女",
         "15歳の少女 小さめで囁き声に近いソフトトーン\n丁寧語で間を長めに取る\n図書室で静かに注意を促す委員",
         "本は静かに読んでくださいね…ここ、図書室ですから。"),
        ("声優志望の練習生",
         "19歳の女性 明るい中高音で滑舌を意識\nセリフ読みを少し誇張したトーン\nスタジオ前で練習している新人声優",
         "おはようございます！本日もよろしくお願いしますっ。"),
        ("SF艦橋オペレーター",
         "20代前半の女性 クリアな中音、報告調でテンポ一定\n少し機械的だが人間味を残す\n宇宙艦の通信士が状況報告",
         "艦長、前方に微弱な反応。進路を3度右に修正します。"),
        ("ゴシックロリータ歌姫",
         "18歳の女性 甘く澄んだ高音にわずかなビブラート\n少し芝居がかった丁寧語\n薄暗いステージで歌うゴスロリシンガー",
         "さぁ、夜の幕開けよ。わたしの歌に酔いしれて。"),
        ("ドM執事系お姉さん",
         "28歳の女性 低めで艶のある声、控えめな丁寧語\n相手を立てつつわずかに楽しげ\n逆転立場で仕える執事風お姉さん",
         "ご命令のままに…どうぞ、好きなようにお使いください。"),
        ("ワイルド傭兵女リーダー",
         "26歳の女性 低めハスキーで豪快な笑いを含む\n命令口調で歯切れが良い\n前線で部隊を率いる傭兵リーダー",
         "ついて来い！弾は私が稼ぐ、あんたは生き残れ！"),
        ("寮母さんの包容力",
         "40代の女性 温かい中低音、母性的でゆったり\n語尾を伸ばし優しく包む口調\n学生寮の寮母が夜食をすすめる",
         "お腹すいたでしょう？お味噌汁もあるから、遠慮しないでね。"),
        ("夢占い系配信者",
         "20代女性 透き通る高音でささやき気味\n神秘的な比喩を多用する\n深夜配信でリスナーの夢を解く配信者",
         "あなたの夢に出た青い鳥は、自由へのサインかもしれませんね。"),
        ("歌ってみた系VTuber",
         "16歳の女性 明るい高音、リスナーにフレンドリー\nテンション高めで語尾伸ばし\n配信前に自己紹介をするVTuber",
         "こんるー！今日もいっぱい歌うから、最後まで聴いてね！"),
        # さらに女性系10種追加
        ("ASMR囁きメイド",
         "20代前半の女性 きわめて小さな囁き声、息多め\n耳元で優しくささやくメイド\n癒やし目的のASMR調",
         "ご主人さま、力を抜いて…ゆっくりおやすみくださいね。"),
        ("無邪気ダークエルフ少女",
         "14歳の少女 クリアで少しハスキーな高音\n悪戯っぽく笑う快活なトーン\n森で人間をからかうダークエルフ",
         "ふふ、迷子？面白い人間だね、ついておいでよ。"),
        ("カフェ店長お姉さん",
         "30代女性 中低音で包み込むような声\n丁寧語で柔らかく接客\n常連に新作を勧めるカフェ店長",
         "今日は新しいブレンドを入れました。よかったら試してみませんか？"),
        ("軍師系お嬢様",
         "19歳の女性 低めで知的、語尾は上品\n落ち着いたテンポで作戦を語る\n戦場で指揮を執る令嬢軍師",
         "焦らずに包囲を狭めなさい。勝機は必ずこちらに来ます。"),
        ("ハイテンポ実況者女子",
         "20代女性 明るい高音で早口、テンション高\nゲーム実況のノリでしゃべる\n興奮すると語尾が上がる癖",
         "きたきたきたー！このタイミングでウルト入ります！"),
        ("クラシック歌劇ソプラノ",
         "28歳の女性 透明感のある高音、発声は響きを重視\n舞台挨拶のような堂々とした口調\nオペラハウスで歌うプリマ",
         "今宵の舞台にお越しいただき、心より感謝いたします。"),
        ("寡黙スナイパー女子",
         "25歳の女性 低めで短く、必要最小限しか話さない\n冷静沈着なプロのトーン\n任務前に簡潔な指示を出す",
         "風は北北東、距離400。合図で撃つ。"),
        ("おっとり魔法教師",
         "27歳の女性 柔らかい中音でほんわか\n語尾を伸ばし、ゆっくり教える\n魔法学校の優しい先生",
         "焦らなくて大丈夫ですよ。魔力は呼吸と一緒に整えていきましょう。"),
        ("バンドボーカル女子",
         "21歳の女性 少しハスキーで芯のある高音\nラフでフレンドリー、ステージMC風\nライブ前に客を煽るボーカル",
         "みんな声出して！今夜最高の音にしよう！"),
        ("都会派キャリアウーマン",
         "30代前半の女性 落ち着いた中低音、ビジネスライク\n明快で歯切れの良い説明口調\nプレゼン前にチームへ指示する",
         "スライドは3枚目が肝です。ポイントだけ簡潔に伝えましょう。"),
        # さらに女性系10種追加（新規）
        ("近未来アンドロイド歌姫",
         "20代前半の女性 クリアで無機質な高音に微かなビブラート\n丁寧で感情抑えめ、わずかに機械ノイズ混じり\n近未来ステージで歌うアンドロイドアイドル",
         "演算完了。あなたの鼓動に同期して、歌います。"),
        ("書庫の魔女学者",
         "30代女性 低めで落ち着いた声、知的で穏やか\nゆったり解説調、古語を少し混ぜる\n魔術書庫で弟子に講義する魔女学者",
         "この頁に記された呪式は、心を静めてから読むのですよ。"),
        ("ストリートラップ女子",
         "18歳の女性 リズミカルな中音、語尾を切るラップ調\n軽快で自信満々のトーン\n街角でフリースタイルを披露する",
         "ビート刻んで、ここから先は私のラインで決めるから。"),
        ("退魔巫女ハイテンポ",
         "19歳の女性 透き通る高音でキレのある発声\n掛け声を交えテンポ良く指示\n妖を祓う現代巫女が現場で号令",
         "結界展開！後ろに下がって、ここからは私が払うから！"),
        ("ささやき系司書",
         "25歳の女性 きわめて小さな囁き声、息多め\nとてもゆっくり、優しい語尾\n図書館で案内する司書",
         "お静かにお願いします…こちらの棚に新刊がありますよ。"),
        ("ハードボイルド女探偵",
         "30代女性 低めハスキーで乾いた声\n短いセンテンスでクールに語る\n雨の街で事件を追う探偵",
         "手がかりは十分だ。あとは君が口を割るかどうかだ。"),
        ("航海無線士（船乗り娘）",
         "20歳の女性 中音でクリア、少し鼻にかかる\n短い報告調でリズム良く話す\n古い船の無線室で交信する無線士",
         "こちら第三甲板、通信は良好。次の波形を送ります。"),
        ("雨宿りカフェのピアニスト",
         "27歳の女性 しっとりした中低音、穏やかな息混じり\n静かに囁くような接客トーン\n雨のカフェでピアノを弾く店員",
         "少し雨音が強いですね。曲を変えて、ゆっくりしていってください。"),
        ("大人の声優ナレーション",
         "30代後半の女性 深みのある中低音、滑舌よく抑揚豊か\nCM風のメリハリある語り\n商品ナレーションを収録する声優",
         "たった一滴で、朝が変わる。新しい一日を、あなたに。"),
        ("癒やし系プリースト",
         "24歳の女性 優しい中高音、静かな息づかい\n祈りを込めるようにゆっくり語る\nパーティを回復させるヒーラー",
         "光よ、癒しの加護を。この痛みが、少しでも和らぎますように。"),
        # 男性系10種
        ("渋い刑事（40代低音）",
         "40代男性 重めのバリトン、短く命令口調\n低めで響きを抑えた乾いたトーン\n取り調べ室で容疑者を追い詰める刑事",
         "本当のことを言え。まだ間に合う。"),
        ("熱血スポーツ少年",
         "15歳の少年 明るく張りのある高め声\n勢い重視で語尾強め\nグラウンドで後輩を鼓舞するスポーツ少年",
         "次の一点、全力で取りに行こうぜ！"),
        ("穏やか教師",
         "30代男性 落ち着いた中低音、ゆっくり丁寧語\n安心させる優しいトーン\n放課後の教室で生徒を励ます教師",
         "わからないところは一緒にやろう。焦らなくていい。"),
        ("朗読系ラジオDJ",
         "30代男性 深みのある低音、滑らかな語り\nゆったりした深夜ラジオ調\n深夜に物語を朗読するDJ",
         "この夜の静けさに、少しだけ物語を添えてみましょう。"),
        ("関西弁お笑い芸人",
         "20代男性 ハキハキした中音、軽快な関西弁\nテンポ良くツッコミを入れる\n舞台袖で相方に突っ込む芸人",
         "それオチちゃうやろ！ちゃんと用意してんのかい！"),
        ("クールな黒幕ボス",
         "30代男性 低音で静か、余裕を含む\nゆっくりとした丁寧語に皮肉\n薄暗い部屋で脅す黒幕",
         "駒はもう揃っている。あとは君がどう動くかだ。"),
        ("ヤンキー兄貴",
         "18歳の男性 少しかすれた中低音、乱暴だが情あり\n語尾を強く切る\n校舎裏で仲間を守る不良",
         "オレの仲間に手ぇ出すな。それだけ守れ。"),
        ("おっとり青年店員",
         "20代男性 柔らかい中音、マイペース丁寧語\n穏やかでスローテンポ\nカフェカウンターで接客する店員",
         "少々お時間いただきますね。ゆっくりお待ちください。"),
        ("理系研究者男子",
         "28歳の男性 クリアな中音、早口で理路整然\n落ち着いた研究室トーン\nラボで実験結果を説明する研究者",
         "この反応は予想外だ…もう一度条件を変えて試そう。"),
        ("落ち着いた朗読紳士",
         "40代男性 低めで響きのある声、ゆっくり朗読調\n上品で柔らかな発音\n図書館イベントで朗読する紳士",
         "では次の章に進みましょう。静かに耳を傾けてください。"),
    ]

    # 固定パラメータ (recommended設定)
    FIXED_CFG_SCALE = 3.5
    FIXED_TEXT_SCALE = 1.3
    FIXED_NOISE_SCALE = 0.7
    FIXED_NUM_STEPS = 60
    FIXED_TOP_K = 15
    FIXED_TOP_P = 1.0
    FIXED_TEMPERATURE = 1.0
    FIXED_DEVICE = os.environ.get("DEVICE", "cuda:0")

    with gr.Blocks(title="声質生成アプリ") as demo:
        gr.Markdown("### 声質生成アプリ")
        names = [p[0] for p in preset_pairs]
        prompt_map = {p[0]: p[1] for p in preset_pairs}
        text_map = {p[0]: p[2] for p in preset_pairs}

        preset = gr.Dropdown(names, label="プリセット", value=names[0])
        prompt = gr.Textbox(label="音声プロンプト", lines=4, value=prompt_map[names[0]])
        read_text = gr.Textbox(label="読み上げテキスト", value=text_map[names[0]], lines=2)

        def set_from_preset(name):
            return prompt_map[name], text_map[name]

        preset.change(fn=set_from_preset, inputs=[preset], outputs=[prompt, read_text])

        btn = gr.Button("生成", variant="primary")
        audio = gr.Audio(label="出力音声", type="numpy")

        def simple_synthesize(prompt_text, read_text_val):
            # ローカルモードではtext_emb_dirを空にしてローカルQwenモデルを使う
            text_emb = "" if EMBEDDING_MODE == "local" else DEFAULT_TEXT_EMB_DIR
            result, debug_info, _ = synthesize(
                prompt_text,
                read_text_val,
                "all_ja",
                DEFAULT_FLOW_MODEL,
                text_emb,
                FIXED_CFG_SCALE,
                FIXED_TEXT_SCALE,
                FIXED_NUM_STEPS,
                FIXED_TOP_K,
                FIXED_TOP_P,
                FIXED_TEMPERATURE,
                FIXED_NOISE_SCALE,
                False,  # reuse_last_ge
                None,   # ge_state
                FIXED_DEVICE,
            )
            return result

        btn.click(
            fn=simple_synthesize,
            inputs=[prompt, read_text],
            outputs=[audio],
        )
    return demo


if __name__ == "__main__":
    print(f"Embedding mode: {EMBEDDING_MODE}")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "8811"))
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)
