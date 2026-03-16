"""パラメータをカスタマイズして生成"""

from kizuna_voice_designer import VoiceDesigner

# ローカルモードで初期化（APIキー不要）
vd = VoiceDesigner(
    device="cuda:0",
    embedding_mode="local",      # ローカルQwenモデル使用
    cfg_scale=5.0,               # CFGスケール（高いほどプロンプトに忠実）
    num_steps=80,                # サンプリングステップ数（多いほど高品質）
    noise_scale=0.6,             # ノイズスケール
)

# 1回目: 声質を生成して保存
audio, sr, embedding = vd.generate(
    prompt="20代後半の女性 月光のように柔らかく耳元で溶けていく声",
    text="今夜だけは、あなたの物語を静かに語らせてください。",
)
vd.save("custom_output.wav", audio, sr)
vd.save_embedding("moon_voice.npy", embedding)

# 2回目: 同じ声で別テキストを生成（embeddingを再利用するので声質が同じ）
audio2, sr, _ = vd.generate(
    prompt="",  # embeddingを渡すのでpromptは不要
    text="星が綺麗ですね。一緒に見ませんか。",
    embedding=embedding,
)
vd.save("custom_output_2.wav", audio2, sr)
