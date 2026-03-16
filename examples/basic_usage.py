"""基本的な使い方 - 1つの音声を生成して保存"""

from kizuna_voice_designer import VoiceDesigner

# 初期化（デフォルト: cuda:0, embedding_mode自動判定）
vd = VoiceDesigner(device="cuda:0")

# 音声生成（audio, sample_rate, embedding の3つが返る）
audio, sr, embedding = vd.generate(
    prompt="30代前半の女性 落ち着きのある透明感ボイスで\n高級ホテルのコンシェルジュ",
    text="こちらのお席でよろしければ、すぐにお飲み物をご用意いたします。",
)

# WAVファイルに保存
vd.save("output.wav", audio, sr)
print(f"Generated: output.wav ({len(audio) / sr:.1f}s)")

# 声質を保存（同じ声で再生成したいとき用）
vd.save_embedding("my_voice.npy", embedding)
print("Saved embedding: my_voice.npy")
