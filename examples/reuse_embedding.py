"""保存した声質で別のテキストを生成する"""

from kizuna_voice_designer import VoiceDesigner

vd = VoiceDesigner(device="cuda:0")

# 保存済みの声質を読み込む
embedding = vd.load_embedding("my_voice.npy")

# 同じ声で別のテキストを生成（何度でも同じ声で生成できる）
texts = [
    "おはようございます。本日も素敵な一日になりますように。",
    "少々お待ちください。ただいまお席をご用意いたします。",
    "ありがとうございました。またのお越しをお待ちしております。",
]

for i, text in enumerate(texts):
    audio, sr, _ = vd.generate(prompt="", text=text, embedding=embedding)
    path = f"same_voice_{i+1}.wav"
    vd.save(path, audio, sr)
    print(f"{path}: {text}")
