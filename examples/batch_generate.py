"""複数キャラクターの音声を一括生成"""

from kizuna_voice_designer import VoiceDesigner

vd = VoiceDesigner(device="cuda:0")

characters = [
    {
        "name": "コンシェルジュ",
        "prompt": "30代前半の女性 落ち着きのある透明感ボイスで\n言葉の間に十分な余白を取りつつ丁寧に\n高級ホテルのラウンジで客に寄り添うコンシェルジュ",
        "text": "こちらのお席でよろしければ、すぐにお飲み物をご用意いたします。",
    },
    {
        "name": "ツンデレ幼馴染",
        "prompt": "16歳の女の子 やや高めの澄んだ声で\n素直になれず早口で誤魔化すツンデレ調\n放課後の帰り道で主人公に文句を言いつつ待っている幼馴染",
        "text": "べ、別に待ってたわけじゃないんだから！早く帰るわよ！",
    },
    {
        "name": "渋い刑事",
        "prompt": "40代男性 重めのバリトン、短く命令口調\n低めで響きを抑えた乾いたトーン\n取り調べ室で容疑者を追い詰める刑事",
        "text": "本当のことを言え。まだ間に合う。",
    },
]

for char in characters:
    print(f"Generating: {char['name']}...")
    audio, sr, embedding = vd.generate(prompt=char["prompt"], text=char["text"])
    path = f"{char['name']}.wav"
    vd.save(path, audio, sr)
    print(f"  -> {path} ({len(audio) / sr:.1f}s)")
