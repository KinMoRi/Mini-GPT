from pathlib import Path
from tokenizer import BPETokenizer

# 원본 텍스트 불러오기
text = Path("data/input.txt").read_text(encoding="utf-8")

# 새로운 Tokenizer 학습
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=8000, min_frequency=2)

# 새로운 BPE 저장
tokenizer.save("Tokenizer/bpe_8000.json")

print("Tokenizer training completed!")
print("Actual vocabulary size:", len(tokenizer.vocab))