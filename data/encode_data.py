import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from pathlib import Path

from Tokenizer.tokenizer import BPETokenizer

text = Path("data/input.txt").read_text(encoding="utf-8")

tokenizer = BPETokenizer()
tokenizer.load("Tokenizer/bpe_8000.json")

data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

torch.save(data, "data/encoded_data.pt")

print("Encoded data saved!")
print("Number of tokens:", len(data))
