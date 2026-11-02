import torch

from Tokenizer.tokenizer import BPETokenizer
from model.gpt import GPT

# ==========================================
# 1. Configuration
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "checkpoints/gpt_checkpoint.pt"

max_new_tokens = 100
temperature = 0.8

# ==========================================
# 2. Load Tokenizer
# ==========================================

tokenizer = BPETokenizer()
tokenizer.load("tokenizer/bpe.json")

# ==========================================
# 3. Load Checkpoint
# ==========================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

vocab_size = checkpoint["vocab_size"]
context_length = checkpoint["context_length"]

model = GPT(
    vocab_size=vocab_size,
    context_length=context_length
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

# ==========================================
# 4. Text Generation
# ==========================================

@torch.no_grad()
def generate(idx, max_new_tokens, temperature=1.0):

    for _ in range(max_new_tokens):

        # Context length 제한
        idx_cond = idx[:, -context_length:]

        # Forward
        logits, loss = model(idx_cond)

        # 마지막 위치의 Logits만 사용
        logits = logits[:, -1, :]

        # Temperature 적용
        logits = logits / temperature

        # 확률분포 계산
        probs = torch.softmax(logits, dim=-1)

        # 다음 토큰 Sampling
        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        # 기존 입력 뒤에 추가
        idx = torch.cat((idx, next_token), dim=1)

    return idx

# ==========================================
# 5. Generate
# ==========================================

prompt = "Poop"

encoded = tokenizer.encode(prompt)

idx = torch.tensor(
    [encoded],
    dtype=torch.long,
    device=device
)

generated = generate(
    idx,
    max_new_tokens=max_new_tokens,
    temperature=temperature
)

output = tokenizer.decode(generated[0].tolist())

print("\n========== GENERATED TEXT ==========\n")
print(output)