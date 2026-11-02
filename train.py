
import torch
from pathlib import Path

from Tokenizer.tokenizer import BPETokenizer
from model.gpt import GPT


# ==========================================
# 1. Hyperparameters
# ==========================================

batch_size = 16
context_length = 128

max_iters = 20
eval_interval = 10
eval_iters = 5

learning_rate = 3e-4
weight_decay = 0.1


# ==========================================
# 2. Device
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ==========================================
# 3. Load Dataset
# ==========================================

data = torch.load(
    "data/encoded_data.pt",
    weights_only=True
)

vocab_size = 8001

split_idx = int(0.9 * len(data))

train_data = data[:split_idx]
val_data = data[split_idx:]


def get_batch(split):
    source = train_data if split == "train" else val_data

    ix = torch.randint(
        len(source) - context_length,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + context_length]
        for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + context_length + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


# ==========================================
# 4. Initialize GPT
# ==========================================

model = GPT(
    vocab_size=vocab_size,
    context_length=context_length
).to(device)


# ==========================================
# 5. Optimizer
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

# ==========================================
# 6. Load Checkpoint
# ==========================================

checkpoint_path = Path("checkpoints/gpt_checkpoint.pt")

start_step = 0

if checkpoint_path.exists():

    print("Loading checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    # 현재 설정한 Hyperparameters 유지
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
        param_group["weight_decay"] = weight_decay

    start_step = checkpoint.get("step", 0)

    print(f"Checkpoint loaded! Starting from step {start_step}")

else:
    print("No checkpoint found. Starting from scratch.")

# ==========================================
# 7. Evaluate Function
# ==========================================

@torch.no_grad()
def estimate_loss():
    model.eval()

    results = {}

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# ==========================================
# 8. Training Loop
# ==========================================

model.train()

for step in range(start_step, start_step + max_iters):

    # Periodically evaluate
    if step % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"Step {step}: "
            f"Train Loss {losses['train']:.4f}, "
            f"Val Loss {losses['val']:.4f}"
        )

    # Get a batch
    x, y = get_batch("train")

    # Forward
    logits, loss = model(x, y)

    # Reset gradients
    optimizer.zero_grad(set_to_none=True)

    # Backward
    loss.backward()

    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )

    # Update parameters
    optimizer.step()


# ==========================================
# 9. Save Checkpoint
# ==========================================

Path("checkpoints").mkdir(exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_size": vocab_size,
        "context_length": context_length,
        "step": start_step + max_iters,
    },
    "checkpoints/gpt_checkpoint.pt"
)

print("Training finished!")
print(f"Total steps: {start_step + max_iters}")
print("Checkpoint saved.")
