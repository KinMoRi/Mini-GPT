
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q, K, V를 한 번의 Linear 연산으로 계산
        self.qkv = nn.Linear(d_model, 3 * d_model)

        self.proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal Mask: 대각선 아래쪽만 허용
        mask = torch.tril(
            torch.ones(context_length, context_length)
        )

        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, context_length, context_length)
        )

    def forward(self, x):
        B, T, C = x.shape

        # [B, T, 3 * C]
        qkv = self.qkv(x)

        # [B, T, 3, n_heads, head_dim]
        qkv = qkv.view(
            B, T, 3, self.n_heads, self.head_dim
        )

        # [3, B, n_heads, T, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        # Attention Score
        # [B, n_heads, T, T]
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 미래 위치에 해당하는 Score를 -inf로 설정
        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Attention Weight
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Weighted Sum of Values
        # [B, n_heads, T, head_dim]
        out = weights @ v

        # Multi-head 결과를 하나로 합침
        # [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out)
        out = self.resid_dropout(out)

        return out


class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model, n_heads, context_length, dropout
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        # Pre-LayerNorm + Residual Connection
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=384,
        n_heads=6,
        n_layers=6,
        context_length=128,
        dropout=0.1
    ):
        super().__init__()

        self.context_length = context_length

        # Token Embedding
        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )

        # Positional Embedding
        self.position_embedding = nn.Embedding(
            context_length, d_model
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model, n_heads, context_length, dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)

        # Language Modeling Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Token Embedding과 Output Weight 공유
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        assert T <= self.context_length

        # Position IDs: [0, 1, ..., T-1]
        positions = torch.arange(T, device=idx.device)

        # Token + Position Embedding
        x = self.token_embedding(idx) + self.position_embedding(positions)

        x = self.dropout(x)

        # Transformer
        x = self.blocks(x)
        x = self.ln_f(x)

        # [B, T, vocab_size]
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss
