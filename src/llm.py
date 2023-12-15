import torch
import torch.nn as nn

from torch.nn import functional as F

torch.manual_seed(1337)


class Head(nn.Module):
    def __init__(
        self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.2
    ) -> None:
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.tensor):
        B, T, C = X.shape

        k = self.key(X)
        q = self.query(X)
        v = self.value(X)

        weights = q @ k.transpose(-2, -1) * C**-0.5

        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)

        weights = self.dropout(weights)  # (B, T, T)

        return weights @ v  # (B, T, T) @ (B, T, C) ~ (B, T, C)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        n_heads: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.tensor):
        out = torch.cat([h(X) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class SquareFeedForward(nn.Module):
    """This module expands the input layer to 4x the input size, performs a ReLU
    non-linearity, and then performs another linear layer back to the original size"""

    def __init__(self, size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.ReLU(),
            nn.Linear(4 * size, size),
            nn.Dropout(dropout),
        )

    def forward(self, X: torch.tensor):
        return self.net(X)


class Block(nn.Module):
    def __init__(self, n_embd: int, block_size: int, n_heads: int) -> None:
        super().__init__()
        # if n_embd is 16 and we have 4 heads, the head size should be 4 so that we
        # have 4 heads of size 4 and we have the same size out as we have in
        head_size = n_embd // n_heads

        if head_size * n_heads != n_embd:
            raise Exception(
                f"No 'head_size' exists that can support n_embd = {n_embd} and n_heads = {n_heads}"
            )

        self.mha = MultiHeadAttention(n_embd, head_size, block_size, n_heads)
        self.ffwd = SquareFeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, X: torch.tensor):
        ln1 = self.ln1(X)
        mha = self.mha(ln1)
        X = X + mha

        ln2 = self.ln2(X)
        ffwd = self.ffwd(ln2)
        X = X + ffwd
        return X


class AttentionIsAllYouNeed(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_heads: int,
        n_layers: int,
        device: str,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.device = device

        self.tok_table = nn.Embedding(vocab_size, n_embd)
        self.pos_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, block_size, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, X: torch.Tensor, y: torch.Tensor = None) -> list[torch.Tensor]:
        B, T = X.shape  # X is (B, T) with integer values that need to be tokenized

        tok_emb = self.tok_table(X)  # (B, T, C) tokenized inputs
        pos_in = torch.arange(T, device=self.device)  # (T,) inputs [0, 1, ...]
        pos_emb = self.pos_table(pos_in)  # (T, C) position inputs

        X = tok_emb + pos_emb
        X = self.blocks(X)
        X = self.ln_f(X)

        logits = self.lm_head(X)

        B, T, C = logits.shape

        logits = logits.view(B * T, C)

        if y is None:
            return logits, None

        y = y.view(B * T)

        loss = F.cross_entropy(logits, y)

        return logits, loss

    def generate(self, X: torch.tensor, N: int):
        for _ in range(N):
            logits, _ = self.forward(X[:, -self.block_size :])
            probas = F.softmax(logits[-1, :], dim=-1)
            pred = torch.multinomial(probas, num_samples=1).reshape(1, -1)
            X = torch.cat((X, pred), dim=1)
        return X
