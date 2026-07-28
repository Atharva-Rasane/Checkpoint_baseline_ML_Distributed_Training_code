import torch
from torch import nn
from torch.nn import functional as F


class TinyGPT(nn.Module):
    def __init__(self, vocab_size=50304, hidden_size=256, num_layers=4, num_heads=4, max_seq_len=256):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.blocks(x, mask=causal_mask)
        logits = self.lm_head(self.norm(x))

        if labels is None:
            return {"logits": logits}

        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
        )
        return {"loss": loss, "logits": logits}


def build_model(vocab_size=50304, seq_len=256, **_):
    return TinyGPT(vocab_size=vocab_size, max_seq_len=seq_len)
