"""Otak Galaxy: model bahasa Transformer yang ditulis dari nol.

Tidak memakai bobot (weights) dari AI lain. Semua yang diketahui Galaxy
berasal dari data yang kamu latihkan sendiri.

Tokenizer: level byte (0-255) ditambah 3 token khusus. Sederhana, bisa
membaca bahasa apa pun dan kode apa pun tanpa perlu kamus kata.
"""
import math
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- tokenizer
EOS = 256      # akhir dokumen / akhir jawaban
USER = 257     # awal pesan pengguna
GALAXY = 258   # awal jawaban Galaxy
VOCAB = 259


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids) -> str:
    return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def chat_tokens(pesan: str, balasan: str | None = None) -> list[int]:
    ids = [USER] + encode(pesan) + [GALAXY]
    if balasan is not None:
        ids += encode(balasan) + [EOS]
    return ids


# ------------------------------------------------------------------- konfig
PRESETS = {
    # nama: (lapisan, kepala, lebar, konteks)  perkiraan jumlah parameter
    "mini": dict(n_layer=4, n_head=4, d_model=128, ctx=256),      # ~0,8 juta (untuk tes)
    "kecil": dict(n_layer=6, n_head=6, d_model=384, ctx=512),     # ~11 juta
    "sedang": dict(n_layer=12, n_head=8, d_model=512, ctx=1024),  # ~38 juta
    "besar": dict(n_layer=16, n_head=12, d_model=768, ctx=1024),  # ~113 juta
}


@dataclass
class Config:
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    ctx: int = 512
    vocab: int = VOCAB
    dropout: float = 0.0


# -------------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(ctx, head_dim, device=None):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(ctx, device=device).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); cos/sin: (T, hd/2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.nh, self.hd = c.n_head, c.d_model // c.n_head
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.dropout = c.dropout

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        pos = 0 if cache is None or cache.get("k") is None else cache["k"].shape[2]
        q = apply_rope(q, cos[pos:pos + T], sin[pos:pos + T])
        k = apply_rope(k, cos[pos:pos + T], sin[pos:pos + T])
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(pos == 0 and T > 1),
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        hidden = int(8 * c.d_model / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.gate = nn.Linear(c.d_model, hidden, bias=False)
        self.up = nn.Linear(c.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n1, self.attn = RMSNorm(c.d_model), Attention(c)
        self.n2, self.mlp = RMSNorm(c.d_model), MLP(c)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, cache)
        return x + self.mlp(self.n2(x))


class Galaxy(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.config = c
        self.embed = nn.Embedding(c.vocab, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.head.weight = self.embed.weight  # berbagi bobot, hemat parameter
        cos, sin = rope_cache(c.ctx, c.d_model // c.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None):
        x = self.embed(idx)
        for i, b in enumerate(self.blocks):
            x = b(x, self.cos, self.sin, None if caches is None else caches[i])
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: list[int], max_new=256, temperature=0.8, top_k=50,
                 n=1, stop_ids=(EOS, USER)):
        """Menghasilkan n kelanjutan sekaligus (batch). Mengembalikan list of list token."""
        self.eval()
        dev = self.cos.device
        max_new = min(max_new, self.config.ctx - 1)
        prompt = prompt[-(self.config.ctx - max_new):]
        idx = torch.tensor([prompt] * n, dtype=torch.long, device=dev)
        caches = [dict() for _ in self.blocks]
        logits, _ = self(idx, caches=caches)
        out = [[] for _ in range(n)]
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        for _ in range(max_new):
            last = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = last.argmax(-1)
            else:
                last = last / temperature
                if top_k:
                    v, _ = torch.topk(last, min(top_k, last.size(-1)))
                    last[last < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(last, -1), 1).squeeze(1)
            for i in range(n):
                if not done[i]:
                    t = int(nxt[i])
                    if t in stop_ids:
                        done[i] = True
                    else:
                        out[i].append(t)
            if bool(done.all()):
                break
            logits, _ = self(nxt.unsqueeze(1), caches=caches)
        return out


# --------------------------------------------------------- simpan / muat
def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save(model: Galaxy, path: str, **meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"config": asdict(model.config), "model": model.state_dict(), "meta": meta}, tmp)
    os.replace(tmp, path)


def load(path: str, device: str = "cpu") -> tuple[Galaxy, dict]:
    ck = torch.load(path, map_location=device, weights_only=True)
    model = Galaxy(Config(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    return model, ck.get("meta", {})
