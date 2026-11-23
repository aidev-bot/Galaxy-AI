"""Data dan pelatihan untuk Galaxy."""
import math
import os
import random
import time
from array import array

import torch

from .model import EOS, Galaxy


# --------------------------------------------------------------------- data
def write_tokens(path: str, ids) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = array("h")
    arr.extend(ids)
    with open(path, "wb") as f:
        arr.tofile(f)


def read_tokens(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    return torch.frombuffer(buf, dtype=torch.int16)  # hemat RAM; diubah ke long per batch


def docs_to_tensor(docs: list[list[int]], min_len: int) -> torch.Tensor:
    """Gabungkan dokumen (dipisah EOS). Diulang kalau terlalu pendek."""
    flat = []
    for d in docs:
        flat.extend(d)
        if not d or d[-1] != EOS:
            flat.append(EOS)
    if not flat:
        raise ValueError("data kosong")
    while len(flat) < min_len:
        flat = flat + flat
    return torch.tensor(flat, dtype=torch.long)


class Mixer:
    """Mengambil potongan acak dari beberapa sumber data sesuai bobotnya."""

    def __init__(self, sources: list[tuple[torch.Tensor, float]], ctx: int, seed=0):
        self.sources = [(t, w) for t, w in sources if len(t) > ctx + 1]
        if not self.sources:
            raise ValueError("semua sumber data lebih pendek dari konteks model")
        self.weights = [w for _, w in self.sources]
        self.ctx = ctx
        self.rng = random.Random(seed)

    def batch(self, bs: int, device: str):
        xs, ys = [], []
        for _ in range(bs):
            t, _ = self.rng.choices(self.sources, weights=self.weights)[0]
            i = self.rng.randrange(0, len(t) - self.ctx - 1)
            chunk = t[i:i + self.ctx + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return torch.stack(xs).long().to(device), torch.stack(ys).long().to(device)


# ---------------------------------------------------------------- pelatihan
@torch.no_grad()
def eval_loss(model: Galaxy, data: torch.Tensor, device: str, batches=20, bs=8, seed=123):
    model.eval()
    mix = Mixer([(data, 1.0)], model.config.ctx, seed=seed)
    total = 0.0
    for _ in range(batches):
        x, y = mix.batch(bs, device)
        with autocast(device):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / batches


def autocast(device: str):
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


def train(model: Galaxy, mixer: Mixer, device: str, steps: int, lr=3e-4, bs=16, accum=1,
          warmup=None, weight_decay=0.1, val_data=None, eval_every=0, log_every=20,
          save_fn=None, time_limit=None):
    """Melatih model. save_fn(step, val_loss) dipanggil setiap eval_every langkah.

    time_limit (detik): latihan berhenti saat waktu habis; jadwal LR ikut
    menyesuaikan agar tetap turun mulus sampai akhir."""
    model.train()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), fused=(device == "cuda"))
    use_scaler = device == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    warmup = max(1, steps // 50) if warmup is None else warmup
    t0, last_val = time.time(), None
    for step in range(1, steps + 1):
        if step < warmup:
            cur = lr * step / warmup
        else:
            prog = (step - warmup) / max(1, steps - warmup)
            if time_limit:
                prog = max(prog, (time.time() - t0) / time_limit)
            prog = min(prog, 1.0)
            cur = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur
        for _ in range(accum):
            x, y = mixer.batch(bs, device)
            with autocast(device):
                _, loss = model(x, y)
            scaler.scale(loss / accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if log_every and step % log_every == 0:
            dt = time.time() - t0
            tok = step * bs * accum * model.config.ctx
            print(f"  langkah {step}/{steps}  loss {loss.item():.3f}  lr {cur:.2e}  "
                  f"{tok / dt:,.0f} token/detik", flush=True)
        habis = bool(time_limit) and time.time() - t0 >= time_limit
        if eval_every and val_data is not None and (
                step % eval_every == 0 or step == steps or habis):
            last_val = eval_loss(model, val_data, device)
            print(f"  >> val loss {last_val:.3f}", flush=True)
            if save_fn:
                save_fn(step, last_val)
        if habis:
            print(f"  waktu habis di langkah {step}", flush=True)
            break
    model.eval()
    return last_val
