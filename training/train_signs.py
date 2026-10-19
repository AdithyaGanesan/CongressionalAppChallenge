"""Train STAT's whole-sign model on ASL Citizen landmarks.

Input: .npz clips from extract_run.py / the Colab notebook (landmarks [T,150], gloss, split, participant).
Output (in --out): stat_sign_model_v2.onnx, sign_classes_v2.json, report.json.

The model's input is exactly what index.html already sends: normalized frames [1, T, 150]
(normalize_frame below is the same as the notebook and the app). Frame-to-frame motion is
computed inside the model, so the app needs no preprocessing changes.

Splits (by signer, so test signers are never seen in training):
  train = official train split + official test split except HOLDOUT signers
  val   = official val split (early stopping)
  test  = every clip of the HOLDOUT test-split signers

Usage: python train_signs.py --data DIR --out DIR [--arch convtx|bilstm] [--epochs 40] [--glosses a.txt b.txt]
"""
import argparse, glob, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------- normalization: identical to CAP.ipynb normalize_frame and index.html ----------------
POSE_SUBSET_ORDER = [11, 12, 13, 14, 15, 16, 23, 24]
L_SH, R_SH = POSE_SUBSET_ORDER.index(11), POSE_SUBSET_ORDER.index(12)
POSE_MIRROR_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]

def normalize_frame(vec):
    out = vec.copy()
    pose = out[126:150].reshape(8, 3)
    center = (pose[L_SH] + pose[R_SH]) / 2.0
    scale = np.linalg.norm(pose[L_SH] - pose[R_SH])
    if scale < 1e-6:
        scale = 1.0
    lh, rh = out[0:63].reshape(21, 3), out[63:126].reshape(21, 3)
    if np.abs(lh).sum() > 0:
        lh = (lh - center) / scale
    if np.abs(rh).sum() > 0:
        rh = (rh - center) / scale
    pose = (pose - center) / scale
    out[0:63], out[63:126], out[126:150] = lh.ravel(), rh.ravel(), pose.ravel()
    return out

# ---------------- augmentation (same three as the notebook, plus a random time crop) ----------------
def mirror_clip(a):
    out = a.copy()
    lh = out[:, 0:63].reshape(-1, 21, 3).copy(); rh = out[:, 63:126].reshape(-1, 21, 3).copy()
    pose = out[:, 126:150].reshape(-1, 8, 3).copy()
    for p in (lh, rh, pose):
        nz = np.abs(p).sum(axis=(1, 2)) > 0
        p[nz, :, 0] *= -1
    out[:, 0:63], out[:, 63:126] = rh.reshape(len(a), -1), lh.reshape(len(a), -1)
    newp = pose.copy()
    for i, j in POSE_MIRROR_PAIRS:
        newp[:, i], newp[:, j] = pose[:, j], pose[:, i]
    out[:, 126:150] = newp.reshape(len(a), -1)
    return out

def jitter_clip(a, rot=0.15, scl=0.1, trans=0.05):
    ang = np.random.uniform(-rot, rot); s = 1 + np.random.uniform(-scl, scl); tx, ty = np.random.uniform(-trans, trans, 2)
    c, sn = np.cos(ang), np.sin(ang)
    out = a.copy()
    for start, n in ((0, 21), (63, 21), (126, 8)):
        p = out[:, start:start + n * 3].reshape(len(a), n, 3)
        nz = (np.abs(p).sum(axis=(1, 2)) > 0)[:, None]
        x, y = p[..., 0].copy(), p[..., 1].copy()
        p[..., 0] = np.where(nz, (x * c - y * sn) * s + tx, x)
        p[..., 1] = np.where(nz, (x * sn + y * c) * s + ty, y)
    return out

def time_warp(a, lo=0.8, hi=1.2):
    T = len(a); nT = max(5, int(round(T * np.random.uniform(lo, hi))))
    idx = np.linspace(0, T - 1, nT)
    i0 = np.floor(idx).astype(int); i1 = np.minimum(i0 + 1, T - 1); w = (idx - i0)[:, None]
    return (a[i0] * (1 - w) + a[i1] * w).astype(np.float32)

MAX_LEN = 64

def crop(a, train):
    if len(a) <= MAX_LEN:
        return a
    s = np.random.randint(0, len(a) - MAX_LEN + 1) if train else (len(a) - MAX_LEN) // 2
    return a[s:s + MAX_LEN]

# ---------------- data ----------------
def load(data_dir, keep_glosses=None):
    recs = []
    for p in sorted(glob.glob(f"{data_dir}/*.npz")):
        if ".tmp" in p:
            continue
        try:
            d = np.load(p)
            g = str(d["gloss"])
            if keep_glosses is not None and g not in keep_glosses:
                continue
            lm = d["landmarks"].astype(np.float32)
            if len(lm) < 3:
                continue
            recs.append({"x": np.stack([normalize_frame(f) for f in lm]), "gloss": g,
                         "split": str(d["split"]), "participant": str(d["participant"])})
        except Exception as e:
            print("skip", p, e)
    return recs

class Clips(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        x, y = self.items[i]
        if self.train:
            x = jitter_clip(x); x = time_warp(x)
            if np.random.rand() < 0.5:
                x = mirror_clip(x)
        x = crop(x, self.train)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), y

def collate(batch):
    L = max(len(x) for x, _ in batch)
    X = torch.zeros(len(batch), L, 150); lens = torch.tensor([len(x) for x, _ in batch])
    for i, (x, _) in enumerate(batch):
        X[i, :len(x)] = x
    return X, lens, torch.tensor([y for _, y in batch])

# ---------------- models ----------------
def masked_mean(h, lens):
    if lens is None:
        return h.mean(1)
    m = (torch.arange(h.size(1), device=h.device)[None, :] < lens[:, None]).float().unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)

class BiLSTM(nn.Module):
    """The current STAT architecture (hidden 64), for comparison."""
    def __init__(self, n_cls, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(150, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden, n_cls))
    def forward(self, x, lens=None):
        if lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.lstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        else:
            h, _ = self.lstm(x)
        return self.fc(masked_mean(h, lens))

class Attention(nn.Module):
    """Plain multi-head self-attention written out by hand so it exports cleanly to ONNX."""
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
    def forward(self, x, key_mask=None):
        # unflatten/flatten instead of view(B, T, ...) so the exported graph keeps T dynamic
        qkv = self.qkv(x).unflatten(-1, (3, self.h, self.dk)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        if key_mask is not None:
            att = att.masked_fill(~key_mask[:, None, None, :], -1e4)
        return self.o((att.softmax(-1) @ v).transpose(1, 2).flatten(2))

class Block(nn.Module):
    def __init__(self, d, heads, k=5, drop=0.2):
        super().__init__()
        self.n1, self.conv = nn.LayerNorm(d), nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.pw = nn.Linear(d, d)
        self.n2, self.att = nn.LayerNorm(d), Attention(d, heads)
        self.n3, self.ff = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(drop)
    def forward(self, x, key_mask=None):
        h = self.n1(x)
        if key_mask is not None:
            h = h * key_mask.unsqueeze(-1).float()
        x = x + self.drop(self.pw(F.gelu(self.conv(h.transpose(1, 2)).transpose(1, 2))))
        x = x + self.drop(self.att(self.n2(x), key_mask))
        return x + self.drop(self.ff(self.n3(x)))

class ConvTx(nn.Module):
    """1D-conv + transformer over landmark frames, with motion (frame differences) added inside."""
    def __init__(self, n_cls, d=192, heads=4, layers=3, max_len=256):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(300, d), nn.LayerNorm(d), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm, self.drop, self.head = nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, n_cls)
    def forward(self, x, lens=None):
        dx = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], 1)
        h = self.inp(torch.cat([x, dx], -1)) + self.pos.narrow(1, 0, x.size(1))
        key_mask = None if lens is None else torch.arange(x.size(1), device=x.device)[None, :] < lens[:, None]
        for b in self.blocks:
            h = b(h, key_mask)
        return self.head(self.drop(masked_mean(self.norm(h), lens)))

class Export(nn.Module):
    """Batch of one, no padding: what the browser runs. Long clips are cut to the positional table."""
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, landmarks):
        return self.m(landmarks)

# ---------------- training ----------------
def evaluate(model, loader):
    model.eval(); top1 = top5 = n = 0; preds = []
    with torch.no_grad():
        for X, lens, y in loader:
            lo = model(X, lens)
            t5 = lo.topk(5, 1).indices
            top1 += (t5[:, 0] == y).sum().item(); top5 += (t5 == y[:, None]).any(1).sum().item(); n += len(y)
            preds += list(zip(y.tolist(), t5[:, 0].tolist()))
    return top1 / max(n, 1), top5 / max(n, 1), preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="convtx", choices=["convtx", "bilstm"])
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--glosses", nargs="*", help="text files listing the glosses to train on (default: all found)")
    ap.add_argument("--holdout", type=int, default=3, help="test-split signers held out for the final test")
    ap.add_argument("--min_train", type=int, default=8, help="drop glosses with fewer training clips")
    ap.add_argument("--old_model", default=None, help="old ONNX model + classes json to score on the same test clips")
    ap.add_argument("--old_classes", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many clips")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); random.seed(0); np.random.seed(0); torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)

    keep = None
    if a.glosses:
        keep = set()
        for f in a.glosses:
            keep |= {g.strip() for g in open(f) if g.strip()}
    t0 = time.time(); recs = load(a.data, keep)
    if a.limit:
        random.shuffle(recs); recs = recs[:a.limit]
    print(f"loaded {len(recs)} clips in {time.time() - t0:.0f}s")

    test_signers = sorted({r["participant"] for r in recs if r["split"] == "test"})
    holdout = set(test_signers[:a.holdout])
    for r in recs:
        r["role"] = "test" if r["participant"] in holdout else ("val" if r["split"] == "val" else "train")
    counts = {}
    for r in recs:
        if r["role"] == "train":
            counts[r["gloss"]] = counts.get(r["gloss"], 0) + 1
    classes = sorted(g for g, c in counts.items() if c >= a.min_train)
    idx = {g: i for i, g in enumerate(classes)}
    split = {k: [(r["x"], idx[r["gloss"]]) for r in recs if r["role"] == k and r["gloss"] in idx] for k in ("train", "val", "test")}
    print(f"{len(classes)} classes; train {len(split['train'])}, val {len(split['val'])}, test {len(split['test'])} "
          f"(held-out signers: {sorted(holdout)})")

    dl = lambda k, tr: DataLoader(Clips(split[k], tr), batch_size=a.batch, shuffle=tr, collate_fn=collate, num_workers=2 if tr else 0)
    tr_l, va_l, te_l = dl("train", True), dl("val", False), dl("test", False)

    model = ConvTx(len(classes)) if a.arch == "convtx" else BiLSTM(len(classes))
    print(f"{a.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05 if a.arch == "convtx" else 1e-4)
    steps = a.epochs * len(tr_l)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    best, since, ckpt = -1, 0, f"{a.out}/best_{a.arch}.pt"
    state_path, start = f"{a.out}/state_{a.arch}.pt", 1
    st = torch.load(state_path, weights_only=False) if os.path.exists(state_path) else None
    if st and st.get("classes") != classes:
        print("saved state is for a different sign list; starting fresh"); st = None
    if st:  # resume after an interruption, from the last finished epoch
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        best, since, start = st["best"], st["since"], st["epoch"] + 1
        print(f"resuming after epoch {st['epoch']} (best val {best:.1%})")
    for ep in range(start, a.epochs + 1):
        model.train(); t = time.time(); tl = tn = 0
        for X, lens, y in tr_l:
            loss = F.cross_entropy(model(X, lens), y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item() * len(y); tn += len(y)
        v1, v5, _ = evaluate(model, va_l)
        mark = ""
        if v1 > best:
            best, since, mark = v1, 0, "  <- saved"; torch.save(model.state_dict(), ckpt)
        else:
            since += 1
        print(f"epoch {ep:2d} loss {tl / tn:.3f} | val top1 {v1:.1%} top5 {v5:.1%} | {time.time() - t:.0f}s{mark}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "since": since, "classes": classes}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        if since >= a.patience:
            print("early stop"); break

    model.load_state_dict(torch.load(ckpt))
    t1, t5, preds = evaluate(model, te_l)
    print(f"\nTEST (held-out signers): top1 {t1:.1%} top5 {t5:.1%}")
    report = {"arch": a.arch, "classes": len(classes), "train_clips": len(split["train"]), "val_clips": len(split["val"]),
              "test_clips": len(split["test"]), "holdout_signers": sorted(holdout), "val_top1": best, "test_top1": t1, "test_top5": t5}

    # score the old model on the same held-out clips, restricted to the signs it knows
    if a.old_model and a.old_classes:
        import onnxruntime as ort
        old_cls = json.load(open(a.old_classes)); old_idx = {g: i for i, g in enumerate(old_cls)}
        sess = ort.InferenceSession(a.old_model)
        items = [(x, classes[y]) for x, y in split["test"] if classes[y] in old_idx]
        o1 = n1 = 0
        for x, g in items:
            lo = sess.run(["logits"], {"landmarks": x[None].astype(np.float32)})[0][0]
            o1 += int(np.argmax(lo) == old_idx[g])
        sub = [(y, p) for y, p in preds if classes[y] in old_idx]
        n1 = sum(y == p for y, p in sub)
        report["old_model_on_its_signs"] = {"clips": len(items), "old_top1": o1 / max(len(items), 1),
                                            "new_top1_same_clips": n1 / max(len(sub), 1)}
        print(f"same held-out clips, old model's {len(old_cls)} signs: old {o1 / max(len(items), 1):.1%} vs new {n1 / max(len(sub), 1):.1%} "
              f"(new model chooses among {len(classes)} signs)")

    # export: batch of one, dynamic length, opset 17, legacy exporter (dynamo=False) like the notebook
    exp = Export(model).eval()
    onnx_path = f"{a.out}/stat_sign_model_v2.onnx"
    torch.onnx.export(exp, torch.randn(1, 30, 150), onnx_path, input_names=["landmarks"], output_names=["logits"],
                      dynamic_axes={"landmarks": {1: "sequence_length"}}, opset_version=17, dynamo=False)
    exp.eval()  # torch.onnx.export restores the caller's mode; make sure dropout is off for the check
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    worst = 0
    for T in (5, 23, 64):
        x = split["test"][0][0][:T] if split["test"] else np.random.randn(T, 150).astype(np.float32)
        with torch.no_grad():
            ref = exp(torch.from_numpy(np.ascontiguousarray(x))[None]).numpy()
        worst = max(worst, float(np.abs(ref - sess.run(["logits"], {"landmarks": x[None].astype(np.float32)})[0]).max()))
    print(f"ONNX export {os.path.getsize(onnx_path) / 1e6:.2f} MB, max diff vs PyTorch {worst:.2e}")
    report["onnx_mb"] = os.path.getsize(onnx_path) / 1e6; report["onnx_max_diff"] = worst
    json.dump(classes, open(f"{a.out}/sign_classes_v2.json", "w"))
    json.dump(report, open(f"{a.out}/report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))

if __name__ == "__main__":
    main()
