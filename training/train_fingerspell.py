"""Train STAT's fingerspelling model v2: a small CTC sequence model on Google's ASL Fingerspelling data.

Data: Kaggle "Google - American Sign Language Fingerspelling Recognition" (asl-fingerspelling, CC BY 4.0):
MediaPipe Holistic landmarks of 67k phrases (addresses, phone numbers, URLs, names) fingerspelled by
94 Deaf signers on phone selfie cameras, plus the supplemental set (53k everyday English sentences, 72 of
the same and other signers). fs_extract.py keeps only both hands' 21 points (x/y/z).

Input to the model is exactly what index.html computes per frame for the dominant hand (domHand63):
x*W/H, y, z*W/H from MediaPipe Tasks HandLandmarker, with frames resampled to 15 fps and frames without
the dominant hand dropped. Normalization (wrist-centered hand shape, wrist path, both divided by the
average hand size over the clip, plus frame-to-frame motion) happens inside the model, so the app only
sends [1, T, 63]. Output: [1, 2T, 60] per-step log-probabilities (index 0 = CTC blank, then the 59
characters of character_to_prediction_index.json); every input frame emits 2 CTC steps so fast
spelling still fits.

Domain gaps handled in training:
  - frame rate: Kaggle clips are ~30 fps; we take every 2nd frame (15 fps, same as the app) with a random
    speed change of 0.7-1.4x
  - aspect ratio: Kaggle videos are portrait phone video; x is multiplied by 9/16 and then by a random
    0.75-1.4 (covers 4:3 and 16:9, portrait); rotation and noise
  - either hand dominant: left-dominant clips are mirrored so every training clip looks like one hand;
    the app runs both the clip and its mirror image and keeps the one with more non-blank CTC steps
    ("auto-mirror" below; on held-out signers it picks the right hand 95% of the time and scores
    the same as knowing the hand, better than picking by average confidence)

Splits (by signer, participant_id, over both sets): 10 random signers are held out for test, 5 for
validation (checkpoint choice), the rest train. Metrics on held-out test signers, greedy CTC decoding:
  - score = competition's normalized Levenshtein, (N - edit distance) / N summed over phrases
  - CER = edit distance / N; letters-only CER compares against the old per-frame letter model
Results: training/report_fs_v2.json.

Usage:
  python train_fingerspell.py train   --npz DIR --csv train.csv,supplemental_metadata.csv --work DIR [--hours 1.8]
  python train_fingerspell.py test    --npz DIR --csv train.csv,supplemental_metadata.csv --work DIR   (writes report)
  python train_fingerspell.py export  --work DIR --out ../models
  python train_fingerspell.py cases   --npz DIR --csv ... --work DIR --cases FILE (browser test input)
  python train_fingerspell.py old     --npz DIR --csv train.csv,supplemental_metadata.csv --work DIR   (old model on the same test set)
"""
import argparse, glob, json, math, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

CHARS = " !#$%&'()*+,-./0123456789:;=?@[_abcdefghijklmnopqrstuvwxyz~"
assert len(CHARS) == 59
C2I = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = blank
NCLS = len(CHARS) + 1
KAGGLE_ASPECT = 9 / 16  # width / height of the (portrait) training videos
SEED = 0
N_TEST_SIGNERS, N_VAL_SIGNERS = 10, 5


# ---------------- data ----------------
def load_data(npz_dir, csv):
    """csv: comma-separated metadata files (train.csv,supplemental_metadata.csv)."""
    import pandas as pd
    meta = pd.concat([pd.read_csv(c) for c in csv.split(",")]).set_index("sequence_id")
    seqs, phrases, pids, src = [], [], [], []
    for f in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
        z = np.load(f)
        kind = "supp" if "supplemental" in os.path.basename(f) else "train"
        for sid in z.files:
            a = z[sid]  # [T, 126] float16: left hand xyz*21, right hand xyz*21
            lh, rh = a[:, :63], a[:, 63:]
            nl, nr = (~np.isnan(lh[:, 0])).sum(), (~np.isnan(rh[:, 0])).sum()
            dom = (lh if nl > nr else rh).astype(np.float32).copy()
            if nl > nr:  # mirror the left hand so every clip has the same chirality
                dom[:, 0::3] = 1.0 - dom[:, 0::3]
            r = meta.loc[int(sid)]
            seqs.append(dom); phrases.append(r.phrase); pids.append(int(r.participant_id)); src.append(kind)
    LOADED_SRC[:] = src
    ALL_SIGNERS[:] = sorted(set(meta.participant_id.astype(int)))
    return seqs, phrases, np.array(pids)


ALL_SIGNERS = []  # every participant_id in the metadata, so the split doesn't depend on what is downloaded
LOADED_SRC = []  # "train" (addresses, phone numbers, URLs) or "supp" (English sentences) per loaded clip


def split_signers(pids):
    u = list(ALL_SIGNERS)
    rng = random.Random(SEED); rng.shuffle(u)
    test, val = set(u[:N_TEST_SIGNERS]), set(u[N_TEST_SIGNERS:N_TEST_SIGNERS + N_VAL_SIGNERS])
    sp = np.array(["test" if p in test else "val" if p in val else "train" for p in pids])
    return sp, sorted(test), sorted(val)


def to_app_frames(dom, train, rng=np.random):
    """Full-rate dominant-hand clip (NaN = missing) -> what the app would send: 15 fps, present frames only."""
    T = len(dom)
    if train:
        step = 2 * rng.uniform(0.7, 1.4)
        idx = np.floor(np.arange(rng.uniform(0, step), T, step)).astype(int)
    else:
        idx = np.arange(0, T, 2)
    x = dom[idx]
    x = x[~np.isnan(x).any(1)]
    if train and len(x) > 8:
        x = x[rng.rand(len(x)) > 0.1]  # dropped detections
    x = x.reshape(-1, 21, 3).copy()
    if len(x) == 0:
        return np.zeros((0, 63), np.float32)
    ax = KAGGLE_ASPECT * (rng.uniform(0.75, 1.4) if train else 1.0)
    x[..., 0] *= ax; x[..., 2] *= ax  # same as index.html: x * W/H, z * W/H
    if train:
        ang = rng.uniform(-0.25, 0.25); c, s = math.cos(ang), math.sin(ang)
        cx, cy = x[..., 0].mean(), x[..., 1].mean()
        dx, dy = x[..., 0] - cx, x[..., 1] - cy
        x[..., 0], x[..., 1] = cx + c * dx - s * dy, cy + s * dx + c * dy
        x[..., 2] *= rng.uniform(0.7, 1.3)
        x += rng.normal(0, 0.002, x.shape)
    return x.reshape(len(x), 63).astype(np.float32)


def encode(phrase):
    return [C2I[c] for c in phrase if c in C2I]


def ctc_fits(n_frames, lab):
    rep = sum(1 for a, b in zip(lab, lab[1:]) if a == b)
    return 2 * n_frames >= len(lab) + rep + 1


# ---------------- model ----------------
class Features(nn.Module):
    """[B, T, 63] app landmarks -> [B, T, 132] normalized shape, wrist path and their motion."""
    def forward(self, x, mask):
        B, T, _ = x.shape
        p = x.view(B, T, 21, 3)
        m = mask.float().unsqueeze(-1)  # [B, T, 1]
        n = m.sum(1).clamp(min=1)  # [B, 1]
        size = torch.sqrt(((p[:, :, 9, :2] - p[:, :, 0, :2]) ** 2).sum(-1) + 1e-8)  # [B, T]
        scale = ((size * m[..., 0]).sum(1, keepdim=True) / n).clamp(min=1e-4)  # [B, 1]
        wrist = p[:, :, 0, :]  # [B, T, 3]
        shape = (p - wrist.unsqueeze(2)).reshape(B, T, 63) / scale.unsqueeze(-1)
        wmean = (wrist * m).sum(1, keepdim=True) / n.unsqueeze(-1)
        path = (wrist - wmean) / scale.unsqueeze(-1)
        f = torch.cat([shape, path], -1) * m
        vel = torch.cat([torch.zeros_like(f[:, :1]), f[:, 1:] - f[:, :-1]], 1)
        vel = vel * m * torch.cat([torch.zeros_like(m[:, :1]), m[:, :-1]], 1)
        return torch.cat([f, vel], -1)


class ConvBlock(nn.Module):
    def __init__(self, d, k=11, drop=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.ln2 = nn.LayerNorm(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        h = F.glu(self.pw1(self.ln(x)), -1) * mask.unsqueeze(-1).float()
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        h = self.pw2(F.silu(self.ln2(h)))
        return x + self.drop(h)


class AttnBlock(nn.Module):
    def __init__(self, d, heads=4, drop=0.1):
        super().__init__()
        self.h = heads
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h)
        a = a + (~mask)[:, None, None, :].float() * -1e4
        h = (a.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.o(h))
        return x + self.drop(self.ff(self.ln2(x)))


class FSModel(nn.Module):
    def __init__(self, d=192, n_stages=3, upsample=2):
        super().__init__()
        self.feat = Features()
        self.inp = nn.Sequential(nn.Linear(132, d), nn.LayerNorm(d))
        self.blocks = nn.ModuleList()
        for _ in range(n_stages):
            self.blocks.extend([ConvBlock(d), ConvBlock(d), AttnBlock(d)])
        self.up = upsample
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.1), nn.Linear(d, NCLS * upsample))

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        h = self.inp(self.feat(x, mask))
        for b in self.blocks:
            h = b(h, mask)
        B, T, _ = h.shape
        return self.head(h).view(B, T * self.up, NCLS).log_softmax(-1)


class Exported(nn.Module):
    """ONNX wrapper: [1, T, 63] -> [1, 2T, 60] log-probs."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, landmarks):
        return self.m(landmarks)


# ---------------- decoding and metrics ----------------
def greedy(lp):
    """lp: [S, C] log-probs -> (string, per-character top-3 [(char, prob)])."""
    best = lp.argmax(-1)
    out, cands, prev = [], [], 0
    for t, b in enumerate(best.tolist()):
        if b != 0 and b != prev:
            out.append(CHARS[b - 1])
            p = lp[t].exp()
            v, i = p.topk(4)
            cands.append([(CHARS[j - 1], float(q)) for q, j in zip(v.tolist(), i.tolist()) if j != 0][:3])
        prev = b
    return "".join(out), cands


def lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def letters(s):
    return "".join(c for c in s if "a" <= c <= "z")


def score(preds, refs):
    N = sum(len(r) for r in refs)
    D = sum(lev(p, r) for p, r in zip(preds, refs))
    return {"score": round((N - D) / N, 4), "cer": round(D / N, 4), "n_phrases": len(refs), "n_chars": N}


@torch.no_grad()
def predict(model, clips, auto_mirror=False):
    model.eval()
    out = []
    for x in clips:
        if len(x) == 0:
            out.append(""); continue
        t = torch.from_numpy(x)[None]
        lp = model(t)[0]
        if auto_mirror:
            m = t.clone().view(1, -1, 21, 3); m[..., 0] = -m[..., 0]
            lp2 = model(m.view(1, -1, 63))[0]
            if (lp2.argmax(-1) != 0).sum() > (lp.argmax(-1) != 0).sum():  # the one that reads more characters
                lp = lp2
        out.append(greedy(lp)[0])
    return out


# ---------------- training ----------------
def batches(idx, lens, bs_frames, rng):
    idx = sorted(idx, key=lambda i: lens[i] + rng.uniform(-8, 8))
    out, cur, mx = [], [], 0
    for i in idx:
        mx2 = max(mx, lens[i])
        if cur and mx2 * (len(cur) + 1) > bs_frames:
            out.append(cur); cur, mx2 = [], lens[i]
        cur.append(i); mx = mx2
    if cur: out.append(cur)
    rng.shuffle(out)
    return out


def collate(xs, labs):
    T = max(len(x) for x in xs)
    X = torch.zeros(len(xs), T, 63); M = torch.zeros(len(xs), T, dtype=torch.bool)
    for i, x in enumerate(xs):
        X[i, :len(x)] = torch.from_numpy(x); M[i, :len(x)] = True
    return X, M, torch.tensor([l for lab in labs for l in lab]), torch.tensor([len(l) for l in labs])


def cmd_train(a):
    torch.set_num_threads(a.threads)
    seqs, phrases, pids = load_data(a.npz, a.csv)
    sp, test_s, val_s = split_signers(pids)
    tr = np.flatnonzero(sp == "train"); va = np.flatnonzero(sp == "val")
    print(f"{len(seqs)} clips; train {len(tr)}, val {len(va)}, test {(sp == 'test').sum()}", flush=True)
    labs = [encode(p) for p in phrases]
    lens = [int((~np.isnan(s[::2, 0])).sum()) for s in seqs]
    rng = np.random.RandomState(SEED); prng = random.Random(SEED)
    va_sub = va[np.random.RandomState(1).permutation(len(va))[:800]]
    va_clips = [to_app_frames(seqs[i], False) for i in va_sub]
    va_refs = [phrases[i] for i in va_sub]

    model = FSModel(d=a.dim, n_stages=a.stages)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    os.makedirs(a.work, exist_ok=True)
    ck = os.path.join(a.work, "last.pt")
    state = {"epoch": 0, "step": 0, "best": -1e9, "hist": []}
    if os.path.exists(ck):
        c = torch.load(ck, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); state = c["state"]
        print(f"resumed after epoch {state['epoch']}", flush=True)
    steps_per_epoch = len(batches(tr, lens, a.bs_frames, prng))
    total = a.epochs * steps_per_epoch
    lr_at = lambda s: a.lr * min(1, s / 500) * (0.5 * (1 + math.cos(math.pi * min(1, s / total))) * 0.97 + 0.03)
    t_end = time.time() + a.hours * 3600
    while state["epoch"] < a.epochs:
        ep = state["epoch"]
        model.train()
        prng.seed(SEED + ep); rng.seed(SEED + ep)
        t0, tot, n = time.time(), 0.0, 0
        for bi, b in enumerate(batches(tr, lens, a.bs_frames, prng)):
            xs, ls = [], []
            for i in b:
                x = to_app_frames(seqs[i], True, rng)
                if len(x) >= 4 and ctc_fits(len(x), labs[i]) and labs[i]:
                    xs.append(x); ls.append(labs[i])
            if not xs: continue
            X, M, Y, YL = collate(xs, ls)
            for g in opt.param_groups: g["lr"] = lr_at(state["step"])
            lp = model(X, M)
            loss = F.ctc_loss(lp.transpose(0, 1), Y, M.sum(1) * model.up, YL, zero_infinity=True)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); state["step"] += 1
            tot += loss.item(); n += 1
            if bi % 200 == 0:
                print(f"ep {ep} batch {bi}/{steps_per_epoch} loss {tot / n:.3f} {time.time() - t0:.0f}s", flush=True)
            if time.time() > t_end:
                print("time limit inside epoch; restarting this epoch next run", flush=True)
                return
        s = score(predict(model, va_clips), va_refs)
        state["hist"].append({"epoch": ep, "loss": round(tot / max(n, 1), 4), **s, "sec": round(time.time() - t0)})
        print("VAL", state["hist"][-1], flush=True)
        state["epoch"] = ep + 1
        if s["score"] > state["best"]:
            state["best"] = s["score"]; torch.save({"model": model.state_dict(), "args": vars(a)}, os.path.join(a.work, "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state, "args": vars(a)}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        if time.time() > t_end - 1.2 * (time.time() - t0):
            print("time limit; resume with the same command", flush=True)
            return
    print("DONE", flush=True)


def load_model(work, name="best.pt"):
    c = torch.load(os.path.join(work, name), weights_only=False)
    m = FSModel(d=c["args"]["dim"], n_stages=c["args"]["stages"])
    m.load_state_dict(c["model"]); m.eval()
    return m


def cmd_test(a):
    torch.set_num_threads(a.threads)
    seqs, phrases, pids = load_data(a.npz, a.csv)
    sp, test_s, val_s = split_signers(pids)
    te = np.flatnonzero(sp == "test")
    model = load_model(a.work)
    clips = [to_app_frames(seqs[i], False) for i in te]
    refs = [phrases[i] for i in te]
    rng = np.random.RandomState(7)
    # unknown chirality: mirror half the clips at random, then let auto-mirror pick
    flip = rng.rand(len(clips)) < 0.5
    mclips = []
    for x, f in zip(clips, flip):
        x = x.copy().reshape(-1, 21, 3)
        if f: x[..., 0] = -x[..., 0]
        mclips.append(x.reshape(-1, 63))
    p_known = predict(model, clips)
    p_auto = predict(model, mclips, auto_mirror=True)
    rep = {}
    if os.path.exists(a.report):
        rep = json.load(open(a.report))
    c = torch.load(os.path.join(a.work, "last.pt"), weights_only=False)
    src = np.array(LOADED_SRC)[te]
    by_src = {}
    for k in ("train", "supp"):
        sel = np.flatnonzero(src == k)
        by_src[k] = {"full_phrase": score([p_auto[j] for j in sel], [refs[j] for j in sel]),
                     "letters_only": score([letters(p_auto[j]) for j in sel], [letters(refs[j]) for j in sel])}
    # how often MediaPipe Holistic found the dominant hand in the clip; many Kaggle clips lose the hand
    # for most frames (the app uses its own HandLandmarker, so the well-tracked rows are closer to it)
    present = np.array([(~np.isnan(seqs[i][:, 0])).mean() for i in te])
    by_track = {}
    for lo, hi in [(0, .3), (.3, .6), (.6, .9), (.9, 1.01)]:
        sel = np.flatnonzero((present >= lo) & (present < hi))
        by_track[f"hand_found_in_{int(lo * 100)}-{min(int(hi * 100), 100)}pct_of_frames"] = {
            "share_of_clips": round(len(sel) / len(te), 3),
            "full_phrase": score([p_auto[j] for j in sel], [refs[j] for j in sel])}
    rep.update({
        "data": "Kaggle asl-fingerspelling (CC BY 4.0): train set (67k addresses/phone numbers/URLs) + "
                "supplemental set (53k English sentences)",
        "split": {"test_signers": test_s, "val_signers": val_s, "rule": "by participant_id, seed 0"},
        "input": "dominant hand, 15 fps, frames without the hand dropped, x*W/H, y, z*W/H (index.html domHand63)",
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "train_history": c["state"]["hist"],
        "test_heldout_signers": {
            "known_hand_full_phrase": score(p_known, refs),
            "auto_mirror_full_phrase": score(p_auto, refs),
            "auto_mirror_letters_only": score([letters(p) for p in p_auto], [letters(r) for r in refs]),
            "by_source_auto_mirror": by_src,
            "by_hand_tracking_auto_mirror": by_track,
        },
        "examples": [{"ref": r, "pred": p} for r, p in list(zip(refs, p_auto))[:15]],
    })
    json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps(rep["test_heldout_signers"], indent=1))


def cmd_old(a):
    """The old per-frame letter MLP + index.html's decodeSpelling, on the same held-out test clips."""
    import onnxruntime as ort
    seqs, phrases, pids = load_data(a.npz, a.csv)
    sp, _, _ = split_signers(pids)
    te = np.flatnonzero(sp == "test")
    if a.limit: te = te[np.random.RandomState(0).permutation(len(te))[:a.limit]]
    sess = ort.InferenceSession(os.path.join(a.models, "stat_fingerspelling_model.onnx"))
    classes = json.load(open(os.path.join(a.models, "fingerspell_classes.json")))
    li = [i for i, c in enumerate(classes) if len(c) == 1 and c.isalpha()]
    L = [classes[i].lower() for i in li]

    def probs(h):  # letterProbs(): wrist-centered / |wrist-MCP|, try the mirror image too
        h = h.reshape(21, 3); h = (h - h[0]) / max(np.linalg.norm(h[9] - h[0]), 1e-6)
        best = None
        for v in (h, h * np.array([-1, 1, 1], np.float32)):
            lg = sess.run(None, {sess.get_inputs()[0].name: v.reshape(1, 63).astype(np.float32)})[0][0]
            p = np.exp(lg - lg.max()); p /= p.sum(); p = p[li]; p /= p.sum()
            if best is None or p.max() > best.max(): best = p
        return best

    preds, refs = [], []
    for i in te:
        d = seqs[i].copy().reshape(-1, 21, 3)
        d[..., 0] *= KAGGLE_ASPECT; d[..., 2] *= KAGGLE_ASPECT  # 30 fps, like a phone at full rate
        lab = [(-1 if np.isnan(f).any() else int(probs(f).argmax())) for f in d]
        sm = []
        for k in range(len(lab)):  # majority filter, window 5
            w = lab[max(0, k - 2):k + 3]; cnt = {}
            for l in w: cnt[l] = cnt.get(l, 0) + 1
            b = max(cnt.items(), key=lambda kv: (kv[1], kv[0] == lab[k]))[0]; sm.append(b)
        out, k = [], 0
        while k < len(sm):  # runs >= letterMinFrames (6); merge equal neighbours (no wrist-move check)
            j = k
            while j < len(sm) and sm[j] == sm[k]: j += 1
            if sm[k] >= 0 and j - k >= 6 and (not out or out[-1] != sm[k]): out.append(sm[k])
            k = j
        preds.append("".join(L[o] for o in out)); refs.append(letters(phrases[i]))
    s = score(preds, refs)
    src = np.array(LOADED_SRC)[te]
    s["by_source"] = {k: score([p for p, q in zip(preds, src) if q == k], [r for r, q in zip(refs, src) if q == k])
                      for k in ("train", "supp") if (src == k).any()}
    rep = json.load(open(a.report)) if os.path.exists(a.report) else {}
    rep["old_model_letters_only"] = {**s, "note": "per-frame MLP (Kaggle ASL Alphabet photos) + decodeSpelling, 30 fps"}
    rep["old_examples"] = [{"ref": r, "pred": p} for r, p in list(zip(refs, preds))[:15]]
    json.dump(rep, open(a.report, "w"), indent=1)
    print(s)


def cmd_cases(a):
    """A few held-out clips (half mirrored) with Python's decoding, for tools/fs_v2_browser_test.mjs."""
    seqs, phrases, pids = load_data(a.npz, a.csv)
    sp, _, _ = split_signers(pids)
    te = np.flatnonzero(sp == "test")
    te = te[np.random.RandomState(3).permutation(len(te))[:a.limit or 8]]
    model = load_model(a.work)
    out = []
    for k, i in enumerate(te):
        x = to_app_frames(seqs[i], False).reshape(-1, 21, 3)
        if k % 2: x[..., 0] = -x[..., 0]
        x = x.reshape(-1, 63)
        out.append({"T": len(x), "x": [round(float(v), 6) for v in x.ravel()], "ref": phrases[i],
                    "python": predict(model, [x.astype(np.float32)], auto_mirror=True)[0]})
    json.dump(out, open(a.cases, "w"))
    print(len(out), "cases ->", a.cases)


def cmd_export(a):
    model = load_model(a.work)
    w = Exported(model).eval()
    path = os.path.join(a.out, "stat_fingerspelling_v2.onnx")
    torch.onnx.export(w, torch.randn(1, 40, 63), path, input_names=["landmarks"], output_names=["logprobs"],
                      dynamic_axes={"landmarks": {1: "T"}, "logprobs": {1: "S"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(path)
    for T in (5, 37, 120):
        x = torch.rand(1, T, 63) * 0.3 + 0.3
        ref = w(x).detach().numpy(); got = s.run(None, {"landmarks": x.numpy()})[0]
        print(T, got.shape, "max abs diff", float(np.abs(ref - got).max()))
    json.dump({"chars": CHARS, "blank": 0, "fps": 15, "upsample": model.up}, open(os.path.join(a.out, "fingerspell_v2.json"), "w"))
    print(path, os.path.getsize(path) / 1e6, "MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test", "export", "old", "cases"])
    ap.add_argument("--cases", default="fs_cases.json")
    ap.add_argument("--npz"); ap.add_argument("--csv"); ap.add_argument("--work", default="fs_work")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    ap.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    ap.add_argument("--report", default=os.path.join(os.path.dirname(__file__), "report_fs_v2.json"))
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--dim", type=int, default=192); ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--bs-frames", dest="bs_frames", type=int, default=4096)
    ap.add_argument("--hours", type=float, default=1.8); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"train": cmd_train, "test": cmd_test, "export": cmd_export, "old": cmd_old, "cases": cmd_cases}[a.cmd](a)
