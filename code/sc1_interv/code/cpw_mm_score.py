"""Delta_image on the generation-regime-controlled sample.

Cell set is the REGISTERED regime rule: cells where both SIGHTED arms are uncapped
(cont_len < 384). The same cells are then scored in the blind arms, so Delta_sighted and
Delta_blind are computed over an identical population.

Positive control: with --sighted-only this must reproduce the published +0.3460.
"""
import os, sys, json
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")
import numpy as np
from j5_common import ChairScorer, read_jsonl
import wia_cpw_common as C
from wia_cpw_score import Units, gap_raw

MAX384, NBOOT, SEED = 384, 4000, 20260817

def load(paths):
    rows = []
    for p in paths:
        assert os.path.isfile(p), f"FATAL_MISSING {p}"
        rows += list(read_jsonl(p))
    return rows

def main():
    chair = ChairScorer("/data/alexmueller/j5_gates/code/synonyms.txt")
    allids, pub, ext, T = C.sample_ids()
    data = C.records(chair, allids, T)
    gt = {d["image_id"]: set(d["gt"]) for d in data}
    objset = lambda t: set(n for (n, _c) in chair.mentions(t))

    S = load([f"{C.OUT_R}/at_corrobpw_b{b}.jsonl" for b in range(C.N_BLOCKS)])
    blind_paths = [f"{C.OUT_R}/at_corrobpw_b{b}_mm.jsonl" for b in range(C.N_BLOCKS)]
    have_blind = all(os.path.isfile(p) for p in blind_paths)
    B = load(blind_paths) if have_blind else []
    print(f"[cpwmm] sighted rows {len(S)}  blind rows {len(B)}  blind_present={have_blind}", flush=True)

    def bycell(rows):
        d = {}
        for r in rows:
            d.setdefault((r["image_id"], r["k"]), {})[r["arm"]] = r
        return d
    cs, cb = bycell(S), bycell(B)

    # REGISTERED regime rule, defined on the sighted arms only
    keys = sorted(k for k, d in cs.items()
                  if set(d) == {"cap1", "cap5"} and all(d[a]["cont_len"] < MAX384 for a in ("cap1", "cap5")))
    if have_blind:
        keys = [k for k in keys if k in cb and set(cb[k]) == {"cap1", "cap5"}]
    imgs = sorted({k[0] for k in keys}); ix = {v: i for i, v in enumerate(imgs)}
    vocab = {}
    print(f"[cpwmm] cells both-sighted-uncapped: {len(keys)}  images {len(imgs)}", flush=True)

    def units(cellmap, arm):
        col = [[], [], [], []]
        for k in keys:
            r = cellmap[k][arm]
            P, Cc = objset(r["prefix_text"]), objset(r["cont_text"])
            g = gt[k[0]]
            for o in P:
                vocab.setdefault(o, len(vocab))
                col[0].append(ix[k[0]]); col[1].append(vocab[o])
                col[2].append(o in g); col[3].append(o in Cc)
        return Units(*col, n_img=len(imgs))

    out = {"n_cells": len(keys), "n_images": len(imgs), "rule": "both sighted arms cont_len<384"}
    U = {("S", a): units(cs, a) for a in ("cap1", "cap5")}
    if have_blind:
        U.update({("B", a): units(cb, a) for a in ("cap1", "cap5")})

    def delta(tag):
        g1 = gap_raw(U[(tag, "cap1")], np.arange(len(U[(tag, "cap1")].tru)))[0]
        g5 = gap_raw(U[(tag, "cap5")], np.arange(len(U[(tag, "cap5")].tru)))[0]
        return g5 - g1, g1, g5

    dS, s1, s5 = delta("S")
    out["Delta_sighted"] = dS; out["gap_cap1_sighted"] = s1; out["gap_cap5_sighted"] = s5
    print(f"[cpwmm] SIGHTED  gap(cap1)={s1:+.4f} gap(cap5)={s5:+.4f}  Delta={dS:+.4f}", flush=True)
    print(f"[cpwmm] POSITIVE CONTROL vs published +0.3460 -> dev {abs(dS-0.3460):.4f}", flush=True)

    if have_blind:
        dB, b1, b5 = delta("B")
        out["Delta_blind"] = dB; out["gap_cap1_blind"] = b1; out["gap_cap5_blind"] = b5
        out["Delta_image"] = dS - dB
        out["ratio_blind_over_sighted"] = dB / dS if dS else None
        print(f"[cpwmm] BLIND    gap(cap1)={b1:+.4f} gap(cap5)={b5:+.4f}  Delta={dB:+.4f}", flush=True)
        print(f"[cpwmm] DELTA_IMAGE {dS-dB:+.4f}   ratio {dB/dS:.3f}", flush=True)

        rng = np.random.default_rng(SEED); n = len(imgs); ds = []
        for _ in range(NBOOT):
            sel = rng.integers(0, n, n)
            gs = {t: {a: gap_raw(U[(t, a)], U[(t, a)].gather(sel))[0] for a in ("cap1", "cap5")}
                  for t in ("S", "B")}
            if any(v is None for t in gs for v in gs[t].values()):
                continue
            ds.append((gs["S"]["cap5"] - gs["S"]["cap1"]) - (gs["B"]["cap5"] - gs["B"]["cap1"]))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        out["Delta_image_ci"] = [float(lo), float(hi)]; out["n_boot_used"] = len(ds)
        print(f"[cpwmm] DELTA_IMAGE 95% CI [{lo:+.4f}, {hi:+.4f}]  (B={len(ds)}, clustered on image)", flush=True)

    json.dump(out, open(f"{C.OUT_R}/cpw_mm_score.json", "w"), indent=1)
    print("CPWMM_SENTINEL_OK", flush=True)

if __name__ == "__main__":
    main()
