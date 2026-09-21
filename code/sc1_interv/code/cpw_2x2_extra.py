"""2x2 on ONE cell population: {grey, mismatch} x {all cells, both-sighted-uncapped cells},
all four arms drawn from the same 3,500-image sample and the same 10,500 cells.

Also reports (a) cap rates of the ablated arms inside the uncapped cells, (b) how often a
scored unit's object is in the SUBSTITUTE image's ground truth under the mismatch arm, and
Delta_image with those units excluded, and (c) a derangement-reconstruction check: under the
mismatch arm the continuation should name objects of the substitute image, not the target.

Positive controls: uncapped sighted Delta must be +0.3460; uncapped grey Delta_image must be
+0.1004 and uncapped mismatch +0.1819 (cpw_blind_score.json, cpw_mm_score.json).
"""
import os, sys, json, random
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

def bycell(rows):
    d = {}
    for r in rows:
        d.setdefault((r["image_id"], r["k"]), {})[r["arm"]] = r
    return d

def main():
    chair = ChairScorer("/data/alexmueller/j5_gates/code/synonyms.txt")
    allids, pub, ext, T = C.sample_ids()
    data = C.records(chair, allids, T)
    gt = {d["image_id"]: set(d["gt"]) for d in data}
    _oc = {}
    def objset(t):
        if t not in _oc: _oc[t] = set(n for (n, _c) in chair.mentions(t))
        return _oc[t]

    S = load([f"{C.OUT_R}/at_corrobpw_b{b}.jsonl" for b in range(C.N_BLOCKS)])
    G = load([f"{C.OUT_R}/at_corrobpw_b{b}_blind.jsonl" for b in range(C.N_BLOCKS)])
    M = load([f"{C.OUT_R}/at_corrobpw_b{b}_mm.jsonl" for b in range(C.N_BLOCKS)])
    print(f"[x2] rows S {len(S)} G {len(G)} M {len(M)}", flush=True)

    # reconstruct the derangement exactly as wia_cpw_gen_mm.py drew it, per block
    sub = {}
    for b in range(C.N_BLOCKS):
        ids = sorted({r["image_id"] for r in M if r["block"] == b})
        rr = random.Random(20260919)
        while True:
            perm = ids[:]; rr.shuffle(perm)
            if all(a != c for a, c in zip(ids, perm)):
                break
        sub.update(dict(zip(ids, perm)))
        print(f"[x2] block {b}: {len(ids)} images deranged", flush=True)

    cs, cg, cm = bycell(S), bycell(G), bycell(M)
    full = lambda d, k: k in d and set(d[k]) == {"cap1", "cap5"}
    ALL = sorted(k for k in cs if full(cs, k) and full(cg, k) and full(cm, k))
    UNC = [k for k in ALL if all(cs[k][a]["cont_len"] < MAX384 for a in ("cap1", "cap5"))]
    print(f"[x2] cells ALL {len(ALL)}  UNC {len(UNC)}", flush=True)
    assert len(UNC) == 2089, f"FATAL_UNC_DRIFT {len(UNC)}"

    # derangement check: under M the continuation should track the substitute's objects
    def track(cmap, keys, which):
        hit = tot = 0
        for k in keys:
            for a in ("cap1", "cap5"):
                o = objset(cmap[k][a]["cont_text"])
                ref = gt[sub[k[0]]] if which == "sub" else gt[k[0]]
                hit += len(o & ref); tot += len(o)
        return hit / max(tot, 1)
    chk = {"M_cont_in_sub_gt": track(cm, ALL, "sub"), "M_cont_in_target_gt": track(cm, ALL, "tgt"),
           "S_cont_in_sub_gt": track(cs, ALL, "sub"), "S_cont_in_target_gt": track(cs, ALL, "tgt")}
    print(f"[x2] DERANGEMENT CHECK {chk}", flush=True)
    assert chk["M_cont_in_sub_gt"] > chk["M_cont_in_target_gt"], "FATAL_DERANGEMENT_NOT_REPRODUCED"

    out = {"n_cells_all": len(ALL), "n_cells_unc": len(UNC), "derangement_check": chk}

    # cap rates of every arm inside each cell set
    for name, keys in (("all", ALL), ("unc", UNC)):
        for tag, cmap in (("sighted", cs), ("grey", cg), ("mismatch", cm)):
            for a in ("cap1", "cap5"):
                out[f"caprate_{name}_{tag}_{a}"] = float(np.mean([cmap[k][a]["cont_len"] >= MAX384 for k in keys]))

    def build(keys, excl_overlap=False):
        imgs = sorted({k[0] for k in keys}); ix = {v: i for i, v in enumerate(imgs)}
        vocab = {}
        def units(cmap, arm):
            col = [[], [], [], []]
            for k in keys:
                r = cmap[k][arm]
                P, Cc = objset(r["prefix_text"]), objset(r["cont_text"])
                g = gt[k[0]]
                for o in P:
                    if excl_overlap and o in gt[sub[k[0]]]:
                        continue
                    vocab.setdefault(o, len(vocab))
                    col[0].append(ix[k[0]]); col[1].append(vocab[o])
                    col[2].append(o in g); col[3].append(o in Cc)
            return Units(*col, n_img=len(imgs))
        U = {(t, a): units(m, a) for t, m in (("S", cs), ("G", cg), ("M", cm)) for a in ("cap1", "cap5")}
        return U, len(imgs)

    def point(U, t):
        g1 = gap_raw(U[(t, "cap1")], np.arange(len(U[(t, "cap1")].tru)))[0]
        g5 = gap_raw(U[(t, "cap5")], np.arange(len(U[(t, "cap5")].tru)))[0]
        return g5 - g1, g1, g5

    def boot(U, n, tags):
        rng = np.random.default_rng(SEED); res = {t: [] for t in tags}; rat = {t: [] for t in tags}
        for _ in range(NBOOT):
            sel = rng.integers(0, n, n)
            gs = {t: {a: gap_raw(U[(t, a)], U[(t, a)].gather(sel))[0] for a in ("cap1", "cap5")}
                  for t in ("S",) + tuple(tags)}
            if any(v is None for t in gs for v in gs[t].values()):
                continue
            dS = gs["S"]["cap5"] - gs["S"]["cap1"]
            for t in tags:
                dB = gs[t]["cap5"] - gs[t]["cap1"]
                res[t].append(dS - dB); rat[t].append(dB / dS)
        return ({t: [float(x) for x in np.percentile(res[t], [2.5, 97.5])] for t in tags},
                {t: [float(x) for x in np.percentile(rat[t], [2.5, 97.5])] for t in tags},
                len(res[tags[0]]))

    for name, keys in (("all", ALL), ("unc", UNC)):
        U, n = build(keys)
        dS, s1, s5 = point(U, "S")
        out[f"{name}_Delta_sighted"] = dS; out[f"{name}_gaps_sighted"] = [s1, s5]
        print(f"[x2] {name}: sighted Delta {dS:+.4f}  (n_img {n})", flush=True)
        ci, rci, nb = boot(U, n, ("G", "M"))
        for t, lab in (("G", "grey"), ("M", "mismatch")):
            dB, b1, b5 = point(U, t)
            out[f"{name}_{lab}"] = {"Delta_blind": dB, "gaps": [b1, b5], "Delta_image": dS - dB,
                                    "Delta_image_ci": ci[t], "ratio": dB / dS, "ratio_ci": rci[t]}
            print(f"[x2] {name} {lab}: Delta_blind {dB:+.4f} Delta_image {dS-dB:+.4f} "
                  f"CI [{ci[t][0]:+.4f},{ci[t][1]:+.4f}] ratio {dB/dS:.3f} [{rci[t][0]:.3f},{rci[t][1]:.3f}]", flush=True)
        out[f"{name}_n_images"] = n; out[f"{name}_n_boot"] = nb

    # overlap of scored units with the substitute image's ground truth
    for name, keys in (("all", ALL), ("unc", UNC)):
        for a in ("cap1", "cap5"):
            pres = [0, 0]; absn = [0, 0]
            for k in keys:
                for o in objset(cs[k][a]["prefix_text"]):
                    ov = o in gt[sub[k[0]]]
                    if o in gt[k[0]]: pres[0] += ov; pres[1] += 1
                    else: absn[0] += ov; absn[1] += 1
            out[f"overlap_{name}_{a}"] = {"present_units_in_sub_gt": pres[0] / pres[1],
                                         "absent_units_in_sub_gt": absn[0] / absn[1],
                                         "n_present": pres[1], "n_absent": absn[1]}
        U, n = build(keys, excl_overlap=True)
        dS = point(U, "S")[0]; dM = point(U, "M")[0]
        ci, rci, nb = boot(U, n, ("M",))
        out[f"{name}_mismatch_excl_overlap"] = {"Delta_sighted": dS, "Delta_blind": dM, "Delta_image": dS - dM,
                                                "Delta_image_ci": ci["M"], "ratio": dM / dS, "ratio_ci": rci["M"]}
        print(f"[x2] {name} mismatch EXCL OVERLAP: Delta_image {dS-dM:+.4f} CI [{ci['M'][0]:+.4f},{ci['M'][1]:+.4f}] "
              f"ratio {dM/dS:.3f}", flush=True)
        print(f"[x2] overlap {name}: {out[f'overlap_{name}_cap1']} | {out[f'overlap_{name}_cap5']}", flush=True)

    pc = [abs(out["unc_Delta_sighted"] - 0.3460) < 5e-4,
          abs(out["unc_grey"]["Delta_image"] - 0.10036) < 5e-4,
          abs(out["unc_mismatch"]["Delta_image"] - 0.18195) < 5e-4]
    out["positive_controls"] = pc
    print(f"[x2] POSITIVE CONTROLS (unc sighted, unc grey, unc mm): {pc}", flush=True)
    json.dump(out, open(f"{C.OUT_R}/cpw_2x2_extra.json", "w"), indent=1)
    assert all(pc), "FATAL_POSITIVE_CONTROL"
    print("X2_SENTINEL_OK", flush=True)

if __name__ == "__main__":
    main()
