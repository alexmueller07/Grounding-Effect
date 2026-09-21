"""L1 scorer -- the ablation ladder on the prefix endpoint (LADDER_PREREG.md sections 3, 5, 6).

Units, `Units.gather`, `gap_raw`, `trunc_text` and the bootstrap machinery are IMPORTED from
`wia_blindscore`, the published lane's own scorer, so they cannot drift from the numbers in
the paper. Nothing there is modified; the md5 of every imported shared module is recorded in
the output so later drift is detectable.

Every Delta in this file -- Delta_primary, Delta_r for each rung, and Delta_image(r) -- is
computed INSIDE ONE bootstrap loop on ONE shared image-clustered draw (pre-reg 3.4), so they
cannot come from different resamples.

Order of operations is the registered one:
    PC1 (frozen sighted + frozen published grey)  ->  PC2 (this lane's grey re-generation)
    ->  degeneracy screens  ->  only then the new rungs' endpoints.

Writes out/lad_l1.json. Sentinel printed by Python; the exit code is never the signal.
"""
import os, sys, json, math, hashlib, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/sc1_gen/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")

import numpy as np
from lad_common import RUNGS, OUT, sentinel, jdump
from j5_common import load_coco_sample, ChairScorer, read_jsonl
import wia_blindscore as WB
from wia_blindscore import Units, gap_raw, trunc_text, _assert_multiplicity, ngram_ratio

OUT_R = "/data/alexmueller/sc1_interv/out"
SIGHTED = f"{OUT_R}/at_corrob.jsonl"
PUB_GREY = f"{OUT_R}/at_blind1.jsonl"
SYN = "/data/alexmueller/j5_gates/code/synonyms.txt"
CAP = 384
WS = "▁"

NBOOT = int(os.environ.get("LAD_NBOOT", "4000"))
SEED = 20260919
MARGIN = 0.10                 # the paper's reporting choice, not a registered bound
WIDTH_CEIL = 0.14             # parent lane's WIDTH_CEIL
N_TRUE_FLOOR = 100
PC1_REF = {"Delta_primary": 0.4859, "Delta_blind": 0.5019, "Delta_image": -0.0160}
PC1_TOL = 5e-3
PC2A_MIN_FRAC = 0.98
PC2B_TOL = 0.010
# degeneracy screen thresholds, pre-reg section 6
SCR_MEDLEN, SCR_EMPTY, SCR_NGRAM, SCR_NTEXT = 8, 0.25, 0.20, 10
SCR_LE3 = 0.25                # Amendment 3: rate of continuations of <= 3 tokens


def md5(p):
    try:
        return hashlib.md5(open(p, "rb").read()).hexdigest()
    except Exception as e:
        return f"ERR {e}"


def tokid_ratio(ids):
    return 1.0 if not ids else len(set(ids)) / len(ids)


def screens(rows):
    L = [r["cont_len"] for r in rows]
    txt = [r["cont_text"] for r in rows]
    from collections import Counter
    top = Counter(txt).most_common(1)
    return {"n": len(rows), "mean_cont_len": float(np.mean(L)),
            "median_cont_len": float(np.median(L)),
            # Reported, NEVER a collapse criterion in either direction (Amendment 3): an arm
            # that stops early is not thereby collapsed, and one that runs to the cap is not
            # thereby healthy.
            "cap_hit_rate": float(np.mean([x >= CAP for x in L])),
            "trunc_rate_at_384": float(np.mean([x >= CAP for x in L])),
            "empty_rate": float(np.mean([x == 0 for x in L])),
            "le3_token_rate": float(np.mean([x <= 3 for x in L])),
            "top_text_share": (top[0][1] / len(txt)) if top else None,
            "top_text": (top[0][0][:160] if top else None),
            "mean_ngram8_ratio": float(np.mean([ngram_ratio(r["cont_text"]) for r in rows])),
            "mean_tokid_ratio": float(np.mean([tokid_ratio(r["cont_ids"]) for r in rows])),
            "n_distinct_texts": len({r["cont_text"] for r in rows})}


def degenerate(s, nt, nf):
    why = []
    if s["median_cont_len"] < SCR_MEDLEN:
        why.append(f"median_len {s['median_cont_len']} < {SCR_MEDLEN}")
    if s["empty_rate"] > SCR_EMPTY:
        why.append(f"empty {s['empty_rate']:.3f} > {SCR_EMPTY}")
    if s["le3_token_rate"] > SCR_LE3:            # Amendment 3: collapse at the BOTTOM
        why.append(f"le3_token_rate {s['le3_token_rate']:.3f} > {SCR_LE3}")
    if s["mean_ngram8_ratio"] < SCR_NGRAM:
        why.append(f"ngram8 {s['mean_ngram8_ratio']:.3f} < {SCR_NGRAM}")
    if s["n_distinct_texts"] < SCR_NTEXT:
        why.append(f"distinct_texts {s['n_distinct_texts']} < {SCR_NTEXT}")
    if nt < N_TRUE_FLOOR or nf < N_TRUE_FLOOR:
        why.append(f"units {nt}/{nf} < {N_TRUE_FLOOR}")
    return why


def load_all(rungs):
    """{arm_name: {(image_id,k): row}} for the sighted pair, the published grey pair and every
    rung pair present on disk."""
    arms, src = {}, {}
    for r in read_jsonl(SIGHTED):
        arms.setdefault(r["arm"], {})[(r["image_id"], r["k"])] = r
    src["cap1"] = src["cap5"] = SIGHTED
    for r in read_jsonl(PUB_GREY):
        arms.setdefault(r["arm"], {})[(r["image_id"], r["k"])] = r
    src["cap1g"] = src["cap5g"] = PUB_GREY
    present = []
    for rg in rungs:
        p, mp = f"{OUT}/lad_f1_{rg}.jsonl", f"{OUT}/lad_f1_{rg}_meta.json"
        if not os.path.exists(p):
            print(f"[lad_score] rung {rg}: no rows on disk, skipped", flush=True)
            continue
        # A generation job may be WRITING this file right now. The real condition for
        # completeness is the meta sentinel plus row-count agreement, never file existence.
        if not os.path.exists(mp):
            print(f"[lad_score] rung {rg}: no meta yet (job still running?), skipped",
                  flush=True)
            continue
        m = json.load(open(mp))
        rows = list(read_jsonl(p))
        if not m.get("sentinel") or m.get("rows") != len(rows):
            print(f"[lad_score] rung {rg}: INCOMPLETE meta.rows={m.get('rows')} "
                  f"on-disk={len(rows)}, skipped", flush=True)
            continue
        for r in rows:
            arms.setdefault(r["arm"], {})[(r["image_id"], r["k"])] = r
        src[f"cap1_{rg}"] = src[f"cap5_{rg}"] = p
        present.append(rg)
    return arms, src, present


def joint(U, names, pairs, nboot, seed):
    """One image-clustered draw per replicate; every arm's gap and rates on that draw."""
    rng = np.random.default_rng(seed)
    n_img = len(U[names[0]].counts)
    acc = None
    for b in range(nboot):
        sel = rng.integers(0, n_img, n_img)
        if b < 20:
            _assert_multiplicity(list(sel), n_img, f"boot#{b}")
        g = {a: U[a].gather(sel) for a in names}
        d = stats_on(U, g, pairs)
        if acc is None:
            acc = {k: [] for k in d}
        for k, v in d.items():
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                acc[k].append(float(v))
    return acc


def stats_on(U, g, pairs):
    """`pairs` = [(label, arm1, arm5)]. The first pair is the sighted reference."""
    out, gapv, rt, rf = {}, {}, {}, {}
    for _lbl, a1, a5 in pairs:
        for a in (a1, a5):
            if a in gapv:
                continue
            v, (nt, nf, ct, cf) = gap_raw(U[a], g[a])
            gapv[a] = v
            rt[a] = ct / nt if nt else None
            rf[a] = cf / nf if nf else None
            out[f"gap[{a}]"] = v
            out[f"r_true[{a}]"] = rt[a]
            out[f"r_false[{a}]"] = rf[a]
    sub = lambda x, y: (None if (x is None or y is None) else x - y)
    # POST HOC (2026-09-21): equal-variance d' and c per arm, their one->five changes, and the
    # image-attributable parts, on the same joint draw. Rates are the scorer's own r_true/r_false.
    from statistics import NormalDist as _ND
    _z = _ND().inv_cdf
    dp, cc = {}, {}
    for a in gapv:
        if rt[a] is not None and rf[a] is not None and 0 < rt[a] < 1 and 0 < rf[a] < 1:
            dp[a] = _z(rt[a]) - _z(rf[a]); cc[a] = -0.5 * (_z(rt[a]) + _z(rf[a]))
        else:
            dp[a] = cc[a] = None
        out[f"dp[{a}]"] = dp[a]; out[f"c[{a}]"] = cc[a]
    for lbl, a1, a5 in pairs:
        out[f"dDp[{lbl}]"] = sub(dp[a5], dp[a1])
        out[f"dC[{lbl}]"] = sub(cc[a5], cc[a1])
    for lbl, _a1, _a5 in pairs[1:]:
        out[f"dDpimg[{lbl}]"] = sub(out[f"dDp[{pairs[0][0]}]"], out[f"dDp[{lbl}]"])
        out[f"dCimg[{lbl}]"] = sub(out[f"dC[{pairs[0][0]}]"], out[f"dC[{lbl}]"])
    for lbl, a1, a5 in pairs:
        out[f"D[{lbl}]"] = sub(gapv[a5], gapv[a1])
        out[f"dH[{lbl}]"] = sub(rt[a5], rt[a1])
        out[f"dF[{lbl}]"] = sub(rf[a5], rf[a1])
    ref = pairs[0][0]
    ladder = []
    for lbl, _a1, _a5 in pairs[1:]:
        v = sub(out[f"D[{ref}]"], out[f"D[{lbl}]"])
        out[f"Dimg[{lbl}]"] = v
        out[f"dHimg[{lbl}]"] = sub(out[f"dH[{ref}]"], out[f"dH[{lbl}]"])
        out[f"dFimg[{lbl}]"] = sub(out[f"dF[{ref}]"], out[f"dF[{lbl}]"])
        if out[f"D[{lbl}]"] is not None and out[f"D[{ref}]"]:
            out[f"ratio[{lbl}]"] = out[f"D[{lbl}]"] / out[f"D[{ref}]"]
        if lbl in LADDER_LABELS and v is not None:
            ladder.append(v)
    out["Dimg_ladder_mean"] = float(np.mean(ladder)) if ladder else None
    # POST HOC (added 2026-09-20, after the mismatch endpoint was read, and labelled as such):
    # the direct paired difference between each ablation's Delta_image and grey's, on the SAME
    # bootstrap draw. Identical to D[grey] - D[rung], so it introduces no new estimator; it is
    # the interval on "does the choice of ablation change the answer".
    if "D[grey]" in out or "D[pubgrey]" in out:
        gref = "D[grey]" if out.get("D[grey]") is not None else "D[pubgrey]"
        for lbl, _a1, _a5 in pairs[1:]:
            if f"D[{lbl}]" in out and out[f"D[{lbl}]"] is not None \
                    and out[gref] is not None:
                out[f"Dimg_minus_greyDimg[{lbl}]"] = out[gref] - out[f"D[{lbl}]"]
    return out


def summarise(point, vals, alpha=0.05):
    lo, hi = (100 * alpha / 2.0, 100 * (1 - alpha / 2.0))
    ci = [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))] if len(vals) else \
        [None, None]
    return {"point": point, "ci": ci,
            "ci_width": None if ci[0] is None else ci[1] - ci[0], "n_boot": len(vals)}


LADDER_LABELS = set()
CH = None


def main():
    global CH, LADDER_LABELS
    print(f"[lad_score] host={socket.gethostname()} B={NBOOT} seed={SEED}", flush=True)
    CH = ChairScorer(SYN)
    data = load_coco_sample(CH, n=500)
    gt = {d["image_id"]: set(d["gt"]) for d in data}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

    arms, src, present = load_all(list(RUNGS) + ["grey", "mismatch", "black"])
    have_grey = "grey" in present
    lrungs = [r for r in present if r != "grey"]
    # Amendment 2: `mismatch` is a SEPARATE registered primary, not one of the
    # degradation ladder. It gets a verdict, screens and an unadjusted interval, but
    # it is excluded from the pooled ladder mean and from the ladder's Bonferroni
    # family, which was fixed in advance at the seven degradation rungs.
    LADDER_LABELS = set(lrungs) - {"mismatch", "black"}

    names = ["cap1", "cap5", "cap1g", "cap5g"]
    pairs = [("sighted", "cap1", "cap5"), ("pubgrey", "cap1g", "cap5g")]
    if have_grey:
        names += ["cap1_grey", "cap5_grey"]
        pairs += [("grey", "cap1_grey", "cap5_grey")]
    for rg in lrungs:
        names += [f"cap1_{rg}", f"cap5_{rg}"]
        pairs += [(rg, f"cap1_{rg}", f"cap5_{rg}")]

    keys = sorted(set.intersection(*[set(arms[a]) for a in names]))
    for a in names:
        assert len(arms[a]) == len(keys), f"FATAL_CELL_SET_MISMATCH {a} {len(arms[a])} {len(keys)}"
    # prefix identity across every arm of every cell
    ndrift = sum(1 for kk in keys for a in names
                 if arms[a][kk]["prefix_text"] != arms[{"1": "cap1", "5": "cap5"}[a[3]]][kk]["prefix_text"])
    assert ndrift == 0, f"FATAL_PREFIX_DRIFT_AT_SCORE {ndrift}"
    print(f"[lad_score] cells {len(keys)} arms {len(names)} prefix_drift {ndrift}", flush=True)

    imgs = sorted({k0 for k0, _ in keys})
    ix = {i: j for j, i in enumerate(imgs)}
    Pof, vocab = {}, {}
    for kk in keys:
        P = set(nm for (nm, _c) in CH.mentions(arms["cap1"][kk]["prefix_text"]))
        Pof[kk] = P
        for o in P:
            vocab.setdefault(o, len(vocab))
    # cap5's prefix is a different text, so its unit set is its own
    Pof5 = {}
    for kk in keys:
        P = set(nm for (nm, _c) in CH.mentions(arms["cap5"][kk]["prefix_text"]))
        Pof5[kk] = P
        for o in P:
            vocab.setdefault(o, len(vocab))

    def build_all(match=None, only=None):
        use = only or names
        cols = {a: [[], [], [], []] for a in use}
        for kk in keys:
            Lof = {}
            if match == "within":
                for _lbl, a1, a5 in pairs:
                    L = min(len(arms[a1][kk]["cont_ids"]), len(arms[a5][kk]["cont_ids"]))
                    Lof[a1] = Lof[a5] = L
            elif isinstance(match, list):
                L = min(len(arms[a][kk]["cont_ids"]) for a in match)
                for a in match:
                    Lof[a] = L
            for a in use:
                r = arms[a][kk]
                if a in Lof:
                    text, _t, _m = trunc_text(tok, r["cont_ids"], Lof[a], WS)
                else:
                    text = r["cont_text"]
                C = set(nm for (nm, _c) in CH.mentions(text))
                P = Pof[kk] if a.startswith("cap1") else Pof5[kk]
                for o in P:
                    cols[a][0].append(ix[kk[0]]); cols[a][1].append(vocab[o])
                    cols[a][2].append(o in gt[kk[0]]); cols[a][3].append(o in C)
        return {a: Units(cols[a][0], cols[a][1], cols[a][2], cols[a][3], len(imgs))
                for a in use}

    res = {"sentinel": True, "host": socket.gethostname(), "B": NBOOT, "seed": SEED,
           "margin": MARGIN, "width_ceil": WIDTH_CEIL, "n_true_floor": N_TRUE_FLOOR,
           "prereg": "SPRINT/LADDER_PREREG.md (Amendments 1-2)",
           "n_cells": len(keys), "n_images": len(imgs), "rungs_present": present,
           "module_md5": {m: md5(p) for m, p in (
               ("wia_blindscore.py", "/data/alexmueller/sc1_interv/code/wia_blindscore.py"),
               ("wia_blind_common.py", "/data/alexmueller/sc1_interv/code/wia_blind_common.py"),
               ("wia_lngen.py", "/data/alexmueller/sc1_interv/code/wia_lngen.py"),
               ("j5_common.py", "/data/alexmueller/j5_gates/code/j5_common.py"),
               ("at_corrob.jsonl", SIGHTED), ("at_blind1.jsonl", PUB_GREY))},
           "sources": src}

    # ---- degeneracy screens, before any endpoint ----------------------------------------
    U = build_all(None)
    allg = {a: np.arange(len(U[a].img)) for a in names}
    scr, ncounts = {}, {}
    for a in names:
        rows = [arms[a][kk] for kk in keys]
        _v, (nt, nf, _ct, _cf) = gap_raw(U[a], allg[a])
        ncounts[a] = {"n_true": int(nt), "n_false": int(nf)}
        s = screens(rows)
        s.update(ncounts[a])
        s["degenerate_why"] = degenerate(s, nt, nf)
        s["DEGENERATE"] = bool(s["degenerate_why"])
        scr[a] = s
    res["screens"] = scr
    for a in names:
        print(f"[lad_score] screen {a:16s} med {scr[a]['median_cont_len']:6.1f} "
              f"empty {scr[a]['empty_rate']:.3f} ng8 {scr[a]['mean_ngram8_ratio']:.3f} "
              f"tokid {scr[a]['mean_tokid_ratio']:.3f} le3 {scr[a]['le3_token_rate']:.3f} "
              f"cap {scr[a]['cap_hit_rate']:.3f} texts {scr[a]['n_distinct_texts']:5d} "
              f"units {scr[a]['n_true']}/{scr[a]['n_false']} "
              f"{'DEGENERATE ' + str(scr[a]['degenerate_why']) if scr[a]['DEGENERATE'] else 'ok'}",
              flush=True)

    # ---- the one joint bootstrap ---------------------------------------------------------
    V = joint(U, names, pairs, NBOOT, SEED)
    pt = stats_on(U, allg, pairs)
    nb7 = max(1, len(LADDER_LABELS))
    res["point"] = {k: v for k, v in pt.items()}
    res["interval"] = {k: summarise(pt.get(k), V.get(k, [])) for k in V}
    # family-wise adjustment over the degradation ladder only; `mismatch`, `grey` and
    # `pubgrey` are single pre-specified comparisons and keep the nominal 95% level.
    res["interval_bonferroni"] = {
        k: summarise(pt.get(k), V.get(k, []),
                     alpha=(0.05 / nb7 if k[5:-1] in LADDER_LABELS else 0.05))
        for k in V if k.startswith("Dimg[")}
    res["bonferroni_family_size"] = nb7
    res["bonferroni_family"] = sorted(LADDER_LABELS)

    # ---- PC1: frozen sighted + frozen published grey -------------------------------------
    pc1 = {"Delta_primary": pt["D[sighted]"], "Delta_blind": pt["D[pubgrey]"],
           "Delta_image": pt["Dimg[pubgrey]"]}
    pc1_dev = {k: abs(pc1[k] - PC1_REF[k]) for k in PC1_REF}
    pc1_pass = all(v <= PC1_TOL for v in pc1_dev.values())
    res["PC1"] = {"measured": pc1, "reference": PC1_REF, "abs_dev": pc1_dev,
                  "tol": PC1_TOL, "PASS": bool(pc1_pass)}
    print(f"[lad_score] PC1 {json.dumps(res['PC1'])}", flush=True)

    # ---- PC2: this lane's grey re-generation --------------------------------------------
    if have_grey:
        pub = {(r["image_id"], r["k"], r["arm"]): r for r in read_jsonl(PUB_GREY)}
        mine = {(r["image_id"], r["k"], r["arm"].replace("_grey", "g")): r
                for r in read_jsonl(f"{OUT}/lad_f1_grey.jsonl")}
        common = sorted(set(pub) & set(mine))
        same, firstdiv = 0, []
        for kk in common:
            a, b = pub[kk]["cont_ids"], mine[kk]["cont_ids"]
            if a == b:
                same += 1
            elif len(firstdiv) < 10:
                j = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]),
                         min(len(a), len(b)))
                firstdiv.append({"key": list(kk), "first_divergent_token": j,
                                 "len_pub": len(a), "len_mine": len(b)})
        frac = same / max(len(common), 1)
        pc2b = abs(pt["Dimg[grey]"] - PC1_REF["Delta_image"])
        res["PC2"] = {"n_compared": len(common), "n_identical": same, "frac_identical": frac,
                      "min_frac": PC2A_MIN_FRAC, "PC2a_PASS": bool(frac >= PC2A_MIN_FRAC),
                      "first_divergences": firstdiv,
                      "Delta_image_regen_grey": pt["Dimg[grey]"],
                      "Delta_blind_regen_grey": pt["D[grey]"],
                      "abs_dev_vs_ref": pc2b, "tol": PC2B_TOL,
                      "PC2b_PASS": bool(pc2b <= PC2B_TOL)}
        res["PC2"]["PASS"] = bool(res["PC2"]["PC2a_PASS"] and res["PC2"]["PC2b_PASS"])
        print(f"[lad_score] PC2 identical {same}/{len(common)} ({frac:.4f})  "
              f"Delta_blind_regen {pt['D[grey]']:.4f}  Delta_image_regen "
              f"{pt['Dimg[grey]']:.4f} (dev {pc2b:.4f})  PASS={res['PC2']['PASS']}", flush=True)
    else:
        res["PC2"] = {"PASS": None, "note": "grey rung not generated yet"}

    # ---- secondaries: length matching ----------------------------------------------------
    # within-condition: L = min inside each pair, set independently per pair (the paper's
    # "within-condition, two-condition L" rule).
    Uw = build_all("within")
    allw = {a: np.arange(len(Uw[a].img)) for a in names}
    Vw = joint(Uw, names, pairs, NBOOT, SEED)
    ptw = stats_on(Uw, allw, pairs)
    res["lenmatch_within"] = {k: summarise(ptw.get(k), Vw.get(k, []))
                              for k in Vw if k.startswith(("D[", "Dimg["))}
    print("[lad_score] lenmatch within: " + "  ".join(
        f"{r}={res['lenmatch_within'][f'Dimg[{r}]']['point']:+.4f}" for r in lrungs),
        flush=True)

    # cross-condition: ONE L per cell, the minimum over the four arms of THAT comparison
    # (sighted pair + the rung's pair), applied to all four -- the paper's own rule, run
    # separately per rung so the rungs do not truncate each other.
    res["lenmatch_cross"] = {}
    for rg in ([r for r in present] if present else []):
        four = ["cap1", "cap5", f"cap1_{rg}", f"cap5_{rg}"]
        p4 = [("sighted", "cap1", "cap5"), (rg, f"cap1_{rg}", f"cap5_{rg}")]
        Uc = build_all(four, only=four)
        allc = {a: np.arange(len(Uc[a].img)) for a in four}
        Vc = joint(Uc, four, p4, NBOOT, SEED)
        ptc = stats_on(Uc, allc, p4)
        res["lenmatch_cross"][rg] = {
            "Delta_image": summarise(ptc.get(f"Dimg[{rg}]"), Vc.get(f"Dimg[{rg}]", [])),
            "Delta_rung": summarise(ptc.get(f"D[{rg}]"), Vc.get(f"D[{rg}]", [])),
            "Delta_primary": summarise(ptc.get("D[sighted]"), Vc.get("D[sighted]", []))}
        print(f"[lad_score] lenmatch cross {rg}: "
              f"{res['lenmatch_cross'][rg]['Delta_image']['point']:+.4f} "
              f"{res['lenmatch_cross'][rg]['Delta_image']['ci']}", flush=True)

    # ---- CP-R: both arms of each contrast uncapped ---------------------------------------
    cpr = {}
    for lbl, a1, a5 in pairs:
        ks = [kk for kk in keys if arms[a1][kk]["cont_len"] < CAP
              and arms[a5][kk]["cont_len"] < CAP]
        cpr[lbl] = {"n_cells": len(ks), "frac": len(ks) / len(keys)}
    res["CP_R_cell_counts"] = cpr

    # ---- registered verdicts --------------------------------------------------------------
    verd = {}

    def cmp_arm(a, cond, refs, label):
        w = []
        R = [scr[x] for x in refs]
        bar_med = min(x["median_cont_len"] for x in R)
        bar_ng = min(x["mean_ngram8_ratio"] for x in R)
        bar_tk = min(x["mean_tokid_ratio"] for x in R)
        bar_em = max(x["empty_rate"] for x in R)
        if scr[a]["median_cont_len"] < bar_med:
            w.append(f"{cond}: median {scr[a]['median_cont_len']} < {label} {bar_med}")
        if scr[a]["mean_ngram8_ratio"] < bar_ng:
            w.append(f"{cond}: ngram8 {scr[a]['mean_ngram8_ratio']:.3f} < {label} {bar_ng:.3f}")
        if scr[a]["mean_tokid_ratio"] < bar_tk:
            w.append(f"{cond}: tokid {scr[a]['mean_tokid_ratio']:.3f} < {label} {bar_tk:.3f}")
        if scr[a]["empty_rate"] > bar_em:
            w.append(f"{cond}: empty {scr[a]['empty_rate']:.3f} > {label} {bar_em:.3f}")
        return w

    for rg in lrungs:
        a1, a5 = f"cap1_{rg}", f"cap5_{rg}"
        d = res["interval"][f"Dimg[{rg}]"]
        db = res["interval_bonferroni"][f"Dimg[{rg}]"]
        deg = scr[a1]["DEGENERATE"] or scr[a5]["DEGENERATE"]
        # originally registered comparison (reported, does not set the verdict)
        worse_grey = (cmp_arm(a1, "cap1", ["cap1g"], "grey")
                      + cmp_arm(a5, "cap5", ["cap5g"], "grey"))
        # Amendment 1: the weaker of the two arms the paper already reads endpoints from
        worse = (cmp_arm(a1, "cap1", ["cap1", "cap1g"], "weaker-ref")
                 + cmp_arm(a5, "cap5", ["cap5", "cap5g"], "weaker-ref"))
        lo, hi = d["ci"]
        p = d["point"]
        # Amendment 3: collapse is tested ABSOLUTELY, by `deg` (Section 6), and the two
        # comparative flags below are diagnostics only. Neither can downgrade a verdict.
        if deg:
            v = "NO-DATA (degenerate arm)"
        elif d["ci_width"] is None or d["ci_width"] > WIDTH_CEIL:
            v = "UNDERPOWERED"
        elif lo > -MARGIN and hi < MARGIN:
            v = "NULL-WITHIN-MARGIN"
        elif lo > 0 and p >= MARGIN:
            v = "IMAGE-CONTRIBUTES"
        elif hi < 0 and p <= -MARGIN:
            v = "REVERSE"
        elif lo > 0 or hi < 0:
            v = "SMALL-BUT-NONZERO"
        else:
            v = "INCONCLUSIVE"
        # What the ORIGINAL (pre-Amendment-3) rule would have returned, recorded so the
        # change of rule is auditable rather than invisible.
        v_orig = ("CONFOUNDED-BY-COLLAPSE" if (v == "IMAGE-CONTRIBUTES" and worse) else v)
        verd[rg] = {"verdict": v,
                    "verdict_under_original_collapse_rule": v_orig,
                    "collapse_rule_changed_verdict": bool(v_orig != v),
                    "Delta_image": d, "Delta_image_bonferroni": db,
                    "Delta_rung": res["interval"][f"D[{rg}]"],
                    "dH_image": res["interval"].get(f"dHimg[{rg}]"),
                    "dF_image": res["interval"].get(f"dFimg[{rg}]"),
                    "more_degenerate_than_grey": worse_grey,
                    "more_degenerate_than_weaker_reference": worse,
                    "degenerate": bool(deg)}
    res["verdict_per_rung"] = verd

    res["POSTHOC_vs_grey"] = {
        k: summarise(pt.get(k), V.get(k, [])) for k in V
        if k.startswith("Dimg_minus_greyDimg[")}
    res["POSTHOC_vs_grey_note"] = (
        "Added after the mismatch endpoint was read; labelled post hoc. Equals "
        "D[grey] - D[rung] on the same draw, i.e. how much the answer depends on which "
        "ablation is used.")
    res["AMENDMENT3_verdicts_changed"] = {
        r: {"corrected": verd[r]["verdict"],
            "original_rule": verd[r]["verdict_under_original_collapse_rule"],
            "comparative_flags_that_would_have_fired":
                verd[r]["more_degenerate_than_weaker_reference"]}
        for r in verd if verd[r]["collapse_rule_changed_verdict"]}
    res["AMENDMENT3_note"] = (
        "Collapse is now tested absolutely (Section 6 + the <=3-token rate). The comparative "
        "flags are diagnostics and cannot downgrade a verdict: in the five-scene condition "
        "BOTH reference arms sit at the 384-token generation budget, so a median-length bar "
        "of min(sighted, grey) equals the ceiling and flags every rung that terminates "
        "naturally; and the diversity ratios fall mechanically with output length, so a rung "
        "that writes more is flagged as writing worse. Both errors ran in the direction that "
        "suppresses the adverse verdict.")
    lad = res["interval"].get("Dimg_ladder_mean")
    res["Delta_image_ladder_mean"] = lad
    nd = [r for r in lrungs if r in LADDER_LABELS and not verd[r]["degenerate"]]
    allnull = bool(nd) and all(verd[r]["verdict"] == "NULL-WITHIN-MARGIN" for r in nd)
    anyadv = any(verd[r]["verdict"] == "IMAGE-CONTRIBUTES"
                 and verd[r]["Delta_image_bonferroni"]["ci"][0] is not None
                 and verd[r]["Delta_image_bonferroni"]["ci"][0] > 0 for r in nd)
    ladder_material = (lad is not None and lad["ci"][0] is not None
                       and lad["point"] is not None and lad["point"] >= MARGIN
                       and lad["ci"][0] > 0)
    if allnull and lad is not None and lad["ci"][0] is not None \
            and lad["ci"][0] > -MARGIN and lad["ci"][1] < MARGIN:
        overall = "LADDER-NULL"
    elif ladder_material or anyadv:
        overall = "NARROW-TO-STRONG-ABLATION"
    else:
        overall = "LADDER-UNINFORMATIVE"
    res["LADDER_VERDICT"] = overall

    # ---- Amendment 2: the in-distribution control, reported beside grey -------------------
    if "mismatch" in verd:
        mv = verd["mismatch"]
        res["MISMATCH_VERDICT"] = mv["verdict"]
        res["MISMATCH_Delta_image"] = mv["Delta_image"]
        # A mismatched REAL image is in distribution and carries full visual information about
        # some scene, just not this one. If Delta_image is material here while it is null under
        # grey, the grey null is partly an artefact of a degenerate input, and that overrides
        # whatever the degradation ladder says.
        if mv["verdict"] == "IMAGE-CONTRIBUTES" and \
                mv["Delta_image"]["point"] is not None and \
                mv["Delta_image"]["point"] >= MARGIN and \
                mv["Delta_image"]["ci"][0] is not None and mv["Delta_image"]["ci"][0] > 0:
            res["OVERALL_VERDICT"] = "IN-DISTRIBUTION-ABLATION-DISAGREES"
        elif mv["verdict"] == "NO-DATA (degenerate arm)":
            res["OVERALL_VERDICT"] = overall + " (mismatch arm degenerate: NO-DATA)"
        else:
            res["OVERALL_VERDICT"] = overall + f" + mismatch:{mv['verdict']}"
    else:
        res["MISMATCH_VERDICT"] = "NOT-RUN"
        res["OVERALL_VERDICT"] = overall

    # descriptive monotonicity against measured severity, if the dose file exists
    dose_p = f"{OUT}/lad_dose.json"
    if os.path.exists(dose_p):
        dose = dict(json.load(open(dose_p))["emb_cos"])
        mm_p = f"{OUT}/lad_dose_mm.json"
        if os.path.exists(mm_p):
            dose.update(json.load(open(mm_p))["emb_cos"])
        res["severity_emb_cos"] = dose
        # Descriptive only, and over the PIXEL ablations: the degradation rungs plus the two
        # uniform-field controls. `mismatch` is a different kind of ablation and a separate
        # registered primary, so putting it on the degradation severity axis would be a
        # category error; its severity is reported beside the others instead.
        px = [r for r in lrungs if r in dose and r != "mismatch"]
        xs = [1.0 - dose[r] for r in px]
        ys = [verd[r]["Delta_image"]["point"] for r in px]
        if have_grey and "grey" in dose:
            xs.append(1.0 - dose["grey"]); ys.append(pt["Dimg[grey]"])
        if len(xs) > 2:
            rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
            res["spearman_severity_vs_Dimage_pixel_rungs"] = float(np.corrcoef(rx, ry)[0, 1])
            res["spearman_n"] = len(xs)

    jdump(f"{OUT}/lad_l1_dprime.json", res)
    print("\n================ L1 LADDER ================", flush=True)
    print(f"PC1 PASS={res['PC1']['PASS']}  PC2 PASS={res['PC2']['PASS']}", flush=True)
    print(f"Delta_primary {pt['D[sighted]']:+.4f} "
          f"{res['interval']['D[sighted]']['ci']}", flush=True)
    print(f"Delta_pubgrey {pt['D[pubgrey]']:+.4f}  Delta_image(pubgrey) "
          f"{pt['Dimg[pubgrey]']:+.4f} {res['interval']['Dimg[pubgrey]']['ci']}", flush=True)
    for rg in lrungs:
        d = verd[rg]["Delta_image"]
        print(f"  {rg:10s} D_rung {res['interval'][f'D[{rg}]']['point']:+.4f}  "
              f"D_image {d['point']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] "
              f"w={d['ci_width']:.4f}  {verd[rg]['verdict']}", flush=True)
    if lad and lad.get("point") is not None:
        print(f"ladder mean Delta_image {lad['point']:+.4f} {lad['ci']}", flush=True)
    print(f"LADDER_VERDICT = {overall}", flush=True)
    print(f"MISMATCH_VERDICT = {res['MISMATCH_VERDICT']}", flush=True)
    print(f"OVERALL_VERDICT = {res['OVERALL_VERDICT']}", flush=True)
    sentinel("SCORE_L1")


if __name__ == "__main__":
    main()
