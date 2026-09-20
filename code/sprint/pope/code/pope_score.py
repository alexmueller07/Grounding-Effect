"""POPE lane -- scoring, intervals and the registered verdict.

Pre-registered in SPRINT/POPE_PREREG.md.  Reads {OUT}/pope_<method>_<pixarm>.jsonl and writes
{OUT}/pope_results.json.  Sentinel POPE_SENTINEL_SCORE_OK.

BOOTSTRAP.  Paired image-level cluster bootstrap, B = 4000, default_rng(20260919).  A replicate
resamples the 500 image ids with replacement and carries ALL of that image's rows -- all three
splits, every arm -- into the replicate together, so every contrast is paired on the same
resampled image multiset.  Multiplicity is preserved.  The resample size is asserted equal to
500 on every replicate and set() never touches a resample.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from statistics import NormalDist

from pope_common import (DEGEN_EMPTY, DEGEN_UNPARSE, DEGEN_YES_HI, DEGEN_YES_LO, HALF_GAIN,
                         MDE_J, METHODS, NBOOT, BOOT_SEED, OUT, PC_ACC_CENTER, PC_ACC_HALF,
                         PC_F1_CENTER, PC_F1_HALF, SPLITS, TRUNC_MAX, CHANCE_ACC_HI)
import pope_pixels

_ND = NormalDist()
LADDER_ORDER = ["sighted", "noise25", "blur4", "noise50", "lowres", "blur16", "pshuffle",
                "noise100", "grey"]


# ------------------------------------------------------------------ parsers
def parse_lenient(text):
    """VERBATIM the official POPE mapping (RUCAIBox/POPE@main evaluate.py, sha256
    f0371dcec5f1bdbde017d1a02f5a9a04c53cb561a2874f3fb8f0718ce3d87036; identical logic in
    haotian-liu/LLaVA llava/eval/eval_pope.py).  Anything without a negation becomes 'yes'."""
    if text.find(".") != -1:
        text = text.split(".")[0]
    text = text.replace(",", "")
    words = text.split(" ")
    if "No" in words or "not" in words or "no" in words:
        return "no"
    return "yes"


_STRICT = re.compile(r"^[\s\W_]*(yes|no)\b", re.IGNORECASE)


def parse_strict(text):
    """The answer must BEGIN with the word yes or the word no.  Anything else is UNPARSEABLE
    and is counted, never silently mapped."""
    m = _STRICT.match(text or "")
    return m.group(1).lower() if m else None


# ------------------------------------------------------------------ metrics
def metrics(C):
    """C: (...,4) array of tp, fp, tn, fn.  Vectorised over leading axes."""
    tp, fp, tn, fn = (C[..., 0].astype(float), C[..., 1].astype(float),
                      C[..., 2].astype(float), C[..., 3].astype(float))
    n = tp + fp + tn + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = (tp + tn) / n
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), np.nan)
        rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-12), np.nan)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), np.nan)
        yes_ratio = (tp + fp) / n
        H = tp / np.maximum(tp + fn, 1e-12)
        F = fp / np.maximum(fp + tn, 1e-12)
        # log-linear (add-one-half) correction, applied to EVERY arm unconditionally
        Hc = (tp + 0.5) / (tp + fn + 1.0)
        Fc = (fp + 0.5) / (fp + tn + 1.0)
    vz = np.vectorize(_ND.inv_cdf)
    zH, zF = vz(Hc), vz(Fc)
    return {"accuracy": acc * 100.0, "precision": prec * 100.0, "recall": rec * 100.0,
            "f1": f1 * 100.0, "yes_ratio": yes_ratio, "H": H, "F": F, "J": H - F,
            "dprime": zH - zF, "c": -0.5 * (zH + zF)}


def identity_check(C, m):
    """FREE CROSS-CHECK on a 1:1 balanced split (POPE's splits are exactly 1500/1500, and
    pooled exactly 4500/4500).  With m = #yes = #no:

        H = Recall,   Accuracy = (H + 1 - F)/2,   so  F = Recall + 1 - 2*Accuracy
        J = 2*Accuracy - 1,   Precision = H/(H+F),   F1 = 2H/(1 + H + F)

    The POPE-standard table and the H/F/J/d'/c decomposition are therefore two coordinate
    systems on the same two rates, and any disagreement between them is a bug in one of the
    two code paths, not a finding.  Asserted on the FULL-SAMPLE point estimates only --
    bootstrap replicates resample images and are not balanced, so the identity does not hold
    there and is not claimed there.
    """
    tp, fp, tn, fn = (float(C[0]), float(C[1]), float(C[2]), float(C[3]))
    npos, nneg = tp + fn, fp + tn
    if npos != nneg or npos == 0:
        return {"balanced": False, "n_pos": npos, "n_neg": nneg}
    H, F, acc = m["H"], m["F"], m["accuracy"] / 100.0
    rec, prec, f1 = m["recall"] / 100.0, m["precision"] / 100.0, m["f1"] / 100.0
    d = {"H_vs_recall": abs(H - rec),
         "F_vs_identity": abs(F - (rec + 1.0 - 2.0 * acc)),
         "J_vs_2acc_minus_1": abs(m["J"] - (2.0 * acc - 1.0))}
    if H + F > 0:
        d["precision_vs_identity"] = abs(prec - H / (H + F))
        d["f1_vs_identity"] = abs(f1 - 2.0 * H / (1.0 + H + F))
    worst = max(d.values())
    assert worst < 1e-9, f"FATAL_IDENTITY_VIOLATION {d}"
    return {"balanced": True, "n_pos": npos, "worst_abs_deviation": worst}


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def pt_ci(point, reps):
    lo, hi = ci(reps)
    return {"point": float(point), "ci95": [lo, hi], "half_width": float((hi - lo) / 2.0)}


# ------------------------------------------------------------------ load
def load_cells():
    cells = {}
    for f in sorted(glob.glob(f"{OUT}/pope_*_*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        if not rows:
            continue
        # NON-DISCRIMINATING-READ GUARD.  A `.get("text", "")` style read of a field that does
        # not exist returns "" for every row, which drives the yes-rate to 0 in EVERY arm --
        # indistinguishable from a genuine always-"No" blind arm.  The field name is therefore
        # asserted, never defaulted, and emptiness is measured rather than assumed away.
        for req in ("answer", "label", "method", "pixarm", "split", "question_id",
                    "image_id", "pid_sha"):
            missing = [i for i, r in enumerate(rows[:50]) if req not in r]
            assert not missing, f"FATAL_MISSING_FIELD {req} in {f} rows {missing[:3]}"
        assert all("answer" in r for r in rows), f"FATAL_MISSING_ANSWER_FIELD {f}"
        n_empty = sum(1 for r in rows if not str(r["answer"]).strip())
        assert n_empty == 0, f"FATAL_EMPTY_ANSWERS {f}: {n_empty}/{len(rows)}"
        assert len({r["method"] for r in rows}) == 1 and \
               len({r["pixarm"] for r in rows}) == 1, f"FATAL_MIXED_CELL {f}"
        key = (rows[0]["method"], rows[0]["pixarm"])
        cells[key] = rows
    assert cells, "FATAL_NO_CELLS"
    return cells


def main():
    cells = load_cells()
    print(f"[score] {len(cells)} cells: {sorted(cells)}", flush=True)

    # ---- master item order and the image index -------------------------------------------
    ref = cells[sorted(cells)[0]]
    master = [(r["split"], r["question_id"]) for r in ref]
    img_of = {(r["split"], r["question_id"]): r["image_id"] for r in ref}
    lab_of = {(r["split"], r["question_id"]): r["label"] for r in ref}
    images = sorted({v for v in img_of.values()})
    iidx = {im: i for i, im in enumerate(images)}
    n_img = len(images)
    print(f"[score] {len(master)} items over {n_img} images", flush=True)

    # ---- G2 token identity + item alignment across cells ---------------------------------
    pid_ref = {(r["split"], r["question_id"]): r["pid_sha"] for r in ref}
    tok_ident, misaligned = True, []
    for k, rows in cells.items():
        if [(r["split"], r["question_id"]) for r in rows] != master:
            misaligned.append(k)
            tok_ident = False
            continue
        for r in rows:
            if r["pid_sha"] != pid_ref[(r["split"], r["question_id"])]:
                tok_ident = False
                misaligned.append((k, r["split"], r["question_id"]))
                break
    assert not misaligned, f"FATAL_G2_TOKEN_IDENTITY {misaligned[:5]}"

    # ---- per-cell predictions, screens, per-image counts ---------------------------------
    counts, screens, levels = {}, {}, {}
    for k, rows in cells.items():
        meta = json.load(open(f"{OUT}/pope_{k[0]}_{k[1]}_meta.json"))
        pred_l = np.array([parse_lenient(r["answer"]) == "yes" for r in rows])
        strict = [parse_strict(r["answer"]) for r in rows]
        unparse = float(np.mean([s is None for s in strict]))
        lab = np.array([r["label"] == "yes" for r in rows])
        spl = np.array([r["split"] for r in rows])
        img = np.array([iidx[r["image_id"]] for r in rows])
        for s in list(SPLITS) + ["pooled"]:
            m = np.ones(len(rows), bool) if s == "pooled" else (spl == s)
            M = np.zeros((n_img, 4), dtype=np.int64)
            np.add.at(M, (img[m], 0), (pred_l[m] & lab[m]).astype(np.int64))
            np.add.at(M, (img[m], 1), (pred_l[m] & ~lab[m]).astype(np.int64))
            np.add.at(M, (img[m], 2), (~pred_l[m] & ~lab[m]).astype(np.int64))
            np.add.at(M, (img[m], 3), (~pred_l[m] & lab[m]).astype(np.int64))
            counts[(k[0], k[1], s)] = M
            lv = {kk: float(vv) for kk, vv in metrics(M.sum(0)).items()}
            lv["_identity"] = identity_check(M.sum(0), lv)
            levels[(k[0], k[1], s)] = lv
        yr = float(pred_l.mean())
        flags = []
        # BOTH constant regimes, named separately.  On a 1:1 balanced split a constant
        # responder scores accuracy 0.50 and is NOT evidence of anything: Lan et al.
        # (arXiv:2605.22903) report blinded LLaVA-1.5-7B on POPE at Acc 0.50 / Prec 0.00 /
        # Rec 0.00 -- an always-"No" responder -- and read the 0.50 as evidence the model is
        # "not independent of visual input".  That is the misreading this paper exists to
        # prevent, and this screen is what stops us making it.
        if yr >= DEGEN_YES_HI:
            flags.append("DEGENERATE-CONSTANT")
            flags.append("DEGENERATE-CONSTANT-YES")
        if yr <= DEGEN_YES_LO:
            flags.append("DEGENERATE-CONSTANT")
            flags.append("DEGENERATE-CONSTANT-NO")
        if unparse > DEGEN_UNPARSE:
            flags.append("DEGENERATE-UNPARSEABLE")
        if meta["empty_rate"] > DEGEN_EMPTY:
            flags.append("DEGENERATE-EMPTY")
        if meta["trunc_rate"] > TRUNC_MAX:
            flags.append("BUDGET-SUSPECT")
        # The screen runs per split too: an arm can be constant on the adversarial split and
        # not on the random one, and the split-level contrast would then be undefined while
        # the pooled one is not.
        per_split_screen = {}
        for s in SPLITS:
            m = (spl == s)
            if not m.any():
                continue
            yrs = float(pred_l[m].mean())
            us = float(np.mean([x is None for x, mm in zip(strict, m) if mm]))
            fl = []
            if yrs >= DEGEN_YES_HI:
                fl += ["DEGENERATE-CONSTANT", "DEGENERATE-CONSTANT-YES"]
            if yrs <= DEGEN_YES_LO:
                fl += ["DEGENERATE-CONSTANT", "DEGENERATE-CONSTANT-NO"]
            if us > DEGEN_UNPARSE:
                fl.append("DEGENERATE-UNPARSEABLE")
            per_split_screen[s] = {"yes_ratio": yrs, "unparseable_rate": us, "flags": fl}
        # agreement between the two parsers, on the items the strict one can read
        agree = [(parse_lenient(r["answer"]) == s) for r, s in zip(rows, strict) if s]
        screens[f"{k[0]}|{k[1]}"] = {
            "yes_ratio": yr, "unparseable_rate": unparse,
            "empty_rate": meta["empty_rate"], "trunc_rate": meta["trunc_rate"],
            "mean_tokens": meta["mean_tokens"], "median_tokens": meta["median_tokens"],
            "n_distinct_answers": meta["n_distinct_answers"],
            "mean_img_mass": meta["mean_img_mass"],
            "pai_applied_steps": meta["pai_applied_steps"],
            "pai_applied_prefill": meta["pai_applied_prefill"],
            "strict_lenient_agreement": float(np.mean(agree)) if agree else float("nan"),
            "flags": flags,
            "per_split": per_split_screen,
            "answer_dist": meta.get("answer_dist"),
            "answer_dist_by_split": meta.get("answer_dist_by_split"),
        }

    # ---- bootstrap draws, shared by every cell -------------------------------------------
    def draws(seed):
        rng = np.random.default_rng(seed)
        D = np.empty((NBOOT, n_img), dtype=np.int32)
        for b in range(NBOOT):
            pick = rng.integers(0, n_img, size=n_img)
            assert pick.size == n_img, "FATAL_BOOTSTRAP_LENGTH"
            cnt = np.bincount(pick, minlength=n_img)
            assert int(cnt.sum()) == n_img, "FATAL_BOOTSTRAP_MULTIPLICITY"
            if b < 20:
                assert int((cnt > 0).sum()) < n_img, "FATAL_BOOTSTRAP_NO_DUPLICATES"
            D[b] = cnt
        return D

    def rep(D, key):
        return metrics(D @ counts[key])

    ident = [v["_identity"] for v in levels.values() if v["_identity"].get("balanced")]
    out = {"decoding_mode_matched": (
               "Every arm uses do_sample=False, num_beams=1, the same registered attention "
               "kernel and the same two-stream loop. vanilla is alpha=0 (sc+0*|sc|, exact in "
               "IEEE) and gamma=1.0 (PAI's own CFG.py short-circuit), so it is kernel- and "
               "loop-matched, not a different code path. NO contrast in this lane compares "
               "across decoding modes."),
           "identity_check": {"n_balanced_cells": len(ident),
                              "worst_abs_deviation": (max(i["worst_abs_deviation"]
                                                          for i in ident) if ident else None)},
           "n_images": n_img, "n_items": len(master), "nboot": NBOOT,
           "boot_seed": BOOT_SEED, "cells": sorted(f"{a}|{b}" for a, b in cells),
           "g2_token_identity": tok_ident, "screens": screens,
           "levels": {f"{a}|{b}|{c}": v for (a, b, c), v in levels.items()},
           "contrasts": {}, "positive_control": {}, "verdict": {}}

    D = draws(BOOT_SEED)
    fullpt = {k: metrics(counts[k].sum(0)) for k in counts}
    reps = {}

    def R(key, seed_D, tagD="main"):
        # keyed by an EXPLICIT label: id() can be recycled after a previous array is freed,
        # which would silently serve replicates from the wrong seed.
        if (key, tagD) not in reps:
            reps[(key, tagD)] = rep(seed_D, key)
        return reps[(key, tagD)]

    # ---- positive control ------------------------------------------------------------------
    vs = ("vanilla", "sighted", "pooled")
    if vs in counts:
        acc = fullpt[vs]["accuracy"]
        f1 = fullpt[vs]["f1"]
        per = {s: fullpt[("vanilla", "sighted", s)]["accuracy"] for s in SPLITS}
        pc_a = (abs(acc - PC_ACC_CENTER) <= PC_ACC_HALF and
                abs(f1 - PC_F1_CENTER) <= PC_F1_HALF)
        pc_a2 = per["random"] >= per["popular"] >= per["adversarial"]
        out["positive_control"]["PC-A"] = {
            "pass": bool(pc_a), "blocking": True,
            "measured_accuracy": float(acc), "measured_f1": float(f1),
            "band_accuracy": [PC_ACC_CENTER - PC_ACC_HALF, PC_ACC_CENTER + PC_ACC_HALF],
            "band_f1": [PC_F1_CENTER - PC_F1_HALF, PC_F1_CENTER + PC_F1_HALF],
            "target": "PAI arXiv:2407.21771 Table 2 single-turn greedy Vanilla LLaVA-1.5-7B, "
                      "3-split average: acc 84.76, F1 85.51"}
        out["positive_control"]["PC-A2"] = {
            "pass": bool(pc_a2), "blocking": True, "per_split_accuracy": per,
            "condition": "random >= popular >= adversarial"}
        if ("vanilla", "grey", "pooled") in counts:
            for s in list(SPLITS) + ["pooled"]:
                d_acc = (fullpt[("vanilla", "sighted", s)]["accuracy"]
                         - fullpt[("vanilla", "grey", s)]["accuracy"])
                r_acc = (R(("vanilla", "sighted", s), D)["accuracy"]
                         - R(("vanilla", "grey", s), D)["accuracy"])
                d_dp = (fullpt[("vanilla", "sighted", s)]["dprime"]
                        - fullpt[("vanilla", "grey", s)]["dprime"])
                r_dp = (R(("vanilla", "sighted", s), D)["dprime"]
                        - R(("vanilla", "grey", s), D)["dprime"])
                out["positive_control"].setdefault("PC-B", {})[s] = pt_ci(d_acc, r_acc)
                out["positive_control"].setdefault("PC-C", {})[s] = pt_ci(d_dp, r_dp)
            out["positive_control"]["PC-B_pass"] = bool(all(
                out["positive_control"]["PC-B"][s]["point"] >= 10.0
                and out["positive_control"]["PC-B"][s]["ci95"][0] > 0 for s in SPLITS))
            out["positive_control"]["PC-C_pass"] = bool(all(
                out["positive_control"]["PC-C"][s]["ci95"][0] > 0 for s in SPLITS))

    # ---- the decomposition -----------------------------------------------------------------
    def decompose(method, ablation, split, Dm, tagD="main"):
        ks = (method, "sighted", split)
        kg = (method, ablation, split)
        vs_, vg_ = ("vanilla", "sighted", split), ("vanilla", ablation, split)
        if not all(k in counts for k in (ks, kg, vs_, vg_)):
            return None
        res = {}
        for q in ("J", "accuracy", "dprime", "c", "H", "F"):
            ds = fullpt[ks][q] - fullpt[vs_][q]
            da = fullpt[kg][q] - fullpt[vg_][q]
            rs = R(ks, Dm, tagD)[q] - R(vs_, Dm, tagD)[q]
            ra = R(kg, Dm, tagD)[q] - R(vg_, Dm, tagD)[q]
            res[q] = {"delta_sighted": pt_ci(ds, rs), "delta_ablated": pt_ci(da, ra),
                      "delta_image": pt_ci(ds - da, rs - ra)}
            if q == "J":
                with np.errstate(divide="ignore", invalid="ignore"):
                    rho = np.where(np.abs(rs) > 1e-9, ra / rs, np.nan)
                res[q]["rho_surviving"] = pt_ci(da / ds if abs(ds) > 1e-9 else float("nan"),
                                                rho)
        return res

    for method in [m for m in METHODS if m != "vanilla"]:
        for ab in [a for a in pope_pixels.PIXEL_ARMS if a != "sighted"]:
            for s in list(SPLITS) + ["pooled"]:
                d = decompose(method, ab, s, D)
                if d:
                    out["contrasts"][f"{method}|{ab}|{s}"] = d

    # ---- registered verdict ------------------------------------------------------------------
    def verdict(method, ablation="grey", split="pooled"):
        key = f"{method}|{ablation}|{split}"
        if key not in out["contrasts"]:
            return {"outcome": "NOT-RUN"}
        d = out["contrasts"][key]["J"]
        ds, da, di = d["delta_sighted"], d["delta_ablated"], d["delta_image"]
        sc_v = screens.get(f"vanilla|{ablation}", {}).get("flags", [])
        sc_m = screens.get(f"{method}|{ablation}", {}).get("flags", [])
        bad = [f for f in sc_v + sc_m
               if f in ("DEGENERATE-CONSTANT", "DEGENERATE-UNPARSEABLE")]
        v = {"delta_sighted": ds, "delta_ablated": da, "delta_image": di,
             "rho_surviving": d["rho_surviving"],
             "ablated_arm_flags": {"vanilla": sc_v, method: sc_m}}
        if bad:
            v["outcome"] = "BLIND-ARM-DEGENERATE"
            v["note"] = ("ablated arm degenerate; delta_image UNDEFINED, not a null and not "
                         "an attribution (prereg 9)")
            return v
        if ds["point"] < MDE_J or (ds["ci95"][0] <= 0 <= ds["ci95"][1]):
            v["outcome"] = "NO-GAIN-TO-DECOMPOSE"
            v["note"] = f"delta_sighted {ds['point']:.4f} below MDE_J={MDE_J} or CI spans 0"
            return v
        if di["half_width"] > ds["point"]:
            v["outcome"] = "UNDERPOWERED"
            v["note"] = (f"CI half-width of delta_image {di['half_width']:.4f} exceeds "
                         f"delta_sighted {ds['point']:.4f}: cannot distinguish all from none")
            return v
        line = HALF_GAIN * ds["point"]
        lo, hi = di["ci95"]
        if lo <= 0 <= hi and hi < line:
            v["outcome"] = "NOT-IMAGE-ATTRIBUTABLE"
        elif lo > 0 and lo > line:
            v["outcome"] = "IMAGE-ATTRIBUTABLE"
        elif lo > 0:
            v["outcome"] = "PARTIAL"
        else:
            v["outcome"] = "CANNOT-RESOLVE"
        v["half_gain_line"] = float(line)
        return v

    # PRIMARY: pai_full, grey ablation, pooled.  CO-PRIMARY (prereg Amendment 1): the same
    # under the mismatched-real-image ablation.  The headline attribution claim requires the
    # two ablations to AGREE; if they do not, the registered outcome is ABLATION-DISAGREEMENT
    # and neither is promoted.
    v_grey = verdict("pai_full", "grey", "pooled")
    v_mis = verdict("pai_full", "mismatch", "pooled")
    out["verdict"]["PRIMARY_grey"] = v_grey
    out["verdict"]["COPRIMARY_mismatch"] = v_mis
    ATTRIB = ("IMAGE-ATTRIBUTABLE", "NOT-IMAGE-ATTRIBUTABLE", "PARTIAL")
    if v_mis.get("outcome") == "NOT-RUN":
        comb = {"outcome": v_grey.get("outcome"),
                "note": "mismatch ablation not run; grey alone, co-primary unmet"}
    elif v_grey.get("outcome") == "BLIND-ARM-DEGENERATE" and \
            v_mis.get("outcome") in ATTRIB:
        comb = {"outcome": v_mis["outcome"],
                "note": "registered contingency: the grey arm is degenerate, so the "
                        "in-distribution mismatched-real-image ablation carries the primary"}
    elif v_grey.get("outcome") == v_mis.get("outcome"):
        comb = {"outcome": v_grey.get("outcome"), "note": "both ablations agree"}
    elif v_grey.get("outcome") in ATTRIB and v_mis.get("outcome") in ATTRIB:
        comb = {"outcome": "ABLATION-DISAGREEMENT",
                "note": f"grey says {v_grey['outcome']}, mismatch says {v_mis['outcome']}; "
                        "neither is promoted (prereg Amendment 1)"}
    else:
        comb = {"outcome": "ABLATION-DISAGREEMENT",
                "note": f"grey={v_grey.get('outcome')} mismatch={v_mis.get('outcome')}"}
    comb["grey"] = v_grey.get("outcome")
    comb["mismatch"] = v_mis.get("outcome")
    out["verdict"]["PRIMARY"] = comb

    for m in [x for x in METHODS if x != "vanilla"]:
        for ab in ("grey", "mismatch"):
            for s in list(SPLITS) + ["pooled"]:
                out["verdict"][f"S_{m}|{ab}|{s}"] = verdict(m, ab, s)

    # ---- S5 blindability ---------------------------------------------------------------------
    for ab in ("grey", "mismatch"):
        if ("vanilla", ab, "pooled") not in counts:
            continue
        b, dpr = {}, {}
        for s in list(SPLITS) + ["pooled"]:
            k = ("vanilla", ab, s)
            b[s] = pt_ci(fullpt[k]["accuracy"], R(k, D)["accuracy"])
            dpr[s] = pt_ci(fullpt[k]["dprime"], R(k, D)["dprime"])
        deg = screens.get(f"vanilla|{ab}", {}).get("flags", [])
        blind = all(b[s]["ci95"][0] <= CHANCE_ACC_HI * 100 for s in SPLITS) and not deg
        out["verdict"][f"S5_blindability_{ab}"] = {
            "outcome": "BLINDABLE" if blind else
                       ("BLIND-ARM-DEGENERATE" if deg else "NOT-BLINDABLE"),
            "vanilla_accuracy": b, "vanilla_dprime": dpr, "chance": 50.0, "flags": deg,
            "yes_ratio": screens.get(f"vanilla|{ab}", {}).get("yes_ratio")}

    # ---- S4 ladder ----------------------------------------------------------------------------
    lad = {}
    for a in LADDER_ORDER:
        k = ("vanilla", a, "pooled")
        if k in counts:
            lad[a] = {"accuracy": pt_ci(fullpt[k]["accuracy"], R(k, D)["accuracy"]),
                      "J": pt_ci(fullpt[k]["J"], R(k, D)["J"]),
                      "yes_ratio": float(fullpt[k]["yes_ratio"]),
                      "dprime": float(fullpt[k]["dprime"])}
    if len(lad) >= 3:
        seq = [lad[a]["J"]["point"] for a in LADDER_ORDER if a in lad]
        lad["_monotone_nonincreasing"] = bool(all(
            seq[i] >= seq[i + 1] - 1e-12 for i in range(len(seq) - 1)))
        lad["_order"] = [a for a in LADDER_ORDER if a in lad]
    out["ladder_vanilla"] = lad

    # ---- alternate seed on the primary ----------------------------------------------------------
    D2 = draws(BOOT_SEED + 1)
    out["alt_seed_primary"] = {"seed": BOOT_SEED + 1}
    for ab in ("grey", "mismatch"):
        alt = decompose("pai_full", ab, "pooled", D2, tagD="alt")
        if alt:
            out["alt_seed_primary"][ab] = alt["J"]

    json.dump(out, open(f"{OUT}/pope_results.json", "w"), indent=1, default=float)
    print(json.dumps({"verdict": out["verdict"], "positive_control": out["positive_control"]},
                     indent=1, default=float), flush=True)
    print("POPE_SENTINEL_SCORE_OK", flush=True)


if __name__ == "__main__":
    main()
