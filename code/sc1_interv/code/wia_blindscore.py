"""WIA-BASELINE scorer (WIA_BASELINE_PREREG.md sections 3-6).

Families 1 and 2, sighted rows frozen, blind rows from this lane. Reports, per family:

  PRIMARY      Delta_sight = Delta_primary - Delta_blind
  CO-PRIMARY   Delta_blind = gap(cap5g) - gap(cap1g)
               gap(cap1g), gap(cap5g)            <- the direct test of SUBMISSION.md L73-74
               Delta_sight_klen, Delta_blind_klen, rho_blind
  SECONDARIES  CP-R, CP-C2, 192 reconstruction (F1), doubled/empty removed, seed alt, per-arm
  NOPFX        base rates from the frozen free-running captions (no generation)
  DIL          the dilution diagnostic (fires no cell)
  GATE_ATTENDS recall(real image) - recall(gray image) on free-running captions

Units / gather / gap_raw / gap_object / delta_object / boot are PORTED UNMODIFIED from
wia_corrobscore.py; trunc_text is wia_lnscore_lm.trunc_text's rule with family 2's one declared
adaptation (word-start marker "G-dot" instead of SentencePiece "_").

Delta_sight is computed INSIDE the same bootstrap loop as its two parents, on the same image draw
(pre-reg section 4). Writes out/wia_baseline.json. Sentinel printed by Python; the exit code is
NEVER the success signal.
"""
import os, sys, json, math, re
import numpy as np
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/sc1_gen/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")
from j5_common import load_coco_sample, ChairScorer, read_jsonl, sentinel

SMOKE = os.environ.get("BLIND_SMOKE", "0") == "1"
OUT_I = "/data/alexmueller/sc1_interv/out"
OUT_G = "/data/alexmueller/sc1_gen/out"
J5OUT = "/data/alexmueller/j5_gates/out"
SUF = "_smoke" if SMOKE else ""
RES = f"{OUT_I}/wia_baseline{SUF}.json"

NBOOT = 200 if SMOKE else 4000
SEED = 20260912
SEED_ALT = 20260913
N_TRUE_FLOOR = 100
WIDTH_CEIL = 0.14                  # pre-reg 5.3, derived
RHO_MIN = {"F1": 0.268, "F2": 0.283}   # pre-reg 5.3 step 5
CTL_TOL = 5e-3
CTL_REF = {"F1": 0.4859, "F2": 0.3780}
SYN = "/data/alexmueller/j5_gates/code/synonyms.txt"

FAM = {
    "F1": {"sighted": f"{OUT_I}/at_corrob{SUF}.jsonl", "blind": f"{OUT_I}/at_blind1{SUF}.jsonl",
           "af": f"{J5OUT}/af.jsonl", "af_gray": f"{OUT_I}/af_gray1{SUF}.jsonl",
           "cap": 384, "ws": "▁", "model": "llava-hf/llava-1.5-7b-hf",
           "arms": ("cap1", "cap5", "cap1g", "cap5g"), "recon192": True},
    "F2": {"sighted": f"{OUT_G}/m2_corrob{SUF}.jsonl", "blind": f"{OUT_G}/m2_blind{SUF}.jsonl",
           "af": f"{OUT_G}/m2_af.jsonl", "af_gray": f"{OUT_G}/m2_af_gray{SUF}.jsonl",
           "cap": 192, "ws": "Ġ", "model": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
           "arms": ("cap1", "cap5", "m2cap1g", "m2cap5g"), "recon192": False},
}


# ------------------------------------------------------------------ ported verbatim
def _assert_multiplicity(sel, n, tag):
    assert len(sel) == n, f"FATAL_BOOTSTRAP_LENGTH {tag}: {len(sel)} != {n}"
    if n >= 20:
        assert len(set(sel)) < n, f"FATAL_BOOTSTRAP_NO_DUPLICATES {tag}"


def trunc_text(tok, ids, L, ws):
    if L is None or L >= len(ids):
        return tok.decode(ids, skip_special_tokens=True), False, False
    text = tok.decode(ids[:L], skip_special_tokens=True)
    nxt = tok.convert_ids_to_tokens(int(ids[L]))
    midword = (not nxt.startswith(ws)) and bool(re.match(r"^[A-Za-z]", nxt))
    if midword:
        t = text.rstrip()
        j = max(t.rfind(" "), t.rfind("\n"))
        text = t[:j + 1] if j >= 0 else ""
    return text, True, midword


class Units:
    """One row per (cell, prefix object). `st` is an optional per-unit stratum index."""

    def __init__(self, img, obj, tru, car, n_img, st=None):
        o = np.argsort(img, kind="stable")
        self.img = np.asarray(img)[o]
        self.obj = np.asarray(obj)[o]
        self.tru = np.asarray(tru)[o].astype(bool)
        self.car = np.asarray(car)[o].astype(bool)
        self.st = np.asarray(st)[o] if st is not None else None
        self.counts = np.bincount(self.img, minlength=n_img)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])

    def gather(self, sel):
        c = self.counts[sel]
        tot = int(c.sum())
        if tot == 0:
            return np.empty(0, dtype=np.int64)
        keep = c > 0
        s, c = self.starts[sel][keep], c[keep]
        out = np.ones(tot, dtype=np.int64)
        out[0] = s[0]
        if len(s) > 1:
            idx = np.cumsum(c)[:-1]
            out[idx] = s[1:] - (s[:-1] + c[:-1]) + 1
        return np.cumsum(out)


def gap_raw(u, g):
    t, c = u.tru[g], u.car[g]
    nt, nf = int(t.sum()), int((~t).sum())
    if nt == 0 or nf == 0:
        return None, (nt, nf, 0, 0)
    ct, cf = int(c[t].sum()), int(c[~t].sum())
    return ct / nt - cf / nf, (nt, nf, ct, cf)


def gap_object(u, g, n_obj):
    o = u.obj[g]
    t, c = u.tru[g], u.car[g]
    nt = np.bincount(o[t], minlength=n_obj)
    ct = np.bincount(o[t], weights=c[t], minlength=n_obj)
    nf = np.bincount(o[~t], minlength=n_obj)
    cf = np.bincount(o[~t], weights=c[~t], minlength=n_obj)
    return nt, ct, nf, cf


def delta_object(uA, uB, gA, gB, n_obj):
    ntA, ctA, nfA, cfA = gap_object(uA, gA, n_obj)
    ntB, ctB, nfB, cfB = gap_object(uB, gB, n_obj)
    stt = (ntA > 0) & (ntB > 0)
    sff = (nfA > 0) & (nfB > 0)
    if not stt.any() or not sff.any():
        return None
    wt, wf = np.minimum(ntA, ntB) * stt, np.minimum(nfA, nfB) * sff
    if wt.sum() == 0 or wf.sum() == 0:
        return None
    rtA = np.where(ntA > 0, ctA / np.where(ntA > 0, ntA, 1), 0.0)
    rtB = np.where(ntB > 0, ctB / np.where(ntB > 0, ntB, 1), 0.0)
    rfA = np.where(nfA > 0, cfA / np.where(nfA > 0, nfA, 1), 0.0)
    rfB = np.where(nfB > 0, cfB / np.where(nfB > 0, nfB, 1), 0.0)
    gA_ = float((wt * rtA).sum() / wt.sum() - (wf * rfA).sum() / wf.sum())
    gB_ = float((wt * rtB).sum() / wt.sum() - (wf * rfB).sum() / wf.sum())
    return gB_ - gA_


def delta_strat(uA, uB, gA, gB, n_s):
    """Equal-weight direct standardisation over a per-unit stratum (pre-reg 5.7 item 2)."""
    def cells(u, g):
        s, t, c = u.st[g], u.tru[g], u.car[g]
        nt = np.bincount(s[t], minlength=n_s)
        ct = np.bincount(s[t], weights=c[t], minlength=n_s)
        nf = np.bincount(s[~t], minlength=n_s)
        cf = np.bincount(s[~t], weights=c[~t], minlength=n_s)
        return nt, ct, nf, cf
    ntA, ctA, nfA, cfA = cells(uA, gA)
    ntB, ctB, nfB, cfB = cells(uB, gB)
    ok = (ntA > 0) & (ntB > 0) & (nfA > 0) & (nfB > 0)
    if not ok.any():
        return None
    w = ok.astype(float) / ok.sum()
    f = lambda nt, ct, nf, cf: float((w * (ct / np.where(nt > 0, nt, 1))).sum()
                                     - (w * (cf / np.where(nf > 0, nf, 1))).sum())
    return f(ntB, ctB, nfB, cfB) - f(ntA, ctA, nfA, cfA)


def ngram_ratio(text, n=8):
    """Distinct-n-gram ratio of a text. 1.0 means no repeated n-gram; low values mean looping.
    Reported for the prefix (pre-reg 5.5) and, additionally, for the continuation."""
    w = (text or "").lower().split()
    if len(w) < n + 1:
        return 1.0
    g = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(g)) / len(g)


def pct(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def joint_boot(units, stats, n_img, nboot=NBOOT, seed=SEED):
    """ONE image-clustered draw per replicate, every statistic evaluated on it.
    `stats` : {name: fn(gather_dict) -> float or None}. Returns {name: list of values}."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in stats}
    for b in range(nboot):
        sel = rng.integers(0, n_img, n_img)
        if b < 20:
            _assert_multiplicity(list(sel), n_img, f"boot#{b}")
        gd = {a: u.gather(sel) for a, u in units.items()}
        for k, fn in stats.items():
            v = fn(gd)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out[k].append(float(v))
    return out


def joint_boot_dict(units, fn, n_img, nboot=NBOOT, seed=SEED):
    """As joint_boot, but `fn(gather_dict)` returns a dict of statistics computed ONCE per
    replicate -- so Delta_sight and its two parents cannot drift apart and nothing is recomputed."""
    rng = np.random.default_rng(seed)
    out = None
    for b in range(nboot):
        sel = rng.integers(0, n_img, n_img)
        if b < 20:
            _assert_multiplicity(list(sel), n_img, f"boot#{b}")
        d = fn({a: u.gather(sel) for a, u in units.items()})
        if out is None:
            out = {k: [] for k in d}
        for k, v in d.items():
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out[k].append(float(v))
    return out or {}


def iv(point, vals):
    ci = pct(vals)
    return {"point": point, "ci": ci,
            "ci_width": None if ci[0] is None else ci[1] - ci[0], "n_boot_used": len(vals)}


def status(point, ci, width, n_true_min, n_false_min, ceil=WIDTH_CEIL):
    if point is None or ci[0] is None:
        return "UNDERPOWERED"
    if n_true_min < N_TRUE_FLOOR or n_false_min < N_TRUE_FLOOR:
        return "UNDERPOWERED"
    if width > ceil:
        return "UNDERPOWERED"
    if ci[0] > 0:
        return "POSITIVE"
    if ci[1] < 0:
        return "NEGATIVE"
    return "NULL"


# ------------------------------------------------------------------ per family
def score_family(key, chair, objset, gt, tok_of):
    cfg = FAM[key]
    A1, A5, B1, B5 = cfg["arms"]
    cap, ws = cfg["cap"], cfg["ws"]
    tok = tok_of(key)

    sr = list(read_jsonl(cfg["sighted"]))
    br = list(read_jsonl(cfg["blind"]))
    by = {}
    for r in sr + br:
        d = by.setdefault((r["image_id"], r["k"]), {})
        assert r["arm"] not in d, f"FATAL_DUPLICATE_ROW {key} {r['image_id']} {r['k']} {r['arm']}"
        assert r["cont_len"] == len(r["cont_ids"]), "FATAL_CONT_LEN_MISMATCH"
        d[r["arm"]] = r
    keys = sorted(by)
    g = {"cells": len(keys), "four_arms": 0, "budget_ok": 0, "prefix_identity": 0,
         "cap5_zero_cycles": 0, "cap1_doubling": 0, "cap_ok": 0, "self_ok": 0}
    for kk in keys:
        d = by[kk]
        assert set(d) == set(cfg["arms"]), f"FATAL_INCOMPLETE_CELL {key} {kk} {sorted(d)}"
        g["four_arms"] += 1
        bs = {d[a]["budget"] for a in d}
        g["budget_ok"] += (len(bs) == 1)
        g["prefix_identity"] += (d[B1]["prefix_text"] == d[A1]["prefix_text"]
                                 and d[B5]["prefix_text"] == d[A5]["prefix_text"])
        g["cap5_zero_cycles"] += (d[A5]["cycles"] == 0 and d[B5]["cycles"] == 0)
        g["cap1_doubling"] += (d[A1]["cycles"] > 0)
        g["cap_ok"] += all(d[a]["cont_len"] <= cap for a in d)
        g["self_ok"] += all(d[a].get("src_id", -1) != kk[0] for a in d)
    n = len(keys)
    assert g["four_arms"] == n and g["budget_ok"] == n, f"FATAL_BUDGET_MISMATCH_AT_SCORE {key}"
    assert g["prefix_identity"] == n, f"FATAL_BLIND_PREFIX_DRIFT {key} {g['prefix_identity']}/{n}"
    assert g["cap5_zero_cycles"] == n and g["cap_ok"] == n and g["self_ok"] == n, \
        f"FATAL_WRITTEN_ROW_GATE {key} {g}"

    imgs = sorted({k0 for k0, _ in keys})
    ix = {i: j for j, i in enumerate(imgs)}
    vocab = {}
    Pof = {}
    for kk in keys:
        for a in cfg["arms"]:
            P = objset(by[kk][a]["prefix_text"])
            Pof[(kk, a)] = P
            for o in P:
                vocab.setdefault(o, len(vocab))
    n_obj = len(vocab)
    # composition, from the written rows
    comp = {}
    for a in cfg["arms"]:
        nP = [len(Pof[(kk, a)]) for kk in keys]
        ment = [len(chair.mentions(by[kk][a]["prefix_text"])) for kk in keys]
        comp[a] = {"mean_nP": float(np.mean(nP)), "mean_mentions": float(np.mean(ment)),
                   "mean_mentions_per_type": float(np.mean([m / p if p else 0.0
                                                            for m, p in zip(ment, nP)]))}

    def cont_ids(r, recon):
        return r["cont_ids"][:192] if recon else r["cont_ids"]

    P_STRATA = 6   # |P| in {1,2,3,4,5,6+} -> index 0..5

    def build(arms, klen_pair=None, cellfilter=None, recon=False):
        ks = [kk for kk in keys if (cellfilter is None or cellfilter(by[kk]))]
        cols = {a: [[], [], [], [], []] for a in arms}
        stt = {a: {"L": [], "raw": [], "tr": 0, "mw": 0, "emp": 0, "cap": 0} for a in arms}
        for kk in ks:
            d = by[kk]
            L = None
            if klen_pair:
                L = min(len(cont_ids(d[a], recon)) for a in klen_pair)
            for a in arms:
                r = d[a]
                ids = cont_ids(r, recon)
                if L is None:
                    text, tr, mw, Le = tok.decode(ids, skip_special_tokens=True), False, False, len(ids)
                else:
                    text, tr, mw = trunc_text(tok, ids, L, ws)
                    Le = min(L, len(ids))
                s = stt[a]
                s["L"].append(Le); s["raw"].append(len(ids)); s["tr"] += tr; s["mw"] += mw
                s["emp"] += (len(ids) == 0); s["cap"] += (len(ids) >= (192 if recon else cap))
                P, C = Pof[(kk, a)], objset(text)
                sidx = min(max(len(P), 1), P_STRATA) - 1
                for o in P:
                    cols[a][0].append(ix[kk[0]]); cols[a][1].append(vocab[o])
                    cols[a][2].append(o in gt[kk[0]]); cols[a][3].append(o in C)
                    cols[a][4].append(sidx)
        U = {a: Units(cols[a][0], cols[a][1], cols[a][2], cols[a][3], len(imgs), st=cols[a][4])
             for a in arms}
        S = {a: {"n_cells": len(ks), "mean_cont_len": float(np.mean(s["raw"])),
                 "median_cont_len": float(np.median(s["raw"])),
                 "mean_matched_len": float(np.mean(s["L"])),
                 "frac_truncated": s["tr"] / max(len(ks), 1),
                 "frac_midword_cut": s["mw"] / max(len(ks), 1),
                 "empty_cont_rate": s["emp"] / max(len(ks), 1),
                 "at_max": s["cap"] / max(len(ks), 1)} for a, s in stt.items()}
        return U, S, ks

    arms4 = list(cfg["arms"])
    Uraw, Sraw, _ = build(arms4)
    nmin = {a: gap_raw(Uraw[a], np.arange(len(Uraw[a].img)))[1] for a in arms4}

    def d_raw(x, y):
        return lambda gd: (None if (gap_raw(Uraw[x], gd[x])[0] is None
                                    or gap_raw(Uraw[y], gd[y])[0] is None)
                           else gap_raw(Uraw[y], gd[y])[0] - gap_raw(Uraw[x], gd[x])[0])

    fP, fB = d_raw(A1, A5), d_raw(B1, B5)

    def _rates(a, gd):
        """(gap, r_true, r_false) for one arm on one draw."""
        g, (nt, nf, ct, cf) = gap_raw(Uraw[a], gd[a])
        return g, (ct / nt if nt else None), (cf / nf if nf else None)

    def all_stats(gd):
        """Every statistic on ONE image draw. Delta_sight is p - b from the same replicate
        (pre-reg section 4); nothing here is recomputed on a different draw.

        The r_true / r_false split is carried separately for BOTH conditions because the gap
        alone cannot distinguish the two readings this lane exists to separate: a symmetric
        crowding mechanism must push both rates DOWN as the number of competing prefix object
        types rises, whereas family 1's regime-controlled cells show r_true rising (+0.110) while
        r_false falls (-0.236). If the blind arm reproduces that asymmetry, the asymmetry is
        image-independent; if it does not, it is image-dependent even where the gap partly
        reproduces."""
        g1, t1, f1 = _rates(A1, gd)
        g5, t5, f5 = _rates(A5, gd)
        b1, tb1, fb1 = _rates(B1, gd)
        b5, tb5, fb5 = _rates(B5, gd)
        p = None if (g1 is None or g5 is None) else g5 - g1
        b = None if (b1 is None or b5 is None) else b5 - b1
        sub = lambda x, y: (None if (x is None or y is None) else x - y)
        d_rt_s, d_rf_s = sub(t5, t1), sub(f5, f1)
        d_rt_b, d_rf_b = sub(tb5, tb1), sub(fb5, fb1)
        return {"P": p, "B": b,
                "S": None if (p is None or b is None) else p - b,
                "ratio": None if (p is None or b is None or p == 0) else b / p,
                "gap_cap1g": b1, "gap_cap5g": b5, "gap_cap1": g1, "gap_cap5": g5,
                "r_true_cap1": t1, "r_true_cap5": t5, "r_false_cap1": f1, "r_false_cap5": f5,
                "r_true_cap1g": tb1, "r_true_cap5g": tb5,
                "r_false_cap1g": fb1, "r_false_cap5g": fb5,
                "d_r_true_sighted": d_rt_s, "d_r_false_sighted": d_rf_s,
                "d_r_true_blind": d_rt_b, "d_r_false_blind": d_rf_b,
                "d_r_true_sight": sub(d_rt_s, d_rt_b), "d_r_false_sight": sub(d_rf_s, d_rf_b),
                "P_c2": delta_object(Uraw[A1], Uraw[A5], gd[A1], gd[A5], n_obj),
                "B_c2": delta_object(Uraw[B1], Uraw[B5], gd[B1], gd[B5], n_obj),
                "P_pm": delta_strat(Uraw[A1], Uraw[A5], gd[A1], gd[A5], P_STRATA),
                "B_pm": delta_strat(Uraw[B1], Uraw[B5], gd[B1], gd[B5], P_STRATA)}

    V = joint_boot_dict(Uraw, all_stats, len(imgs), seed=SEED)
    allg = {a: np.arange(len(Uraw[a].img)) for a in arms4}
    pt = all_stats(allg)
    rho = float(np.corrcoef(V["P"][:min(len(V["P"]), len(V["B"]))],
                            V["B"][:min(len(V["P"]), len(V["B"]))])[0, 1]) \
        if len(V["P"]) > 2 and len(V["B"]) > 2 else None

    # Length-matched. Each contrast is truncated within its OWN pair (L_cell = min over that
    # pair), but BOTH are carried through ONE bootstrap loop on a shared image draw, so
    # Delta_sight_klen is a difference on a common draw exactly as the raw one is (pre-reg 5.2).
    Uk1, Sk1, _ = build(arms4, klen_pair=(A1, A5))
    Uk2, Sk2, _ = build(arms4, klen_pair=(B1, B5))
    Uk = {"A1k": Uk1[A1], "A5k": Uk1[A5], "B1k": Uk2[B1], "B5k": Uk2[B5]}

    def klen_stats(gd):
        g1, g5 = gap_raw(Uk["A1k"], gd["A1k"])[0], gap_raw(Uk["A5k"], gd["A5k"])[0]
        b1, b5 = gap_raw(Uk["B1k"], gd["B1k"])[0], gap_raw(Uk["B5k"], gd["B5k"])[0]
        p = None if (g1 is None or g5 is None) else g5 - g1
        b = None if (b1 is None or b5 is None) else b5 - b1
        return {"Pk": p, "Bk": b, "Sk": None if (p is None or b is None) else p - b}

    Vk = joint_boot_dict(Uk, klen_stats, len(imgs), seed=SEED)
    allk = {a: np.arange(len(u.img)) for a, u in Uk.items()}
    ptk = klen_stats(allk)

    # CP-R: both arms of the contrast uncapped -- separately per contrast
    def both_uncapped(pair):
        return lambda d: all(d[a]["cont_len"] < cap for a in pair)
    out_cpr = {}
    for lbl, pair in (("CP-R_sighted", (A1, A5)), ("CP-R_blind", (B1, B5))):
        Ur, Sr, ksr = build(arms4, cellfilter=both_uncapped(pair))
        if ksr:
            f = lambda gd, x=pair[0], y=pair[1]: (
                None if (gap_raw(Ur[x], gd[x])[0] is None or gap_raw(Ur[y], gd[y])[0] is None)
                else gap_raw(Ur[y], gd[y])[0] - gap_raw(Ur[x], gd[x])[0])
            Vr = joint_boot(Ur, {"d": f}, len(imgs), seed=SEED)
            allr = {a: np.arange(len(Ur[a].img)) for a in arms4}
            nt = {a: gap_raw(Ur[a], allr[a])[1] for a in arms4}
            out_cpr[lbl] = dict(iv(f(allr), Vr["d"]), n_cells=len(ksr),
                                n_true={a: nt[a][0] for a in pair},
                                n_false={a: nt[a][1] for a in pair},
                                r_true={a: nt[a][2] / max(nt[a][0], 1) for a in pair},
                                r_false={a: nt[a][3] / max(nt[a][1], 1) for a in pair},
                                per_arm={a: Sr[a] for a in pair})
            out_cpr[lbl]["status"] = status(out_cpr[lbl]["point"], out_cpr[lbl]["ci"],
                                            out_cpr[lbl]["ci_width"] or 9,
                                            min(nt[a][0] for a in pair),
                                            min(nt[a][1] for a in pair))
    # ---- empty-continuation and degeneracy accounting -----------------------------------
    # The scorer does NOT drop empty continuations: an empty continuation has an empty carried
    # set, so each of its prefix objects enters the r_true / r_false DENOMINATOR and never the
    # numerator, which shrinks that arm's gap by exactly the unit share sitting in empty cells.
    # A sibling lane was found today to have a 9.6x empty-rate asymmetry between its primary
    # arms. The exact arithmetic correction and the registered empty-removed subset are both
    # reported below, and the divergence between them is a reportable quantity.
    empty_acc = {}
    for a in arms4:
        ce = [kk for kk in keys if by[kk][a]["cont_len"] == 0]
        nt_e = sum(1 for kk in ce for o in Pof[(kk, a)] if o in gt[kk[0]])
        nf_e = sum(1 for kk in ce for o in Pof[(kk, a)] if o not in gt[kk[0]])
        nt_all, nf_all = nmin[a][0], nmin[a][1]
        gA, sA = gap_raw(Uraw[a], allg[a])
        ft = nt_e / nt_all if nt_all else 0.0
        ff = nf_e / nf_all if nf_all else 0.0
        rt_c = (sA[2] / (nt_all - nt_e)) if (nt_all - nt_e) > 0 else None
        rf_c = (sA[3] / (nf_all - nf_e)) if (nf_all - nf_e) > 0 else None
        L = np.array([by[kk][a]["cont_len"] for kk in keys], dtype=float)
        rc = np.array([ngram_ratio(by[kk][a]["cont_text"]) for kk in keys])
        rp = np.array([ngram_ratio(by[kk][a]["prefix_text"]) for kk in keys])
        empty_acc[a] = {
            "empty_cells": len(ce), "empty_cont_rate": len(ce) / max(len(keys), 1),
            "true_units_in_empty_cells": nt_e, "false_units_in_empty_cells": nf_e,
            "frac_true_units_in_empty_cells": ft, "frac_false_units_in_empty_cells": ff,
            "gap_as_scored": gA,
            "gap_arithmetically_corrected": (None if (rt_c is None or rf_c is None)
                                             else rt_c - rf_c),
            "r_true_as_scored": sA[2] / max(nt_all, 1), "r_true_corrected": rt_c,
            "r_false_as_scored": sA[3] / max(nf_all, 1), "r_false_corrected": rf_c,
            "mean_cont_len": float(L.mean()), "median_cont_len": float(np.median(L)),
            "p10_cont_len": float(np.percentile(L, 10)),
            "p90_cont_len": float(np.percentile(L, 90)),
            "frac_cont_len_le_3": float((L <= 3).mean()),
            "frac_cont_len_le_10": float((L <= 10).mean()),
            "mean_distinct_8gram_ratio_cont": float(rc.mean()),
            "frac_cont_8gram_ratio_below_0.5": float((rc < 0.5).mean()),
            "mean_distinct_8gram_ratio_prefix": float(rp.mean())}

    # registered empty-removed subset (pre-reg 5.5): cells where EVERY arm produced output
    Uer, Ser, kser = build(arms4, cellfilter=lambda d: all(d[a]["cont_len"] > 0 for a in arms4))
    empty_removed = {"n_cells": len(kser), "n_cells_dropped": len(keys) - len(kser),
                     "per_arm": Ser}
    if kser:
        def er_stats(gd):
            g1, g5 = gap_raw(Uer[A1], gd[A1])[0], gap_raw(Uer[A5], gd[A5])[0]
            b1, b5 = gap_raw(Uer[B1], gd[B1])[0], gap_raw(Uer[B5], gd[B5])[0]
            p = None if (g1 is None or g5 is None) else g5 - g1
            b = None if (b1 is None or b5 is None) else b5 - b1
            return {"P": p, "B": b, "S": None if (p is None or b is None) else p - b,
                    "gap_cap1": g1, "gap_cap5": g5, "gap_cap1g": b1, "gap_cap5g": b5}
        Ver = joint_boot_dict(Uer, er_stats, len(imgs), seed=SEED)
        aller = {a: np.arange(len(Uer[a].img)) for a in arms4}
        pter = er_stats(aller)
        empty_removed.update({k: iv(pter[k], Ver[k]) for k in pter})
        ner = {a: gap_raw(Uer[a], aller[a])[1] for a in arms4}
        empty_removed["n_true"] = {a: ner[a][0] for a in arms4}
        empty_removed["n_false"] = {a: ner[a][1] for a in arms4}

    # 192 reconstruction (family 1 only)
    recon = None
    if cfg["recon192"]:
        Ur, Sr, _ = build(arms4, recon=True)
        fPr = lambda gd: (None if (gap_raw(Ur[A1], gd[A1])[0] is None
                                   or gap_raw(Ur[A5], gd[A5])[0] is None)
                          else gap_raw(Ur[A5], gd[A5])[0] - gap_raw(Ur[A1], gd[A1])[0])
        fBr = lambda gd: (None if (gap_raw(Ur[B1], gd[B1])[0] is None
                                   or gap_raw(Ur[B5], gd[B5])[0] is None)
                          else gap_raw(Ur[B5], gd[B5])[0] - gap_raw(Ur[B1], gd[B1])[0])
        Vr = joint_boot(Ur, {"P": fPr, "B": fBr,
                             "S": lambda gd: (None if (fPr(gd) is None or fBr(gd) is None)
                                              else fPr(gd) - fBr(gd))},
                        len(imgs), seed=SEED)
        allr = {a: np.arange(len(Ur[a].img)) for a in arms4}
        recon = {k: iv(v(allr), Vr[k]) for k, v in
                 (("P", fPr), ("B", fBr),
                  ("S", lambda gd: (None if (fPr(gd) is None or fBr(gd) is None)
                                    else fPr(gd) - fBr(gd))))}
        recon["per_arm"] = Sr

    # seed sensitivity
    Valt = joint_boot_dict(Uraw, lambda gd: {k: all_stats(gd)[k] for k in ("P", "B", "S")},
                           len(imgs), seed=SEED_ALT)

    # ---- NOPFX: the frozen no-prefix captions (pre-reg 5.6) -------------------------------
    afr = {r["image_id"]: r for r in read_jsonl(cfg["af"])}
    nop_cols = {a: [[], [], [], [], []] for a in (A1, A5)}
    for kk in keys:
        free = objset(afr[kk[0]]["text"])
        for a in (A1, A5):
            P = Pof[(kk, a)]
            sidx = min(max(len(P), 1), P_STRATA) - 1
            for o in P:
                nop_cols[a][0].append(ix[kk[0]]); nop_cols[a][1].append(vocab[o])
                nop_cols[a][2].append(o in gt[kk[0]]); nop_cols[a][3].append(o in free)
                nop_cols[a][4].append(sidx)
    Un = {a: Units(*nop_cols[a][:4], n_img=len(imgs), st=nop_cols[a][4]) for a in (A1, A5)}
    fn = lambda gd: (None if (gap_raw(Un[A1], gd[A1])[0] is None
                              or gap_raw(Un[A5], gd[A5])[0] is None)
                     else gap_raw(Un[A5], gd[A5])[0] - gap_raw(Un[A1], gd[A1])[0])
    Vn = joint_boot(Un, {"d": fn, "g1": lambda gd: gap_raw(Un[A1], gd[A1])[0],
                         "g5": lambda gd: gap_raw(Un[A5], gd[A5])[0]}, len(imgs), seed=SEED)
    alln = {a: np.arange(len(Un[a].img)) for a in (A1, A5)}
    nopfx = {"delta_noprefix": iv(fn(alln), Vn["d"]),
             "gap_np_cap1": iv(gap_raw(Un[A1], alln[A1])[0], Vn["g1"]),
             "gap_np_cap5": iv(gap_raw(Un[A5], alln[A5])[0], Vn["g5"]),
             "per_arm": {a: {"n_true": gap_raw(Un[a], alln[a])[1][0],
                             "n_false": gap_raw(Un[a], alln[a])[1][1],
                             "b_true": gap_raw(Un[a], alln[a])[1][2]
                             / max(gap_raw(Un[a], alln[a])[1][0], 1),
                             "b_false": gap_raw(Un[a], alln[a])[1][3]
                             / max(gap_raw(Un[a], alln[a])[1][1], 1)} for a in (A1, A5)},
             "note": ("one free caption per image, shared across k and arms; gap_np differs "
                      "between arms only through P. Not a substitute for the blind arm.")}

    # ---- GATE_ATTENDS (pre-reg 3.3) --------------------------------------------------------
    attends = None
    if os.path.exists(cfg["af_gray"]):
        gr = {r["image_id"]: r for r in read_jsonl(cfg["af_gray"])}
        ids_common = [i for i in imgs if i in gr and i in afr]
        rec_r = {i: [len(objset(afr[i]["text"]) & gt[i]) / max(len(gt[i]), 1)]
                 for i in ids_common}
        rec_g = {i: [len(objset(gr[i]["text"]) & gt[i]) / max(len(gt[i]), 1)]
                 for i in ids_common}
        rng = np.random.default_rng(SEED)
        nI = len(ids_common)
        vals = []
        for b in range(NBOOT):
            sel = rng.integers(0, nI, nI)
            if b < 20:
                _assert_multiplicity(list(sel), nI, f"attends#{b}")
            vals.append(float(np.mean([rec_r[ids_common[s]][0] for s in sel])
                              - np.mean([rec_g[ids_common[s]][0] for s in sel])))
        d_rec = float(np.mean([rec_r[i][0] for i in ids_common])
                      - np.mean([rec_g[i][0] for i in ids_common]))
        ident = float(np.mean([list(gr[i]["gen_ids"]) == list(afr[i]["gen_ids"])
                               for i in ids_common]))
        ci = pct(vals)
        attends = {"delta_recall": d_rec, "ci": ci, "n_images": nI,
                   "recall_real": float(np.mean([rec_r[i][0] for i in ids_common])),
                   "recall_gray": float(np.mean([rec_g[i][0] for i in ids_common])),
                   "identical_caption_rate": ident,
                   "mean_len_real": float(np.mean([afr[i]["gen_len"] for i in ids_common])),
                   "mean_len_gray": float(np.mean([gr[i]["gen_len"] for i in ids_common])),
                   "pass": bool(ci[0] is not None and ci[0] > 0 and ident <= 0.50)}

    # ---- DIL diagnostic (pre-reg 5.7) ------------------------------------------------------
    dil = {"strata_definition": "|P| in {1,2,3,4,5,6+}", "by_nP": {}, "by_len_quartile": {},
           "spearman": {}}
    for a in arms4:
        u = allg[a]
        tab = {}
        for s in range(P_STRATA):
            m = Uraw[a].st == s
            t, c = Uraw[a].tru[m], Uraw[a].car[m]
            nt, nf = int(t.sum()), int((~t).sum())
            tab[str(s + 1) if s < P_STRATA - 1 else f"{P_STRATA}+"] = {
                "n_true": nt, "n_false": nf,
                "r_true": float(c[t].mean()) if nt else None,
                "r_false": float(c[~t].mean()) if nf else None,
                "gap": (float(c[t].mean() - c[~t].mean()) if (nt and nf) else None)}
        dil["by_nP"][a] = tab
        # cell-level association
        per_cell = []
        for kk in keys:
            P = Pof[(kk, a)]
            if not P:
                continue
            C = objset(by[kk][a]["cont_text"])
            per_cell.append((len(P), by[kk][a]["cont_len"],
                             sum(1 for o in P if o in C) / len(P)))
        if per_cell:
            arr = np.array(per_cell, dtype=float)
            def _rank(v):
                """Tie-averaged ranks. |P| is a small integer with heavy ties, so the naive
                argsort-of-argsort rank would break ties arbitrarily and distort rho."""
                o = np.argsort(v, kind="stable")
                r = np.empty(len(v), dtype=float)
                r[o] = np.arange(len(v), dtype=float)
                sv = v[o]
                i = 0
                while i < len(sv):
                    j = i
                    while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                        j += 1
                    if j > i:
                        r[o[i:j + 1]] = (i + j) / 2.0
                    i = j + 1
                return r

            def sp(x, y):
                rx, ry = _rank(x), _rank(y)
                if rx.std() == 0 or ry.std() == 0:
                    return None
                return float(np.corrcoef(rx, ry)[0, 1])
            dil["spearman"][a] = {"carry_vs_nP": sp(arr[:, 0], arr[:, 2]),
                                  "carry_vs_cont_len": sp(arr[:, 1], arr[:, 2]),
                                  "n_cells": len(per_cell)}
            q = np.quantile(arr[:, 1], [0.25, 0.5, 0.75])
            qt = {}
            for qi in range(4):
                lo = -1 if qi == 0 else q[qi - 1]
                hi = q[qi] if qi < 3 else 1e9
                m = (arr[:, 1] > lo) & (arr[:, 1] <= hi)
                qt[f"Q{qi+1}"] = {"n_cells": int(m.sum()),
                                  "mean_cont_len": float(arr[m, 1].mean()) if m.any() else None,
                                  "mean_carry": float(arr[m, 2].mean()) if m.any() else None}
            dil["by_len_quartile"][a] = qt

    # ---- assemble --------------------------------------------------------------------------
    n_true_min_P = min(nmin[A1][0], nmin[A5][0])
    n_false_min_P = min(nmin[A1][1], nmin[A5][1])
    n_true_min_B = min(nmin[B1][0], nmin[B5][0])
    n_false_min_B = min(nmin[B1][1], nmin[B5][1])
    res = {"family": key, "model": cfg["model"], "cap": cap, "n_cells": n,
           "n_images": len(imgs), "n_prefix_object_types": n_obj,
           "gates": g, "composition_per_arm": comp, "per_arm_regime": Sraw,
           "PRIMARY_Delta_sight": dict(iv(pt["S"], V["S"]), rho_PB=rho,
                                       rho_threshold=RHO_MIN[key]),
           "Delta_primary": iv(pt["P"], V["P"]),
           "Delta_blind": iv(pt["B"], V["B"]),
           "rho_blind_ratio": iv(pt["ratio"], V["ratio"]),
           "gap_cap1g": iv(pt["gap_cap1g"], V["gap_cap1g"]),
           "gap_cap5g": iv(pt["gap_cap5g"], V["gap_cap5g"]),
           "gap_cap1": iv(pt["gap_cap1"], V["gap_cap1"]),
           "gap_cap5": iv(pt["gap_cap5"], V["gap_cap5"]),
           "Delta_primary_klen": dict(iv(ptk["Pk"], Vk["Pk"]), per_arm=Sk1),
           "Delta_blind_klen": dict(iv(ptk["Bk"], Vk["Bk"]), per_arm=Sk2),
           "Delta_sight_klen": iv(ptk["Sk"], Vk["Sk"]),
           "RATE_SPLIT": {k: iv(pt[k], V[k]) for k in
                          ("r_true_cap1", "r_true_cap5", "r_false_cap1", "r_false_cap5",
                           "r_true_cap1g", "r_true_cap5g", "r_false_cap1g", "r_false_cap5g",
                           "d_r_true_sighted", "d_r_false_sighted",
                           "d_r_true_blind", "d_r_false_blind",
                           "d_r_true_sight", "d_r_false_sight")},
           "RATE_SPLIT_NOTE": ("A symmetric crowding (dilution) mechanism must move BOTH rates "
                               "down as distinct prefix object types roughly double. "
                               "d_r_true_* > 0 with d_r_false_* < 0 is an asymmetric signature "
                               "that crowding alone does not produce. Reported for the sighted "
                               "and blind conditions on the same image draw."),
           "CP-C2_sighted": iv(pt["P_c2"], V["P_c2"]),
           "CP-C2_blind": iv(pt["B_c2"], V["B_c2"]),
           "DIL_P_standardised_sighted": iv(pt["P_pm"], V["P_pm"]),
           "DIL_P_standardised_blind": iv(pt["B_pm"], V["B_pm"]),
           "CP-R": out_cpr, "recon192": recon,
           "scorer_drops_empty_continuations": False,
           "EMPTY_AND_DEGENERACY": empty_acc,
           "empty_removed": empty_removed,
           "seed_alt": {k: iv(pt[k], Valt[k]) for k in ("P", "B", "S")},
           "n_units": {a: {"n_true": nmin[a][0], "n_false": nmin[a][1]} for a in arms4},
           "NOPFX": nopfx, "GATE_ATTENDS": attends, "DIL": dil}

    res["status_Delta_primary"] = status(pt["P"], res["Delta_primary"]["ci"],
                                         res["Delta_primary"]["ci_width"] or 9,
                                         n_true_min_P, n_false_min_P, ceil=9)
    res["status_Delta_blind"] = status(pt["B"], res["Delta_blind"]["ci"],
                                       res["Delta_blind"]["ci_width"] or 9,
                                       n_true_min_B, n_false_min_B)
    res["status_Delta_sight"] = status(pt["S"], res["PRIMARY_Delta_sight"]["ci"],
                                       res["PRIMARY_Delta_sight"]["ci_width"] or 9,
                                       min(n_true_min_P, n_true_min_B),
                                       min(n_false_min_P, n_false_min_B))

    # ---- pre-reg 6: the cell --------------------------------------------------------------
    ctl_dev = abs(pt["P"] - CTL_REF[key]) if pt["P"] is not None else 9
    res["POSITIVE_CONTROL"] = {"reference": CTL_REF[key], "recomputed": pt["P"],
                               "abs_dev": ctl_dev, "tol": CTL_TOL, "pass": ctl_dev <= CTL_TOL}
    sB, sS = res["status_Delta_blind"], res["status_Delta_sight"]
    ciB, ciS = res["Delta_blind"]["ci"], res["PRIMARY_Delta_sight"]["ci"]
    if not res["POSITIVE_CONTROL"]["pass"]:
        cell, why = "NO-DATA", (f"estimator positive control failed: recomputed {pt['P']} vs "
                                f"{CTL_REF[key]} (dev {ctl_dev:.5f} > {CTL_TOL})")
    elif attends is not None and not attends["pass"]:
        cell, why = "IMAGE-INERT-MODEL", "GATE_ATTENDS failed"
    elif sB == "UNDERPOWERED":
        cell, why = "NO-DATA-ON-BLIND", f"Delta_blind width {res['Delta_blind']['ci_width']}"
    elif sS == "UNDERPOWERED":
        cell, why = ("NO-DATA-ON-SIGHT",
                     f"Delta_sight width {res['PRIMARY_Delta_sight']['ci_width']} > "
                     f"{WIDTH_CEIL}; realised rho_PB {rho} against threshold {RHO_MIN[key]}")
    elif sS == "NEGATIVE":
        cell, why = "SIGHT-REVERSED", "Delta_sight CI entirely below 0 at power"
    elif sB == "NEGATIVE":
        cell, why = "BLIND-REVERSED", "Delta_blind CI entirely below 0 at power"
    elif sB == "NULL" and sS == "POSITIVE":
        cell, why = "SIGHT-EARNED", "Delta_blind contains 0 and Delta_sight excludes 0 from above"
    elif sB == "POSITIVE" and sS == "POSITIVE":
        cell, why = "SIGHT-PARTIAL", "both Delta_blind and Delta_sight exclude 0 from above"
    elif sB == "POSITIVE" and sS == "NULL":
        cell, why = "ENDPOINT-DILUTION", ("Delta_blind excludes 0 from above and Delta_sight "
                                          "contains 0: the contrast reproduces with no visual "
                                          "information and the image adds nothing detectable")
    else:
        cell, why = "NO-DATA", f"unhandled combination Delta_blind={sB} Delta_sight={sS}"
    res["PRIMARY_CELL"] = cell
    res["PRIMARY_CELL_WHY"] = why
    def _contains_zero(d):
        c = d["ci"]
        return c[0] is not None and c[1] is not None and c[0] <= 0 <= c[1]
    res["L73_ASSERTION"] = ("PASSES" if (_contains_zero(res["gap_cap1g"])
                                         and _contains_zero(res["gap_cap5g"])) else "FAILS")
    res["L73_ASSERTION_WHY"] = (f"gap(blind one-scene) {res['gap_cap1g']['point']} "
                                f"{res['gap_cap1g']['ci']}; gap(blind five-scene) "
                                f"{res['gap_cap5g']['point']} {res['gap_cap5g']['ci']}; "
                                "SUBMISSION.md L73-74 asserts both are zero")
    return res


def main():
    chair = ChairScorer(SYN)
    data = load_coco_sample(chair, n=500)
    gt = {d["image_id"]: set(d["gt"]) for d in data}
    objset = lambda t: set(nm for (nm, _c) in chair.mentions(t))
    toks = {}

    def tok_of(key):
        if key not in toks:
            from transformers import AutoTokenizer
            toks[key] = AutoTokenizer.from_pretrained(FAM[key]["model"])
        return toks[key]

    out = {"sentinel": True, "smoke": SMOKE, "width_ceil": WIDTH_CEIL,
           "n_true_floor": N_TRUE_FLOOR, "seed": SEED, "B": NBOOT, "families": {}}
    for key in ("F1", "F2"):
        if not (os.path.exists(FAM[key]["sighted"]) and os.path.exists(FAM[key]["blind"])):
            print(f"[blindscore] {key}: blind rows absent, skipped", flush=True)
            continue
        r = score_family(key, chair, objset, gt, tok_of)
        out["families"][key] = r
        print(f"[blindscore] {key} CELL={r['PRIMARY_CELL']}  L73={r['L73_ASSERTION']}\n"
              f"   Delta_primary {r['Delta_primary']['point']:.4f} {r['Delta_primary']['ci']}\n"
              f"   Delta_blind   {r['Delta_blind']['point']:.4f} {r['Delta_blind']['ci']} "
              f"w={r['Delta_blind']['ci_width']:.4f} [{r['status_Delta_blind']}]\n"
              f"   Delta_sight   {r['PRIMARY_Delta_sight']['point']:.4f} "
              f"{r['PRIMARY_Delta_sight']['ci']} w={r['PRIMARY_Delta_sight']['ci_width']:.4f} "
              f"rho_PB={r['PRIMARY_Delta_sight']['rho_PB']} [{r['status_Delta_sight']}]\n"
              f"   ratio_blind   {r['rho_blind_ratio']['point']:.4f} "
              f"{r['rho_blind_ratio']['ci']}\n"
              f"   gap(cap1g) {r['gap_cap1g']['point']:.4f} {r['gap_cap1g']['ci']}  "
              f"gap(cap5g) {r['gap_cap5g']['point']:.4f} {r['gap_cap5g']['ci']}", flush=True)
    json.dump(out, open(RES, "w"), indent=1)
    sentinel("BLIND_SCORE_SMOKE" if SMOKE else "BLIND_SCORE")


if __name__ == "__main__":
    main()
