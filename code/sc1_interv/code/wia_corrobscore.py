"""WIA-CORROBORATION scorer (WIA_CORROBORATION_PREREG.md sections 5 and 6).

gap(arm) = r_true - r_false over prefix objects. counts/gap semantics copied from
wia_redunscore.py; trunc_text copied VERBATIM from wia_lnscore_lm.py; the paired image-clustered
bootstrap and _assert_multiplicity copied from wia_garblescore.py / wia_redunscore.py
(multiplicity preserved -- no set(), no unique(), no dict keyed on image id on the resampling
path; assert len(sel) == n).

PRIMARY    Delta_corroboration = gap(cap5) - gap(cap1), raw, all cells, 384 regime.
CO-PRIM R  the same restricted to cells where BOTH arms are uncapped (cont_len < 384).
CO-PRIM C1 direct standardisation over the three frozen base-rate tertiles, equal weights.
CO-PRIM C2 common-support object-type standardisation, weights min(n_cap1(o), n_cap5(o)).
CO-PRIM M  gap(cap5m) - gap(cap1) -- only if the census BRANCH generated cap5m.

Writes out/wia_corrob.json. Sentinel J5_SENTINEL_CORROB_SCORE_OK. Exit code is never the signal.
"""
import os, sys, json, math, re
import numpy as np
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")
from j5_common import load_coco_sample, ChairScorer, read_jsonl, sentinel, N_IMAGES

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT_R = "/data/alexmueller/sc1_interv/out"
SUF = "_smoke" if SMOKE else ""
IN_TAG = "smoke_at_corrob" if SMOKE else "at_corrob"
CENSUS_PATH = f"{OUT_R}/corrob_census{SUF}.json"
RES = f"{OUT_R}/wia_corrob_smoke.json" if SMOKE else f"{OUT_R}/wia_corrob.json"
# POSITIVE CONTROL ONLY (inert unless set): re-run this estimator on the predecessor's frozen
# rows renamed wrongasm->cap1 / wrongmulticap->cap5, so the new vectorised Units/bootstrap can be
# checked against wia_redunscore.py's independently written estimator. Never used on lane data.
if os.environ.get("CORROB_CTL_TAG"):
    IN_TAG = os.environ["CORROB_CTL_TAG"]
    RES = f"{OUT_R}/{IN_TAG}_score.json"

NBOOT = 200 if SMOKE else 4000
SEED = 20260909
SEED_ALT = 20260908
N_TRUE_FLOOR = 100
WIDTH_CEIL = 0.13
AT_MAX_GUARD = 0.25                  # prereg 6, length artefact guard
MAX_NEW_384 = 384
MAX_NEW_REF = 192
EXPLORATORY = 0.4876                 # predecessor point estimate of Delta_corroboration, raw
EXPLORATORY_HALFWIDTH = 0.0567

ARMS = ("cap1", "cap5", "cap5m")


# --------------------------------------------------------------------------- helpers
def rep_ratio(text, n=8):
    w = text.lower().split()
    if len(w) < n + 1:
        return 1.0
    g = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(g)) / len(g)


def _assert_multiplicity(sel, n, tag):
    assert len(sel) == n, f"FATAL_BOOTSTRAP_LENGTH {tag}: {len(sel)} != {n}"
    if n >= 20:
        assert len(set(sel)) < n, f"FATAL_BOOTSTRAP_NO_DUPLICATES {tag}"


def trunc_text(tok, ids, L):
    """VERBATIM wia_lnscore_lm.trunc_text."""
    if L is None or L >= len(ids):
        return tok.decode(ids, skip_special_tokens=True), False, False
    text = tok.decode(ids[:L], skip_special_tokens=True)
    nxt = tok.convert_ids_to_tokens(int(ids[L]))
    midword = (not nxt.startswith("▁")) and bool(re.match(r"^[A-Za-z]", nxt))
    if midword:
        t = text.rstrip()
        j = max(t.rfind(" "), t.rfind("\n"))
        text = t[:j + 1] if j >= 0 else ""
    return text, True, midword


# --------------------------------------------------------------------------- unit tables
class Units:
    """Flat unit table for one arm: one row per (cell, prefix object).

    img   index into the common image list
    obj   index into the common object vocabulary
    tru   1 if the object is in the TARGET image's COCO ground truth
    car   1 if the object is mentioned in the (possibly length-matched) continuation
    """

    def __init__(self, img, obj, tru, car, n_img):
        o = np.argsort(img, kind="stable")
        self.img = np.asarray(img)[o]
        self.obj = np.asarray(obj)[o]
        self.tru = np.asarray(tru)[o].astype(bool)
        self.car = np.asarray(car)[o].astype(bool)
        self.counts = np.bincount(self.img, minlength=n_img)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])

    def gather(self, sel):
        """Indices of every unit belonging to the resampled image list `sel` (with multiplicity)."""
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


def gap_tertile(u, g, tert, n_t=3):
    """CP-C1: equal-weight direct standardisation over base-rate tertiles.
    Returns (gap, per-stratum counts) or (None, ...) if no stratum survives."""
    s = tert[u.obj[g]]
    t, c = u.tru[g], u.car[g]
    nt = np.bincount(s[t], minlength=n_t)
    ct = np.bincount(s[t], weights=c[t], minlength=n_t)
    nf = np.bincount(s[~t], minlength=n_t)
    cf = np.bincount(s[~t], weights=c[~t], minlength=n_t)
    return (nt, ct, nf, cf)


def gap_object(u, g, n_obj):
    """CP-C2 raw per-object-type counts."""
    o = u.obj[g]
    t, c = u.tru[g], u.car[g]
    nt = np.bincount(o[t], minlength=n_obj)
    ct = np.bincount(o[t], weights=c[t], minlength=n_obj)
    nf = np.bincount(o[~t], minlength=n_obj)
    cf = np.bincount(o[~t], weights=c[~t], minlength=n_obj)
    return nt, ct, nf, cf


def delta_tertile(uA, uB, gA, gB, tert):
    """gap^ew(B) - gap^ew(A) with IDENTICAL weights, strata with n==0 in either arm dropped."""
    ntA, ctA, nfA, cfA = gap_tertile(uA, gA, tert)
    ntB, ctB, nfB, cfB = gap_tertile(uB, gB, tert)
    ok = (ntA > 0) & (ntB > 0) & (nfA > 0) & (nfB > 0)
    if not ok.any():
        return None, 0
    w = ok.astype(float) / ok.sum()
    gA_ = float((w * (ctA / np.where(ntA > 0, ntA, 1))).sum()
                - (w * (cfA / np.where(nfA > 0, nfA, 1))).sum())
    gB_ = float((w * (ctB / np.where(ntB > 0, ntB, 1))).sum()
                - (w * (cfB / np.where(nfB > 0, nfB, 1))).sum())
    return gB_ - gA_, int(ok.sum())


def delta_object(uA, uB, gA, gB, n_obj):
    """gap^cs(B) - gap^cs(A): common support per object type, weights min(nA, nB)."""
    ntA, ctA, nfA, cfA = gap_object(uA, gA, n_obj)
    ntB, ctB, nfB, cfB = gap_object(uB, gB, n_obj)
    st = (ntA > 0) & (ntB > 0)
    sf = (nfA > 0) & (nfB > 0)
    if not st.any() or not sf.any():
        return None, (0, 0)
    wt = np.minimum(ntA, ntB) * st
    wf = np.minimum(nfA, nfB) * sf
    if wt.sum() == 0 or wf.sum() == 0:
        return None, (int(st.sum()), int(sf.sum()))
    rtA = np.where(ntA > 0, ctA / np.where(ntA > 0, ntA, 1), 0.0)
    rtB = np.where(ntB > 0, ctB / np.where(ntB > 0, ntB, 1), 0.0)
    rfA = np.where(nfA > 0, cfA / np.where(nfA > 0, nfA, 1), 0.0)
    rfB = np.where(nfB > 0, cfB / np.where(nfB > 0, nfB, 1), 0.0)
    gA_ = float((wt * rtA).sum() / wt.sum() - (wf * rfA).sum() / wf.sum())
    gB_ = float((wt * rtB).sum() / wt.sum() - (wf * rfB).sum() / wf.sum())
    return gB_ - gA_, (int(st.sum()), int(sf.sum()))


def boot(uA, uB, stat, n_img, nboot=NBOOT, seed=SEED, tag=""):
    """Paired image-clustered percentile bootstrap. `stat(gA, gB) -> float or None`."""
    rng = np.random.default_rng(seed)
    vals = []
    for b in range(nboot):
        sel = rng.integers(0, n_img, n_img)
        if b < 20:
            _assert_multiplicity(list(sel), n_img, f"{tag}#{b}")
        v = stat(uA.gather(sel), uB.gather(sel))
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return [None, None], 0
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], len(vals)


def contrast(name, uA, uB, n_img, kind, tert=None, n_obj=None, seed=SEED, extra=None):
    """kind in {'raw','tertile','object'}. A = cap1 (reference), B = cap5-family.
    Delta is ALWAYS gap(B) - gap(A), so H1 predicts Delta > 0 (prereg section 0)."""
    allA = np.arange(len(uA.img)); allB = np.arange(len(uB.img))
    if kind == "raw":
        f = lambda ga, gb: (None if (gap_raw(uA, ga)[0] is None or gap_raw(uB, gb)[0] is None)
                            else gap_raw(uB, gb)[0] - gap_raw(uA, ga)[0])
        aux = {}
    elif kind == "tertile":
        f = lambda ga, gb: delta_tertile(uA, uB, ga, gb, tert)[0]
        ntA, ctA, nfA, cfA = gap_tertile(uA, allA, tert)
        ntB, ctB, nfB, cfB = gap_tertile(uB, allB, tert)
        aux = {"n_strata_used": delta_tertile(uA, uB, allA, allB, tert)[1],
               "n_true_per_stratum_A": [int(x) for x in ntA],
               "n_false_per_stratum_A": [int(x) for x in nfA],
               "n_true_per_stratum_B": [int(x) for x in ntB],
               "n_false_per_stratum_B": [int(x) for x in nfB],
               "r_true_per_stratum_A": [float(c / n) if n else None for c, n in zip(ctA, ntA)],
               "r_false_per_stratum_A": [float(c / n) if n else None for c, n in zip(cfA, nfA)],
               "r_true_per_stratum_B": [float(c / n) if n else None for c, n in zip(ctB, ntB)],
               "r_false_per_stratum_B": [float(c / n) if n else None for c, n in zip(cfB, nfB)]}
    else:
        f = lambda ga, gb: delta_object(uA, uB, ga, gb, n_obj)[0]
        s = delta_object(uA, uB, allA, allB, n_obj)[1]
        ntA, _, nfA, _ = gap_object(uA, allA, n_obj)
        ntB, _, nfB, _ = gap_object(uB, allB, n_obj)
        st = (ntA > 0) & (ntB > 0); sf = (nfA > 0) & (nfB > 0)
        aux = {"n_types_true": s[0], "n_types_false": s[1],
               "n_types_true_A_total": int((ntA > 0).sum()),
               "n_types_true_B_total": int((ntB > 0).sum()),
               "n_types_false_A_total": int((nfA > 0).sum()),
               "n_types_false_B_total": int((nfB > 0).sum()),
               "frac_true_units_retained_A": float(ntA[st].sum() / ntA.sum()) if ntA.sum() else None,
               "frac_true_units_retained_B": float(ntB[st].sum() / ntB.sum()) if ntB.sum() else None,
               "frac_false_units_retained_A": float(nfA[sf].sum() / nfA.sum()) if nfA.sum() else None,
               "frac_false_units_retained_B": float(nfB[sf].sum() / nfB.sum()) if nfB.sum() else None}
    pt = f(allA, allB)
    ci, nb = boot(uA, uB, f, n_img, seed=seed, tag=name)
    gA, sA = gap_raw(uA, allA)
    gB, sB = gap_raw(uB, allB)
    out = {"name": name, "kind": kind, "point": pt, "ci": ci,
           "ci_width": None if ci[0] is None else ci[1] - ci[0],
           "n_boot_used": nb, "seed": seed, "n_images": n_img,
           "gap_cap1_raw": gA, "gap_capX_raw": gB,
           "n_true_A": sA[0], "n_false_A": sA[1], "n_true_B": sB[0], "n_false_B": sB[1],
           "r_true_A": sA[2] / sA[0] if sA[0] else None,
           "r_false_A": sA[3] / sA[1] if sA[1] else None,
           "r_true_B": sB[2] / sB[0] if sB[0] else None,
           "r_false_B": sB[3] / sB[1] if sB[1] else None}
    out.update(aux)
    if extra:
        out.update(extra)
    return out


def status(c, is_primary=False):
    """prereg section 6 status of one contrast."""
    if c is None:
        return "NOT_RUN"
    if c["point"] is None or c["ci"][0] is None:
        return "UNDERPOWERED"
    if min(c["n_true_A"], c["n_true_B"]) < N_TRUE_FLOOR:
        return "UNDERPOWERED"
    if is_primary and c["ci_width"] > WIDTH_CEIL:
        return "UNDERPOWERED"
    if c["ci"][0] > 0:
        return "POSITIVE"
    if c["ci"][1] < 0:
        return "NEGATIVE"
    return "NULL"


def readout(prim_s, cps):
    """prereg section 6. cps: {label: status} for CP-R, CP-C1, CP-C2, CP-M."""
    comp_labels = ("CP-C1", "CP-C2", "CP-M")
    reg_labels = ("CP-R",)
    evaluable = {k: v for k, v in cps.items()
                 if v in ("POSITIVE", "NULL", "NEGATIVE")}
    comp = {k: v for k, v in evaluable.items() if k in comp_labels}
    reg = {k: v for k, v in evaluable.items() if k in reg_labels}
    if prim_s == "UNDERPOWERED":
        return "NO-DATA", "primary underpowered (n_true floor or CI width ceiling)"
    if prim_s == "NULL":
        return "NULL-AT-POWER", "primary CI contains 0 at power"
    if prim_s == "NEGATIVE":
        return "REVERSED-AT-POWER", "primary CI entirely below 0 at power"
    # prim_s == POSITIVE
    all_pos = all(v == "POSITIVE" for v in evaluable.values())
    if all_pos and comp and reg:
        return "CORROBORATION-CONFIRMED", "primary and every evaluable co-primary POSITIVE"
    if all_pos:
        miss = []
        if not comp:
            miss.append("no composition co-primary evaluable")
        if not reg:
            miss.append("no regime co-primary evaluable")
        return "CORROBORATION-CONFIRMED-PARTIAL", "; ".join(miss)
    if comp and not any(v == "POSITIVE" for v in comp.values()):
        return "COMPOSITION-ARTEFACT", ("no evaluable composition co-primary is POSITIVE: "
                                        + json.dumps(comp))
    if comp and all(v == "POSITIVE" for v in comp.values()) and reg \
            and not all(v == "POSITIVE" for v in reg.values()):
        return "REGIME-ARTEFACT", "composition survives, regime control does not: " + json.dumps(reg)
    failing = {k: v for k, v in evaluable.items() if v != "POSITIVE"}
    return "CORROBORATION-ATTENUATED", "control(s) not POSITIVE: " + json.dumps(failing)


# --------------------------------------------------------------------------- main
def main():
    from wia_garble_common import load_tok
    tok = load_tok()
    chair = ChairScorer("/data/alexmueller/j5_gates/code/synonyms.txt")
    data = load_coco_sample(chair, n=N_IMAGES)
    gt = {d["image_id"]: set(d["gt"]) for d in data}
    objset = lambda t: set(n for (n, _c) in chair.mentions(t))

    census = json.load(open(CENSUS_PATH))
    cuts = census["base_rate"]["tertile_cuts"]
    pi_top = dict(census["base_rate"]["top10"])
    rows = list(read_jsonl(os.path.join(OUT_R, f"{IN_TAG}.jsonl")))
    by_cell = {}
    for r in rows:
        assert r["arm"] in ARMS, f"FATAL_UNKNOWN_ARM {r['arm']}"
        assert r["cont_len"] == len(r["cont_ids"]), "FATAL_CONT_LEN_MISMATCH"
        d = by_cell.setdefault((r["image_id"], r["k"]), {})
        assert r["arm"] not in d, f"FATAL_DUPLICATE_ROW {r}"
        d[r["arm"]] = r
    for key, d in by_cell.items():
        assert "cap1" in d and "cap5" in d, f"FATAL_INCOMPLETE_CELL {key}"
        bs = {d[a]["budget"] for a in d}
        assert len(bs) == 1, f"FATAL_BUDGET_MISMATCH_AT_SCORE {key} {bs}"
        for a in d:
            assert len(d[a]["prefix_ids"]) == d[a]["budget"], \
                f"FATAL_BUDGET_MISMATCH_AT_SCORE {key} {a}"
        for a in ("cap5", "cap5m"):
            if a in d:
                assert d[a]["cycles"] == 0, f"FATAL_DOUBLING_IN_NEW_ARM {key} {a}"

    # ---- rebuild pi(o) so the frozen cuts can be re-asserted (prereg 4.1) -----------------
    from j5_common import OUT as J5OUT
    af = list(read_jsonl(f"{J5OUT}/af.jsonl"))
    pi = {}
    for r in af:
        for o in objset(r["text"]):
            pi[o] = pi.get(o, 0) + 1
    pi = {o: c / len(af) for o, c in pi.items()}
    for o, v in pi_top.items():
        assert abs(pi.get(o, -1) - v) < 1e-12, f"FATAL_TERTILE_DRIFT base rate {o}"

    keys = sorted(by_cell)
    imgs = sorted({k[0] for k in keys})
    img_ix = {i: j for j, i in enumerate(imgs)}
    vocab = {}
    for key in keys:
        for a, r in by_cell[key].items():
            for o in objset(r["prefix_text"]):
                vocab.setdefault(o, len(vocab))
    n_obj = len(vocab)
    tert = np.zeros(n_obj, dtype=np.int64)
    for o, ix in vocab.items():
        p = pi.get(o, 0.0)
        tert[ix] = 0 if p <= cuts[0] else (1 if p <= cuts[1] else 2)
    print(f"[score] {len(rows)} rows, {len(keys)} cells, {len(imgs)} images, "
          f"{n_obj} prefix-object types; tertile sizes "
          f"{[int((tert==i).sum()) for i in range(3)]}", flush=True)

    def cont_of(r, regime, L):
        ids = r["cont_ids"][:MAX_NEW_REF] if regime == 192 else r["cont_ids"]
        if L is None:
            return tok.decode(ids, skip_special_tokens=True), False, False, len(ids)
        t, tr, mw = trunc_text(tok, ids, L)
        return t, tr, mw, min(L, len(ids))

    def build(armA, armB, regime, klen, cellfilter=None):
        """Unit tables for the two arms over the cells where both exist and cellfilter passes."""
        ks = [k for k in keys if armA in by_cell[k] and armB in by_cell[k]]
        if cellfilter is not None:
            ks = [k for k in ks if cellfilter(by_cell[k], regime)]
        if not ks:
            return None, None, None, None
        cols = {armA: [[], [], [], []], armB: [[], [], [], []]}
        st = {armA: {"L": [], "Lr": [], "tr": 0, "mw": 0, "emp": 0, "cap": 0, "c192": 0,
                     "rep": [], "cyc": 0, "b": []},
              armB: {"L": [], "Lr": [], "tr": 0, "mw": 0, "emp": 0, "cap": 0, "c192": 0,
                     "rep": [], "cyc": 0, "b": []}}
        for k in ks:
            d = by_cell[k]
            if klen:
                L = min(len(d[a]["cont_ids"][:MAX_NEW_REF] if regime == 192 else d[a]["cont_ids"])
                        for a in (armA, armB))
            else:
                L = None
            for a in (armA, armB):
                r = d[a]
                text, tr, mw, Le = cont_of(r, regime, L)
                raw = r["cont_ids"][:MAX_NEW_REF] if regime == 192 else r["cont_ids"]
                s = st[a]
                s["L"].append(Le); s["Lr"].append(len(raw)); s["tr"] += tr; s["mw"] += mw
                s["emp"] += (len(raw) == 0)
                s["cap"] += (len(raw) >= (MAX_NEW_REF if regime == 192 else MAX_NEW_384))
                s["c192"] += (len(raw) >= MAX_NEW_REF)
                s["rep"].append(rep_ratio(r["prefix_text"])); s["cyc"] += (r["cycles"] > 0)
                s["b"].append(r["budget"])
                P, C = objset(r["prefix_text"]), objset(text)
                g = gt[k[0]]
                for o in P:
                    cols[a][0].append(img_ix[k[0]]); cols[a][1].append(vocab[o])
                    cols[a][2].append(o in g); cols[a][3].append(o in C)
        uA = Units(*cols[armA], n_img=len(imgs))
        uB = Units(*cols[armB], n_img=len(imgs))
        stats = {a: {"n_cells": len(ks), "mean_cont_len": float(np.mean(s["Lr"])),
                     "mean_matched_len": float(np.mean(s["L"])),
                     "frac_truncated": s["tr"] / len(ks),
                     "frac_midword_cut": s["mw"] / len(ks),
                     "empty_cont_rate": s["emp"] / len(ks),
                     "at_max": s["cap"] / len(ks), "at_192": s["c192"] / len(ks),
                     "doubling_rate": s["cyc"] / len(ks),
                     "mean_distinct_8gram_ratio": float(np.mean(s["rep"])),
                     "mean_budget": float(np.mean(s["b"]))}
                 for a, s in st.items()}
        return uA, uB, stats, ks

    def both_uncapped_for(armA, armB):
        """prereg CP-R: BOTH arms OF THIS CONTRAST uncapped. Other arms in the cell are
        irrelevant and must not remove the cell."""
        def f(d, regime):
            cap = MAX_NEW_REF if regime == 192 else MAX_NEW_384
            return all(len(d[a]["cont_ids"][:MAX_NEW_REF] if regime == 192
                           else d[a]["cont_ids"]) < cap for a in (armA, armB))
        return f

    has_m = any("cap5m" in by_cell[k] for k in keys)
    res = {"sentinel": True, "input_tag": IN_TAG, "smoke": SMOKE, "n_rows": len(rows),
           "n_cells": len(keys), "n_images": len(imgs),
           "per_arm_rows": {a: sum(1 for r in rows if r["arm"] == a) for a in ARMS
                            if any(r["arm"] == a for r in rows)},
           "cap5m_present": has_m, "census_branch": census["BRANCH"],
           "census_f_match": census["f_match"],
           "manipulation_gate": census["GATE"],
           "tertile_cuts": cuts,
           "_bootstrap": {"unit": "image, clustered over derangements", "B": NBOOT,
                          "seed": SEED, "multiplicity": "PRESERVED -- no set(), no unique()",
                          "assertion": "len(sel) == n, and duplicates present"},
           "thresholds": {"n_true_floor": N_TRUE_FLOOR, "width_ceiling": WIDTH_CEIL,
                          "at_max_guard": AT_MAX_GUARD},
           "arm_stats": {}, "contrasts": {}}

    # ---------------- PRIMARY and co-primaries, 384 regime ---------------------------------
    uA, uB, stats, ks = build("cap1", "cap5", 384, False)
    res["arm_stats"]["384_all"] = stats
    PRIM = contrast("PRIMARY", uA, uB, len(imgs), "raw", extra={"n_cells": len(ks)})
    kA, kB, kstats, _ = build("cap1", "cap5", 384, True)
    PRIM_KLEN = contrast("PRIMARY_klen", kA, kB, len(imgs), "raw")
    res["arm_stats"]["384_klen"] = kstats
    CP_C1 = contrast("CP-C1", uA, uB, len(imgs), "tertile", tert=tert)
    CP_C2 = contrast("CP-C2", uA, uB, len(imgs), "object", n_obj=n_obj)
    rA, rB, rstats, rks = build("cap1", "cap5", 384, False,
                                     cellfilter=both_uncapped_for("cap1", "cap5"))
    res["arm_stats"]["384_both_uncapped"] = rstats
    CP_R = (contrast("CP-R", rA, rB, len(imgs), "raw", extra={"n_cells": len(rks)})
            if rA is not None else None)
    CP_M = None
    if has_m:
        mA, mB, mstats, mks = build("cap1", "cap5m", 384, False)
        res["arm_stats"]["384_cap5m"] = mstats
        CP_M = contrast("CP-M", mA, mB, len(imgs), "raw", extra={"n_cells": len(mks)})

    res["contrasts"]["PRIMARY"] = PRIM
    res["contrasts"]["PRIMARY_klen"] = PRIM_KLEN
    res["contrasts"]["CP-R"] = CP_R
    res["contrasts"]["CP-C1"] = CP_C1
    res["contrasts"]["CP-C2"] = CP_C2
    res["contrasts"]["CP-M"] = CP_M

    # ---------------- secondaries ------------------------------------------------------------
    sec = {}
    sA, sB, sstats, sks = build("cap1", "cap5", 192, False)
    sec["PRIMARY_recon192"] = contrast("PRIMARY_recon192", sA, sB, len(imgs), "raw",
                                       extra={"n_cells": len(sks)})
    res["arm_stats"]["192_all"] = sstats
    kA2, kB2, _, _ = build("cap1", "cap5", 192, True)
    sec["PRIMARY_recon192_klen"] = contrast("PRIMARY_recon192_klen", kA2, kB2, len(imgs), "raw")
    if rA is not None:
        sec["CP-C1_both_uncapped"] = contrast("CP-C1_bu", rA, rB, len(imgs), "tertile", tert=tert)
        sec["CP-C2_both_uncapped"] = contrast("CP-C2_bu", rA, rB, len(imgs), "object",
                                              n_obj=n_obj)
        sec["CP-R_klen"] = None
        kr, krB, _, _ = build("cap1", "cap5", 384, True,
                              cellfilter=both_uncapped_for("cap1", "cap5"))
        if kr is not None:
            sec["CP-R_klen"] = contrast("CP-R_klen", kr, krB, len(imgs), "raw")
    # per-tertile contrasts
    allA, allB = np.arange(len(uA.img)), np.arange(len(uB.img))
    per_t = {}
    for s in range(3):
        mA_ = np.zeros(3, dtype=bool); mA_[s] = True
        selA = allA[mA_[tert[uA.obj]]]
        selB = allB[mA_[tert[uB.obj]]]
        gA, cA = gap_raw(uA, selA)
        gB, cB = gap_raw(uB, selB)
        per_t[f"tertile_{s+1}"] = {"gap_cap1": gA, "gap_cap5": gB,
                                   "delta": None if (gA is None or gB is None) else gB - gA,
                                   "n_true_cap1": cA[0], "n_false_cap1": cA[1],
                                   "n_true_cap5": cB[0], "n_false_cap5": cB[1]}
    sec["per_tertile"] = per_t
    sec["PRIMARY_seed_alt"] = contrast("PRIMARY_seed_alt", uA, uB, len(imgs), "raw",
                                       seed=SEED_ALT)
    res["secondaries"] = sec

    # ---------------- readout ----------------------------------------------------------------
    prim_s = status(PRIM, is_primary=True)
    cps = {"CP-R": status(CP_R), "CP-C1": status(CP_C1), "CP-C2": status(CP_C2),
           "CP-M": status(CP_M)}
    cell, why = readout(prim_s, cps)
    at_max_cap5 = res["arm_stats"]["384_all"]["cap5"]["at_max"]
    kill = False
    if PRIM["ci"][0] is not None and PRIM_KLEN["ci"][0] is not None:
        raw_ex = (PRIM["ci"][0] > 0) or (PRIM["ci"][1] < 0)
        kl_ex = (PRIM_KLEN["ci"][0] > 0) or (PRIM_KLEN["ci"][1] < 0)
        sgn_raw = np.sign(PRIM["point"]) if raw_ex else 0
        sgn_kl = np.sign(PRIM_KLEN["point"]) if kl_ex else 0
        # prereg 6: "the FIRED CELL is DOWNGRADED to KILL-ARTEFACT" -- a downgrade presupposes a
        # claim, so the guard applies only to claim-bearing cells (same scope as the
        # predecessor's guard, which ran only on REDUNDANCY / LISTING). Recorded as a wording
        # clarification made before the confirmatory data existed.
        CLAIM_CELLS = ("CORROBORATION-CONFIRMED", "CORROBORATION-CONFIRMED-PARTIAL",
                       "CORROBORATION-ATTENUATED", "REVERSED-AT-POWER")
        if sgn_raw != sgn_kl and at_max_cap5 > AT_MAX_GUARD and cell in CLAIM_CELLS:
            kill = True
    mag = ("LARGER_THAN_EXPLORATORY" if (PRIM["point"] or 0) > EXPLORATORY + EXPLORATORY_HALFWIDTH
           else "SMALLER_THAN_EXPLORATORY"
           if (PRIM["point"] or 0) < EXPLORATORY - EXPLORATORY_HALFWIDTH else "CONSISTENT")
    res["STATUS"] = {"PRIMARY": prim_s, **cps}
    res["CELL"] = "KILL-ARTEFACT" if kill else cell
    res["CELL_WHY"] = ("raw and klen primaries exclude 0 with opposite signs while "
                       f"at_max(cap5) = {at_max_cap5:.3f} > {AT_MAX_GUARD}") if kill else why
    res["MAGNITUDE_FLAG"] = mag
    res["EXPLORATORY_REFERENCE"] = {"point": EXPLORATORY, "halfwidth": EXPLORATORY_HALFWIDTH,
                                    "note": "descriptive only; changes no cell (prereg 6)"}

    json.dump(res, open(RES, "w"), indent=1)
    print(f"[score] wrote {RES}", flush=True)
    for tag in ("384_all", "384_both_uncapped", "192_all"):
        for a, s in (res["arm_stats"].get(tag) or {}).items():
            print(f"[score] {tag:20s} {a:6s} cells {s['n_cells']:5d} cont {s['mean_cont_len']:7.2f} "
                  f"at_max {s['at_max']:.3f} at192 {s['at_192']:.3f} empty {s['empty_cont_rate']:.4f} "
                  f"dbl {s['doubling_rate']:.4f}", flush=True)
    def fmt(nm, c):
        if not c:
            return f"[score] {nm:14s} NOT_RUN"
        f = lambda x, d=4: ("  None " if x is None else f"{x:+.{d}f}")
        return (f"[score] {nm:14s} {f(c['point'])} [{f(c['ci'][0])},{f(c['ci'][1])}] "
                f"w {'None' if c['ci_width'] is None else '%.4f' % c['ci_width']} "
                f"nT {c['n_true_A']}/{c['n_true_B']} nF {c['n_false_A']}/{c['n_false_B']} "
                f"gap1 {f(c['gap_cap1_raw'])} gapX {f(c['gap_capX_raw'])} "
                f"rT {f(c['r_true_A'])}/{f(c['r_true_B'])} "
                f"rF {f(c['r_false_A'])}/{f(c['r_false_B'])}")
    for nm in ("PRIMARY", "PRIMARY_klen", "CP-R", "CP-C1", "CP-C2", "CP-M"):
        print(fmt(nm, res["contrasts"].get(nm)), flush=True)
    for nm in ("PRIMARY_recon192", "PRIMARY_recon192_klen", "CP-R_klen",
               "CP-C1_both_uncapped", "CP-C2_both_uncapped", "PRIMARY_seed_alt"):
        print(fmt(nm, res["secondaries"].get(nm)), flush=True)
    print(f"[score] per_tertile {json.dumps(res['secondaries']['per_tertile'])}", flush=True)
    print(f"[score] STATUS {json.dumps(res['STATUS'])}", flush=True)
    print(f"[score] CELL = {res['CELL']} :: {res['CELL_WHY']}", flush=True)
    print(f"[score] MAGNITUDE_FLAG = {mag}", flush=True)
    sentinel("CORROB_SCORE")


if __name__ == "__main__":
    main()
