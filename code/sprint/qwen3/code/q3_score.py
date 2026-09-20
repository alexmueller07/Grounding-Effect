"""Lane `qwen3` scorer. Units / gather / gap_raw / delta_object / the paired image-clustered
bootstrap are ported from f3_score.py (themselves ported from wia_corrobscore.py) unmodified,
so the scorer's first-order bias is identical to every other model's and cancels in every
contrast.

Order is enforced by control flow:
  1. ESTIMATOR POSITIVE CONTROLS on families 1 and 3's FROZEN rows. If either misses its
     published point estimate by more than 5e-3 the lane reports NO-DATA and stops.
  2. manipulation gates, recomputed FROM THE WRITTEN JSONL, never from generator meta.
  3. the contrasts: Delta_primary, Delta_blind, Delta_image = primary - blind, plus the
     per-condition rates H and F.
  4. the full degeneracy screen, per arm.
"""
import os, sys, json, math, re
sys.path.insert(0, "/data/alexmueller/sprint/qwen3/code")
import numpy as np
from q3_common import (M5, M5_SHA, OUT, load_coco_sample, ChairScorer, SYN, MAX_NEW_384,
                       MAX_NEW_192, jload, sentinel)
from transformers import AutoTokenizer

NBOOT = 4000
SEED = 20260919
SEED_ALT = 20260920
N_TRUE_FLOOR = 100
N_FALSE_FLOOR = 100
MAX_384, MAX_192 = MAX_NEW_384, MAX_NEW_192
WS = "Ġ"                       # byte-level BPE word-start marker
CTL = {"fam1": ("/data/alexmueller/sc1_interv/out/at_corrob.jsonl", 0.4859),
       "fam3": ("/data/alexmueller/sc1_fam3/out/f3_sighted.jsonl", 0.1598)}
SIGHT = ("cap1", "cap5")
BLIND = ("cap1g", "cap5g")
DEGEN_MEDIAN = 12                   # the harness gate's HG-5 thresholds, reused as the
DEGEN_LE3 = 0.30                    # readout's degeneracy screen (QWEN3_PREREG.md section 6)


def rep_ratio(text, n=8):
    w = text.lower().split()
    if len(w) < n + 1:
        return 1.0
    g = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(g)) / len(g)


def uniq_tok_ratio(ids):
    return (len(set(ids)) / len(ids)) if len(ids) else None


def _mult(sel, n, tag):
    assert len(sel) == n, f"FATAL_BOOTSTRAP_LENGTH {tag}: {len(sel)} != {n}"
    if n >= 20:
        assert len(set(sel)) < n, f"FATAL_BOOTSTRAP_NO_DUPLICATES {tag}"


def trunc_text(tok, ids, L):
    if L is None or L >= len(ids):
        return tok.decode(ids, skip_special_tokens=True), False, False
    text = tok.decode(ids[:L], skip_special_tokens=True)
    nxt = tok.convert_ids_to_tokens(int(ids[L]))
    midword = (not nxt.startswith(WS)) and bool(re.match(r"^[A-Za-z]", nxt))
    if midword:
        t = text.rstrip()
        j = max(t.rfind(" "), t.rfind("\n"))
        text = t[:j + 1] if j >= 0 else ""
    return text, True, midword


class Units:
    def __init__(self, img, obj, tru, car, n_img):
        o = np.argsort(img, kind="stable")
        self.img = np.asarray(img, dtype=np.int64)[o]
        self.obj = np.asarray(obj, dtype=np.int64)[o]
        self.tru = np.asarray(tru)[o].astype(bool)
        self.car = np.asarray(car)[o].astype(bool)
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
    o, t, c = u.obj[g], u.tru[g], u.car[g]
    return (np.bincount(o[t], minlength=n_obj),
            np.bincount(o[t], weights=c[t], minlength=n_obj),
            np.bincount(o[~t], minlength=n_obj),
            np.bincount(o[~t], weights=c[~t], minlength=n_obj))


def delta_object(uA, uB, gA, gB, n_obj):
    ntA, ctA, nfA, cfA = gap_object(uA, gA, n_obj)
    ntB, ctB, nfB, cfB = gap_object(uB, gB, n_obj)
    st = (ntA > 0) & (ntB > 0)
    sf = (nfA > 0) & (nfB > 0)
    if not st.any() or not sf.any():
        return None, (0, 0)
    wt, wf = np.minimum(ntA, ntB) * st, np.minimum(nfA, nfB) * sf
    if wt.sum() == 0 or wf.sum() == 0:
        return None, (int(st.sum()), int(sf.sum()))
    d = lambda c, n: np.where(n > 0, c / np.where(n > 0, n, 1), 0.0)
    gA_ = float((wt * d(ctA, ntA)).sum() / wt.sum() - (wf * d(cfA, nfA)).sum() / wf.sum())
    gB_ = float((wt * d(ctB, ntB)).sum() / wt.sum() - (wf * d(cfB, nfB)).sum() / wf.sum())
    return gB_ - gA_, (int(st.sum()), int(sf.sum()))


def boot(stat, n_img, nboot=NBOOT, seed=SEED, tag=""):
    rng = np.random.default_rng(seed)
    v = []
    for b in range(nboot):
        sel = rng.integers(0, n_img, n_img)
        if b < 20:
            _mult(list(sel), n_img, f"{tag}#{b}")
        x = stat(sel)
        if x is not None and not (isinstance(x, float) and math.isnan(x)):
            v.append(x)
    if not v:
        return [None, None], 0
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], len(v)


def contrast(name, uA, uB, n_img, kind="raw", n_obj=None, seed=SEED, extra=None):
    allA, allB = np.arange(len(uA.img)), np.arange(len(uB.img))
    if kind == "raw":
        def f(gA, gB):
            a, b = gap_raw(uA, gA)[0], gap_raw(uB, gB)[0]
            return None if (a is None or b is None) else b - a
        aux = {}
    else:
        def f(gA, gB):
            return delta_object(uA, uB, gA, gB, n_obj)[0]
        s = delta_object(uA, uB, allA, allB, n_obj)[1]
        aux = {"n_types_true": s[0], "n_types_false": s[1]}
    pt = f(allA, allB)
    ci, nb = boot(lambda sel: f(uA.gather(sel), uB.gather(sel)), n_img, seed=seed, tag=name)
    gA, sA = gap_raw(uA, allA)
    gB, sB = gap_raw(uB, allB)
    out = {"name": name, "kind": kind, "point": pt, "ci": ci,
           "ci_width": None if ci[0] is None else ci[1] - ci[0], "n_boot_used": nb, "seed": seed,
           "n_images": n_img, "gap_A": gA, "gap_B": gB,
           "n_true_A": sA[0], "n_false_A": sA[1], "n_true_B": sB[0], "n_false_B": sB[1],
           "H_A": sA[2] / sA[0] if sA[0] else None, "F_A": sA[3] / sA[1] if sA[1] else None,
           "H_B": sB[2] / sB[0] if sB[0] else None, "F_B": sB[3] / sB[1] if sB[1] else None}
    out.update(aux)
    if extra:
        out.update(extra)
    return out


def status(c):
    if c is None or c["point"] is None or c["ci"][0] is None:
        return "NOT_RUN"
    if min(c["n_true_A"], c["n_true_B"]) < N_TRUE_FLOOR:
        return "UNDERPOWERED"
    if min(c["n_false_A"], c["n_false_B"]) < N_FALSE_FLOOR:
        return "UNDERPOWERED"
    if c["ci"][0] > 0:
        return "POSITIVE"
    if c["ci"][1] < 0:
        return "NEGATIVE"
    return "NULL"


def build_tables(by_cell, keys, arms, tok, gt, objset, vocab, img_ix, n_img,
                 regime=MAX_384, klen=False, cellfilter=None):
    ks = [k for k in keys if all(a in by_cell[k] for a in arms)]
    if cellfilter is not None:
        ks = [k for k in ks if cellfilter(by_cell[k])]
    if not ks:
        return None, None, None
    cols = {a: [[], [], [], []] for a in arms}
    st = {a: {"L": [], "Lr": [], "tr": 0, "mw": 0, "emp": 0, "cap": 0, "c192": 0, "rep": [],
              "cyc": 0, "b": [], "nP": [], "nM": [], "utr": [], "le3": 0} for a in arms}
    for k in ks:
        d = by_cell[k]
        ids_of = lambda r: (r["cont_ids"][:MAX_192] if regime == MAX_192 else r["cont_ids"])
        L = min(len(ids_of(d[a])) for a in arms) if klen else None
        for a in arms:
            r = d[a]
            raw = ids_of(r)
            if L is None:
                text, tr, mw = tok.decode(raw, skip_special_tokens=True), False, False
                Le = len(raw)
            else:
                text, tr, mw = trunc_text(tok, raw, L)
                Le = min(L, len(raw))
            s = st[a]
            s["L"].append(Le); s["Lr"].append(len(raw)); s["tr"] += tr; s["mw"] += mw
            s["emp"] += (len(raw) == 0); s["le3"] += (len(raw) <= 3)
            s["cap"] += (len(raw) >= regime)
            s["c192"] += (len(raw) >= MAX_192)
            u = uniq_tok_ratio(raw)
            if u is not None:
                s["utr"].append(u)
            s["rep"].append(rep_ratio(r["prefix_text"])); s["cyc"] += (r["cycles"] > 0)
            s["b"].append(r["budget"])
            P, C = objset(r["prefix_text"]), objset(text)
            s["nP"].append(len(P)); s["nM"].append(r["_nment"])
            g = gt[k[0]]
            for o in P:
                cols[a][0].append(img_ix[k[0]]); cols[a][1].append(vocab[o])
                cols[a][2].append(o in g); cols[a][3].append(o in C)
    U = {a: Units(*cols[a], n_img=n_img) for a in arms}
    stats = {a: {"n_cells": len(ks),
                 "mean_cont_len": float(np.mean(s["Lr"])),
                 "median_cont_len": float(np.median(s["Lr"])),
                 "mean_matched_len": float(np.mean(s["L"])),
                 "truncation_rate_at_budget": s["cap"] / len(ks),
                 "at_192": s["c192"] / len(ks),
                 "empty_rate": s["emp"] / len(ks),
                 "frac_le3": s["le3"] / len(ks),
                 "mean_unique_token_ratio": float(np.mean(s["utr"])) if s["utr"] else None,
                 "median_unique_token_ratio": (float(np.median(s["utr"])) if s["utr"] else None),
                 "n_rows_with_tokens": len(s["utr"]),
                 "frac_len_matched_truncated": s["tr"] / len(ks),
                 "frac_midword_cut": s["mw"] / len(ks),
                 "doubling_rate": s["cyc"] / len(ks),
                 "mean_distinct_8gram_ratio_prefix": float(np.mean(s["rep"])),
                 "mean_budget": float(np.mean(s["b"])),
                 "mean_distinct_prefix_objects": float(np.mean(s["nP"])),
                 "mean_total_mentions": float(np.mean(s["nM"])),
                 "mean_mentions_per_type": (float(np.mean(s["nM"])) / float(np.mean(s["nP"]))
                                            if np.mean(s["nP"]) else None)}
             for a, s in st.items()}
    return U, stats, ks


def positive_control(gt, objset):
    res = {}
    for fam, (path, ref) in CTL.items():
        rows = jload(path)
        by = {}
        for r in rows:
            if r["arm"] in ("cap1", "cap5"):
                by.setdefault((r["image_id"], r["k"]), {})[r["arm"]] = r
        keys = sorted(k for k, d in by.items() if "cap1" in d and "cap5" in d)
        imgs = sorted({k[0] for k in keys}); ix = {i: j for j, i in enumerate(imgs)}
        vocab, cols = {}, {"cap1": [[], [], [], []], "cap5": [[], [], [], []]}
        for k in keys:
            for a in ("cap1", "cap5"):
                r = by[k][a]
                P, C = objset(r["prefix_text"]), objset(r["cont_text"])
                g = gt[k[0]]
                for o in P:
                    vocab.setdefault(o, len(vocab))
                    cols[a][0].append(ix[k[0]]); cols[a][1].append(vocab[o])
                    cols[a][2].append(o in g); cols[a][3].append(o in C)
        uA = Units(*cols["cap1"], n_img=len(imgs))
        uB = Units(*cols["cap5"], n_img=len(imgs))
        pt = (gap_raw(uB, np.arange(len(uB.img)))[0] - gap_raw(uA, np.arange(len(uA.img)))[0])
        dev = abs(pt - ref)
        res[fam] = {"point": pt, "published": ref, "dev": dev, "pass": bool(dev <= 5e-3),
                    "n_cells": len(keys), "n_images": len(imgs)}
        print(f"[ctl] {fam}: {pt:+.5f} vs published {ref:+.4f}  dev {dev:.6f}  "
              f"{'PASS' if dev <= 5e-3 else 'FAIL'}", flush=True)
    return res


def main():
    chair = ChairScorer(SYN)
    data = load_coco_sample(chair, n=500)
    gt = {d["image_id"]: set(d["gt"]) for d in data}
    objset = lambda t: set(n for (n, _c) in chair.mentions(t))
    tok = AutoTokenizer.from_pretrained(M5, revision=M5_SHA)
    assert tok.convert_ids_to_tokens(
        tok(" cat", add_special_tokens=False)["input_ids"])[0].startswith(WS), \
        "FATAL_WORD_START_MARKER"

    res = {"sentinel": True, "model": M5, "revision": M5_SHA,
           "thresholds": {"n_true_floor": N_TRUE_FLOOR, "n_false_floor": N_FALSE_FLOOR,
                          "degen_median": DEGEN_MEDIAN, "degen_le3": DEGEN_LE3},
           "_bootstrap": {"unit": "image, clustered over derangements", "B": NBOOT,
                          "seed": SEED, "multiplicity": "PRESERVED -- no set(), no unique()"}}

    res["positive_control"] = positive_control(gt, objset)
    if not all(v["pass"] for v in res["positive_control"].values()):
        res["CELL"] = "NO-DATA"
        res["why"] = "estimator positive control FAILED: " + json.dumps(
            {k: v["dev"] for k, v in res["positive_control"].items()})
        json.dump(res, open(f"{OUT}/q3_score.json", "w"), indent=1)
        print("[score] CELL = NO-DATA (positive control failed)", flush=True)
        sentinel("SCORE")
        return
    if os.environ.get("Q3_CTL_ONLY") == "1":
        json.dump(res, open(f"{OUT}/q3_ctl_only.json", "w"), indent=1)
        sentinel("CTL_ONLY")
        return

    census = json.load(open(f"{OUT}/q3_census.json"))
    att = json.load(open(f"{OUT}/q3_attends.json"))
    gate = json.load(open(f"{OUT}/q3_gate.json"))
    rows = jload(f"{OUT}/q3_sighted.jsonl") + jload(f"{OUT}/q3_blind.jsonl")
    by_cell = {}
    for r in rows:
        r["_nment"] = len(chair.mentions(r["prefix_text"]))
        d = by_cell.setdefault((r["image_id"], r["k"]), {})
        assert r["arm"] not in d, f"FATAL_DUPLICATE_ROW {r['image_id']} {r['k']} {r['arm']}"
        d[r["arm"]] = r
    keys = sorted(by_cell)
    ALL = SIGHT + BLIND

    g = {"n_cells": len(keys), "n_rows": len(rows)}
    g["four_arms_every_cell"] = all(set(by_cell[k]) == set(ALL) for k in keys)
    g["budget_equal_within_cell"] = all(
        len({by_cell[k][a]["budget"] for a in ALL}) == 1 for k in keys)
    g["plen_equals_budget"] = all(r["plen"] == r["budget"] for r in rows)
    g["prefix_ntok_equals_budget"] = all(by_cell[k][a]["prefix_ntok"] == by_cell[k][a]["budget"]
                                         for k in keys for a in ALL)
    g["blind_prefix_identical"] = {
        "cap1": sum(by_cell[k]["cap1g"]["prefix_text"] == by_cell[k]["cap1"]["prefix_text"]
                    for k in keys),
        "cap5": sum(by_cell[k]["cap5g"]["prefix_text"] == by_cell[k]["cap5"]["prefix_text"]
                    for k in keys), "n": len(keys)}
    g["zero_doubling_five_scene"] = all(by_cell[k][a]["cycles"] == 0
                                        for k in keys for a in ("cap5", "cap5g"))
    g["one_scene_doubling_rate"] = float(np.mean([by_cell[k]["cap1"]["cycles"] > 0
                                                  for k in keys]))
    g["unit_count_cap1_sources"] = all(by_cell[k][a]["n_src"] == 1
                                       for k in keys for a in ("cap1", "cap1g"))
    g["unit_count_cap5_sources"] = all(by_cell[k][a]["n_src"] == 5 and
                                       len(set(by_cell[k][a]["srcs"])) == 5
                                       for k in keys for a in ("cap5", "cap5g"))
    g["no_self_pairing"] = all(r["src_id"] != r["image_id"] for r in rows)
    g["cap_not_exceeded"] = max(r["cont_len"] for r in rows) <= MAX_384
    g["cont_len_consistent"] = all(r["cont_len"] == len(r["cont_ids"]) for r in rows)
    g["gray_flag_correct"] = all(by_cell[k][a]["gray"] == (a in BLIND)
                                 for k in keys for a in ALL)
    hard = ["four_arms_every_cell", "budget_equal_within_cell", "plen_equals_budget",
            "prefix_ntok_equals_budget", "zero_doubling_five_scene", "unit_count_cap1_sources",
            "unit_count_cap5_sources", "no_self_pairing", "cap_not_exceeded",
            "cont_len_consistent", "gray_flag_correct"]
    g["ALL_HARD_PASS"] = all(bool(g[h]) for h in hard) and \
        g["blind_prefix_identical"]["cap1"] == len(keys) and \
        g["blind_prefix_identical"]["cap5"] == len(keys) and \
        g["one_scene_doubling_rate"] < 0.05
    res["gates_from_jsonl"] = g
    res["GATE_SCENE"] = census["GATE_SCENE"]
    res["GATE_ATTENDS"] = att["GATE_ATTENDS"]
    res["HARNESS_GATE"] = {k: v for k, v in gate.items()
                           if k in ("VERDICT", "why", "cut_f", "thresholds",
                                    "HG1_template", "HG2_prefill_lands", "HG4_free_length",
                                    "HG5_assembled", "HG6_prefix_binds")}
    res["HARNESS_GATE"]["HG3_self_continuation"] = {
        k: v for k, v in gate["HG3_self_continuation"].items() if k != "per_row"}
    res["HARNESS_GATE"]["HG3_calibration_llava15"] = {
        k: v for k, v in gate["HG3_calibration_llava15"].items() if k != "per_row"}
    print("[score] gates " + json.dumps({k: v for k, v in g.items() if k != "n_rows"}),
          flush=True)
    assert g["ALL_HARD_PASS"], "FATAL_JSONL_GATE " + json.dumps(g)

    imgs = sorted({k[0] for k in keys})
    img_ix = {i: j for j, i in enumerate(imgs)}
    n_img = len(imgs)
    vocab = {}
    for k in keys:
        for a in ALL:
            for o in objset(by_cell[k][a]["prefix_text"]):
                vocab.setdefault(o, len(vocab))
    n_obj = len(vocab)

    bt = lambda arms, **kw: build_tables(by_cell, keys, arms, tok, gt, objset, vocab, img_ix,
                                         n_img, **kw)
    res["arm_stats"] = {}
    US, stS, ksS = bt(SIGHT)
    res["arm_stats"]["384_sighted"] = stS
    UB, stB, ksB = bt(BLIND)
    res["arm_stats"]["384_blind"] = stB
    print(f"[score] {len(rows)} rows, {len(keys)} cells, {n_img} images, {n_obj} types",
          flush=True)

    C = {}
    C["PRIMARY"] = contrast("PRIMARY", US["cap1"], US["cap5"], n_img,
                            extra={"n_cells": len(ksS)})
    UK, stK, _ = bt(SIGHT, klen=True)
    res["arm_stats"]["384_sighted_klen"] = stK
    C["PRIMARY_klen"] = contrast("PRIMARY_klen", UK["cap1"], UK["cap5"], n_img)
    C["BLIND"] = contrast("BLIND", UB["cap1g"], UB["cap5g"], n_img, extra={"n_cells": len(ksB)})
    UBK, _, _ = bt(BLIND, klen=True)
    C["BLIND_klen"] = contrast("BLIND_klen", UBK["cap1g"], UBK["cap5g"], n_img)
    C["CP-C2_objmatched"] = contrast("CP-C2_objmatched", US["cap1"], US["cap5"], n_img,
                                     kind="object", n_obj=n_obj)
    unc = lambda d: all(len(d[a]["cont_ids"]) < MAX_384 for a in SIGHT)
    UR, stR, ksR = bt(SIGHT, cellfilter=unc)
    res["arm_stats"]["384_both_uncapped"] = stR
    C["CP-R_uncapped"] = (contrast("CP-R_uncapped", UR["cap1"], UR["cap5"], n_img,
                                   extra={"n_cells": len(ksR)}) if UR else None)
    noe = lambda d: all(len(d[a]["cont_ids"]) > 0 for a in SIGHT)
    UE, _, ksE = bt(SIGHT, cellfilter=noe)
    C["empty_removed"] = (contrast("empty_removed", UE["cap1"], UE["cap5"], n_img,
                                   extra={"n_cells": len(ksE)}) if UE else None)
    C["PRIMARY_seed_alt"] = contrast("PRIMARY_seed_alt", US["cap1"], US["cap5"], n_img,
                                     seed=SEED_ALT)
    U9, st9, ks9 = bt(SIGHT, regime=MAX_192)
    res["arm_stats"]["192_sighted"] = st9
    C["PRIMARY_recon192"] = contrast("PRIMARY_recon192", U9["cap1"], U9["cap5"], n_img,
                                     extra={"n_cells": len(ks9)})

    # Delta_image: ONE joint image resample driving all four arms.
    def joint(sel):
        a = gap_raw(US["cap1"], US["cap1"].gather(sel))[0]
        b = gap_raw(US["cap5"], US["cap5"].gather(sel))[0]
        c = gap_raw(UB["cap1g"], UB["cap1g"].gather(sel))[0]
        d = gap_raw(UB["cap5g"], UB["cap5g"].gather(sel))[0]
        if None in (a, b, c, d):
            return None
        return (b - a) - (d - c)
    ptS = C["PRIMARY"]["point"] - C["BLIND"]["point"]
    ciS, nbS = boot(joint, n_img, seed=SEED, tag="DELTA_IMAGE")
    C["DELTA_IMAGE"] = {"name": "DELTA_IMAGE", "kind": "raw", "point": ptS, "ci": ciS,
                        "ci_width": None if ciS[0] is None else ciS[1] - ciS[0],
                        "n_boot_used": nbS, "seed": SEED, "n_images": n_img,
                        "n_true_A": C["PRIMARY"]["n_true_A"], "n_true_B": C["PRIMARY"]["n_true_B"],
                        "n_false_A": C["PRIMARY"]["n_false_A"],
                        "n_false_B": C["PRIMARY"]["n_false_B"],
                        "note": "primary minus blind; crosses the two generation jobs"}

    # the image's contribution WITHIN each condition (sighted - blind), same joint resample
    for cond, (sa, ba) in (("one_scene", ("cap1", "cap1g")), ("five_scene", ("cap5", "cap5g"))):
        def within(sel, sa=sa, ba=ba):
            x = gap_raw(US[sa], US[sa].gather(sel))[0]
            y = gap_raw(UB[ba], UB[ba].gather(sel))[0]
            return None if (x is None or y is None) else x - y
        p_ = (gap_raw(US[sa], np.arange(len(US[sa].img)))[0]
              - gap_raw(UB[ba], np.arange(len(UB[ba].img)))[0])
        ci_, nb_ = boot(within, n_img, seed=SEED, tag=f"WITHIN_{cond}")
        C[f"WITHIN_{cond}"] = {"name": f"WITHIN_{cond}", "point": p_, "ci": ci_,
                               "ci_width": None if ci_[0] is None else ci_[1] - ci_[0],
                               "n_boot_used": nb_, "seed": SEED, "n_images": n_img}
    res["contrasts"] = C
    res["status"] = {k: status(v) for k, v in C.items()
                     if v and "n_true_A" in v}

    # ---- per-condition rates H and F --------------------------------------------------------
    per_cond = {}
    for a, U in (("cap1", US["cap1"]), ("cap5", US["cap5"]),
                 ("cap1g", UB["cap1g"]), ("cap5g", UB["cap5g"])):
        gp, s = gap_raw(U, np.arange(len(U.img)))
        per_cond[a] = {"H": s[2] / s[0] if s[0] else None, "F": s[3] / s[1] if s[1] else None,
                       "J": gp, "n_true": s[0], "n_false": s[1]}
    res["per_condition_rates"] = per_cond

    # ---- degeneracy screen -------------------------------------------------------------------
    screen, degen_fail = {}, []
    for tag, stt in (("sighted", stS), ("blind", stB)):
        for a, s in stt.items():
            screen[a] = {k: s[k] for k in ("n_cells", "median_cont_len", "mean_cont_len",
                                           "truncation_rate_at_budget", "empty_rate",
                                           "frac_le3", "mean_unique_token_ratio",
                                           "median_unique_token_ratio")}
            ok = (s["median_cont_len"] >= DEGEN_MEDIAN and s["frac_le3"] <= DEGEN_LE3)
            screen[a]["pass"] = bool(ok)
            if not ok:
                degen_fail.append(a)
    screen["ALL_PASS"] = not degen_fail
    screen["failing_arms"] = degen_fail
    res["degeneracy_screen"] = screen

    # ---- readout ------------------------------------------------------------------------------
    di = C["DELTA_IMAGE"]
    lo, hi = di["ci"]
    if degen_fail:
        cell = "UNINTERPRETABLE-DEGENERATE"
        why = (f"arms {degen_fail} fail the degeneracy screen. Harness-gate verdict was "
               f"{gate['VERDICT']}: HG-1/HG-2/HG-3 status decides whether this is the model or "
               f"our harness.")
    elif lo is not None and lo > 0:
        cell = "IMAGE-ATTRIBUTABLE"
        why = ("Delta_image excludes zero and is positive: on this model the rise IS "
               "image-attributable. This bounds the paper's claim and must be reported as a "
               "bound in the main text, as Kosmos-2 is.")
    elif hi is not None and hi < 0:
        cell = "NEGATIVE-DELTA-IMAGE"
        why = ("Delta_image excludes zero and is negative: removing the image makes the rise "
               "LARGER, as on Kosmos-2.")
    elif lo is not None and lo > -0.10 and hi < 0.10:
        cell = "REPLICATES-NULL"
        why = ("Delta_image is null within the paper's +/-0.10 reference margin while "
               "Delta_blind is large: the paper's pattern reproduces on a 2025-generation model.")
    else:
        cell = "NULL-BUT-IMPRECISE"
        why = ("Delta_image's interval contains zero but is wider than the paper's +/-0.10 "
               "reference margin, so it does not resolve the question either way.")
    res["CELL"] = cell
    res["why"] = why
    json.dump(res, open(f"{OUT}/q3_score.json", "w"), indent=1)
    print("[score] per_condition_rates " + json.dumps(per_cond, indent=1), flush=True)
    print("[score] degeneracy " + json.dumps(screen, indent=1), flush=True)
    for k in ("PRIMARY", "PRIMARY_klen", "BLIND", "BLIND_klen", "DELTA_IMAGE",
              "WITHIN_one_scene", "WITHIN_five_scene", "CP-C2_objmatched", "CP-R_uncapped",
              "empty_removed", "PRIMARY_recon192", "PRIMARY_seed_alt"):
        v = C.get(k)
        if v and v["point"] is not None:
            print(f"[score] {k:22s} {v['point']:+.4f} "
                  f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]", flush=True)
    print(f"[score] CELL = {cell}\n[score] {why}", flush=True)
    sentinel("SCORE")


if __name__ == "__main__":
    main()
