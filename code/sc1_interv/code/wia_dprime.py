"""TASK-2: signal-detection decomposition of `gap = r_true - r_false`.

`gap` is hit-rate minus false-alarm-rate (Youden's J; `Pr` in the recognition-memory
literature). Snodgrass & Corwin (1988) prescribe reporting a DISCRIMINATION measure and a
BIAS measure separately, because "the difference is unchanged" is not "discriminability is
unchanged". This script computes, for every arm of every blind-capable lane, in BOTH
conditions:

    d' = z(r_true) - z(r_false)          discrimination
    c  = -0.5 * (z(r_true) + z(r_false)) bias   (c > 0 = conservative = carries less)

BOUNDARY HANDLING: the standard log-linear correction, Hautus (1995) / Snodgrass & Corwin
(1988), applied to EVERY arm unconditionally (never only to the boundary ones, which would
bias the comparison between arms):

    r_true_corr = (carried_true + 0.5) / (n_true + 1)
    r_false_corr = (carried_false + 0.5) / (n_false + 1)

Uncorrected rates and uncorrected d'/c are reported alongside so the correction's effect is
visible, and every replicate in which an uncorrected rate would hit 0 or 1 (z infinite) is
counted.

BOOTSTRAP: the lanes' own image-clustered resamples, reproduced exactly -- same generator
(np.random.default_rng), same seed per lane (F1/F2 20260912 from wia_blindscore.py, CX
20260911 from wia_cxsight.py), same B=4000, same draw order, so replicate b here is the same
image multiset as replicate b there. Multiplicity is preserved: the draw is
rng.integers(0, n, n) and `assert len(sel) == n` fires on EVERY replicate. No set(), no
unique(), no dict keyed on image id anywhere on the resampling path (the duplicate assertion
on the first 20 replicates reads sel but never rebinds it).

POSITIVE CONTROL: every arm's n_true / n_false / r_true / r_false is recomputed from the raw
.jsonl rows and checked against the published per-arm values of each lane. No result JSON is
trusted; the published numbers are inlined as constants below.

Writes out/wia_dprime.json. Sentinel J5_SENTINEL_DPRIME_OK.
"""
import os, sys, json, math, collections
from statistics import NormalDist
import numpy as np
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/sc1_gen/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")
from j5_common import load_coco_sample, ChairScorer, read_jsonl, sentinel
from wia_corrobscore import Units, gap_raw

OUT_I = "/data/alexmueller/sc1_interv/out"
OUT_G = "/data/alexmueller/sc1_gen/out"
RES = f"{OUT_I}/wia_dprime.json"
SYN = "/data/alexmueller/j5_gates/code/synonyms.txt"
NB = int(os.environ.get("DPRIME_NBOOT", "0")) or 4000

_ND = NormalDist()
def Z(p):
    return _ND.inv_cdf(p)

# ---------------------------------------------------------------- lane definitions
LANES = {
    "F1": {
        "desc": "baseline lane family 1, llava-1.5-7b, cap 384, k in 1..3, 1500 cells",
        "files": {"s": f"{OUT_I}/at_corrob.jsonl", "b": f"{OUT_I}/at_blind1.jsonl"},
        "model": "llava-hf/llava-1.5-7b-hf", "seed": 20260912, "n_img": 500,
        # (label, condition, row-arm-name)
        "arms": [("cap1", "s", "cap1"), ("cap5", "s", "cap5"),
                 ("cap1", "b", "cap1g"), ("cap5", "b", "cap5g")],
        "contrast": ("cap1", "cap5"),
        "published": {("cap1", "s"): (528, 2332, 0.6761363636363636, 0.4879931389365352),
                      ("cap5", "s"): (892, 4988, 0.7802690582959642, 0.10625501202886929),
                      ("cap1", "b"): (528, 2332, 0.4678030303030303, 0.4897084048027444),
                      ("cap5", "b"): (892, 4988, 0.7085201793721974, 0.2285485164394547)},
        "published_delta": {"s": 0.4859, "b": 0.5019},
        "published_delta_ci": {"s": [0.4292, 0.5416], "b": [0.4441, 0.5565]},
    },
    "F2": {
        "desc": "baseline lane family 2, llava-onevision-qwen2-0.5b-ov, cap 192, 1500 cells",
        "files": {"s": f"{OUT_G}/m2_corrob.jsonl", "b": f"{OUT_G}/m2_blind.jsonl"},
        "model": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", "seed": 20260912, "n_img": 500,
        "arms": [("cap1", "s", "cap1"), ("cap5", "s", "cap5"),
                 ("cap1", "b", "m2cap1g"), ("cap5", "b", "m2cap5g")],
        "contrast": ("cap1", "cap5"),
        "published": {("cap1", "s"): (554, 2452, 0.5577617328519856, 0.5171288743882545),
                      ("cap5", "s"): (1020, 6308, 0.5470588235294118, 0.12840837032339886),
                      ("cap1", "b"): (554, 2452, 0.20577617328519857, 0.32300163132137033),
                      ("cap5", "b"): (1020, 6308, 0.39215686274509803, 0.09654407102092581)},
        "published_delta": {"s": 0.3780, "b": 0.4128},
        "published_delta_ci": {"s": [0.3201, 0.4335], "b": [0.3609, 0.4640]},
    },
    "CX": {
        "desc": "cap1x lane, llava-1.5-7b, cap 384, k in 1..10, 2166 cells / 496 images",
        "files": {"s": f"{OUT_I}/at_cx.jsonl", "b": f"{OUT_I}/at_cx_blind.jsonl"},
        "model": "llava-hf/llava-1.5-7b-hf", "seed": 20260911, "n_img": None,
        "arms": [(a, c, a) for c in ("s", "b")
                 for a in ("cap1", "cap1y", "cap1x", "cap5")],
        "contrast": ("cap1", "cap5"),
        "contrast2": ("cap1y", "cap1x"),
        "published": {("cap1", "s"): (1135, 4202, 0.7154, 0.5233),
                      ("cap1y", "s"): (1129, 4117, 0.7068, 0.5409),
                      ("cap1x", "s"): (1129, 4117, 0.7724, 0.5443),
                      ("cap5", "s"): (1490, 8170, 0.7913, 0.0988),
                      ("cap1", "b"): (1135, 4202, 0.4678, 0.4681),
                      ("cap1y", "b"): (1129, 4117, 0.4748, 0.5013),
                      ("cap1x", "b"): (1129, 4117, 0.5872, 0.5689),
                      ("cap5", "b"): (1490, 8170, 0.7101, 0.1995)},
        "published_delta": {"s": 0.5004, "b": 0.5108},
        "published_delta_ci": {"s": [0.4618, 0.5382], "b": [0.4700, 0.5522]},
        "pub_tol": 1e-4,   # published to 4 dp in the lane log
    },
}


def _logit(p):
    return math.log(p / (1.0 - p))


def sdt(nt, nf, ct, cf):
    """(raw rates, log-linear corrected rates, d', c, uncorrected d'/c or None)."""
    rt, rf = ct / nt, cf / nf
    h, f = (ct + 0.5) / (nt + 1), (cf + 0.5) / (nf + 1)
    zh, zf = Z(h), Z(f)
    d, cbias = zh - zf, -0.5 * (zh + zf)
    if 0.0 < rt < 1.0 and 0.0 < rf < 1.0:
        zu, zfu = Z(rt), Z(rf)
        du, cu = zu - zfu, -0.5 * (zu + zfu)
    else:
        du = cu = None
    return rt, rf, h, f, d, cbias, du, cu


def load_lane(name, chair, objset, gt):
    cfg = LANES[name]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    rows = {c: list(read_jsonl(p)) for c, p in cfg["files"].items()}
    cell = {c: {} for c in rows}
    for c in rows:
        for r in rows[c]:
            d = cell[c].setdefault((r["image_id"], r["k"]), {})
            assert r["arm"] not in d, f"FATAL_DUPLICATE_ROW {name} {r['image_id']} {r['k']}"
            assert r["cont_len"] == len(r["cont_ids"]), "FATAL_CONT_LEN_MISMATCH"
            d[r["arm"]] = r
    keys = sorted(set(cell["s"]) & set(cell["b"]))
    assert len(keys) == len(cell["s"]) == len(cell["b"]), \
        f"FATAL_CELL_SET_MISMATCH {name} {len(keys)} {len(cell['s'])} {len(cell['b'])}"
    imgs = sorted({k[0] for k in keys})
    ix = {i: j for j, i in enumerate(imgs)}
    if cfg["n_img"] is not None:
        assert len(imgs) == cfg["n_img"], f"FATAL_N_IMG {name} {len(imgs)}"

    drift = txtmiss = 0
    drift_on = {}
    retok = {"pairs": 0, "ids_equal": 0, "len_equal": 0,
             "plen_present": 0, "plen_equal": 0, "retok_minus_plen": {}}
    cols = {(a, c): [[], [], [], []] for (a, c, _r) in cfg["arms"]}
    lens = collections.defaultdict(list)
    for k in keys:
        for (lab, c, arm) in cfg["arms"]:
            r = cell[c][k][arm]
            s_arm = [x for (l2, c2, x) in cfg["arms"] if l2 == lab and c2 == "s"][0]
            rs = cell["s"][k][s_arm]
            if "prefix_ids" in r and "prefix_ids" in rs:
                drift += (list(r["prefix_ids"]) != list(rs["prefix_ids"]))
                drift_on["ids"] = True
            else:
                drift += (r["prefix_text"] != rs["prefix_text"])
                drift_on["text"] = True
                # The written rows carry no prefix token ids, so bit-identity cannot be read
                # off disk. Strongest available substitute: re-tokenise BOTH prefix_texts with
                # this family's own tokenizer and compare the id sequences, and compare the
                # recorded prefix token count `plen` on the two sides.
                if c != "s":
                    ids_b = tok(r["prefix_text"], add_special_tokens=False)["input_ids"]
                    ids_s = tok(rs["prefix_text"], add_special_tokens=False)["input_ids"]
                    retok["pairs"] += 1
                    retok["ids_equal"] += (list(ids_b) == list(ids_s))
                    retok["len_equal"] += (len(ids_b) == len(ids_s))
                    if "plen" in r and "plen" in rs:
                        retok["plen_present"] += 1
                        retok["plen_equal"] += (r["plen"] == rs["plen"])
                        retok["retok_minus_plen"][len(ids_b) - r["plen"]] = \
                            retok["retok_minus_plen"].get(len(ids_b) - r["plen"], 0) + 1
            text = tok.decode(r["cont_ids"], skip_special_tokens=True)
            txtmiss += (text != r.get("cont_text", text))
            lens[(lab, c)].append(len(r["cont_ids"]))
            P, C = objset(r["prefix_text"]), objset(text)
            g = gt[k[0]]
            for o in P:
                cols[(lab, c)][0].append(ix[k[0]])
                cols[(lab, c)][1].append(0)
                cols[(lab, c)][2].append(o in g)
                cols[(lab, c)][3].append(o in C)
    assert drift == 0, f"FATAL_BLIND_PREFIX_DRIFT {name} {drift}"
    U = {kk: Units(*cols[kk], n_img=len(imgs)) for kk in cols}
    meta = {"n_cells": len(keys), "n_images": len(imgs),
            "prefix_identity_checked_on": sorted(drift_on),
            "prefix_retokenisation_check": retok,
            "decode_vs_cont_text_mismatches": txtmiss,
            "mean_cont_len": {f"{a}_{c}": float(np.mean(v)) for (a, c), v in lens.items()}}
    return U, len(imgs), meta


def run_lane(name, U, n_img, meta):
    cfg = LANES[name]
    labs = [(a, c) for (a, c, _r) in cfg["arms"]]
    tol = cfg.get("pub_tol", 1e-6)

    def one(gd):
        out, bad = {}, 0
        for kk in labs:
            _g, (nt, nf, ct, cf) = gap_raw(U[kk], gd[kk])
            if nt == 0 or nf == 0:
                return None, 0
            rt, rf, h, f, d, cb, du, cu = sdt(nt, nf, ct, cf)
            bad += (rt in (0.0, 1.0)) + (rf in (0.0, 1.0))
            nm = f"{kk[0]}_{kk[1]}"
            out[f"r_true_{nm}"], out[f"r_false_{nm}"] = rt, rf
            out[f"gap_{nm}"] = rt - rf
            out[f"dprime_{nm}"], out[f"c_{nm}"] = d, cb
            # SCALE SENSITIVITY: the same two corrected rates under a logit link instead of
            # the probit link d' assumes. An interaction that survives both is not an artefact
            # of choosing the Gaussian scale; one that does not, is.
            lh, lf = _logit(h), _logit(f)
            out[f"dlogit_{nm}"], out[f"clogit_{nm}"] = lh - lf, -0.5 * (lh + lf)
        A, B = cfg["contrast"]
        pairs = [("51", A, B)]
        if "contrast2" in cfg:
            pairs.append(("x", cfg["contrast2"][0], cfg["contrast2"][1]))
        for tag, lo, hi in pairs:
            for c in ("s", "b"):
                for q in ("dprime", "c", "gap", "dlogit", "clogit"):
                    out[f"D{tag}_{q}_{c}"] = out[f"{q}_{hi}_{c}"] - out[f"{q}_{lo}_{c}"]
            for q in ("dprime", "c", "gap", "dlogit", "clogit"):
                out[f"D{tag}_{q}_sight"] = out[f"D{tag}_{q}_s"] - out[f"D{tag}_{q}_b"]
        for a in sorted({l for (l, _c) in labs}):
            for q in ("dprime", "c", "gap", "r_true", "r_false", "dlogit", "clogit"):
                out[f"sight_{q}_{a}"] = out[f"{q}_{a}_s"] - out[f"{q}_{a}_b"]
        return out, bad

    allg = {kk: np.arange(len(U[kk].img)) for kk in U}
    point, _b0 = one(allg)
    assert point is not None, f"FATAL_POINT_UNDEFINED {name}"

    # ---- positive control, recomputed from the raw rows -----------------------------
    ctl, ctl_ok = {}, True
    for kk in labs:
        _g, (nt, nf, ct, cf) = gap_raw(U[kk], allg[kk])
        pnt, pnf, prt, prf = cfg["published"][kk]
        dev = {"n_true": [nt, pnt], "n_false": [nf, pnf],
               "r_true_dev": abs(ct / nt - prt), "r_false_dev": abs(cf / nf - prf)}
        good = (nt == pnt and nf == pnf and dev["r_true_dev"] <= tol
                and dev["r_false_dev"] <= tol)
        ctl[f"{kk[0]}_{kk[1]}"] = dict(dev, PASS=bool(good))
        ctl_ok &= good
    for c in ("s", "b"):
        dev = abs(point[f"D51_gap_{c}"] - cfg["published_delta"][c])
        ctl[f"Delta_gap_{c}"] = {"recomputed": point[f"D51_gap_{c}"],
                                 "published": cfg["published_delta"][c],
                                 "abs_dev": dev, "PASS": bool(dev <= 5e-4)}
        ctl_ok &= dev <= 5e-4

    # ---- bootstrap: the lane's own image-clustered draws -----------------------------
    rng = np.random.default_rng(cfg["seed"])
    acc, nbad, ndrop, ndup_checked = collections.defaultdict(list), 0, 0, 0
    for b in range(NB):
        sel = rng.integers(0, n_img, n_img)
        assert len(sel) == n_img, f"FATAL_BOOTSTRAP_LENGTH {name}#{b}"
        if b < 20:
            assert len(set(sel.tolist())) < n_img, f"FATAL_BOOTSTRAP_NO_DUPLICATES {name}#{b}"
            ndup_checked += 1
        d, bad = one({kk: U[kk].gather(sel) for kk in U})
        if d is None:
            ndrop += 1
            continue
        nbad += bad
        for k2, v in d.items():
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                acc[k2].append(v)
    ci = {k2: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
          for k2, v in acc.items()}

    # BOOTSTRAP-IDENTITY CONTROL: same generator, same seed, same draw order as the lane, so
    # at full B the recomputed CI of the gap contrast must reproduce the lane's published CI.
    cictl, cictl_ok = {}, True
    for c in ("s", "b"):
        got, want = ci[f"D51_gap_{c}"], cfg["published_delta_ci"][c]
        dev = max(abs(got[0] - want[0]), abs(got[1] - want[1]))
        good = dev <= 5e-5
        cictl[f"D51_gap_{c}"] = {"recomputed_ci": got, "published_ci": want,
                                 "max_abs_dev": dev, "PASS": bool(good)}
        cictl_ok &= good

    # ---- boundary / correction report -------------------------------------------------
    bnd = {}
    for kk in labs:
        _g, (nt, nf, ct, cf) = gap_raw(U[kk], allg[kk])
        rt, rf, h, f, d, cb, du, cu = sdt(nt, nf, ct, cf)
        nm = f"{kk[0]}_{kk[1]}"
        bnd[nm] = {"n_true": nt, "n_false": nf, "carried_true": ct, "carried_false": cf,
                   "r_true_raw": rt, "r_false_raw": rf,
                   "r_true_corrected": h, "r_false_corrected": f,
                   "shift_r_true": h - rt, "shift_r_false": f - rf,
                   "dist_to_boundary": min(rt, 1 - rt, rf, 1 - rf),
                   "dprime_corrected": d, "dprime_uncorrected": du,
                   "dprime_correction_effect": None if du is None else d - du,
                   "c_corrected": cb, "c_uncorrected": cu,
                   "c_correction_effect": None if cu is None else cb - cu}
    return {"desc": cfg["desc"], "meta": meta, "seed": cfg["seed"], "B": NB,
            "n_boot_dropped": ndrop, "n_boundary_rate_events": nbad,
            "n_replicates_duplicate_checked": ndup_checked,
            "POSITIVE_CONTROL": ctl, "POSITIVE_CONTROL_PASS": bool(ctl_ok),
            "BOOTSTRAP_IDENTITY_CONTROL": cictl,
            "BOOTSTRAP_IDENTITY_PASS": bool(cictl_ok) if NB == 4000 else None,
            "boundary_and_correction": bnd,
            "point": point, "ci": ci,
            "ci_width": {k2: ci[k2][1] - ci[k2][0] for k2 in ci}}


def main():
    chair = ChairScorer(SYN)
    data = load_coco_sample(chair, n=500)
    gt = {d["image_id"]: set(d["gt"]) for d in data}
    objset = lambda t: set(nm for (nm, _c) in chair.mentions(t))
    out = {"sentinel": True, "B": NB,
           "correction": "log-linear: (hits+0.5)/(n_signal+1), (fa+0.5)/(n_noise+1), "
                         "applied to EVERY arm unconditionally",
           "z": "statistics.NormalDist().inv_cdf",
           "z_check_0.975": Z(0.975), "lanes": {}}
    assert abs(Z(0.975) - 1.959963984540054) < 1e-12, "FATAL_Z_WRONG"
    for name in ("F1", "F2", "CX"):
        cfg = LANES[name]
        miss = [p for p in cfg["files"].values() if not os.path.exists(p)]
        if miss:
            print(f"[dprime] {name}: rows absent {miss}, SKIPPED", flush=True)
            out["lanes"][name] = {"SKIPPED": miss}
            continue
        U, n_img, meta = load_lane(name, chair, objset, gt)
        r = run_lane(name, U, n_img, meta)
        out["lanes"][name] = r
        print(f"[dprime] === {name} :: {cfg['desc']}", flush=True)
        print(f"[dprime] {name} POSITIVE_CONTROL_PASS={r['POSITIVE_CONTROL_PASS']} "
              f"cells={meta['n_cells']} imgs={meta['n_images']} "
              f"decode_mismatch={meta['decode_vs_cont_text_mismatches']} "
              f"boundary_events={r['n_boundary_rate_events']} dropped={r['n_boot_dropped']} "
              f"BOOTSTRAP_IDENTITY={r['BOOTSTRAP_IDENTITY_PASS']} "
              f"{json.dumps({k3: round(v3['max_abs_dev'], 8) for k3, v3 in r['BOOTSTRAP_IDENTITY_CONTROL'].items()})} "
              f"prefix_check={meta['prefix_identity_checked_on']} "
              f"retok={json.dumps(meta['prefix_retokenisation_check'])}", flush=True)
        assert r["POSITIVE_CONTROL_PASS"], f"FATAL_POSITIVE_CONTROL_FAILED {name}"
        P, CI = r["point"], r["ci"]
        for (a, c, _x) in cfg["arms"]:
            nm = f"{a}_{c}"
            b = r["boundary_and_correction"][nm]
            print(f"[dprime] {name} ARM {a:6s} {'SIGHTED' if c=='s' else 'BLIND  '} "
                  f"rT {b['r_true_raw']:.4f} rF {b['r_false_raw']:.4f} "
                  f"gap {P['gap_'+nm]:+.4f} [{CI['gap_'+nm][0]:+.4f},{CI['gap_'+nm][1]:+.4f}] "
                  f"d' {P['dprime_'+nm]:+.4f} [{CI['dprime_'+nm][0]:+.4f},"
                  f"{CI['dprime_'+nm][1]:+.4f}] "
                  f"c {P['c_'+nm]:+.4f} [{CI['c_'+nm][0]:+.4f},{CI['c_'+nm][1]:+.4f}] "
                  f"| corr_effect d' {b['dprime_correction_effect']:+.5f} "
                  f"dist_bnd {b['dist_to_boundary']:.4f}", flush=True)
        for k2 in sorted(P):
            if k2.startswith(("D51_", "Dx_", "sight_")):
                print(f"[dprime] {name} {k2:22s} {P[k2]:+.4f} "
                      f"[{CI[k2][0]:+.4f},{CI[k2][1]:+.4f}] w {r['ci_width'][k2]:.4f} "
                      f"{'EXCLUDES0' if (CI[k2][0] > 0 or CI[k2][1] < 0) else 'contains0'}",
                      flush=True)
    json.dump(out, open(RES if NB == 4000 else RES.replace(".json", "_smoke.json"), "w"),
              indent=1)
    print(f"[dprime] wrote {RES} (B={NB})", flush=True)
    sentinel("DPRIME" if NB == 4000 else "DPRIME_SMOKE")


if __name__ == "__main__":
    main()
