"""L2 scorer -- the ablation ladder on standard CHAIR (LADDER_PREREG.md section 4).

`units`, `chair_s`, `chair_i` and `sdt_of` are IMPORTED from `md_score`, the published lane's
own scorer, so the CHAIR port, the occurrence unit, the 80-category census and the log-linear
correction cannot drift from the numbers in Appendix G. Nothing there is modified.

Order: PC3 (re-score the frozen mitdecomp arms) -> denominator screen -> only then, at USABLE
rungs, the PAI contrast.

THE PRIMARY IS DENOMINATOR-FREE. CHAIR's denominator is the arm's own mention set, so a
sighted-vs-ablated CHAIR contrast conflates a rate change with a composition change
(Appendix G.7's own argument). The registered primary is therefore J = H - F on the FIXED
500 x 80 = 40,000-unit census; CHAIR_i is reported as a secondary with that caveat attached.

Writes out/lad_l2.json. Sentinel printed by Python.
"""
import os, sys, json, math, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/alexmueller/mitdecomp/code")
sys.path.insert(0, "/data/alexmueller/paialpha/code")
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")

import numpy as np
from lad_common import RUNGS, OUT, sentinel, jdump
from md_common import SYN, MODEL, OUT as MDOUT
from md_score import units, chair_s, chair_i, sdt_of
from j5_common import ChairScorer, read_jsonl

NBOOT = int(os.environ.get("LAD_NBOOT", "4000"))
SEED = 20260919
USABLE_MIN_CAPS = 250          # of 500
USABLE_MIN_OCC = 1000          # vs the sighted vanilla arm's 3,827
PC3_REF = {"CHAIR_s_vanilla": 45.00, "CHAIR_i_vanilla": 12.78, "CHAIR_i_pai05": 7.2,
           "H_vanilla": 0.78, "F_vanilla": 0.010}
PC3_TOL_CHAIR, PC3_TOL_RATE = 0.05, 5e-3
GAIN_SURVIVE_FRAC = 0.5


def load(path):
    rows = read_jsonl(path)
    rows.sort(key=lambda r: r["image_id"])
    return rows


def boot_joint(fns, n, nboot=NBOOT, seed=SEED):
    """One image-clustered draw per replicate, every statistic on that draw. No set(), no
    unique(), no dict keyed on image id: multiplicity is the whole point."""
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in fns}
    for b in range(nboot):
        sel = rng.integers(0, n, n)
        assert len(sel) == n, f"FATAL_BOOTSTRAP_LENGTH {b}"
        if b < 20 and n >= 20:
            assert len(set(sel.tolist())) < n, f"FATAL_BOOTSTRAP_NO_DUPLICATES {b}"
        for k, f in fns.items():
            v = f(sel)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                acc[k].append(float(v))
    return acc


def _topshare(rows):
    from collections import Counter
    c = Counter(r["text"] for r in rows).most_common(1)
    return (c[0][1] / len(rows)) if c else None


def ivl(point, vals):
    ci = ([float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
          if len(vals) else [None, None])
    return {"point": point, "ci": ci, "n_boot": len(vals)}


def main():
    print(f"[lad_cscore] host={socket.gethostname()} B={NBOOT}", flush=True)
    from transformers import AutoTokenizer
    chair = ChairScorer(SYN)
    cats = sorted({v for v in chair.inverse.values()})
    assert len(cats) == 80, f"FATAL_NOT_80_CATEGORIES {len(cats)}"
    tok = AutoTokenizer.from_pretrained(MODEL)

    res = {"sentinel": True, "host": socket.gethostname(), "B": NBOOT, "seed": SEED,
           "prereg": "SPRINT/LADDER_PREREG.md @ 09ddcf6 (Amendment 1)",
           "usable_thresholds": {"min_captions_with_mention": USABLE_MIN_CAPS,
                                 "min_mention_occurrences": USABLE_MIN_OCC}}

    # ---- PC3: the frozen mitdecomp arms ---------------------------------------------------
    froz = {a: load(f"{MDOUT}/md_{a}.jsonl") for a in ("vanilla", "pai05")}
    n = len(froz["vanilla"])
    ids0 = [r["image_id"] for r in froz["vanilla"]]
    assert [r["image_id"] for r in froz["pai05"]] == ids0, "FATAL_IMAGE_SET_MISMATCH_FROZEN"
    Uf = {a: units(froz[a], chair, cats, tok) for a in froz}
    allsel = np.arange(n)
    sv, sp = sdt_of(Uf["vanilla"], allsel), sdt_of(Uf["pai05"], allsel)
    meas = {"CHAIR_s_vanilla": chair_s(Uf["vanilla"], allsel),
            "CHAIR_i_vanilla": chair_i(Uf["vanilla"], allsel),
            "CHAIR_i_pai05": chair_i(Uf["pai05"], allsel),
            "H_vanilla": sv["H"], "F_vanilla": sv["F"]}
    dev = {k: abs(meas[k] - PC3_REF[k]) for k in PC3_REF}
    ok = all(dev[k] <= (PC3_TOL_RATE if k.startswith(("H_", "F_")) else PC3_TOL_CHAIR)
             for k in dev)
    res["PC3"] = {"measured": meas, "reference": PC3_REF, "abs_dev": dev, "PASS": bool(ok),
                  "tol_chair": PC3_TOL_CHAIR, "tol_rate": PC3_TOL_RATE,
                  "n_images": n,
                  "sighted": {"vanilla": sv, "pai05": sp,
                              "gain_i": chair_i(Uf["vanilla"], allsel)
                              - chair_i(Uf["pai05"], allsel),
                              "dJ": sp["J"] - sv["J"]}}
    print(f"[lad_cscore] PC3 {json.dumps(res['PC3']['measured'])} PASS={ok}", flush=True)

    gs_fns = {"gain_i": lambda s: chair_i(Uf["vanilla"], s) - chair_i(Uf["pai05"], s),
              "dJ": lambda s: sdt_of(Uf["pai05"], s)["J"] - sdt_of(Uf["vanilla"], s)["J"]}
    Vs = boot_joint(gs_fns, n)
    res["sighted_contrast"] = {
        "gain_i": ivl(res["PC3"]["sighted"]["gain_i"], Vs["gain_i"]),
        "dJ": ivl(res["PC3"]["sighted"]["dJ"], Vs["dJ"])}
    print(f"[lad_cscore] sighted gain_i {json.dumps(res['sighted_contrast']['gain_i'])}",
          flush=True)

    # ---- denominator screen ----------------------------------------------------------------
    screen, U = {}, {}
    L2_RUNGS = list(RUNGS) + ["mismatch", "black"]
    for rg in L2_RUNGS:
        p, mp = f"{OUT}/lad_c_vanilla_{rg}.jsonl", f"{OUT}/lad_c_vanilla_{rg}_meta.json"
        if not os.path.exists(p):
            continue
        # A generation job may be WRITING this file. The real condition for completeness is
        # the meta sentinel plus row-count agreement, never file existence.
        if not os.path.exists(mp):
            print(f"[lad_cscore] {rg}: no meta yet (job still running?), skipped", flush=True)
            continue
        rows = load(p)
        m = json.load(open(mp))
        if not m.get("sentinel") or m.get("n") != len(rows):
            print(f"[lad_cscore] {rg}: INCOMPLETE meta.n={m.get('n')} disk={len(rows)}, "
                  f"skipped", flush=True)
            continue
        assert [r["image_id"] for r in rows] == ids0, f"FATAL_IMAGE_SET_MISMATCH {rg}"
        u = units(rows, chair, cats, tok)
        U[("vanilla", rg)] = u
        occ = int(u["tot"].sum())
        caps = int((u["tot"] > 0).sum())
        screen[rg] = {"mention_occurrences": occ, "captions_with_mention": caps,
                      "n": len(rows),
                      "median_tokens": float(np.median([r["n_tokens"] for r in rows])),
                      "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])),
                      "n_distinct_texts": len({r["text"] for r in rows}),
                      "truncation_rate": float(np.mean([r["truncated"] for r in rows])),
                      "empty_rate": float(np.mean([r["n_tokens"] == 0 for r in rows])),
                      "CHAIR_i": (None if occ == 0 else chair_i(u, allsel)),
                      "CHAIR_s": chair_s(u, allsel),
                      "le3_token_rate": float(np.mean([r["n_tokens"] <= 3 for r in rows])),
                      "top_text_share": _topshare(rows),
                      "DEFINED": occ >= 1,
                      "USABLE": bool(caps >= USABLE_MIN_CAPS and occ >= USABLE_MIN_OCC)}
        screen[rg]["verdict"] = ("USABLE" if screen[rg]["USABLE"]
                                 else ("DENOMINATOR-TOO-THIN" if screen[rg]["DEFINED"]
                                       else "UNDEFINED (0/0)"))
        print(f"[lad_cscore] screen {rg:10s} occ {occ:6d} caps {caps:4d}/500 "
              f"CHAIR_i {screen[rg]['CHAIR_i']} med_tok {screen[rg]['median_tokens']:.0f} "
              f"-> {screen[rg]['verdict']}", flush=True)
    # the published grey arm, as the calibration point
    if os.path.exists(f"{MDOUT}/md_vanilla_blind.jsonl"):
        gb = load(f"{MDOUT}/md_vanilla_blind.jsonl")
        ug = units(gb, chair, cats, tok)
        screen["grey_published"] = {"mention_occurrences": int(ug["tot"].sum()),
                                    "captions_with_mention": int((ug["tot"] > 0).sum()),
                                    "n": len(gb),
                                    "median_tokens": float(np.median([r["n_tokens"]
                                                                      for r in gb])),
                                    "n_distinct_texts": len({r["text"] for r in gb}),
                                    "verdict": "UNDEFINED (0/0)"}
    # Amendment 3: a severity floor, so "PAI's gain survives at rung X" cannot be read as
    # surviving a trivial degradation. Registered BEFORE emb_cos was measured, and anchored on
    # the mismatch arm rather than on an absolute cosine, because pooled patch embeddings are
    # anisotropic and an absolute threshold would not be scale-free: a rung is MATERIAL iff its
    # mean emb_cos is at or below the midpoint between the identity rung (1.0) and the
    # mismatched-real-image rung, i.e. it has travelled at least half the distance to "a
    # different real image".
    dose_p = f"{OUT}/lad_dose.json"
    if os.path.exists(dose_p):
        dose = dict(json.load(open(dose_p))["emb_cos"])
        # `mismatch` is not a pixel transform, so lad_dose.py cannot produce it; lad_dose_mm.py
        # computes it as the cosine between the clean embeddings of A and its substitute B.
        mm_p = f"{OUT}/lad_dose_mm.json"
        if os.path.exists(mm_p):
            dose.update(json.load(open(mm_p))["emb_cos"])
        floor = (1.0 + dose["mismatch"]) / 2.0 if "mismatch" in dose else None
        res["severity"] = {"emb_cos": dose, "material_floor_emb_cos": floor,
                           "anchor": "midpoint between identity (1.0) and mismatch"}
        for rg, s_ in screen.items():
            c = dose.get(rg)
            s_["emb_cos"] = c
            s_["MATERIAL"] = (None if (c is None or floor is None) else bool(c <= floor))
    else:
        res["severity"] = {"note": "lad_dose.json absent; severity floor not applied"}
    res["denominator_screen"] = screen

    # ---- the contrast at USABLE rungs -------------------------------------------------------
    out = {}
    for rg, s in screen.items():
        if rg == "grey_published" or not s.get("USABLE"):
            continue
        pv, pm = f"{OUT}/lad_c_pai05_{rg}.jsonl", f"{OUT}/lad_c_pai05_{rg}_meta.json"
        if not (os.path.exists(pv) and os.path.exists(pm)):
            out[rg] = {"verdict": "PAI-NOT-YET-RUN"}
            continue
        rp = load(pv)
        mp2 = json.load(open(pm))
        if not mp2.get("sentinel") or mp2.get("n") != len(rp):
            out[rg] = {"verdict": "PAI-INCOMPLETE"}
            continue
        assert [r["image_id"] for r in rp] == ids0, f"FATAL_IMAGE_SET_MISMATCH pai05 {rg}"
        up = units(rp, chair, cats, tok)
        uv = U[("vanilla", rg)]
        sv_r, sp_r = sdt_of(uv, allsel), sdt_of(up, allsel)
        gain_pt = chair_i(uv, allsel) - chair_i(up, allsel)
        dJ_pt = sp_r["J"] - sv_r["J"]
        fns = {"gain_i_r": lambda s: chair_i(uv, s) - chair_i(up, s),
               "dJ_r": lambda s: sdt_of(up, s)["J"] - sdt_of(uv, s)["J"],
               "gain_i_image": lambda s: ((chair_i(Uf["vanilla"], s) - chair_i(Uf["pai05"], s))
                                          - (chair_i(uv, s) - chair_i(up, s))),
               "dJ_image": lambda s: ((sdt_of(Uf["pai05"], s)["J"]
                                       - sdt_of(Uf["vanilla"], s)["J"])
                                      - (sdt_of(up, s)["J"] - sdt_of(uv, s)["J"]))}
        V = boot_joint(fns, n)
        gs = res["PC3"]["sighted"]["gain_i"]
        g = ivl(gain_pt, V["gain_i_r"])
        gi = ivl(gs - gain_pt, V["gain_i_image"])
        dj = ivl(dJ_pt, V["dJ_r"])
        dji = ivl(res["PC3"]["sighted"]["dJ"] - dJ_pt, V["dJ_image"])
        if g["ci"][0] is not None and g["ci"][0] > 0 and gain_pt >= GAIN_SURVIVE_FRAC * gs:
            v = "GAIN-SURVIVES-ABLATION"
        elif (g["ci"][0] is not None and g["ci"][0] <= 0 <= g["ci"][1]
              and res["sighted_contrast"]["gain_i"]["ci"][0] > 0
              and gi["ci"][0] is not None and gi["ci"][0] > 0):
            v = "GAIN-IS-IMAGE-ATTRIBUTABLE"
        else:
            v = "CANNOT-RESOLVE"
        mat = screen[rg].get("MATERIAL")
        if v == "GAIN-SURVIVES-ABLATION" and mat is False:
            v = "GAIN-SURVIVES-BUT-ABLATION-NOT-MATERIAL"
        out[rg] = {"verdict": v, "material_ablation": mat,
                   "emb_cos": screen[rg].get("emb_cos"),
                   "gain_i_rung": g, "gain_i_image": gi,
                   "gain_i_sighted": res["sighted_contrast"]["gain_i"],
                   "dJ_rung": dj, "dJ_image": dji,
                   "levels": {"vanilla": {k: sv_r[k] for k in ("H", "F", "J", "dprime", "c")},
                              "pai05": {k: sp_r[k] for k in ("H", "F", "J", "dprime", "c")},
                              "CHAIR_i_vanilla": chair_i(uv, allsel),
                              "CHAIR_i_pai05": chair_i(up, allsel),
                              "CHAIR_s_vanilla": chair_s(uv, allsel),
                              "CHAIR_s_pai05": chair_s(up, allsel)},
                   "denominator_caveat": "CHAIR_i's denominator is the arm's own mention set; "
                                         "the primary here is the fixed-unit J"}
        print(f"[lad_cscore] {rg:10s} gain_i {gain_pt:+.3f} {g['ci']}  "
              f"gain_i_image {gi['point']:+.3f} {gi['ci']}  dJ {dJ_pt:+.4f} {dj['ci']}  "
              f"-> {v}", flush=True)
    res["contrast_at_usable_rungs"] = out
    usable = [r for r, s in screen.items() if r != "grey_published" and s.get("USABLE")]
    res["L2_VERDICT"] = ("NO-USABLE-RUNG" if not usable
                         else ("PAI-PENDING" if all(v.get("verdict") == "PAI-NOT-YET-RUN"
                                                    for v in out.values())
                               else "; ".join(f"{k}:{v['verdict']}" for k, v in out.items())))
    jdump(f"{OUT}/lad_l2.json", res)
    print(f"[lad_cscore] L2_VERDICT = {res['L2_VERDICT']}", flush=True)
    sentinel("SCORE_L2")


if __name__ == "__main__":
    main()
