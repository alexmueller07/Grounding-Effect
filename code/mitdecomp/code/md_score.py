"""Mitigation-method CHAIR decomposition -- scoring, decomposition and the pre-registered verdicts.

Pre-registered in CAMPAIGN/MITIGATION_DECOMP_PREREG.md (commit ce3ed30, amendments d2ff7b1).
Nothing in this file may be tuned after an endpoint exists.

WHAT IS MEASURED

  CHAIR (Rohrbach et al. 2018, this project's frozen port). Mentions counted per OCCURRENCE,
  not deduped per caption:
      CHAIR_i = hallucinated occurrences / all object occurrences
      CHAIR_s = captions with >= 1 hallucinated occurrence / captions
  Ground truth per image is the union of the COCO instance categories and the objects named in
  the five human captions -- Rohrbach's definition.

  THE SDT CENSUS. Over the 80 COCO categories, per image, each category is PRESENT (in the
  ground truth) or ABSENT, and is MENTIONED or not:
      H = Pr(mention | present)      F = Pr(mention | absent)
      J = H - F        d' = Z(H) - Z(F)        c = -0.5 * [Z(H) + Z(F)]
  Rates are log-linear corrected, (hits + 0.5)/(n + 1), applied UNCONDITIONALLY to every arm.
  Z and sdt are imported from sc1_interv/code/wia_dprime.py rather than rewritten.

  TWO DIFFERENT UNITS, deliberately. CHAIR counts occurrences (the literature's unit); the
  census counts per-image category presence (the natural unit for detection theory). No
  quantity is silently converted between them.

BOOTSTRAP. The resampling unit is the IMAGE. Every arm is gathered with the SAME index vector
so contrasts stay paired, and each selected image contributes its whole block of 80 category
cells, carrying the within-image clustering. Multiplicity is preserved: `rng.integers(0, n, n)`
with `assert len(sel) == n` on every replicate. There is deliberately NO set(), NO unique() and
NO dict keyed on image id anywhere on this path -- a set() around a resample once silently
turned it into a 63.2% subsample here and made nine intervals ~24% too narrow.

Writes {OUT}/md_result.json. Sentinel MD_SENTINEL_SCORE_OK.
"""
import os, sys, json, math, socket
from statistics import NormalDist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/alexmueller/paialpha/code")
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")

import numpy as np

from md_common import (OUT, SYN, MODEL, ARMS, SIGHTED, BLIND, METHODS, BLIND_OF, ALL_ARMS,
                       NBOOT, BOOT_SEED, MDE_CHAIR_I, MDE_CHAIR_S, DELTA_DPRIME, DELTA_C,
                       DELTA_H, WINDOWS, PC_A_S_CENTER, PC_A_S_HALF, PC_A_I_CENTER,
                       PC_A_I_HALF, PC_B_MIN, arm_tag)                              # noqa
from wia_dprime import Z, sdt                                                       # noqa
from j5_common import ChairScorer, read_jsonl                                       # noqa

PHI = NormalDist().cdf
CORRECTION_MATERIAL_THRESH = 0.05      # prereg Amendment 2
# Smoke-only: lets a pre-submission end-to-end check score the MD_LIMIT-tagged arms. The
# default 0 is the registered path; md_score.sbatch independently asserts all six arms have
# 500 rows with agreeing meta before the real run, so a smoke artefact cannot be scored as one.
LIMIT = int(os.environ.get("MD_LIMIT", "0"))


def _mult(sel, n, tag):
    assert len(sel) == n, f"FATAL_BOOTSTRAP_LENGTH {tag}: {len(sel)} != {n}"
    if n >= 20:
        assert len(set(sel)) < n, f"FATAL_BOOTSTRAP_NO_DUPLICATES {tag}"


def distinct_ngram_ratio(ids, n=8):
    if len(ids) < n:
        return 1.0
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(g)) / len(g)


def max_token_run(ids):
    best = cur = 0
    prev = None
    for t in ids:
        cur = cur + 1 if t == prev else 1
        prev = t
        best = max(best, cur)
    return best


# ---------------------------------------------------------------- per-image unit tables
def units(rows, chair, cats, tok, window=None):
    """Per-image arrays. Returns dict of length-n float arrays.

    CHAIR (occurrence unit):  hal, tot, any_hal
    SDT census (category unit per image): nt, nf, ct, cf
    """
    hal, tot, anyh, nt, nf, ct, cf, nlen = ([] for _ in range(8))
    ncat = len(cats)
    for r in rows:
        text = r["text"] if window is None else tok.decode(r["gen_ids"][:window],
                                                           skip_special_tokens=True)
        gt = set(r["gt"])
        ment = chair.mentions(text)                       # list of (node, char) OCCURRENCES
        bad = sum(1 for (nd, _c) in ment if nd not in gt)
        hal.append(bad)
        tot.append(len(ment))
        anyh.append(1.0 if bad > 0 else 0.0)
        mset = {nd for (nd, _c) in ment}                  # DISTINCT categories mentioned
        npres = len(gt)
        nt.append(npres)
        nf.append(ncat - npres)
        ct.append(len(gt & mset))
        cf.append(len(mset - gt))
        nlen.append(len(r["gen_ids"]) if window is None else min(window, len(r["gen_ids"])))
    f = lambda x: np.array(x, float)
    return {"hal": f(hal), "tot": f(tot), "any": f(anyh), "nt": f(nt), "nf": f(nf),
            "ct": f(ct), "cf": f(cf), "nlen": f(nlen)}


def chair_s(u, sel):
    return 100.0 * float(u["any"][sel].mean())


def chair_i(u, sel):
    t = float(u["tot"][sel].sum())
    return float("nan") if t == 0 else 100.0 * float(u["hal"][sel].sum()) / t


def sdt_of(u, sel):
    """(H, F, J, d', c, logOR, d'_uncorr, c_uncorr) pooled over the resampled images."""
    nt, nf = float(u["nt"][sel].sum()), float(u["nf"][sel].sum())
    ct, cf = float(u["ct"][sel].sum()), float(u["cf"][sel].sum())
    if nt <= 0 or nf <= 0:
        return None
    rt, rf, h, f, d, c, du, cu = sdt(nt, nf, ct, cf)
    lo = math.log(h / (1 - h)) - math.log(f / (1 - f))
    return {"H": h, "F": f, "J": h - f, "dprime": d, "c": c, "logOR": lo,
            "H_raw": rt, "F_raw": rf, "J_raw": rt - rf,
            "dprime_uncorr": du, "c_uncorr": cu}


def J_of(d, c):
    """J as a function of (d', c) under the equal-variance normal model: prereg eq. J."""
    return PHI(d / 2.0 - c) - PHI(-d / 2.0 - c)


def boot(fn, n, tag, nboot=NBOOT, seed=BOOT_SEED):
    """fn(sel) -> scalar or None. ONE index vector per replicate, shared by every arm."""
    rng = np.random.default_rng(seed)
    v, drop = [], 0
    for b in range(nboot):
        sel = rng.integers(0, n, n)
        # Length is asserted on EVERY replicate, as the pre-registration requires -- it is
        # O(1) on an ndarray, so there is no reason to sample it. The duplicate check builds a
        # set and is therefore run on the first 20 only; that set is a READ of sel and never
        # rebinds it, so it cannot turn the resample into a subsample (which is exactly the
        # bug that once made nine intervals here ~24% too narrow).
        assert len(sel) == n, f"FATAL_BOOTSTRAP_LENGTH {tag}#{b}: {len(sel)} != {n}"
        if b < 20:
            _mult(list(sel), n, f"{tag}#{b}")
        x = fn(sel)
        if x is None or (isinstance(x, float) and math.isnan(x)):
            drop += 1
        else:
            v.append(float(x))
    if not v:
        return {"ci95": [None, None], "n_boot": 0, "n_dropped": drop}
    return {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "n_boot": len(v), "n_dropped": drop}


def pt_ci(fn, allsel, n, tag):
    b = boot(fn, n, tag)
    p = fn(allsel)
    return {"point": (None if p is None or (isinstance(p, float) and math.isnan(p)) else float(p)),
            "ci95": b["ci95"], "n_boot": b["n_boot"], "n_dropped": b["n_dropped"],
            "ci_width": (None if b["ci95"][0] is None else b["ci95"][1] - b["ci95"][0])}


# ---------------------------------------------------------------- verdict helpers
def _ci_above(ci, t):
    return ci[0] is not None and ci[0] > t


def _ci_below(ci, t):
    return ci[1] is not None and ci[1] < t


def _ci_within(ci, t):
    return ci[0] is not None and ci[0] >= -t and ci[1] <= t


def _ci_excludes_zero(ci):
    return ci[0] is not None and (ci[0] > 0 or ci[1] < 0)


def primary1(gain_i, dd, dc):
    """prereg 5.1. Gate on there being a gain at all, then decompose."""
    if not _ci_above(gain_i["ci95"], MDE_CHAIR_I):
        return "NO_GAIN_TO_DECOMPOSE"
    no_sep_gain = _ci_within(dd["ci95"], DELTA_DPRIME) or _ci_below(dd["ci95"], 0.0)
    real_crit = _ci_excludes_zero(dc["ci95"]) and abs(dc["point"]) > DELTA_C
    if _ci_above(dd["ci95"], DELTA_DPRIME):
        return "REFUTES"
    if no_sep_gain and real_crit:
        return "SUPPORTS"
    return "CANNOT_RESOLVE"


def primary2(gain_blind, thin):
    """prereg 5.2."""
    if thin:
        return "BLIND_DENOMINATOR_TOO_THIN"
    if _ci_above(gain_blind["ci95"], MDE_CHAIR_I):
        return "SURVIVES_BLINDING"
    if _ci_within(gain_blind["ci95"], MDE_CHAIR_I):
        return "DIES_UNDER_BLINDING"
    return "CANNOT_RESOLVE"


def primary3(dh, extrap):
    """prereg 5.3."""
    if extrap:
        return "EXTRAPOLATION"
    if _ci_within(dh["ci95"], DELTA_H):
        return "ON_CURVE"
    if _ci_above(dh["ci95"], DELTA_H):
        return "ABOVE_CURVE"
    if _ci_below(dh["ci95"], -DELTA_H):
        return "BELOW_CURVE"
    return "CANNOT_RESOLVE"


def main():
    from transformers import AutoTokenizer
    print(f"[md_score] host={socket.gethostname()}", flush=True)
    chair = ChairScorer(SYN)
    cats = sorted({v for v in chair.inverse.values()})
    assert len(cats) == 80, f"FATAL_NOT_80_CATEGORIES {len(cats)}"
    tok = AutoTokenizer.from_pretrained(MODEL)

    arms, metas = {}, {}
    for a in ALL_ARMS:
        rows = read_jsonl(f"{OUT}/{arm_tag(a, LIMIT)}.jsonl")
        rows.sort(key=lambda r: r["image_id"])
        arms[a] = rows
        metas[a] = json.load(open(f"{OUT}/{arm_tag(a, LIMIT)}_meta.json"))
    ids0 = [r["image_id"] for r in arms[ALL_ARMS[0]]]
    for a in ALL_ARMS:
        assert [r["image_id"] for r in arms[a]] == ids0, f"FATAL_IMAGE_SET_MISMATCH {a}"
    n = len(ids0)
    assert n > 0, "FATAL_NO_ROWS"
    print(f"[md_score] {n} images x {len(ALL_ARMS)} arms x {len(cats)} categories", flush=True)

    U = {a: units(arms[a], chair, cats, tok) for a in ALL_ARMS}
    UW = {w: units(arms["vanilla"], chair, cats, tok, window=w) for w in WINDOWS}
    allsel = np.arange(n)

    gates_p = f"{OUT}/md_gates.json"
    out = {"n_images": n, "n_categories": len(cats), "model": MODEL,
           "prereg": "CAMPAIGN/MITIGATION_DECOMP_PREREG.md @ ce3ed30 (+d2ff7b1)",
           "nboot": NBOOT, "boot_seed": BOOT_SEED, "host": socket.gethostname(),
           "job_id": os.environ.get("SLURM_JOB_ID", ""),
           "vcd": "DROPPED -- prereg Amendment 1: no published CHAIR eval, sampling-only, "
                  "paper/repo disagree on beta/T/noise model, release was a no-op past "
                  "token 1 until c637c85a",
           "gates": json.load(open(gates_p)) if os.path.exists(gates_p) else None,
           "arms": {}, "contrasts": {}, "primaries": {}, "curve": {}}

    # Construct validity is printed BEFORE any level or contrast, deliberately: if alpha did
    # not move the generations, nothing below it means anything.
    g = out["gates"] or {}
    g3 = g.get("G3") or {}
    print(f"  [G3 alpha liveness] status={g3.get('status')} "
          f"frac_differ(a0 vs a05)={g3.get('frac_differ_a0_vs_a05')} "
          f"determinism={g3.get('determinism_frac_differ_a0_vs_a0rerun')}", flush=True)

    # ---------------- per-arm levels, SDT, and the mandatory confound columns
    for a in ALL_ARMS:
        rows, u = arms[a], U[a]
        s = sdt_of(u, allsel)
        mass = np.array([r["img_mass"] for r in rows], float)
        corr_eff = (None if s["dprime_uncorr"] is None
                    else abs(s["dprime"] - s["dprime_uncorr"]))
        blk = {
            "alpha": ARMS[a]["alpha"], "gamma": ARMS[a]["gamma"], "blind": ARMS[a]["blind"],
            "CHAIR_s": chair_s(u, allsel), "CHAIR_i": chair_i(u, allsel),
            "CHAIR_s_ci95": boot(lambda sl, u=u: chair_s(u, sl), n, f"cs_{a}")["ci95"],
            "CHAIR_i_ci95": boot(lambda sl, u=u: chair_i(u, sl), n, f"ci_{a}")["ci95"],
            "H": s["H"], "F": s["F"], "J": s["J"], "dprime": s["dprime"], "c": s["c"],
            "logOR": s["logOR"], "H_raw": s["H_raw"], "F_raw": s["F_raw"],
            "dprime_uncorr": s["dprime_uncorr"], "c_uncorr": s["c_uncorr"],
            "dprime_correction_effect": corr_eff,
            "CORRECTION_MATERIAL": bool(corr_eff is not None
                                        and corr_eff > CORRECTION_MATERIAL_THRESH),
            "J_ci95": boot(lambda sl, u=u: sdt_of(u, sl)["J"], n, f"J_{a}")["ci95"],
            "dprime_ci95": boot(lambda sl, u=u: sdt_of(u, sl)["dprime"], n, f"d_{a}")["ci95"],
            "c_ci95": boot(lambda sl, u=u: sdt_of(u, sl)["c"], n, f"c_{a}")["ci95"],
            # pre-registered confound guards, reported regardless of outcome
            "truncation_rate": float(np.mean([r["truncated"] for r in rows])),
            "eos_rate": float(np.mean([r["hit_eos"] for r in rows])),
            "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])),
            "median_tokens": float(np.median([r["n_tokens"] for r in rows])),
            "mean_obj_mentions": float(u["tot"].mean()),
            "mean_distinct_cats_mentioned": float((u["ct"] + u["cf"]).mean()),
            "mean_distinct8": float(np.mean([distinct_ngram_ratio(r["gen_ids"]) for r in rows])),
            "max_token_run_p95": float(np.percentile(
                [max_token_run(r["gen_ids"]) for r in rows], 95)),
            "img_attn_mass_decode": float(mass.mean()),
            "n_rows": len(rows), "meta_job_id": metas[a].get("job_id", ""),
        }
        out["arms"][a] = blk
        print(f"  {a:14s} CHAIR_s={blk['CHAIR_s']:6.2f} CHAIR_i={blk['CHAIR_i']:5.2f} "
              f"H={blk['H']:.4f} F={blk['F']:.4f} J={blk['J']:+.4f} "
              f"d'={blk['dprime']:+.4f} c={blk['c']:+.4f} len={blk['mean_tokens']:6.1f} "
              f"obj={blk['mean_obj_mentions']:5.2f}", flush=True)

    # ---------------- arm-liveness diagnostics (reported, never used to rescue a verdict)
    # G3 established that alpha moves the generations on REAL images. It did not test grey
    # ones, and it should not be assumed: PAI amplifies attention to the image token columns,
    # and a flat grey image still occupies all 576 of them. If pai05_blind turned out to be
    # caption-identical to vanilla_blind, then any blind "gain" would be a no-op arm rather
    # than a measurement, and PRIMARY-2 would be uninterpretable in the same way a dead hook
    # would make PRIMARY-1 uninterpretable. Reported either way.
    out["arm_liveness"] = {}
    for a, b in (("vanilla", "pai02"), ("vanilla", "pai05"), ("pai02", "pai05"),
                 ("vanilla_blind", "pai02_blind"), ("vanilla_blind", "pai05_blind"),
                 ("pai02_blind", "pai05_blind"), ("vanilla", "vanilla_blind"),
                 ("pai05", "pai05_blind")):
        out["arm_liveness"][f"{a}_vs_{b}"] = float(np.mean(
            [x["text"] != y["text"] for x, y in zip(arms[a], arms[b])]))
    out["arm_liveness"]["blind_alpha_is_live"] = bool(
        out["arm_liveness"]["vanilla_blind_vs_pai05_blind"] >= 0.90)
    print("  arm liveness (fraction of captions differing):", flush=True)
    for k, v in out["arm_liveness"].items():
        if isinstance(v, float):
            print(f"    {k:34s} {v:.4f}", flush=True)

    # ---------------- positive control (blocking)
    va = out["arms"]["vanilla"]
    pcb = pt_ci(lambda sl: chair_s(U["vanilla"], sl) - chair_s(U["pai05"], sl),
                allsel, n, "PC_B")
    out["positive_control"] = {
        "PC_A_CHAIR_s": {"value": va["CHAIR_s"],
                         "band": [PC_A_S_CENTER - PC_A_S_HALF, PC_A_S_CENTER + PC_A_S_HALF],
                         "pass": bool(abs(va["CHAIR_s"] - PC_A_S_CENTER) <= PC_A_S_HALF)},
        "PC_A_CHAIR_i": {"value": va["CHAIR_i"],
                         "band": [PC_A_I_CENTER - PC_A_I_HALF, PC_A_I_CENTER + PC_A_I_HALF],
                         "pass": bool(abs(va["CHAIR_i"] - PC_A_I_CENTER) <= PC_A_I_HALF)},
        "PC_A_sources": "PAI 2407.21771v1 T4 46.2/13.8; T1 46.6/13.4; VISTA 2502.03628 46.4",
        "PC_A_licence": "passing licenses ONLY 'the harness measures something that behaves "
                        "like CHAIR at the published level'. Our own appendix.tex:122 states "
                        "absolute rates from this port are not comparable to published CHAIR.",
        "PC_B_pai05_effect": {**pcb, "min_required": PC_B_MIN,
                              "pass": bool(pcb["point"] is not None
                                           and pcb["point"] >= PC_B_MIN
                                           and _ci_above(pcb["ci95"], 0.0)),
                              "source": "PAI T4 46.2-24.6=21.6; T1 46.6-24.8=21.8"},
    }
    out["positive_control"]["pass"] = bool(
        out["positive_control"]["PC_A_CHAIR_s"]["pass"]
        and out["positive_control"]["PC_A_CHAIR_i"]["pass"]
        and out["positive_control"]["PC_B_pai05_effect"]["pass"])
    print(f"  PC-A CHAIR_s={va['CHAIR_s']:.2f} in {out['positive_control']['PC_A_CHAIR_s']['band']} "
          f"-> {out['positive_control']['PC_A_CHAIR_s']['pass']}", flush=True)
    print(f"  PC-A CHAIR_i={va['CHAIR_i']:.2f} in {out['positive_control']['PC_A_CHAIR_i']['band']} "
          f"-> {out['positive_control']['PC_A_CHAIR_i']['pass']}", flush=True)
    print(f"  PC-B vanilla-pai05 CHAIR_s={pcb['point']:+.2f} {pcb['ci95']} "
          f"-> {out['positive_control']['PC_B_pai05_effect']['pass']}", flush=True)

    # ---------------- vanilla's own truncation curve (PRIMARY-3 instrument)
    Fv = np.array([sdt_of(UW[w], allsel)["F"] for w in WINDOWS])
    Hv = np.array([sdt_of(UW[w], allsel)["H"] for w in WINDOWS])
    out["curve"] = {
        "windows": list(WINDOWS),
        "F_vanilla": [float(x) for x in Fv], "H_vanilla": [float(x) for x in Hv],
        "CHAIR_i_vanilla": [chair_i(UW[w], allsel) for w in WINDOWS],
        "CHAIR_s_vanilla": [chair_s(UW[w], allsel) for w in WINDOWS],
        "mean_obj_mentions": [float(UW[w]["tot"].mean()) for w in WINDOWS],
        "F_monotone": bool(np.all(np.diff(Fv) >= 0)),
        "note": "truncating vanilla lowers mention propensity without changing the model, the "
                "image or the decoding -- a pure criterion move. Asymmetry stated in prereg "
                "5.3: lying ON this curve is strong evidence of a criterion move; lying ABOVE "
                "it is weaker evidence of a discrimination gain, since the method could sit "
                "on a better but still purely criterion-driven trajectory.",
    }
    print(f"  curve F_vanilla monotone={out['curve']['F_monotone']}  "
          f"F range [{Fv.min():.4f}, {Fv.max():.4f}]", flush=True)

    def curve_stat(sel, um, key):
        """H (or CHAIR) of truncated vanilla at the W* where its F equals the method's F."""
        fm = sdt_of(um, sel)["F"]
        fv = np.array([sdt_of(UW[w], sel)["F"] for w in WINDOWS])
        if fm < fv.min() or fm > fv.max():
            return None
        order = np.argsort(fv)
        if key == "H":
            yv = np.array([sdt_of(UW[w], sel)["H"] for w in WINDOWS])
        elif key == "CHAIR_i":
            yv = np.array([chair_i(UW[w], sel) for w in WINDOWS])
        else:
            yv = np.array([chair_s(UW[w], sel) for w in WINDOWS])
        return float(np.interp(fm, fv[order], yv[order]))

    # ---------------- contrasts and the three primaries, per method
    uv, uvb = U["vanilla"], U["vanilla_blind"]
    for m in METHODS:
        um, umb = U[m], U[BLIND_OF[m]]
        C = {}
        C["GAIN_CHAIR_i"] = pt_ci(lambda sl: chair_i(uv, sl) - chair_i(um, sl),
                                  allsel, n, f"gi_{m}")
        C["GAIN_CHAIR_s"] = pt_ci(lambda sl: chair_s(uv, sl) - chair_s(um, sl),
                                  allsel, n, f"gs_{m}")
        C["delta_J"] = pt_ci(lambda sl: sdt_of(um, sl)["J"] - sdt_of(uv, sl)["J"],
                             allsel, n, f"dJ_{m}")
        C["delta_dprime"] = pt_ci(lambda sl: sdt_of(um, sl)["dprime"] - sdt_of(uv, sl)["dprime"],
                                  allsel, n, f"dd_{m}")
        C["delta_c"] = pt_ci(lambda sl: sdt_of(um, sl)["c"] - sdt_of(uv, sl)["c"],
                             allsel, n, f"dc_{m}")
        C["delta_logOR"] = pt_ci(lambda sl: sdt_of(um, sl)["logOR"] - sdt_of(uv, sl)["logOR"],
                                 allsel, n, f"dl_{m}")
        C["delta_H"] = pt_ci(lambda sl: sdt_of(um, sl)["H"] - sdt_of(uv, sl)["H"],
                             allsel, n, f"dH_{m}")
        C["delta_F"] = pt_ci(lambda sl: sdt_of(um, sl)["F"] - sdt_of(uv, sl)["F"],
                             allsel, n, f"dF_{m}")

        # counterfactual split of delta J: move only c, or only d'
        def _cf(sl, which):
            sv, sm = sdt_of(uv, sl), sdt_of(um, sl)
            base = J_of(sv["dprime"], sv["c"])
            if which == "c":
                return J_of(sv["dprime"], sm["c"]) - base
            return J_of(sm["dprime"], sv["c"]) - base
        C["delta_J_criterion_only"] = pt_ci(lambda sl: _cf(sl, "c"), allsel, n, f"cfc_{m}")
        C["delta_J_separation_only"] = pt_ci(lambda sl: _cf(sl, "d"), allsel, n, f"cfd_{m}")

        # blinding
        C["GAIN_CHAIR_i_blind"] = pt_ci(lambda sl: chair_i(uvb, sl) - chair_i(umb, sl),
                                        allsel, n, f"gib_{m}")
        C["GAIN_CHAIR_s_blind"] = pt_ci(lambda sl: chair_s(uvb, sl) - chair_s(umb, sl),
                                        allsel, n, f"gsb_{m}")
        C["delta_sight_CHAIR_i"] = pt_ci(
            lambda sl: (chair_i(uv, sl) - chair_i(um, sl))
                       - (chair_i(uvb, sl) - chair_i(umb, sl)), allsel, n, f"ds_{m}")
        C["delta_dprime_blind"] = pt_ci(
            lambda sl: sdt_of(umb, sl)["dprime"] - sdt_of(uvb, sl)["dprime"],
            allsel, n, f"ddb_{m}")

        # Retention ratio, promised in prereg 5.2. It handles the scale problem that the
        # absolute MDE does not: a blind arm's CHAIR_i sits on a completely different
        # baseline from a sighted one, so "1.5 points" is a much smaller relative effect
        # there. CAVEAT, recorded with the number: this is a ratio whose denominator is
        # itself estimated, so its bootstrap distribution is heavy-tailed and the percentile
        # interval widens sharply as the sighted gain approaches zero. Replicates with a
        # non-positive denominator are dropped and COUNTED (n_dropped); if many are dropped
        # the ratio is not interpretable and the difference `delta_sight_CHAIR_i` is the
        # quantity to read instead.
        def _rr(sl):
            gs = chair_i(uv, sl) - chair_i(um, sl)
            gb = chair_i(uvb, sl) - chair_i(umb, sl)
            if math.isnan(gs) or math.isnan(gb) or gs <= 0:
                return None
            return gb / gs
        C["retention_ratio_CHAIR_i"] = pt_ci(_rr, allsel, n, f"rr_{m}")

        # truncation curve
        C["H_vanilla_at_matched_F"] = pt_ci(lambda sl: curve_stat(sl, um, "H"),
                                            allsel, n, f"hm_{m}")
        C["delta_H_at_matched_F"] = pt_ci(
            lambda sl: (None if curve_stat(sl, um, "H") is None
                        else sdt_of(um, sl)["H"] - curve_stat(sl, um, "H")),
            allsel, n, f"dhm_{m}")
        C["CHAIR_i_vanilla_at_matched_F"] = pt_ci(lambda sl: curve_stat(sl, um, "CHAIR_i"),
                                                  allsel, n, f"cim_{m}")
        C["delta_CHAIR_i_vs_matched_truncation"] = pt_ci(
            lambda sl: (None if curve_stat(sl, um, "CHAIR_i") is None
                        else chair_i(um, sl) - curve_stat(sl, um, "CHAIR_i")),
            allsel, n, f"dcim_{m}")
        extrap = C["delta_H_at_matched_F"]["point"] is None

        thin = out["arms"][BLIND_OF[m]]["mean_obj_mentions"] < 1.0 \
            or out["arms"]["vanilla_blind"]["mean_obj_mentions"] < 1.0
        p1 = primary1(C["GAIN_CHAIR_i"], C["delta_dprime"], C["delta_c"])
        # link-function companion: probit (d') vs logit (logOR) must agree on the key row
        d_no_gain = _ci_within(C["delta_dprime"]["ci95"], DELTA_DPRIME) \
            or _ci_below(C["delta_dprime"]["ci95"], 0.0)
        l_no_gain = _ci_within(C["delta_logOR"]["ci95"], DELTA_DPRIME) \
            or _ci_below(C["delta_logOR"]["ci95"], 0.0)
        if p1 in ("SUPPORTS", "REFUTES") and (d_no_gain != l_no_gain):
            p1 = "LINK_SENSITIVE"
        p2 = primary2(C["GAIN_CHAIR_i_blind"], thin)
        p3 = primary3(C["delta_H_at_matched_F"], extrap)

        out["contrasts"][m] = C
        out["primaries"][m] = {
            "PRIMARY_1_decomposition": p1,
            "PRIMARY_2_blinding": p2,
            "PRIMARY_3_truncation_curve": p3,
            "link_probit_says_no_sep_gain": bool(d_no_gain),
            "link_logit_says_no_sep_gain": bool(l_no_gain),
            "informativeness_confounded": bool(
                out["arms"][m]["mean_obj_mentions"]
                < 0.5 * out["arms"]["vanilla"]["mean_obj_mentions"]),
            "blind_denominator_thin": bool(thin),
        }
        print(f"\n  == {m} ==", flush=True)
        for k in ("GAIN_CHAIR_i", "GAIN_CHAIR_s", "delta_J", "delta_dprime", "delta_c",
                  "delta_logOR", "delta_J_criterion_only", "delta_J_separation_only",
                  "GAIN_CHAIR_i_blind", "delta_sight_CHAIR_i",
                  "retention_ratio_CHAIR_i", "delta_H_at_matched_F",
                  "delta_CHAIR_i_vs_matched_truncation"):
            v = C[k]
            if v["point"] is None:
                print(f"    {k:38s}  n/a", flush=True)
            else:
                print(f"    {k:38s} {v['point']:+8.4f} "
                      f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]", flush=True)
        print(f"    PRIMARY-1 {p1} | PRIMARY-2 {p2} | PRIMARY-3 {p3}", flush=True)

    # ---------------- overall verdict per prereg 5.4
    def combine(pr):
        p1, p2, p3 = (pr["PRIMARY_1_decomposition"], pr["PRIMARY_2_blinding"],
                      pr["PRIMARY_3_truncation_curve"])
        if p1 == "NO_GAIN_TO_DECOMPOSE":
            return "NO_GAIN_TO_DECOMPOSE"
        supports = (p1 == "SUPPORTS")
        agrees = p2 == "SURVIVES_BLINDING" or p3 == "ON_CURVE"
        refutes_any = (p1 == "REFUTES") or (p3 == "ABOVE_CURVE")
        if supports and agrees and not refutes_any:
            return "SUPPORTS_PAPER_ASSERTION"
        if refutes_any and not supports:
            return "REFUTES_PAPER_ASSERTION"
        return "CANNOT_RESOLVE"

    for m in METHODS:
        out["primaries"][m]["COMBINED"] = combine(out["primaries"][m])

    gates_ok = bool((out["gates"] or {}).get("all_blocking_pass", False))
    if (g3.get("status") == "DEAD_HOOK_EXPERIMENT_VOID"):
        final = "EXPERIMENT_VOID_DEAD_HOOK"
    elif not gates_ok:
        final = "GATES_FAILED_NO_VERDICT"
    elif not out["positive_control"]["pass"]:
        final = "POSITIVE_CONTROL_FAILED_STOP"
    else:
        final = out["primaries"]["pai05"]["COMBINED"]
    out["FINAL_VERDICT"] = final
    out["FINAL_VERDICT_basis"] = {
        "primary_method": "pai05",
        "reason": "pai05 is PAI at the alpha its own paper prescribes for LLaVA and is the "
                  "only arm with a published CHAIR effect to decompose; pai02 is reported "
                  "alongside because it is what the released code and README actually run.",
        "per_method": {m: out["primaries"][m] for m in METHODS},
        "gates_all_blocking_pass": gates_ok,
        "positive_control_pass": out["positive_control"]["pass"]}

    json.dump(out, open(f"{OUT}/md_result{'_lim%d' % LIMIT if LIMIT else ''}.json", "w"), indent=1)
    print(f"\n  POSITIVE CONTROL: {'PASS' if out['positive_control']['pass'] else 'FAIL'}",
          flush=True)
    print(f"  FINAL VERDICT: {final}", flush=True)
    print("MD_SENTINEL_SCORE_OK", flush=True)


if __name__ == "__main__":
    main()
