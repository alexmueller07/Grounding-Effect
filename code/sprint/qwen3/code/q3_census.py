"""Lane `qwen3` census. STIMULUS-SIDE ONLY: no model is loaded, nothing is generated, no
continuation is read, no `gap` is computed.

Executes the SAME selection rule that chose the cut for Qwen2.5-VL (WIA_FAM3_PREREG.md section 1)
and for Kosmos-2 (WIA_FAM4_PREREG.md section 1), imported unchanged rather than copying either
model's realised number:

  per-cell budget b_i = floor(f . L_i), L_i this model's OWN free-running length;
  the cut is the LARGEST f on a 0.005 grid over [0.050, 0.500] at which all four hold
    (a) strictly fewer than 5% of cap1 prefixes have to double (cycles > 0)
    (b) no cap5 cell suffers inventory shortfall
    (c) the minimum per-cell budget is >= 8 tokens
    (d) realised prefix-object unit counts >= 100 true and >= 100 false PER ARM;
  screen every grid point with two cheap per-cell lengths, then verify the candidate AND the
  next larger grid point with the REAL builders on all cells; the screen decides nothing.
  If no f satisfies all four, the lane stops at NO-DATA and reports which condition binds.

Also runs GATE_SCENE and freezes an md5 per cap1/cap5 prefix so the generator can prove it
built the same stimulus. Writes out/q3_census.json and out/q3_cells.json.
"""
import os, sys, json, time, hashlib
sys.path.insert(0, "/data/alexmueller/sprint/qwen3/code")
import numpy as np
from q3_common import (M5, M5_SHA, OUT, load_coco_sample, ChairScorer, SYN, MIN_LEN, cut_of,
                       jload, sentinel)
from j5_common import build_prefix_ids
from wia_redun_common import (sources, permuted, assemble_nodouble, asm_units, asm_caps_ordered,
                              N_SRC, S_CAPPICK, S_JOIN_CAP, s_word, s_obj)
from transformers import AutoTokenizer

GRID = [round(0.050 + 0.005 * i, 3) for i in range(int((0.500 - 0.050) / 0.005) + 1)]
ASM_SEED_BASE = 1950
DOUBLE_GATE = 0.05
MIN_BUDGET = 8
UNIT_FLOOR = 100
KDER = 3
NBOOT = 4000
SEED = 20260919


def boot_diff(a, b, nboot=NBOOT, seed=SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    rng = np.random.default_rng(seed)
    v = []
    for i in range(nboot):
        sel = rng.integers(0, n, n)
        assert len(sel) == n, "FATAL_BOOTSTRAP_LENGTH"
        if i < 20:
            assert len(set(sel.tolist())) < n, "FATAL_BOOTSTRAP_NO_DUPLICATES"
        v.append(a[sel].mean() - b[sel].mean())
    return float(a.mean() - b.mean()), [float(np.percentile(v, 2.5)),
                                        float(np.percentile(v, 97.5))]


def main():
    t0 = time.time()
    chair = ChairScorer(SYN)
    recs = load_coco_sample(chair, n=500)
    by_id = {r["image_id"]: r for r in recs}
    objset = lambda t: set(n for (n, _c) in chair.mentions(t))
    gt = {r["image_id"]: set(r["gt"]) for r in recs}
    tok = AutoTokenizer.from_pretrained(M5, revision=M5_SHA)
    assert tok.name_or_path == M5, f"FATAL_TOK_NAME {tok.name_or_path}"
    ws = tok.convert_ids_to_tokens(tok(" cat", add_special_tokens=False)["input_ids"])[0]
    assert ws.startswith("Ġ"), f"FATAL_WORD_START_MARKER {ws!r}"

    af = {r["image_id"]: r for r in jload(f"{OUT}/q3_af.jsonl")}
    assert set(af) == set(by_id), "FATAL_IMAGE_SET_DRIFT"
    excluded = sorted(i for i in af if af[i]["gen_len"] < MIN_LEN)
    keep_ids = sorted(i for i in af if af[i]["gen_len"] >= MIN_LEN)
    print(f"[census] images {len(af)}  excluded L<{MIN_LEN}: {len(excluded)} {excluded[:20]}",
          flush=True)
    sids = sorted(by_id)
    pos = {i: j for j, i in enumerate(sids)}

    cells = []
    for iid in keep_ids:
        L = af[iid]["gen_len"]
        for k in range(1, KDER + 1):
            srcs = sources(sids, pos, iid, k)
            caps1 = asm_caps_ordered(by_id[srcs[0]]["human_caps"], iid * 7 + ASM_SEED_BASE + k)
            nat = len(tok(". ".join(caps1) + ".", add_special_tokens=False)["input_ids"])
            per_src = [permuted(asm_units(by_id[s]["human_caps"]),
                                iid * 7 + S_CAPPICK + k + 10 * j) for j, s in enumerate(srcs)]
            allu = [u for lst in per_src for u in lst]
            inv = len(tok(". ".join(allu) + ".", add_special_tokens=False)["input_ids"])
            cells.append({"image_id": iid, "k": k, "L": L, "srcs": srcs,
                          "nat_len": nat, "inv_len": inv})
    print(f"[census] {len(cells)} cells; screen lengths done {time.time()-t0:.0f}s", flush=True)

    natv = np.array([c["nat_len"] for c in cells])
    invv = np.array([c["inv_len"] for c in cells])
    Lv = np.array([c["L"] for c in cells])
    grid_tab = []
    for f in GRID:
        b = np.floor(f * Lv).astype(int)
        grid_tab.append({"f": f, "mean_budget": float(b.mean()), "min": int(b.min()),
                         "max": int(b.max()),
                         "cap1_doubling_screen": float((natv < b).mean()),
                         "cap5_shortfall_screen": int((invv < b).sum())})
    ok_screen = [g for g in grid_tab
                 if g["cap1_doubling_screen"] < DOUBLE_GATE and g["cap5_shortfall_screen"] == 0
                 and g["min"] >= MIN_BUDGET]
    assert ok_screen, "FATAL_NO_FEASIBLE_CUT_ON_SCREEN"
    cand = max(g["f"] for g in ok_screen)
    nxt = round(cand + 0.005, 3)
    print(f"[census] screen candidate f={cand}  next={nxt}", flush=True)

    def exact(f):
        n_dbl = n_short = 0
        bmin = 10 ** 9
        rows = []
        for c in cells:
            b = cut_of(c["L"], f)
            bmin = min(bmin, b)
            iid, k, srcs = c["image_id"], c["k"], c["srcs"]
            a_ids, a_txt, a_cyc = build_prefix_ids(tok, by_id[srcs[0]]["human_caps"], b,
                                                   seed=iid * 7 + ASM_SEED_BASE + k)
            per_src = [permuted(asm_units(by_id[s]["human_caps"]),
                                iid * 7 + S_CAPPICK + k + 10 * j) for j, s in enumerate(srcs)]
            mc = assemble_nodouble(tok, per_src, b, iid * 7 + S_JOIN_CAP + k)
            n_dbl += (a_cyc > 0); n_short += bool(mc["shortfall"])
            rows.append((c, b, a_ids, a_txt, a_cyc, mc, per_src))
        return {"f": f, "n_cells": len(cells), "cap1_doubling": n_dbl / len(cells),
                "n_doubled": n_dbl, "cap5_shortfall": n_short, "min_budget": bmin}, rows

    ex_c, rows_c = exact(cand)
    print(f"[census] EXACT at {cand}: {json.dumps(ex_c)}", flush=True)
    ok_c = (ex_c["cap1_doubling"] < DOUBLE_GATE and ex_c["cap5_shortfall"] == 0
            and ex_c["min_budget"] >= MIN_BUDGET)
    ex_n = None
    if nxt <= GRID[-1]:
        ex_n, _ = exact(nxt)
        print(f"[census] EXACT at {nxt} (must FAIL): {json.dumps(ex_n)}", flush=True)
        ok_n = (ex_n["cap1_doubling"] < DOUBLE_GATE and ex_n["cap5_shortfall"] == 0
                and ex_n["min_budget"] >= MIN_BUDGET)
    else:
        ok_n = False
        print("[census] candidate is the grid maximum; no larger point exists to falsify",
              flush=True)
    walked = []
    while not ok_c:
        walked.append({"f": cand, "exact": ex_c})
        cand = round(cand - 0.005, 3)
        assert cand >= GRID[0], "FATAL_NO_FEASIBLE_CUT_EXACT"
        ex_c, rows_c = exact(cand)
        ok_c = (ex_c["cap1_doubling"] < DOUBLE_GATE and ex_c["cap5_shortfall"] == 0
                and ex_c["min_budget"] >= MIN_BUDGET)
        print(f"[census] walked down to {cand}: {json.dumps(ex_c)}", flush=True)
        ok_n = True
    assert not ok_n, f"FATAL_NOT_LARGEST {nxt} also passes exactly"
    CUT = cand
    print(f"[census] CUT = {CUT}", flush=True)

    cell_out = []
    nt1 = nf1 = nt5 = nf5 = 0
    sw1, sw5, so1, so5 = {}, {}, {}, {}
    P1, P5, M1_, M5_ = [], [], [], []
    for (c, b, a_ids, a_txt, a_cyc, mc, per_src) in rows_c:
        iid, k, srcs = c["image_id"], c["k"], c["srcs"]
        g = gt[iid]
        p1, p5 = objset(a_txt), objset(mc["text"])
        nt1 += len(p1 & g); nf1 += len(p1 - g)
        nt5 += len(p5 & g); nf5 += len(p5 - g)
        P1.append(len(p1)); P5.append(len(p5))
        M1_.append(len(chair.mentions(a_txt))); M5_.append(len(chair.mentions(mc["text"])))
        caps1 = asm_caps_ordered(by_id[srcs[0]]["human_caps"], iid * 7 + ASM_SEED_BASE + k)
        u1, u5 = caps1, mc["caps_ordered"]
        for d, v in ((sw1, s_word(u1)), (sw5, s_word(u5)),
                     (so1, s_obj(u1, objset)), (so5, s_obj(u5, objset))):
            if v is not None:
                d.setdefault(iid, []).append(v)
        cell_out.append({"image_id": iid, "k": k, "L": c["L"], "budget": b, "srcs": srcs,
                         "cap1_md5": hashlib.md5(a_txt.encode()).hexdigest(),
                         "cap5_md5": hashlib.md5(mc["text"].encode()).hexdigest(),
                         "cap1_cycles": a_cyc, "cap5_topups": mc["topups"],
                         "cap1_n_units": len(asm_units(by_id[srcs[0]]["human_caps"])),
                         "cap5_n_units": mc["n_units"]})

    imgs = sorted(set(sw1) & set(sw5))
    a = [float(np.mean(sw1[i])) for i in imgs]
    b_ = [float(np.mean(sw5[i])) for i in imgs]
    dsw, ciw = boot_diff(a, b_)
    imgs_o = sorted(set(so1) & set(so5))
    ao = [float(np.mean(so1[i])) for i in imgs_o]
    bo = [float(np.mean(so5[i])) for i in imgs_o]
    dso, cio = boot_diff(ao, bo)
    gate_scene = bool(dsw > 0 and ciw[0] > 0)
    gate_units = bool(min(nt1, nt5) >= UNIT_FLOOR and min(nf1, nf5) >= UNIT_FLOOR)

    out = {"sentinel": True, "model": M5, "revision": M5_SHA, "CUT": CUT, "grid": grid_tab,
           "screen_candidate": max(g["f"] for g in ok_screen), "walked_down": walked,
           "exact_at_cut": ex_c, "exact_at_next": ex_n, "next_f": nxt,
           "n_cells": len(cells), "n_images": len(keep_ids),
           "excluded_below_MIN_LEN": excluded,
           "GATE_SCENE": {"pass": gate_scene, "S_word_cap1": float(np.mean(a)),
                          "S_word_cap5": float(np.mean(b_)), "diff": dsw, "ci": ciw,
                          "S_obj_cap1": float(np.mean(ao)), "S_obj_cap5": float(np.mean(bo)),
                          "S_obj_diff": dso, "S_obj_ci": cio,
                          "n_images_word": len(imgs), "n_images_obj": len(imgs_o),
                          "nboot": NBOOT, "seed": SEED},
           "expected_units": {"n_true_cap1": nt1, "n_false_cap1": nf1,
                              "n_true_cap5": nt5, "n_false_cap5": nf5,
                              "floor": UNIT_FLOOR, "pass": gate_units},
           "composition": {"mean_P_cap1": float(np.mean(P1)), "mean_P_cap5": float(np.mean(P5)),
                           "mean_mentions_cap1": float(np.mean(M1_)),
                           "mean_mentions_cap5": float(np.mean(M5_)),
                           "mentions_per_type_cap1": float(np.mean(M1_)) / float(np.mean(P1)),
                           "mentions_per_type_cap5": float(np.mean(M5_)) / float(np.mean(P5)),
                           "arm_ratio_P": float(np.mean(P5)) / float(np.mean(P1))},
           "budget": {"mean": float(np.mean([r[1] for r in rows_c])),
                      "sd": float(np.std([r[1] for r in rows_c])),
                      "min": int(min(r[1] for r in rows_c)),
                      "max": int(max(r[1] for r in rows_c))},
           "reference_budgets": {"fam1": 44.603, "fam2": 50.272, "fam3": 40.744, "fam4": 36.142},
           "secs": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/q3_census.json", "w"), indent=1)
    json.dump({"sentinel": True, "cut": CUT, "cells": cell_out},
              open(f"{OUT}/q3_cells.json", "w"), indent=1)
    print("[census] GATE_SCENE", json.dumps(out["GATE_SCENE"]), flush=True)
    print("[census] expected_units", json.dumps(out["expected_units"]), flush=True)
    print("[census] composition", json.dumps(out["composition"]), flush=True)
    print("[census] budget", json.dumps(out["budget"]), flush=True)
    for g in grid_tab:
        print(f"[grid] f={g['f']:.3f} mean_b={g['mean_budget']:.2f} [{g['min']},{g['max']}] "
              f"dbl_screen={g['cap1_doubling_screen']:.4f} short={g['cap5_shortfall_screen']}",
              flush=True)
    assert gate_scene, "FATAL_GATE_SCENE"
    assert gate_units, f"FATAL_UNIT_FLOOR {out['expected_units']}"
    sentinel("CENSUS")


if __name__ == "__main__":
    main()
