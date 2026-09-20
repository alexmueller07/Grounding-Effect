"""WIA-CORROB-POWER generator (WIA_CORROB_POWER_PREREG.md sections 3, 4).

Two arms, exactly as the parent lane built them:

    cap1   five COCO captions of ONE other scene,  published build_prefix_ids   (== wrongasm)
    cap5   one COCO caption from each of FIVE other scenes, assemble_nodouble   (== wrongmulticap)

There is no cap5m arm in this lane. The only deliberate change to generation is the one the parent
already made: wia_lngen.MAX_NEW_CONT is overridden 192 -> 384. j5_common.py is NOT edited.

The run is split into N_BLOCKS concurrent jobs BY IMAGE (block = pos(i) mod N_BLOCKS), so both
arms of every cell are always built and generated inside the same job, under the same sort order
and the same batch size -- the parent's "one job" requirement is about arms sharing batch
composition, and that is preserved exactly (prereg 3).

BLOCK=<b> selects the block. SMOKE=1 runs 8 published + 8 extension images at k=1 and ignores
BLOCK. Writes {OUT_R}/at_corrobpw_b<b>.jsonl (+ meta). Sentinel printed by Python; the exit code
is NEVER the success signal.
"""
import os, sys, json, time, hashlib
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")
from j5_common import (ChairScorer, build_prefix_ids, read_jsonl, JsonlWriter, sentinel,
                       MODEL, OUT, BS)
import wia_lngen
from wia_lngen import load_model_lowmem, gen_batch
from wia_redun_common import sources, permuted, assemble_nodouble, asm_units, S_CAPPICK, S_JOIN_CAP
import wia_cpw_common as C
import numpy as np
import torch

SMOKE = os.environ.get("SMOKE", "0") == "1"
BLOCK = int(os.environ.get("BLOCK", "0"))
TAG = "smoke_at_corrobpw" if SMOKE else f"at_corrobpw_b{BLOCK}"
TAG = TAG + "_mm"
CELLS_PATH = f"{C.OUT_R}/corrobpw_cells.json"
CENSUS_PATH = f"{C.OUT_R}/corrobpw_census.json"
REF_ROWS = f"{C.OUT_R}/at_redun.jsonl"
MAX_NEW_384 = 384
MAX_NEW_REF = 192
ASM_SEED_BASE = 1950
N_SMOKE_PUB = 8
N_SMOKE_EXT = 8


def main():
    t00 = time.time()
    assert wia_lngen.MAX_NEW_CONT == MAX_NEW_REF, \
        f"FATAL_UNEXPECTED_BASE_CAP {wia_lngen.MAX_NEW_CONT}"
    wia_lngen.MAX_NEW_CONT = MAX_NEW_384
    print(f"[cpwgM] MAX_NEW_CONT override {MAX_NEW_REF} -> {wia_lngen.MAX_NEW_CONT}", flush=True)

    census = json.load(open(CENSUS_PATH))
    assert census["GATE"]["cap1_vs_cap5"], "FATAL_MANIPULATION_GATE_CAP1_CAP5"
    assert census["blocks"] == C.N_BLOCKS, "FATAL_BLOCK_COUNT_DRIFT"
    cells_all = json.load(open(CELLS_PATH))["cells"]

    chair = ChairScorer("/data/alexmueller/j5_gates/code/synonyms.txt")
    allids, pub, ext, T = C.sample_ids()
    pubset = set(pub)
    data = C.records(chair, allids, T)
    by_id = {d["image_id"]: d for d in data}
    pos = {iid: j for j, iid in enumerate(allids)}
    pos500 = {iid: j for j, iid in enumerate(pub)}

    if SMOKE:
        keep = set(pub[:N_SMOKE_PUB]) | set(ext[:N_SMOKE_EXT])
        cells = [c for c in cells_all if c["image_id"] in keep and c["k"] == 1]
    else:
        cells = [c for c in cells_all if c["block"] == BLOCK]
    print(f"[cpwgM] BLOCK={BLOCK} smoke={SMOKE} cells {len(cells)} of {len(cells_all)}", flush=True)
    assert cells, "FATAL_NO_CELLS"

    ref = {}
    for r in read_jsonl(REF_ROWS):
        if r["arm"] in ("wrongasm", "wrongmulticap"):
            ref[(r["image_id"], r["k"], r["arm"])] = r

    model, proc, tok = load_model_lowmem()
    print(f"[cpwgM] MODEL={MODEL} tokenizer={tok.__class__.__name__} vocab={tok.vocab_size} "
          f"t={time.time()-t00:.0f}s", flush=True)
    assert "llava-1.5-7b" in MODEL, f"FATAL_WRONG_MODEL: {MODEL}"

    # ---- CONSTRUCTION GATE: the published `wrong` arm rebuilds bit-for-bit (over S500) -------
    wof = {pub[i]: pub[(i + 1) % len(pub)] for i in range(len(pub))}
    at_rows = [r for r in read_jsonl(f"{OUT}/at.jsonl") if r.get("kind") == "wrong"]
    nchk = nbad = 0
    for r in at_rows:
        iid, f = r["image_id"], r["cut_f"]
        _p, ptext, _c = build_prefix_ids(tok, by_id[wof[iid]]["human_caps"],
                                         r["cut_tokens"], seed=iid * 7 + int(f * 100))
        nchk += 1
        nbad += (ptext != r["prefix_text"])
    print(f"[cpwgM] CONSTRUCTION GATE: rebuilt {nchk}, mismatches {nbad}", flush=True)
    assert nchk > 0 and nbad == 0, f"FATAL_CONSTRUCTION_GATE nchk={nchk} nbad={nbad}"

    # ---- build every item -------------------------------------------------------------------
    items, st = [], {"ncell": 0, "npub": 0, "asm_cycles_rows": 0, "asm_cycles_total": 0,
                     "drift_prefix": 0, "drift_md5": 0, "drift_srcs": 0, "budget": []}
    for c in cells:
        iid, k, b = c["image_id"], c["k"], c["budget"]
        is_pub = iid in pubset
        srcs = sources(pub, pos500, iid, k) if is_pub else sources(allids, pos, iid, k)
        st["drift_srcs"] += (srcs != c["srcs"])
        st["ncell"] += 1
        st["npub"] += is_pub
        st["budget"].append(b)

        a_ids, a_text, a_cyc = build_prefix_ids(tok, by_id[srcs[0]]["human_caps"], b,
                                                seed=iid * 7 + ASM_SEED_BASE + k)
        st["asm_cycles_rows"] += (a_cyc > 0)
        st["asm_cycles_total"] += a_cyc
        per_src = [permuted(asm_units(by_id[s]["human_caps"]),
                            iid * 7 + S_CAPPICK + k + 10 * j) for j, s in enumerate(srcs)]
        mc = assemble_nodouble(tok, per_src, b, iid * 7 + S_JOIN_CAP + k)
        assert not mc["shortfall"], f"FATAL_CAP5_SHORTFALL {iid} {k}"
        assert len(a_ids) == len(mc["ids"]) == b, f"FATAL_BUDGET_MISMATCH {iid} {k}"
        st["drift_md5"] += (hashlib.md5(a_text.encode()).hexdigest() != c["cap1_md5"])
        st["drift_md5"] += (hashlib.md5(mc["text"].encode()).hexdigest() != c["cap5_md5"])
        if is_pub:
            st["drift_prefix"] += (list(ref[(iid, k, "wrongasm")]["prefix_ids"]) != list(a_ids))
            st["drift_prefix"] += (list(ref[(iid, k, "wrongmulticap")]["prefix_ids"])
                                   != list(mc["ids"]))

        for arm, pids, ptext, cyc, nu, tu in (
                ("cap1", a_ids, a_text, a_cyc, len(asm_units(by_id[srcs[0]]["human_caps"])), 0),
                ("cap5", mc["ids"], mc["text"], 0, mc["n_units"], mc["topups"])):
            items.append({"image_id": iid, "src_id": srcs[0], "srcs": srcs, "k": k, "arm": arm,
                          "is_pub": bool(is_pub), "block": c["block"],
                          "file": by_id[iid]["file"], "prefix_ids": list(pids),
                          "prefix_text": ptext, "budget": b, "cycles": cyc,
                          "n_units": nu, "topups": tu})

    print(f"[cpwgM] SRCS GATE drift {st['drift_srcs']}", flush=True)
    print(f"[cpwgM] PREFIX GATE drift {st['drift_prefix']} / {2*st['npub']} published prefixes",
          flush=True)
    print(f"[cpwgM] CENSUS MD5 GATE drift {st['drift_md5']} / {2*st['ncell']}", flush=True)
    assert st["drift_srcs"] == 0, f"FATAL_CELL_DRIFT srcs {st['drift_srcs']}"
    assert st["drift_prefix"] == 0, f"FATAL_PREFIX_DRIFT {st['drift_prefix']}"
    assert st["drift_md5"] == 0, f"FATAL_CENSUS_DRIFT {st['drift_md5']}"
    B = np.array(st["budget"])
    print(f"[cpwgM] cells {st['ncell']} ({st['npub']} published) rows {len(items)} "
          f"budget mean {B.mean():.3f} sd {B.std():.3f}", flush=True)

    # ---- MISMATCH: swap each cell's image for a deranged partner's, prefix untouched ----------
    import random as _rnd
    _ids = sorted({it["image_id"] for it in items})
    _r = _rnd.Random(20260919)
    while True:
        _perm = _ids[:]; _r.shuffle(_perm)
        if all(a != b for a, b in zip(_ids, _perm)):
            break
    _sub = dict(zip(_ids, _perm))
    assert len(set(_sub.values())) == len(_sub), "FATAL_MM_NOT_BIJECTIVE"
    assert all(k != v for k, v in _sub.items()), "FATAL_MM_SELF_PAIRED"
    _file = {it["image_id"]: it["file"] for it in items}
    _nsw = 0
    for it in items:
        it["sub_image_id"] = _sub[it["image_id"]]
        it["file"] = _file[it["sub_image_id"]]
        _nsw += 1
    print(f"[cpwgM] GATE_MISMATCH_DERANGED ok, n={len(_sub)}, rows_swapped={_nsw}", flush=True)

    # ---- generate ---------------------------------------------------------------------------
    w = JsonlWriter(f"{C.OUT_R}/{TAG}.jsonl")
    items.sort(key=lambda it: len(it["prefix_ids"]))
    bs = max(4, BS // 2)
    t0 = time.time()
    for b0 in range(0, len(items), bs):
        chunk = items[b0:b0 + bs]
        conts = gen_batch(model, proc, tok, chunk)
        for it, ids in zip(chunk, conts):
            w.write({"image_id": it["image_id"], "src_id": it["src_id"], "srcs": it["srcs"],
                     "k": it["k"], "arm": it["arm"], "is_pub": it["is_pub"],
                     "block": it["block"], "budget": it["budget"], "cycles": it["cycles"],
                     "n_units": it["n_units"], "topups": it["topups"],
                     "prefix_ids": it["prefix_ids"], "prefix_text": it["prefix_text"],
                     "cont_ids": ids, "cont_text": tok.decode(ids, skip_special_tokens=True),
                     "cont_len": len(ids)})
        w.flush()
        if (b0 // bs) % 50 == 0 or b0 + len(chunk) >= len(items):
            el = time.time() - t0
            done = b0 + len(chunk)
            print(f"[cpwgM] b{BLOCK} {done}/{len(items)} t={el:.0f}s "
                  f"eta={el/max(done,1)*(len(items)-done):.0f}s", flush=True)
    w.close()

    rows = read_jsonl(f"{C.OUT_R}/{TAG}.jsonl")
    assert len(rows) == len(items), f"FATAL_ROWS_WRITTEN {len(rows)} != {len(items)}"
    per_arm = {}
    for r in rows:
        per_arm[r["arm"]] = per_arm.get(r["arm"], 0) + 1
    assert set(per_arm) == {"cap1", "cap5"}, f"FATAL_ARMS {sorted(per_arm)}"
    assert per_arm["cap1"] == per_arm["cap5"] == st["ncell"], f"FATAL_ARMS_UNBALANCED {per_arm}"
    assert max(r["cont_len"] for r in rows) <= MAX_NEW_384, "FATAL_CAP_EXCEEDED"

    agree = {}
    for a, ra in (("cap1", "wrongasm"), ("cap5", "wrongmulticap")):
        tot = same = 0
        for r in rows:
            if r["arm"] != a or not r["is_pub"]:
                continue
            key = (r["image_id"], r["k"], ra)
            if key in ref:
                tot += 1
                same += (list(ref[key]["cont_ids"]) == list(r["cont_ids"][:MAX_NEW_REF]))
        agree[a] = {"checked": tot, "identical": same, "frac": same / tot if tot else None}
    print(f"[cpwgM] 192-RECONSTRUCTION AGREEMENT vs at_redun.jsonl on published cells "
          f"(reported, not gated): {agree}", flush=True)

    cl = {a: float(np.mean([r["cont_len"] for r in rows if r["arm"] == a])) for a in per_arm}
    am = {a: float(np.mean([r["cont_len"] >= MAX_NEW_384 for r in rows if r["arm"] == a]))
          for a in per_arm}
    emp = {a: float(np.mean([r["cont_len"] == 0 for r in rows if r["arm"] == a]))
           for a in per_arm}
    bu = sum(1 for k in {(r["image_id"], r["k"]) for r in rows}
             if all(x["cont_len"] < MAX_NEW_384
                    for x in rows if (x["image_id"], x["k"]) == k)) if SMOKE else None
    print(f"[cpwgM] mean cont len {cl}", flush=True)
    print(f"[cpwgM] at_max(384) {am}   empty {emp}", flush=True)

    json.dump({"sentinel": True, "model": MODEL, "block": BLOCK, "smoke": SMOKE,
               "rows": len(rows), "per_arm": per_arm, "n_cells": st["ncell"],
               "n_cells_published": st["npub"], "loader": "lowmem",
               "n_images": len({it["image_id"] for it in items}),
               "max_new_cont": MAX_NEW_384, "max_new_cont_reference": MAX_NEW_REF,
               "cap1_rows_doubled": st["asm_cycles_rows"],
               "cap1_cycles_total": st["asm_cycles_total"],
               "budget_tokens": {"mean": float(B.mean()), "sd": float(B.std()),
                                 "min": int(B.min()), "max": int(B.max())},
               "srcs_gate_drift": st["drift_srcs"], "prefix_gate_drift": st["drift_prefix"],
               "census_md5_drift": st["drift_md5"],
               "construction_gate": {"checked": nchk, "mismatches": nbad},
               "recon192_agreement_vs_at_redun": agree,
               "mean_cont_len_per_arm": cl, "at_max_384_per_arm": am,
               "empty_cont_rate_per_arm": emp, "smoke_both_uncapped_cells": bu},
              open(f"{C.OUT_R}/{TAG}_meta.json", "w"), indent=1)
    sentinel("CPW_GEN_SMOKE" if SMOKE else f"CPW_GEN_B{BLOCK}")


if __name__ == "__main__":
    main()
