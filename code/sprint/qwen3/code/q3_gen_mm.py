"""Lane `qwen3` generator. Port of f3_gen.py, model swapped.

Q3_ARMSET=sighted -> arms cap1 , cap5   on the TARGET image      -> out/q3_sighted.jsonl
Q3_ARMSET=blind   -> arms cap1g, cap5g  on a FLAT MID-GRAY image -> out/q3_blind.jsonl

Both arms of a set are generated in ONE q3_cont call at a FIXED batch size, so batch composition
is shared between the two arms of every contrast, and -- because the prefixes are bit-identical
between the sets and q3_cont sorts by prefix length -- between the sighted and blind runs too.

The blind set RE-DERIVES its prefixes from the same builders and seeds rather than copying the
sighted rows, so prefix identity across the pair is an independent check, not a tautology; the
scorer re-asserts it from the written JSONL.

Gates before any generation: harness-gate verdict, host, census sentinel + GATE_SCENE + unit
floor, cut drift, cell drift, census md5 drift, budget equality, doubling rate, frozen-template
drift, tokenizer identity, model identity.
"""
import os, sys, json, time, hashlib
sys.path.insert(0, "/data/alexmueller/sprint/qwen3/code")
import numpy as np
from q3_common import (M5, M5_SHA, WHICH, OUT, load_m5, q3_cont, QUESTION, load_coco_sample,
                       ChairScorer, SYN, MIN_LEN, cut_of, BS_CONT, MAX_NEW_384, MAX_NEW_192,
                       jdump, jload, sentinel)
from gen_common import prompt_for
from j5_common import build_prefix_ids
from wia_redun_common import (sources, permuted, assemble_nodouble, asm_units,
                              N_SRC, S_CAPPICK, S_JOIN_CAP)
from transformers import AutoTokenizer

ARMSET = os.environ.get("Q3_ARMSET", "")
SMOKE = os.environ.get("Q3_SMOKE", "0") == "1"
NS = 8
ASM_SEED_BASE = 1950
DOUBLE_GATE = 0.05
KDER = 3
SETS = {"sighted": (("cap1", "cap5"), False), "blind": (("cap1g", "cap5g"), True),
        "mm": (("cap1m", "cap5m"), False)}
OK_VERDICTS = ("RUN-ENDPOINT", "RUN-ENDPOINT-DEGENERACY-ATTRIBUTED")


def main():
    t0 = time.time()
    assert ARMSET in SETS, f"FATAL_ARMSET {ARMSET!r}"
    arms, gray = SETS[ARMSET]
    assert os.uname().nodename == "vgi1", f"FATAL_WRONG_NODE {os.uname().nodename}"

    gate = json.load(open(f"{OUT}/q3_gate.json"))
    assert gate["sentinel"], "FATAL_NO_GATE_SENTINEL"
    assert gate["VERDICT"] in OK_VERDICTS, f"FATAL_GATE_VERDICT {gate['VERDICT']}"
    print(f"[gen:{ARMSET}] harness gate verdict {gate['VERDICT']}", flush=True)

    census = json.load(open(f"{OUT}/q3_census.json"))
    cf = json.load(open(f"{OUT}/q3_cells.json"))
    assert census["sentinel"] and cf["sentinel"], "FATAL_CENSUS_NO_SENTINEL"
    assert census["GATE_SCENE"]["pass"], "FATAL_GATE_SCENE"
    assert census["expected_units"]["pass"], "FATAL_UNIT_FLOOR"
    CUT = cf["cut"]
    assert abs(census["CUT"] - CUT) < 1e-12, "FATAL_CUT_DRIFT"
    assert abs(gate["cut_f"] - CUT) < 1e-12, "FATAL_GATE_CUT_DRIFT"
    assert census["exact_at_cut"]["cap1_doubling"] < DOUBLE_GATE, "FATAL_DOUBLING_OVER_5PCT"
    cells_ref = {(c["image_id"], c["k"]): c for c in cf["cells"]}
    print(f"[gen:{ARMSET}] host={os.uname().nodename} CUT={CUT} cells={len(cells_ref)} "
          f"arms={arms} gray={gray} bs={BS_CONT} max_new={MAX_NEW_384}", flush=True)

    chair = ChairScorer(SYN)
    recs = load_coco_sample(chair, n=500)
    by_id = {r["image_id"]: r for r in recs}
    sids = sorted(by_id)
    pos = {i: j for j, i in enumerate(sids)}
    af = {r["image_id"]: r for r in jload(f"{OUT}/q3_af.jsonl")}
    assert set(af) == set(by_id), "FATAL_IMAGE_SET_DRIFT"

    tok = AutoTokenizer.from_pretrained(M5, revision=M5_SHA)
    assert tok.name_or_path == M5, "FATAL_WRONG_TOKENIZER_NAME"

    work = sorted({k[0] for k in cells_ref})
    if SMOKE:
        work = work[:NS]
    st = {"ncell": 0, "cyc1": 0, "topups": 0, "drift_md5": 0, "drift_cell": 0, "budget": []}
    items = []
    for iid in work:
        L = af[iid]["gen_len"]
        assert L >= MIN_LEN, f"FATAL_SHORT_L {iid}: {L}"
        for k in range(1, (1 if SMOKE else KDER) + 1):
            rc = cells_ref.get((iid, k))
            assert rc is not None, f"FATAL_CELL_MISSING {iid} {k}"
            srcs = sources(sids, pos, iid, k)
            b = cut_of(L, CUT)
            st["drift_cell"] += (rc["srcs"] != srcs) + (rc["budget"] != b) + (rc["L"] != L)
            st["ncell"] += 1
            st["budget"].append(b)

            a_ids, a_txt, a_cyc = build_prefix_ids(tok, by_id[srcs[0]]["human_caps"], b,
                                                   seed=iid * 7 + ASM_SEED_BASE + k)
            per_src = [permuted(asm_units(by_id[s]["human_caps"]),
                                iid * 7 + S_CAPPICK + k + 10 * j) for j, s in enumerate(srcs)]
            mc = assemble_nodouble(tok, per_src, b, iid * 7 + S_JOIN_CAP + k)
            assert not mc["shortfall"], f"FATAL_CAP5_SHORTFALL {iid} {k}"
            assert len(a_ids) == len(mc["ids"]) == b, f"FATAL_BUDGET_MISMATCH {iid} {k}"
            st["drift_md5"] += (hashlib.md5(a_txt.encode()).hexdigest() != rc["cap1_md5"])
            st["drift_md5"] += (hashlib.md5(mc["text"].encode()).hexdigest() != rc["cap5_md5"])
            st["cyc1"] += (a_cyc > 0)
            st["topups"] += (mc["topups"] > 0)

            common = {"image_id": iid, "k": k, "L": L, "budget": b, "cut_f": CUT, "srcs": srcs,
                      "src_id": srcs[0], "file": by_id[iid]["file"], "gray": gray}
            items.append(dict(common, arm=arms[0], prefix_ids=list(a_ids), cycles=a_cyc, plen=b,
                              prefix_ntok=len(a_ids), n_src=1,
                              n_units=len(asm_units(by_id[srcs[0]]["human_caps"])), topups=0))
            items.append(dict(common, arm=arms[1], prefix_ids=list(mc["ids"]), cycles=0, plen=b,
                              prefix_ntok=len(mc["ids"]), n_src=N_SRC,
                              n_units=mc["n_units"], topups=mc["topups"]))

    print(f"[gen:{ARMSET}] CELL GATE drift {st['drift_cell']}  CENSUS MD5 GATE drift "
          f"{st['drift_md5']}", flush=True)
    assert st["drift_cell"] == 0, f"FATAL_CELL_DRIFT {st['drift_cell']}"
    assert st["drift_md5"] == 0, f"FATAL_CENSUS_DRIFT {st['drift_md5']}"
    B = np.array(st["budget"])
    dbl = st["cyc1"] / st["ncell"]
    print(f"[gen:{ARMSET}] cells {st['ncell']} rows {len(items)} budget {B.mean():.3f} "
          f"sd {B.std():.3f} [{B.min()},{B.max()}] cap1 doubling {st['cyc1']}/{st['ncell']} "
          f"= {dbl:.4f}  cap5 topup cells {st['topups']}", flush=True)
    if not SMOKE:
        assert dbl < DOUBLE_GATE, f"FATAL_DOUBLING_OVER_5PCT {dbl}"

    if ARMSET == "mm":
        # MISMATCH: every cell sees another sample image under a fixed-point-free permutation
        import random as _rnd
        _ids = sorted({it["image_id"] for it in items})
        _r = _rnd.Random(20260919)
        while True:
            _perm = _ids[:]; _r.shuffle(_perm)
            if all(x != y for x, y in zip(_ids, _perm)):
                break
        _sub = dict(zip(_ids, _perm))
        assert len(set(_sub.values())) == len(_sub), "FATAL_MM_NOT_BIJECTIVE"
        assert all(x != y for x, y in _sub.items()), "FATAL_MM_SELF_PAIRED"
        for it in items:
            it["sub_image_id"] = _sub[it["image_id"]]
            it["file"] = by_id[it["sub_image_id"]]["file"]
        assert all(it["file"] != by_id[it["image_id"]]["file"] for it in items), "FATAL_MM_FILE_UNCHANGED"
        print(f"[gen:mm] GATE_MISMATCH_DERANGED ok, n={len(_sub)}, rows={len(items)}", flush=True)

    model, proc, gtok, meta = load_m5()
    assert meta["model"] == M5 and meta["revision"] == M5_SHA, "FATAL_WRONG_MODEL"
    assert meta["cls"] == "Qwen3VLForConditionalGeneration", "FATAL_WRONG_CLS"
    frozen = json.load(open(f"{OUT}/q3_template.json"))
    p = prompt_for(WHICH, proc, QUESTION)
    assert p == frozen["template"], f"FATAL_TEMPLATE_DRIFT {p!r}"
    nid = sum(1 for it in items[:100]
              if gtok.decode(it["prefix_ids"]) == tok.decode(it["prefix_ids"]))
    assert nid == min(100, len(items)), "FATAL_TOKENIZER_DECODE_MISMATCH"
    print(f"[gen:{ARMSET}] model loaded, template OK, tokenizer OK  {time.time()-t0:.0f}s",
          flush=True)

    rows = q3_cont(model, proc, gtok, items, gray=gray, max_new=MAX_NEW_384, bs=BS_CONT)
    assert len(rows) == len(items), f"FATAL_ROW_COUNT {len(rows)} != {len(items)}"
    per_arm = {}
    for r in rows:
        per_arm[r["arm"]] = per_arm.get(r["arm"], 0) + 1
    assert set(per_arm) == set(arms), f"FATAL_ARMS {sorted(per_arm)}"
    assert per_arm[arms[0]] == per_arm[arms[1]] == st["ncell"], f"FATAL_ARMS_UNBALANCED {per_arm}"
    assert len({(r["image_id"], r["k"], r["arm"]) for r in rows}) == len(rows), "FATAL_DUP_ROW"
    assert max(r["cont_len"] for r in rows) <= MAX_NEW_384, "FATAL_CAP_EXCEEDED"
    assert all(r["cont_len"] == len(r["cont_ids"]) for r in rows), "FATAL_CONT_LEN_MISMATCH"
    assert all(r["src_id"] != r["image_id"] for r in rows), "FATAL_SELF_PAIRED"
    assert all(r["plen"] == r["budget"] for r in rows), "FATAL_BUDGET_MISMATCH_AT_WRITE"
    assert all(r["prefix_ntok"] == r["budget"] for r in rows), "FATAL_PREFIX_NTOK_AT_WRITE"
    bycell = {}
    for r in rows:
        bycell.setdefault((r["image_id"], r["k"]), []).append(r["plen"])
    assert all(len(set(v)) == 1 and len(v) == 2 for v in bycell.values()), \
        "FATAL_BUDGET_MISMATCH_ACROSS_ARMS"

    sfx = "_smoke" if SMOKE else ""
    path = f"{OUT}/q3_{ARMSET}{sfx}.jsonl"
    jdump(path, rows)
    back = jload(path)
    assert len(back) == len(rows), f"FATAL_ROWS_WRITTEN {len(back)} != {len(rows)}"

    def st_(a, key):
        return float(np.mean([r[key] for r in rows if r["arm"] == a]))

    def md_(a, key):
        return float(np.median([r[key] for r in rows if r["arm"] == a]))

    m = {"sentinel": True, "family": "M5-qwen3-vl-8b", "model": M5, "revision": M5_SHA,
         "armset": ARMSET, "arms": list(arms), "gray": gray, "cut_f": CUT, "smoke": SMOKE,
         "host": os.uname().nodename, "max_new_cont": MAX_NEW_384, "bs": BS_CONT,
         "n_cells": st["ncell"], "rows": len(rows), "kder": 1 if SMOKE else KDER,
         "per_arm_rows": per_arm, "template": p, "gate_verdict": gate["VERDICT"],
         "per_arm": {a: {"mean_budget": st_(a, "budget"), "mean_cont_len": st_(a, "cont_len"),
                         "median_cont_len": md_(a, "cont_len"),
                         "frac_le3": float(np.mean([r["cont_len"] <= 3 for r in rows
                                                    if r["arm"] == a])),
                         "doubling_rate": float(sum(1 for r in rows if r["arm"] == a
                                                    and r["cycles"] > 0) / st["ncell"]),
                         "empty_cont_rate": float(sum(1 for r in rows if r["arm"] == a
                                                      and r["cont_len"] == 0) / st["ncell"]),
                         "at_max_384": float(sum(1 for r in rows if r["arm"] == a
                                                 and r["cont_len"] >= MAX_NEW_384) / st["ncell"]),
                         "at_max_192": float(sum(1 for r in rows if r["arm"] == a
                                                 and r["cont_len"] >= MAX_NEW_192) / st["ncell"])}
                     for a in arms},
         "cap5_topup_cells": st["topups"], "secs": round(time.time() - t0, 1)}
    json.dump(m, open(f"{OUT}/q3_{ARMSET}{sfx}_meta.json", "w"), indent=1)
    print(f"[gen:{ARMSET}] meta {json.dumps(m['per_arm'])}", flush=True)
    sentinel(f"GEN_{ARMSET.upper()}{'_SMOKE' if SMOKE else ''}")


if __name__ == "__main__":
    main()
