"""L1 -- the ablation ladder on the paper's prefix endpoint (LADDER_PREREG.md section 3).

One rung at a time (or a comma-separated list, model loaded once), two arms per rung:

    cap1_<rung>   prefix bit-identical to the frozen sighted cap1
    cap5_<rung>   prefix bit-identical to the frozen sighted cap5

The sighted arms are NOT regenerated: Delta_primary comes from the frozen
sc1_interv/out/at_corrob.jsonl, exactly as the published analysis reads it.

This file mirrors wia_blindgen.py's build order, stable sort, batch size and gates line for
line. The ONLY change is the pixels: `wia_blind_common._img(path, gray)` becomes
`RungImages.get(path, image_id)`. Nothing under sc1_interv/, j5_gates/ or paialpha/ is written.

ENV
  LAD_RUNGS   comma-separated rung names (default: every rung + the grey control)
  LAD_SMOKE   1 -> first 16 image ids, K = 1, _smoke-suffixed files
  LAD_BS      batch size, default 8 (the published blind run's `max(4, BS // 2)`)

Writes out/lad_f1_<rung>.jsonl and out/lad_f1_<rung>_meta.json. Sentinel printed by Python;
the exit code is NEVER the success signal.
"""
import os, sys, json, time, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")

import numpy as np
import torch

from lad_common import (RungImages, ALL_RUNGS, UNIFORM_RUNGS, OUT, sentinel, jdump,
                        canon, derangement, MISMATCH_SEED)
from j5_common import (load_coco_sample, ChairScorer, build_prefix_ids, read_jsonl,
                       JsonlWriter, MODEL, OUT as J5OUT, N_IMAGES, BS)
import wia_lngen
from wia_lngen import load_model_lowmem
from wia_redun_common import (sources, permuted, assemble_nodouble, asm_units,
                              S_CAPPICK, S_JOIN_CAP)
from wia_blind_common import mirror_gate_f1

SMOKE = os.environ.get("LAD_SMOKE", "0") == "1"
N_SMOKE = 16
OUT_R = "/data/alexmueller/sc1_interv/out"          # READ ONLY
SUF = "_smoke" if SMOKE else ""
SIGHTED = f"{OUT_R}/at_corrob.jsonl"
CELLS_PATH = f"{OUT_R}/corrob_cells.json"
KDER = 1 if SMOKE else 3
MIN_LEN = 32
ASM_SEED_BASE = 1950
MAX_NEW_384 = 384
MAX_NEW_REF = 192
N_MIRROR = 32
EXPECT_ROWS = 3000


# ---------------------------------------------------------------- generation
def gen_batch_rung(model, proc, tok, items, imgs, max_new, pixgate=None):
    """LINE-FOR-LINE mirror of wia_blind_common.gen_batch_gray. The ONLY change is the image
    source: `imgs.get(file, image_id)` instead of `_img(file, gray)`. `imgs=None` means the
    clean canonical image, which GATE_MIRROR asserts is the sighted path."""
    from j5_common import prompt_text
    from PIL import Image
    with torch.no_grad():
        if imgs is None:
            ims = [Image.open(it["file"]).convert("RGB") for it in items]
        else:
            ims = [imgs.get(it["file"], it["image_id"]) for it in items]
        enc = proc(images=ims, text=[prompt_text()] * len(items), return_tensors="pt",
                   padding=True)
        base_ids = enc["input_ids"]
        pix = enc["pixel_values"].to("cuda", torch.bfloat16)
        if pixgate is not None:
            pixgate(items, pix, proc)
        rows, masks = [], []
        maxlen = max(base_ids.shape[1] + len(it["prefix_ids"]) for it in items)
        pad = tok.pad_token_id
        for i, it in enumerate(items):
            row = base_ids[i].tolist() + list(it["prefix_ids"])
            need = maxlen - len(row)
            rows.append([pad] * need + row)
            masks.append([0] * need + [1] * len(row))
        input_ids = torch.tensor(rows, device="cuda")
        attn = torch.tensor(masks, device="cuda")
        gen = model.generate(input_ids=input_ids, attention_mask=attn, pixel_values=pix,
                             do_sample=False, max_new_tokens=max_new, pad_token_id=pad)
        new = gen[:, input_ids.shape[1]:]
        outs = []
        for row in new:
            ids = [int(t) for t in row if int(t) != pad]
            if ids and ids[-1] == tok.eos_token_id:
                ids = ids[:-1]
            outs.append(ids)
        return outs


def build_items(tok, by_id, sids, all_sids, pos, af, cells_ref, ref, arm_of):
    items, st = [], {"ncell": 0, "drift_prefix": 0, "budget": [], "cyc1": 0, "topups": 0}
    for iid in sids:
        if af[iid]["gen_len"] < MIN_LEN:
            continue
        for k in range(1, KDER + 1):
            rc = cells_ref.get((iid, k))
            assert rc is not None, f"FATAL_CELL_MISSING {iid} {k}"
            srcs = sources(all_sids, pos, iid, k)
            assert rc["srcs"] == srcs, f"FATAL_CELL_DRIFT srcs {iid} {k}"
            b = rc["budget"]
            st["ncell"] += 1
            st["budget"].append(b)
            a_ids, a_text, a_cyc = build_prefix_ids(tok, by_id[srcs[0]]["human_caps"], b,
                                                    seed=iid * 7 + ASM_SEED_BASE + k)
            per_src = [permuted(asm_units(by_id[s]["human_caps"]),
                                iid * 7 + S_CAPPICK + k + 10 * j) for j, s in enumerate(srcs)]
            mc = assemble_nodouble(tok, per_src, b, iid * 7 + S_JOIN_CAP + k)
            assert not mc["shortfall"], f"FATAL_CAP5_SHORTFALL {iid} {k}"
            assert len(a_ids) == len(mc["ids"]) == b, f"FATAL_BUDGET_MISMATCH {iid} {k}"
            st["cyc1"] += (a_cyc > 0)
            st["topups"] += (mc["topups"] > 0)
            for base, pids, ptext, cyc, nu, tu in (
                    ("cap1", a_ids, a_text, a_cyc,
                     len(asm_units(by_id[srcs[0]]["human_caps"])), 0),
                    ("cap5", mc["ids"], mc["text"], 0, mc["n_units"], mc["topups"])):
                rr = ref[(iid, k, base)]
                st["drift_prefix"] += (list(rr["prefix_ids"]) != list(pids))
                st["drift_prefix"] += (rr["prefix_text"] != ptext)
                st["drift_prefix"] += (rr["budget"] != b)
                items.append({"image_id": iid, "src_id": srcs[0], "srcs": srcs, "k": k,
                              "arm": arm_of[base], "sighted_arm": base,
                              "file": by_id[iid]["file"], "prefix_ids": list(pids),
                              "prefix_text": ptext, "budget": b, "cycles": cyc,
                              "n_units": nu, "topups": tu})
    return items, st


def run_rung(rung, model, proc, tok, items, sighted_order, bs, st, sub=None):
    tag = f"lad_f1_{rung}{SUF}"
    path, mpath = f"{OUT}/{tag}.jsonl", f"{OUT}/{tag}_meta.json"
    want = len(items)
    have = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    mn = -1
    if os.path.exists(mpath):
        try:
            mn = json.load(open(mpath)).get("rows", -1)
        except Exception:
            mn = -1
    if have == want and mn == want:
        print(f"[lad_gen] rung {rung} already complete ({have} rows), skipping", flush=True)
        return None
    print(f"[lad_gen] === rung {rung} (have {have} rows, meta {mn}) ===", flush=True)

    arm_of = {"cap1": f"cap1_{rung}", "cap5": f"cap5_{rung}"}
    ritems = [dict(it, arm=arm_of[it["sighted_arm"]]) for it in items]
    order = [(it["image_id"], it["k"], it["sighted_arm"]) for it in ritems]
    if not SMOKE:
        assert order == sighted_order, "FATAL_BATCH_ORDER_DRIFT"

    imgs = RungImages(proc, rung, sub=sub)
    submeta = None
    if rung == "mismatch":
        # GATE_MISMATCH_DERANGED: a bijection with no fixed point, and never the target image.
        assert sub, "FATAL_MISMATCH_WITHOUT_SUBSTITUTION_MAP"
        assert all(int(v[0]) != int(k) for k, v in sub.items()), "FATAL_MISMATCH_SELF_PAIRED"
        assert len({v[0] for v in sub.values()}) == len(sub), "FATAL_MISMATCH_NOT_BIJECTIVE"
        # How often does the substituted image happen to BE one of the cell's prefix sources?
        # Registered as a reported diagnostic plus an exclusion sensitivity (Amendment 2).
        coll = sum(1 for it in ritems if int(sub[int(it["image_id"])][0]) in
                   {int(x) for x in it["srcs"]})
        submeta = {"n": len(sub), "seed": MISMATCH_SEED,
                   "cells_where_substitute_is_a_prefix_source": coll,
                   "frac": coll / max(len(ritems), 1),
                   "map_sample": {str(k): int(v[0]) for k, v in list(sub.items())[:10]}}
        print(f"[lad_gen] GATE_MISMATCH_DERANGED ok, n={len(sub)}, "
              f"substitute-is-a-prefix-source in {coll}/{len(ritems)} rows", flush=True)

    # ---- GATE_RUNG_APPLIED, on the real tensors of the first chunk ------------------------
    gate = {"rows_changed": 0, "rows": 0, "cross_image_distinct": 0, "pairs": 0}

    def pixgate(chunk, pix, proc_):
        if gate["rows"] >= len(chunk):
            return
        from PIL import Image
        clean = proc_.image_processor(
            images=[canon(proc_, Image.open(it["file"]).convert("RGB")) for it in chunk],
            return_tensors="pt")["pixel_values"].to(pix.device, pix.dtype)
        for i in range(len(chunk)):
            gate["rows"] += 1
            gate["rows_changed"] += int(not bool(torch.equal(pix[i], clean[i])))
        seen = {}
        for i, it in enumerate(chunk):
            key = it["image_id"]
            if key in seen:
                continue
            seen[key] = i
        ks = list(seen)
        for a in range(len(ks) - 1):
            gate["pairs"] += 1
            gate["cross_image_distinct"] += int(
                not bool(torch.equal(pix[seen[ks[a]]], pix[seen[ks[a + 1]]])))

    w = JsonlWriter(path)
    t1 = time.time()
    for b0 in range(0, len(ritems), bs):
        chunk = ritems[b0:b0 + bs]
        conts = gen_batch_rung(model, proc, tok, chunk, imgs, MAX_NEW_384,
                               pixgate=pixgate if b0 == 0 else None)
        if b0 == 0:
            assert gate["rows_changed"] == gate["rows"], \
                f"FATAL_RUNG_NOT_APPLIED {rung} {gate}"
            if rung in UNIFORM_RUNGS:
                # A uniform field of ANY input size resizes to the SAME canonical tensor, so
                # every row must be bit-identical. Asserted positively, not skipped.
                assert gate["cross_image_distinct"] == 0, \
                    f"FATAL_UNIFORM_RUNG_NOT_CONSTANT {rung} {gate}"
            elif gate["pairs"]:
                assert gate["cross_image_distinct"] == gate["pairs"], \
                    f"FATAL_RUNG_CONSTANT_ACROSS_IMAGES {rung} {gate}"
            print(f"[lad_gen] GATE_RUNG_APPLIED {rung} {json.dumps(gate)}", flush=True)
        for it, ids in zip(chunk, conts):
            w.write({"image_id": it["image_id"], "src_id": it["src_id"], "srcs": it["srcs"],
                     "k": it["k"], "arm": it["arm"], "sighted_arm": it["sighted_arm"],
                     "rung": rung, "budget": it["budget"], "cycles": it["cycles"],
                     "n_units": it["n_units"], "topups": it["topups"],
                     "prefix_ids": it["prefix_ids"], "prefix_text": it["prefix_text"],
                     "cont_ids": ids, "cont_text": tok.decode(ids, skip_special_tokens=True),
                     "cont_len": len(ids),
                     "sub_image_id": (None if sub is None
                                      else int(sub[int(it["image_id"])][0]))})
        w.flush()
        if (b0 // bs) % 40 == 0 or b0 + len(chunk) >= len(ritems):
            el, done = time.time() - t1, b0 + len(chunk)
            print(f"[lad_gen] {rung} {done}/{len(ritems)} t={el:.0f}s "
                  f"eta={el/max(done,1)*(len(ritems)-done):.0f}s", flush=True)
    w.close()

    rows = list(read_jsonl(path))
    assert len(rows) == len(ritems), f"FATAL_ROWS_WRITTEN {len(rows)} != {len(ritems)}"
    per_arm = {}
    for r in rows:
        per_arm[r["arm"]] = per_arm.get(r["arm"], 0) + 1
    assert set(per_arm) == set(arm_of.values()), f"FATAL_ARMS {sorted(per_arm)}"
    assert len(set(per_arm.values())) == 1, f"FATAL_ARMS_UNBALANCED {per_arm}"
    assert len({(r["image_id"], r["k"], r["arm"]) for r in rows}) == len(rows), "FATAL_DUP_ROW"
    assert max(r["cont_len"] for r in rows) <= MAX_NEW_384, "FATAL_CAP_EXCEEDED"
    assert all(r["cont_len"] == len(r["cont_ids"]) for r in rows), "FATAL_CONT_LEN_MISMATCH"
    assert all(r["src_id"] != r["image_id"] for r in rows), "FATAL_SELF_PAIRED"
    assert all(len(r["prefix_ids"]) == r["budget"] for r in rows), "FATAL_BUDGET_AT_WRITE"

    def s_(a, key):
        return [r[key] for r in rows if r["arm"] == a]

    meta = {"sentinel": True, "rung": rung, "tag": tag, "model": MODEL, "smoke": SMOKE,
            "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "max_new_cont": MAX_NEW_384, "bs": bs, "rows": len(rows), "kder": KDER,
            "n_cells": st["ncell"], "per_arm_rows": per_arm,
            "prefix_drift": st["drift_prefix"], "batch_order_identical": (not SMOKE),
            "gate_rung_applied": gate, "rung_cache": imgs.stats(), "mismatch": submeta,
            "prereg": "SPRINT/LADDER_PREREG.md @ f186bcf",
            "per_arm": {a: {"mean_cont_len": float(np.mean(s_(a, "cont_len"))),
                            "median_cont_len": float(np.median(s_(a, "cont_len"))),
                            "at_max_384": float(np.mean([x >= MAX_NEW_384
                                                         for x in s_(a, "cont_len")])),
                            "empty_cont_rate": float(np.mean([x == 0
                                                              for x in s_(a, "cont_len")])),
                            "n_distinct_texts": len(set(s_(a, "cont_text")))}
                        for a in per_arm},
            "secs": round(time.time() - t1, 1)}
    jdump(mpath, meta)
    print(f"[lad_gen] {rung} per_arm {json.dumps(meta['per_arm'])}", flush=True)
    return meta


def main():
    t0 = time.time()
    assert socket.gethostname().startswith("vgi1"), f"FATAL_WRONG_NODE {socket.gethostname()}"
    rungs = [r.strip() for r in
             os.environ.get("LAD_RUNGS", ",".join(ALL_RUNGS)).split(",") if r.strip()]
    for r in rungs:
        assert r in ALL_RUNGS, f"FATAL_UNKNOWN_RUNG {r}"
    assert "identity" not in rungs, "FATAL_IDENTITY_IS_THE_SIGHTED_ARM"
    bs = int(os.environ.get("LAD_BS", str(max(4, BS // 2))))

    assert wia_lngen.MAX_NEW_CONT == MAX_NEW_REF, \
        f"FATAL_UNEXPECTED_BASE_CAP {wia_lngen.MAX_NEW_CONT}"
    wia_lngen.MAX_NEW_CONT = MAX_NEW_384          # the sighted lane's one deliberate override
    print(f"[lad_gen] host={socket.gethostname()} rungs={rungs} bs={bs} smoke={SMOKE}",
          flush=True)

    sighted = list(read_jsonl(SIGHTED))
    assert len(sighted) == EXPECT_ROWS, f"FATAL_SIGHTED_ROWS {len(sighted)}"
    plen = [len(r["prefix_ids"]) for r in sighted]
    assert all(plen[i] <= plen[i + 1] for i in range(len(plen) - 1)), "FATAL_SIGHTED_NOT_SORTED"
    ref = {(r["image_id"], r["k"], r["arm"]): r for r in sighted}
    assert len(ref) == EXPECT_ROWS, "FATAL_SIGHTED_DUP_ROW"
    sighted_order = [(r["image_id"], r["k"], r["arm"]) for r in sighted]
    cells_ref = {(c["image_id"], c["k"]): c for c in json.load(open(CELLS_PATH))["cells"]}

    chair = ChairScorer("/data/alexmueller/j5_gates/code/synonyms.txt")
    data = load_coco_sample(chair, n=N_IMAGES)
    by_id = {d["image_id"]: d for d in data}
    all_sids = sorted(by_id)
    sids = all_sids[:N_SMOKE] if SMOKE else all_sids
    pos = {iid: j for j, iid in enumerate(all_sids)}
    af = {r["image_id"]: r for r in read_jsonl(f"{J5OUT}/af.jsonl")}
    assert set(af) == set(by_id), "FATAL_AF_IMAGE_SET_DRIFT"

    model, proc, tok = load_model_lowmem()
    assert "llava-1.5-7b" in MODEL, f"FATAL_WRONG_MODEL {MODEL}"

    items, st = build_items(tok, by_id, sids, all_sids, pos, af, cells_ref, ref,
                            {"cap1": "cap1", "cap5": "cap5"})
    print(f"[lad_gen] PREFIX GATE drift {st['drift_prefix']} / {3*len(items)}", flush=True)
    assert st["drift_prefix"] == 0, f"FATAL_BLIND_PREFIX_DRIFT {st['drift_prefix']}"
    items.sort(key=lambda it: len(it["prefix_ids"]))          # stable, as the sighted lane
    if not SMOKE:
        assert len(items) == EXPECT_ROWS, f"FATAL_ITEMS {len(items)}"

    nchk, nid = mirror_gate_f1(model, proc, tok, items[:N_MIRROR], bs, MAX_NEW_384)
    print(f"[lad_gen] GATE_MIRROR_IDENTICAL {nid}/{nchk}", flush=True)
    assert nchk == min(N_MIRROR, len(items)) and nid == nchk, \
        f"FATAL_MIRROR_NOT_IDENTICAL {nid}/{nchk}"

    sub = None
    if "mismatch" in rungs:
        d = derangement(all_sids)
        sub = {int(k): (int(v), by_id[v]["file"]) for k, v in d.items()}

    done = []
    for r in rungs:
        m = run_rung(r, model, proc, tok, items,
                     [(i, k, a) for (i, k, a) in sighted_order], bs, st,
                     sub=(sub if r == "mismatch" else None))
        done.append({"rung": r, "skipped": m is None,
                     "secs": None if m is None else m["secs"]})
        print(f"[lad_gen] cumulative t={time.time()-t0:.0f}s {json.dumps(done)}", flush=True)
    sentinel("GEN_F1_SMOKE" if SMOKE else "GEN_F1")


if __name__ == "__main__":
    main()
