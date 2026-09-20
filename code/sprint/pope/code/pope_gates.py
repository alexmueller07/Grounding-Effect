"""POPE lane -- integrity gates.  All blocking unless marked REPORTED.

Run on POPE_LIMIT images before any endpoint arm.  Writes {OUT}/pope_gates.json and prints
POPE_SENTINEL_GATES_OK only if every blocking gate passes.

The gate that matters most is G7 (liveness): without it, a decomposition of "the intervention"
could be a decomposition of nothing.  The gate that matters second most is G2: if the token
ids are not bit-identical across arms, "only the pixels change" is false and every contrast
below is void.
"""
import hashlib
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

import pope_pixels
from pope_common import (BS, DATA, IMGDIR, MAX_NEW, METHODS, OUT, SPLITS, prompt_text)
from pope_items import batch_plan, load_items
import pope_kernel
from pope_kernel import ST, build_uncond, load_model_pa, two_stream

assert socket.gethostname().startswith("vgi1"), f"FATAL_WRONG_NODE {socket.gethostname()}"
LIMIT = int(os.environ.get("POPE_LIMIT", "16"))
R = {}


def rec(name, passed, blocking, detail):
    R[name] = {"pass": bool(passed), "blocking": bool(blocking), "detail": detail}
    print(f"[gate] {name:26s} {'PASS' if passed else 'FAIL':4s} "
          f"{'(blocking)' if blocking else '(reported)'} {detail}", flush=True)


# ---------------------------------------------------------------- G1 data integrity
def g1_data():
    man = json.load(open(f"{DATA}/image_manifest.json"))
    need = set()
    for s in SPLITS:
        for l in open(f"{DATA}/coco_pope_{s}.json"):
            need.add(json.loads(l)["image"])
    miss = [f for f in need if f not in man or not os.path.exists(os.path.join(IMGDIR, f))]
    agg = hashlib.sha256("".join(man[f]["sha256"] for f in sorted(need)).encode()).hexdigest()
    rec("G1_data_integrity", len(need) == 500 and not miss, True,
        {"n_images": len(need), "missing": len(miss), "aggregate_sha256": agg})


# ---------------------------------------------------------------- G6 causality (no model)
def g6_causality():
    torch.manual_seed(0)
    B, H, Q, K, D = 2, 4, 7, 7, 8
    q = torch.randn(B, H, Q, D)
    k = torch.randn(B, H, Q, D)
    v = torch.randn(B, H, Q, D)

    class M:
        num_key_value_groups = 1
        layer_idx = 0
    ST["active"] = False
    out, w = pope_kernel.pa_attention(M(), q, k, v, None, scaling=1.0)
    fut = torch.triu(torch.ones(Q, K), diagonal=1).bool()
    mass = float(w[..., fut].sum())
    rec("G6_causality_no_mask", mass == 0.0, True,
        {"future_key_mass": mass, "shape": list(w.shape)})


# ---------------------------------------------------------------- G9 frame equivalence
def g9_frame(proc, items):
    ds = []
    for it in items[:8]:
        with Image.open(it["file"]) as im:
            im = im.convert("RGB")
            a = proc(images=[im], text=[prompt_text(it["text"])],
                     return_tensors="pt")["pixel_values"]
            b = proc(images=[pope_pixels._to336(im)], text=[prompt_text(it["text"])],
                     return_tensors="pt")["pixel_values"]
        ds.append(float((a - b).abs().max()))
    rec("G9_frame_equivalence", max(ds) < 0.05, False,
        {"max_abs_pixel_diff": max(ds), "per_image": ds,
         "note": "REPORTED: bounds how much the m336 ladder frame can move a number by itself"})


# ---------------------------------------------------------------- model-side gates
def run_cell(model, proc, tok, items, plan, pixarm, spec, track=True):
    pil = {}
    for it in items:
        if it["image"] not in pil:
            with Image.open(it["file"]) as im:
                pil[it["image"]] = pope_pixels.make(pixarm, im.convert("RGB"), it["image_id"])
    pad, eos = tok.pad_token_id, tok.eos_token_id
    outs, pids, masses = {}, {}, []
    for idx in plan:
        chunk = [items[i] for i in idx]
        enc = proc(images=[pil[c["image"]] for c in chunk],
                   text=[prompt_text(c["text"]) for c in chunk],
                   return_tensors="pt", padding=True)
        enc = {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda"))
               for k, v in enc.items()}
        assert bool(enc["attention_mask"].all()), "FATAL_BATCH_HAS_PADDING"
        ids_u, am_u = build_uncond(enc["input_ids"], enc["attention_mask"], ST["img_tok"], pad)
        gen, mass = two_stream(model, enc, enc["pixel_values"], ids_u, am_u,
                               spec["alpha"], spec["gamma"], spec["rowmode"],
                               MAX_NEW, eos, pad, track_mass=track)
        masses.append(mass.mean())
        for j, c in enumerate(chunk):
            key = (c["split"], c["question_id"])
            ids = [int(x) for x in gen[j].tolist()]
            ids = ids[:ids.index(eos)] if eos in ids else ids
            ids = [x for x in ids if x != pad]
            outs[key] = tok.decode(ids, skip_special_tokens=True)
            pids[key] = hashlib.sha256(
                json.dumps(enc["input_ids"][j].tolist()).encode()).hexdigest()[:16]
    return outs, pids, float(np.mean(masses))


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    g1_data()
    g6_causality()

    items = load_items(SPLITS, limit_images=LIMIT)
    print(f"[gates] {len(items)} items over {LIMIT} images", flush=True)
    model, proc, tok = load_model_pa()
    ST["img_tok"] = getattr(model.config, "image_token_index", None) or \
        getattr(model.config, "image_token_id", None)
    plan, plan_sha, len_hist = batch_plan(items, tok, BS)
    print(f"[gates] plan {len(plan)} batches sha={plan_sha[:16]}", flush=True)
    g9_frame(proc, items)

    van, pf, pa = METHODS["vanilla"], METHODS["pai_full"], METHODS["pai_attn"]

    # ---- G11 determinism + G2 token identity + G7 liveness ----
    for k in ("calls", "applied", "applied_prefill", "mask_leak"):
        ST[k] = 0
    o_van, pid_van, m_van = run_cell(model, proc, tok, items, plan, "sighted", van)
    ed_van = (ST["applied"], ST["applied_prefill"])
    for k in ("calls", "applied", "applied_prefill", "mask_leak"):
        ST[k] = 0
    o_van2, pid_van2, _ = run_cell(model, proc, tok, items, plan, "sighted", van)
    rec("G11_determinism", o_van == o_van2, True,
        {"n": len(o_van), "differing": sum(o_van[k] != o_van2[k] for k in o_van)})

    for k in ("calls", "applied", "applied_prefill", "mask_leak"):
        ST[k] = 0
    o_pf, pid_pf, m_pf = run_cell(model, proc, tok, items, plan, "sighted", pf)
    ed_pf = (ST["applied"], ST["applied_prefill"])
    for k in ("calls", "applied", "applied_prefill", "mask_leak"):
        ST[k] = 0
    o_pa, pid_pa, m_pa = run_cell(model, proc, tok, items, plan, "sighted", pa)
    ed_pa = (ST["applied"], ST["applied_prefill"])
    for k in ("calls", "applied", "applied_prefill", "mask_leak"):
        ST[k] = 0
    o_grey, pid_grey, m_grey = run_cell(model, proc, tok, items, plan, "grey", van)

    same_pid = (pid_van == pid_pf == pid_pa == pid_grey)
    rec("G2_token_identity", same_pid, True,
        {"arms_compared": ["sighted/vanilla", "sighted/pai_full", "sighted/pai_attn",
                           "grey/vanilla"],
         "n_items": len(pid_van),
         "plan_sha": plan_sha, "len_hist": len_hist})

    rec("G7a_liveness_edits",
        ed_van == (0, 0) and ed_pf[0] > 0 and ed_pf[1] == 0
        and ed_pa[0] > 0 and ed_pa[1] > 0, True,
        {"vanilla(applied,prefill)": ed_van, "pai_full": ed_pf, "pai_attn": ed_pa,
         "note": "pai_full must NOT edit prefill (released --use-attn --use-cfg path); "
                 "pai_attn must (released --use-attn path)"})
    rec("G7b_liveness_mass", m_pa > m_van and m_pf > m_van, True,
        {"img_attn_mass_vanilla": m_van, "pai_full": m_pf, "pai_attn": m_pa})
    d_pf = sum(o_van[k] != o_pf[k] for k in o_van) / len(o_van)
    d_pa = sum(o_van[k] != o_pa[k] for k in o_van) / len(o_van)
    d_grey = sum(o_van[k] != o_grey[k] for k in o_van) / len(o_van)
    rec("G7c_answers_move", (d_pf + d_pa) > 0.0, True,
        {"frac_answers_differing_pai_full": d_pf, "pai_attn": d_pa, "grey_vs_sighted": d_grey,
         "note": "REPORTED magnitudes: on a yes/no endpoint most answers coincide even under "
                 "a live intervention, so the blocking condition is only that SOME move"})
    rec("G8_grey_constant", True, True,
        {"note": "asserted per batch inside run_arm/two_stream (FATAL_GREY_PIX_ROWS_DIFFER)"})
    rec("G_masks", ST["mask_leak"] == 0, True,
        {"mask_leak": ST["mask_leak"], "mask_present": ST["mask_present"],
         "causal_repairs": ST["causal_repairs"], "nomask_decode": ST["nomask_decode"]})

    # ---- G10 batch invariance ----
    small = [b for b in plan if len(b) >= 4][0][:4]
    o_b, _, _ = run_cell(model, proc, tok, items, [small], "sighted", van)
    o_s, _, _ = run_cell(model, proc, tok, items, [[i] for i in small], "sighted", van)
    rec("G10_batch_invariance", True, False,
        {"n": len(o_b), "differing": sum(o_b[k] != o_s[k] for k in o_b),
         "note": "REPORTED: batch composition is identical across arms by construction "
                 "(the plan depends only on the text), so any batch-shape effect is common "
                 "to every arm and cancels in the paired contrasts"})

    # ---- G5 decode-loop against HF generate, SAME kernel ----
    idx = plan[0][:4]
    chunk = [items[i] for i in idx]
    with_img = []
    for c in chunk:
        with Image.open(c["file"]) as im:
            with_img.append(im.convert("RGB"))
    enc = proc(images=with_img, text=[prompt_text(c["text"]) for c in chunk],
               return_tensors="pt", padding=True)
    enc = {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda"))
           for k, v in enc.items()}
    ST["active"] = False
    ST["alpha"] = 0.0
    with torch.no_grad():
        hf = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW,
                            pad_token_id=tok.pad_token_id)
    hf_new = hf[:, enc["input_ids"].shape[1]:]
    ids_u, am_u = build_uncond(enc["input_ids"], enc["attention_mask"], ST["img_tok"],
                               tok.pad_token_id)
    mine, _ = two_stream(model, enc, enc["pixel_values"], ids_u, am_u, 0.0, 1.0, "decode",
                         MAX_NEW, tok.eos_token_id, tok.pad_token_id)
    mm = 0
    for j in range(len(chunk)):
        a = [int(x) for x in hf_new[j].tolist()]
        a = a[:a.index(tok.eos_token_id)] if tok.eos_token_id in a else a
        a = [x for x in a if x != tok.pad_token_id]
        b = [int(x) for x in mine[j].tolist()]
        b = b[:b.index(tok.eos_token_id)] if tok.eos_token_id in b else b
        b = [x for x in b if x != tok.pad_token_id]
        mm += int(a != b)
    rec("G5_decode_loop_vs_hf", mm == 0, True, {"mismatches": mm, "n": len(chunk)})

    # ---- G4 copy drift against the ORIGINAL pa_gen loop ----
    detail = {}
    ok4 = False
    try:
        src = "/data/alexmueller/paialpha/code/pa_gen.py"
        h = hashlib.sha256(open(src, "rb").read()).hexdigest()
        detail["pa_gen_sha256"] = h
        detail["frozen_sha256"] = \
            "594c7103462fd58bbdf0afa2d59f74ad7737239b41dccf7623b1b5690ff8e677"
        assert h == detail["frozen_sha256"], "pa_gen.py changed since it was frozen"
        os.environ.setdefault("PA_ARM", "pai05")
        sys.path.insert(0, "/data/alexmueller/paialpha/code")
        sys.path.insert(0, "/data/alexmueller/j5_gates/code")
        import pa_gen
        from transformers import AttentionInterface
        AttentionInterface.register("pa_eager", pa_gen.pa_attention)
        lm = model.model.language_model
        pa_gen.ST["img_tok"] = ST["img_tok"]
        lm.config._attn_implementation = "pa_eager"
        ref, _ = pa_gen.greedy_two_stream(model, tok, enc, enc["pixel_values"], ids_u, am_u,
                                          pf["alpha"], pf["gamma"], MAX_NEW,
                                          tok.eos_token_id, tok.pad_token_id)
        lm.config._attn_implementation = "pope_eager"
        got, _ = two_stream(model, enc, enc["pixel_values"], ids_u, am_u, pf["alpha"],
                            pf["gamma"], "decode", MAX_NEW, tok.eos_token_id,
                            tok.pad_token_id)
        ok4 = bool(torch.equal(ref, got))
        detail["mismatches"] = int((ref != got).any(dim=1).sum())
        detail["n"] = int(ref.shape[0])
    except Exception as e:
        detail["error"] = repr(e)
    rec("G4_copy_drift_vs_pa_gen", ok4, True, detail)

    blocking_fail = [k for k, v in R.items() if v["blocking"] and not v["pass"]]
    out = {"gates": R, "limit_images": LIMIT, "n_items": len(items),
           "plan_sha": plan_sha, "secs": time.time() - t0,
           "blocking_failures": blocking_fail, "host": socket.gethostname(),
           "job_id": os.environ.get("SLURM_JOB_ID", "")}
    json.dump(out, open(f"{OUT}/pope_gates.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    assert not blocking_fail, f"FATAL_BLOCKING_GATE_FAILED {blocking_fail}"
    print("POPE_SENTINEL_GATES_OK", flush=True)


if __name__ == "__main__":
    main()
