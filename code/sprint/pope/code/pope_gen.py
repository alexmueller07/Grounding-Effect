"""POPE lane -- generation for one (method, pixel arm) cell.

Pre-registered in SPRINT/POPE_PREREG.md, committed before any generation row existed.

ENV
  POPE_METHOD   vanilla | pai_full | pai_attn        (pope_common.METHODS)
  POPE_PIXARMS  comma-separated pixel arms           (pope_pixels.PIXEL_ARMS)
  POPE_SPLITS   comma-separated, default all three
  POPE_LIMIT    optional int, first N images (gates / smoke only)
  POPE_BS       batch size, default 16

Writes {OUT}/pope_{method}_{pixarm}.jsonl and _meta.json per pixel arm.
Sentinel POPE_SENTINEL_GEN_OK after the LAST requested arm.
"""
import hashlib
import json
import os
from collections import Counter
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

import pope_pixels
from pope_common import (BS, MAX_NEW, METHODS, MODEL, OUT, SPLITS, prompt_text, tag,
                         LAYER_LO, LAYER_HI)
from pope_items import batch_plan, load_items
from pope_kernel import ST, build_uncond, load_model_pa, two_stream

assert socket.gethostname().startswith("vgi1"), f"FATAL_WRONG_NODE {socket.gethostname()}"


def encode(proc, chunk_items, pil_cache):
    ims = [pil_cache[it["image"]] for it in chunk_items]
    txt = [prompt_text(it["text"]) for it in chunk_items]
    enc = proc(images=ims, text=txt, return_tensors="pt", padding=True)
    enc = {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda"))
           for k, v in enc.items()}
    # ZERO-PADDING GATE. The batch plan groups by exact token length, so a pad here means the
    # plan and the processor disagree and every downstream token-identity claim is void.
    assert bool(enc["attention_mask"].all()), "FATAL_BATCH_HAS_PADDING"
    return enc


def run_arm(model, proc, tok, items, plan, pixarm, method, spec, meta_common):
    pil_cache = {}
    deran = None
    if pixarm == "mismatch":
        # A DIFFERENT real COCO image, fixed seeded derangement, recorded in the meta.
        by_id = {it["image_id"]: it for it in items}
        deran = pope_pixels.derangement(by_id.keys())
        for it in items:
            if it["image"] not in pil_cache:
                src = by_id[deran[it["image_id"]]]
                assert src["image_id"] != it["image_id"], "FATAL_MISMATCH_SELF_PAIR"
                with Image.open(src["file"]) as im:
                    pil_cache[it["image"]] = im.convert("RGB")
    else:
        for it in items:
            if it["image"] not in pil_cache:
                with Image.open(it["file"]) as im:
                    pil_cache[it["image"]] = pope_pixels.make(pixarm, im.convert("RGB"),
                                                              it["image_id"])
    assert len(pil_cache) == len({it["image"] for it in items}), "FATAL_PIL_CACHE"

    for k in ("calls", "applied", "applied_prefill", "n_edited", "mask_present",
              "causal_repairs", "nomask_decode", "mask_leak"):
        ST[k] = 0
    ST["check_mask"] = True

    pad, eos = tok.pad_token_id, tok.eos_token_id
    T = tag(method, pixarm)
    rows = []
    t0 = time.time()
    n_trunc = 0
    grey_ref = None
    mass_acc = np.zeros(32)
    mass_n = 0
    for bi, idx in enumerate(plan):
        chunk = [items[i] for i in idx]
        enc = encode(proc, chunk, pil_cache)
        pix = enc["pixel_values"]
        if pixarm == "grey":
            # The grey field resizes to the SAME 336x336 tensor whatever the source size, so
            # every blind row must be bit-identical. Checked on the real tensors.
            if grey_ref is None:
                grey_ref = pix[0].clone()
            assert bool((pix == grey_ref[None]).all()), "FATAL_GREY_PIX_ROWS_DIFFER"
        ids_u, am_u = build_uncond(enc["input_ids"], enc["attention_mask"], ST["img_tok"], pad)
        gen, mass = two_stream(model, enc, pix, ids_u, am_u, spec["alpha"], spec["gamma"],
                               spec["rowmode"], MAX_NEW, eos, pad)
        mass_acc += mass.mean(axis=0)
        mass_n += 1
        for j, it in enumerate(chunk):
            ids = [int(x) for x in gen[j].tolist()]
            hit_eos = eos in ids
            ids = ids[:ids.index(eos)] if hit_eos else ids
            ids = [x for x in ids if x != pad]
            text = tok.decode(ids, skip_special_tokens=True)
            trunc = (not hit_eos) and len(ids) >= MAX_NEW
            n_trunc += int(trunc)
            pid = enc["input_ids"][j].tolist()
            rows.append({
                "split": it["split"], "question_id": it["question_id"],
                "image": it["image"], "image_id": it["image_id"],
                "label": it["label"], "question": it["text"],
                "method": method, "pixarm": pixarm,
                "answer": text, "gen_ids": ids, "n_tokens": len(ids),
                "hit_eos": bool(hit_eos), "truncated": bool(trunc),
                "img_mass": float(mass[j].mean()),
                "pid_sha": hashlib.sha256(
                    json.dumps(pid).encode()).hexdigest()[:16],
            })
        if bi % 50 == 0 or bi == len(plan) - 1:
            print(f"[pope_gen] {T} batch {bi+1}/{len(plan)} rows={len(rows)} "
                  f"{time.time()-t0:.0f}s trunc={n_trunc}", flush=True)

    rows.sort(key=lambda r: (r["split"], r["question_id"]))
    with open(f"{OUT}/{T}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = dict(meta_common)
    meta.update({
        "tag": T, "method": method, "pixarm": pixarm,
        "alpha": spec["alpha"], "gamma": spec["gamma"], "rowmode": spec["rowmode"],
        "n_rows": len(rows), "n_truncated": n_trunc,
        "trunc_rate": n_trunc / max(1, len(rows)),
        "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])),
        "median_tokens": float(np.median([r["n_tokens"] for r in rows])),
        "empty_rate": float(np.mean([r["n_tokens"] == 0 for r in rows])),
        "n_distinct_answers": len({r["answer"] for r in rows}),
        # Full answer distribution, per arm and per split.  POPE's own paper reports a
        # yes-ratio; a blinded arm that answers "yes" to nearly everything is a degenerate
        # regime, not a score, and only the raw distribution shows that.
        "answer_dist": Counter(r["answer"] for r in rows).most_common(20),
        "answer_dist_by_split": {
            s: Counter(r["answer"] for r in rows if r["split"] == s).most_common(10)
            for s in sorted({r["split"] for r in rows})},
        "derangement": ({str(k): v for k, v in sorted(deran.items())}
                        if deran is not None else None),
        "mean_img_mass": float(np.mean([r["img_mass"] for r in rows])),
        "img_mass_by_layer": [float(x) for x in (mass_acc / max(1, mass_n))],
        "attn_calls": ST["calls"], "pai_applied_steps": ST["applied"],
        "pai_applied_prefill": ST["applied_prefill"], "n_edited_first": ST["n_edited"],
        "mask_present": ST["mask_present"], "causal_repairs": ST["causal_repairs"],
        "nomask_decode": ST["nomask_decode"], "mask_leak": ST["mask_leak"],
        "pixel_arm_desc": pope_pixels.PIXEL_ARMS[pixarm][2],
        "pixel_arm_frame": pope_pixels.PIXEL_ARMS[pixarm][1],
        "secs": time.time() - t0,
    })
    assert ST["mask_leak"] == 0, f"FATAL_MASK_LEAK {ST['mask_leak']}"
    assert ST["mask_present"] + ST["causal_repairs"] > 0, "FATAL_NO_PREFILL_SEEN"
    if spec["alpha"] > 0:
        assert ST["applied"] > 0, "FATAL_PAI_NEVER_APPLIED"
        if spec["rowmode"] == "lastrow":
            assert ST["applied_prefill"] > 0, "FATAL_PAI_NEVER_APPLIED_AT_PREFILL"
        else:
            assert ST["applied_prefill"] == 0, "FATAL_PAI_APPLIED_AT_PREFILL_IN_DECODE_MODE"
    json.dump(meta, open(f"{OUT}/{T}_meta.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "img_mass_by_layer"}), flush=True)
    return meta


def main():
    method = os.environ["POPE_METHOD"]
    assert method in METHODS, f"FATAL_UNKNOWN_METHOD {method}"
    spec = METHODS[method]
    pixarms = os.environ["POPE_PIXARMS"].split(",")
    for a in pixarms:
        assert a in pope_pixels.PIXEL_ARMS, f"FATAL_UNKNOWN_PIXARM {a}"
    splits = tuple(os.environ.get("POPE_SPLITS", ",".join(SPLITS)).split(","))
    limit = int(os.environ.get("POPE_LIMIT", "0"))
    assert 0.0 <= spec["alpha"] < 1.0, f"FATAL_ALPHA_OUT_OF_BOUND {spec['alpha']}"

    os.makedirs(OUT, exist_ok=True)
    items = load_items(splits, limit_images=limit)
    print(f"[pope_gen] host={socket.gethostname()} method={method} arms={pixarms} "
          f"splits={splits} items={len(items)} limit={limit} max_new={MAX_NEW} bs={BS} "
          f"alpha={spec['alpha']} gamma={spec['gamma']} rowmode={spec['rowmode']}", flush=True)

    model, proc, tok = load_model_pa()
    ST["img_tok"] = getattr(model.config, "image_token_index", None) or \
        getattr(model.config, "image_token_id", None)
    assert ST["img_tok"] is not None, "FATAL_NO_IMAGE_TOKEN"
    plan, plan_sha, len_hist = batch_plan(items, tok, BS)
    print(f"[pope_gen] plan: {len(plan)} batches, sha={plan_sha[:16]}, "
          f"len_hist={len_hist}", flush=True)

    meta_common = {"model": MODEL, "prompt_template": prompt_text("<QUESTION>"),
                   "max_new": MAX_NEW, "bs": BS, "splits": list(splits),
                   "layers": [LAYER_LO, LAYER_HI], "n_items": len(items),
                   "plan_sha": plan_sha, "plan_batches": len(plan),
                   "pix_seed": pope_pixels.PIX_SEED, "grey_rgb": list(pope_pixels.GRAY),
                   "host": socket.gethostname(), "limit_images": limit,
                   "job_id": os.environ.get("SLURM_JOB_ID", ""),
                   "torch": torch.__version__,
                   "operator_is_prior_work": "arXiv 2407.21771 Eq.3 (+Eq.4 when gamma!=1)",
                   "prereg": "SPRINT/POPE_PREREG.md"}
    for a in pixarms:
        run_arm(model, proc, tok, items, plan, a, method, spec, meta_common)
    print("POPE_SENTINEL_GEN_OK", flush=True)


if __name__ == "__main__":
    main()
