"""Mitigation-method CHAIR decomposition -- generation for one arm.

Pre-registered in CAMPAIGN/MITIGATION_DECOMP_PREREG.md (commit ce3ed30), committed before any
generation for this experiment existed.

THE INTERVENTIONS ARE PRIOR WORK AND ARE NOT CLAIMED HERE.

  PAI (arXiv:2407.21771v1, ECCV 2024)
    Eq. 3  attention amplification, pre-softmax, image key columns:
             A~[n, j] <- A~[n, j] + alpha * |A~[n, j]|   for j in the image span
    Eq. 4  image-centric logit refine (CFG.py):
             out = gamma * (log p_cond - log p_second) + log p_second
             out = out.masked_fill(log p_cond < log(0.1) + max(log p_cond), -inf)
           with a short-circuit `if guidance_scale == 1: return log_softmax(scores)` that
           makes gamma == 1 PAI's own exact no-op.
    PAI's second stream is the SAME prompt with the image span deleted.

  VCD (arXiv:2311.16922) is NOT run -- see md_common.py and pre-registration Amendment 1.
  It has no published CHAIR evaluation, its published decoding is multinomial sampling and
  its released code cannot run greedy at all, its paper and repo disagree on beta, on T and
  on the noise model itself, and the release was a no-op past token 1 for its first eight
  months. Approximating it was forbidden by the pre-registration; skipping it is the
  registered action.

WHY THIS FILE REUSES pa_gen RATHER THAN RE-IMPLEMENTING. pa_gen.py's kernel (`pa_attention`,
including the mandatory causality repair) and its two-stream decode loop are already gate-
verified. Re-typing them would create a second thing that can drift. The ONLY addition here is
the grey-image arm, which changes the pixels and nothing else; gate G2c asserts that
md_two_stream is token-identical to pa_gen.greedy_two_stream on real images.

CAUSALITY. `transformers` hands a registered AttentionInterface `attention_mask=None` and
expects the kernel to enforce causality itself. A kernel honouring only the mask it is handed
attends BIDIRECTIONALLY during prefill; this project retracted eight claims to that bug. The
repair lives in pa_gen.pa_attention and is counted and gated.

ENV
  MD_ARM     one of md_common.ARMS
  MD_LIMIT   optional int, first N images (gates / smoke only)
  MD_BS      batch size, default 8

Writes {OUT}/{tag}.jsonl and {tag}_meta.json. Sentinel MD_SENTINEL_GEN_OK.
"""
import os, sys, json, time, socket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/alexmueller/paialpha/code")
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")

# pa_gen reads os.environ["PA_ARM"] at import time and would raise KeyError without it. We
# never use pa_gen's own ARM/ALPHA/GAMMA module globals -- every parameter is passed in
# explicitly below -- so this only satisfies the import.
os.environ.setdefault("PA_ARM", "vanilla")

import numpy as np
import torch
import torch.nn.functional as F

from md_common import (MODEL, OUT, SYN, N_IMAGES, SAMPLE_SEED, MAX_NEW, LAYER_LO, LAYER_HI,
                       CFG_PLAUS, ARMS, prompt_text, arm_tag, BS)                   # noqa
import pa_gen
from pa_gen import ST, load_model_pa, build_uncond, pos_ids, combine                # noqa
from wia_blind_common import gray_image, GRAY                                       # noqa
from j5_common import load_coco_sample, ChairScorer, JsonlWriter, image_token_id    # noqa

assert socket.gethostname().startswith("vgi1"), f"FATAL_WRONG_NODE {socket.gethostname()}"

N_LAYERS = 32


# ------------------------------------------------------------------ the decode loop
@torch.no_grad()
def md_two_stream(model, tok, enc_c, pix, ids_u, am_u, alpha, gamma, max_new, eos_id, pad_id,
                  pix_u=None, track_mass=True):
    """Lockstep greedy decode of the conditional stream and a second stream.

    LINE-FOR-LINE mirror of pa_gen.greedy_two_stream. `pix_u` (the second stream's pixel
    values) is retained as an explicit argument so the second stream's content is visible at
    the call site rather than implied; every arm run here passes pix_u=None, which is PAI's
    image-deleted prompt. Gate G2c asserts token-identity with pa_gen.greedy_two_stream.

    The SAME chosen token is appended to both streams, which is what makes the second stream's
    log-probability the contrastive term both methods subtract.
    """
    ids_c, am_c = enc_c["input_ids"], enc_c["attention_mask"]
    B = ids_c.shape[0]
    ST["alpha"] = alpha
    ST["vis"] = (ids_c == ST["img_tok"])
    assert int(ST["vis"].sum(1).min()) == 576, "FATAL_IMAGE_BLOCK_NOT_576"          # G5
    ST["mass"] = torch.zeros((B, N_LAYERS), device=ids_c.device) if track_mass else None
    ST["cnt"] = 0
    ST["active"] = True
    ST["step"] = -1

    kw_u = {} if pix_u is None else {"pixel_values": pix_u}

    # ---- prefill, both streams
    ST["cond"] = True
    out_c = model(input_ids=ids_c, attention_mask=am_c, position_ids=pos_ids(am_c),
                  pixel_values=pix, use_cache=True)
    cache_c = out_c.past_key_values
    ST["cond"] = False
    out_u = model(input_ids=ids_u, attention_mask=am_u, position_ids=pos_ids(am_u),
                  use_cache=True, **kw_u)
    cache_u = out_u.past_key_values

    lc, lu = out_c.logits[:, -1, :], out_u.logits[:, -1, :]
    done = torch.zeros(B, dtype=torch.bool, device=ids_c.device)
    gen = []
    for t in range(max_new):
        ST["step"] = t
        nxt = combine(lc, lu, gamma).argmax(dim=-1)
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        gen.append(nxt)
        done = done | (nxt == eos_id)
        if bool(done.all()) or t == max_new - 1:
            break
        step = nxt[:, None]
        am_c = torch.cat([am_c, (~done)[:, None].long()], dim=1)
        am_u = torch.cat([am_u, (~done)[:, None].long()], dim=1)
        ST["cond"] = True
        out_c = model(input_ids=step, attention_mask=am_c, past_key_values=cache_c,
                      position_ids=pos_ids(am_c)[:, -1:], use_cache=True)
        cache_c = out_c.past_key_values
        ST["cond"] = False
        out_u = model(input_ids=step, attention_mask=am_u, past_key_values=cache_u,
                      position_ids=pos_ids(am_u)[:, -1:], use_cache=True)
        cache_u = out_u.past_key_values
        lc, lu = out_c.logits[:, -1, :], out_u.logits[:, -1, :]

    ST["active"] = False
    mass = (ST["mass"] / max(1, ST["cnt"])).float().cpu().numpy() if track_mass else None
    return torch.stack(gen, dim=1), mass


def encode(proc, chunk, blind):
    from PIL import Image
    ims = [gray_image(d["file"]) if blind else Image.open(d["file"]).convert("RGB")
           for d in chunk]
    enc = proc(images=ims, text=[prompt_text()] * len(chunk), return_tensors="pt", padding=True)
    return {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda"))
            for k, v in enc.items()}


def main():
    ARM = os.environ["MD_ARM"]
    assert ARM in ARMS, f"FATAL_UNKNOWN_ARM {ARM}"
    spec = ARMS[ARM]
    ALPHA, GAMMA, BLIND, SECOND = (spec["alpha"], spec["gamma"], spec["blind"], spec["second"])
    LIMIT = int(os.environ.get("MD_LIMIT", "0"))
    TAG = arm_tag(ARM, LIMIT)

    # HARD BOUND. At alpha >= 1 the edit maps a masked column (filled with finfo.min, which IS
    # finite) to >= 0 and silently un-masks padded / future keys.
    assert 0.0 <= ALPHA < 1.0, f"FATAL_ALPHA_OUT_OF_BOUND {ALPHA}"

    print(f"[md_gen] host={socket.gethostname()} arm={ARM} alpha={ALPHA} gamma={GAMMA} "
          f"blind={BLIND} second={SECOND} layers=[{LAYER_LO},{LAYER_HI}) max_new={MAX_NEW} "
          f"tag={TAG}", flush=True)
    chair = ChairScorer(SYN)
    data = load_coco_sample(chair, n=N_IMAGES, seed=SAMPLE_SEED)
    # ONE deterministic order in every arm so batch composition is byte-identical across arms.
    data.sort(key=lambda d: d["image_id"])
    if LIMIT:
        data = data[:LIMIT]
    assert data, "FATAL_NO_ITEMS"
    print(f"[md_gen] {len(data)} images, sample_seed={SAMPLE_SEED}", flush=True)

    model, proc, tok = load_model_pa()
    ST["img_tok"] = image_token_id(model)
    pad, eos = tok.pad_token_id, tok.eos_token_id
    w = JsonlWriter(f"{OUT}/{TAG}.jsonl")
    t0 = time.time()
    n_trunc = 0
    blind_pix_ref = None
    for b0 in range(0, len(data), BS):
        chunk = data[b0:b0 + BS]
        enc = encode(proc, chunk, BLIND)
        pix = enc["pixel_values"]

        # G6: a flat grey of any input size resizes to the SAME 336x336 tensor, so every blind
        # row must be bit-identical. Checked here, on the real tensors, not assumed.
        if BLIND:
            if blind_pix_ref is None:
                blind_pix_ref = pix[0].clone()
            assert bool(torch.equal(pix[0], blind_pix_ref)), "FATAL_BLIND_PIX_NOT_CONSTANT"
            assert bool((pix == blind_pix_ref[None]).all()), "FATAL_BLIND_PIX_ROWS_DIFFER"

        assert SECOND == "noimg", f"FATAL_UNKNOWN_SECOND {SECOND}"
        ids_u, am_u = build_uncond(enc["input_ids"], enc["attention_mask"], ST["img_tok"], pad)
        pix_u = None

        gen, mass = md_two_stream(model, tok, enc, pix, ids_u, am_u, ALPHA, GAMMA,
                                  MAX_NEW, eos, pad, pix_u=pix_u)
        for i, d in enumerate(chunk):
            ids = [int(x) for x in gen[i].tolist()]
            hit_eos = eos in ids
            ids = ids[:ids.index(eos)] if hit_eos else ids
            ids = [x for x in ids if x != pad]
            text = tok.decode(ids, skip_special_tokens=True)
            trunc = (not hit_eos) and len(ids) >= MAX_NEW
            n_trunc += int(trunc)
            w.write({"image_id": d["image_id"], "arm": ARM, "alpha": ALPHA, "gamma": GAMMA,
                     "blind": BLIND, "second": SECOND, "layers": [LAYER_LO, LAYER_HI],
                     "text": text, "gen_ids": ids, "n_tokens": len(ids),
                     "hit_eos": bool(hit_eos), "truncated": bool(trunc), "gt": d["gt"],
                     "img_mass": [float(x) for x in mass[i]]})
        done = min(b0 + BS, len(data))
        print(f"[md_gen] {done}/{len(data)} imgs  {time.time()-t0:.0f}s  trunc={n_trunc}",
              flush=True)
    w.close()

    meta = {"arm": ARM, "alpha": ALPHA, "gamma": GAMMA, "blind": BLIND, "second": SECOND,
            "layers": [LAYER_LO, LAYER_HI], "model": MODEL, "n": len(data),
            "max_new": MAX_NEW, "sample_seed": SAMPLE_SEED, "n_truncated": n_trunc,
            "prompt": prompt_text(), "bs": BS, "gray_rgb": list(GRAY) if BLIND else None,
            "attn_calls": ST["calls"], "pai_applied_steps": ST["applied"],
            "n_edited_first": ST["n_edited"], "mask_present": ST["mask_present"],
            "causal_repairs": ST["causal_repairs"], "nomask_decode": ST["nomask_decode"],
            "mask_leak": ST["mask_leak"], "host": socket.gethostname(),
            "job_id": os.environ.get("SLURM_JOB_ID", ""), "secs": time.time() - t0,
            "vcd_dropped_reason": "no published CHAIR eval; sampling-only; paper/repo "
                                  "disagree on beta, T and the noise model; release was a "
                                  "no-op past token 1 until c637c85a (prereg Amendment 1)",
            "operator_is_prior_work": "arXiv 2407.21771v1 Eq.3 + Eq.4 (PAI)",
            "prereg": "CAMPAIGN/MITIGATION_DECOMP_PREREG.md @ ce3ed30"}
    assert ST["mask_leak"] == 0, f"FATAL_MASK_LEAK {ST['mask_leak']}"
    assert ST["mask_present"] + ST["causal_repairs"] > 0, "FATAL_NO_PREFILL_SEEN"
    if ALPHA > 0:
        assert ST["applied"] > 0, "FATAL_PAI_NEVER_APPLIED"
    json.dump(meta, open(f"{OUT}/{TAG}_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1), flush=True)
    print("MD_SENTINEL_GEN_OK", flush=True)


if __name__ == "__main__":
    main()
