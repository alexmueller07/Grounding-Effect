"""PAI alpha-misconfiguration test -- generation for one arm.

Pre-registered in CAMPAIGN/PAI_ALPHA_PREREG.md (commit 1f47296), committed before any
generation for this experiment existed.

THE INTERVENTION IS PRIOR WORK AND IS NOT CLAIMED. Both stages are PAI's
(arXiv:2407.21771v1, ECCV 2024), verified at source:

  Eq. 3, attention amplification, pre-softmax, image key columns, last query row:
      A~[n, j] <- A~[n, j] + alpha * |A~[n, j]|      for j = m+1 .. m+n_V

  Eq. 4, image-centric logit refine (CFG.py in the released repo):
      out = gamma * (log p_cond - log p_uncond) + log p_uncond
      out = out.masked_fill(log p_cond < log(0.1) + max(log p_cond), -inf)
  with a short-circuit `if guidance_scale == 1: return log_softmax(scores)` that makes
  gamma == 1 PAI's own exact no-op.

WHY A HAND-WRITTEN DECODE LOOP. PAI's second stage needs a parallel unconditional stream
(same prompt, image span deleted) advanced in lockstep with the conditional one. Writing the
loop explicitly makes both streams auditable and, more importantly, lets ALL THREE ARMS share
one execution path: alpha = 0.0 is `sc + 0.0*|sc|` which is `sc` exactly in IEEE, and
gamma = 1.0 takes PAI's own short-circuit. The vanilla arm is therefore a kernel-matched and
loop-matched no-op, not a different code path. This project has had a baseline diverge at
token 6 purely from running a different path from the gated arms. pa_gates.py checks the loop
against HF `generate` token-for-token before any arm is trusted.

CAUSALITY. `transformers` hands a registered AttentionInterface `attention_mask=None` and
expects the kernel to enforce causality itself. A kernel that honours only the mask it is
handed attends BIDIRECTIONALLY during prefill; this project retracted eight claims to that
bug. The repair below is mandatory and counted; pa_gates.py measures residual future-key mass.

ENV
  PA_ARM     vanilla | pai02 | pai05      (alpha/gamma come from pa_common.ARMS)
  PA_LIMIT   optional int, first N images (gates / smoke only)
  PA_BS      batch size, default 8
  SMOKE      1 to tag output as a smoke run

Writes {OUT}/{tag}.jsonl and {tag}_meta.json. Sentinel PA_SENTINEL_GEN_OK.
"""
import os, sys, json, time, socket

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/alexmueller/j5_gates/code")

import numpy as np
import torch
import torch.nn.functional as F

from pa_common import (MODEL, OUT, SYN, N_IMAGES, SAMPLE_SEED, MAX_NEW, LAYER_LO, LAYER_HI,
                       CFG_PLAUS, ARMS, prompt_text, arm_tag, BS)          # noqa
from j5_common import (load_coco_sample, ChairScorer, JsonlWriter, image_token_id)  # noqa

assert socket.gethostname().startswith("vgi1"), f"FATAL_WRONG_NODE {socket.gethostname()}"

ARM = os.environ["PA_ARM"]
assert ARM in ARMS, f"FATAL_UNKNOWN_ARM {ARM}"
ALPHA = ARMS[ARM]["alpha"]
GAMMA = ARMS[ARM]["gamma"]
LIMIT = int(os.environ.get("PA_LIMIT", "0"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
TAG = arm_tag(ARM, SMOKE, LIMIT)
N_LAYERS = 32

# HARD BOUND. At alpha >= 1 the edit maps a masked column (filled with finfo.min, which IS
# finite) to >= 0 and silently un-masks padded / future keys. The registered grid tops at 0.5.
assert 0.0 <= ALPHA < 1.0, f"FATAL_ALPHA_OUT_OF_BOUND {ALPHA}"

ST = {"active": False, "cond": True, "step": -1, "vis": None, "alpha": 0.0,
      "calls": 0, "applied": 0, "n_edited": 0, "mask_present": 0, "causal_repairs": 0,
      "nomask_decode": 0, "mask_leak": 0, "check_mask": True,
      "mass": None, "cnt": 0, "future_mass": 0.0}


def pa_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    from transformers.models.llama.modeling_llama import repeat_kv
    key_s = repeat_kv(key, module.num_key_value_groups)
    val_s = repeat_kv(value, module.num_key_value_groups)
    sc = torch.matmul(query, key_s.transpose(2, 3)) * scaling
    if attention_mask is not None:
        m = attention_mask
        if m.shape[-1] != key_s.shape[-2]:
            m = m[..., : key_s.shape[-2]]
        sc = sc + m
        ST["mask_present"] += 1
    elif query.shape[2] > 1:
        # CAUSALITY REPAIR -- mandatory, see module docstring.
        _q, _K = query.shape[2], key_s.shape[2]
        _i = torch.arange(_q, device=query.device).view(-1, 1) + (_K - _q)
        _j = torch.arange(_K, device=query.device).view(1, -1)
        sc = sc.masked_fill(_j > _i, torch.finfo(sc.dtype).min)
        ST["causal_repairs"] += 1
    else:
        ST["nomask_decode"] += 1
    ST["calls"] += 1

    lidx = getattr(module, "layer_idx", None)
    decode = (query.shape[2] == 1)

    # PAI Eq. 3. Conditional stream only, decode steps only, layers [LAYER_LO, LAYER_HI).
    # Decode-only reproduces the released code: chair_eval.py's guard is
    # `use_attn and not use_cfg`, and llama_modify sets use_cfg=True, which CFGLogits only
    # clears inside its first __call__ -- so with the README's `--use-attn --use-cfg` the
    # prompt-encoding pass is never amplified. Identical in both PAI arms, so it cannot
    # confound the primary contrast.
    do_pai = (ST["active"] and ST["cond"] and decode and ST["alpha"] > 0.0
              and lidx is not None and LAYER_LO <= lidx < LAYER_HI)
    if do_pai:
        K = sc.shape[-1]
        vm = ST["vis"]
        if vm.shape[1] < K:
            vm = F.pad(vm, (0, K - vm.shape[1]))
        vcol = vm[:, None, None, :].bool()
        # MASKED-COLUMN GUARD. Exclude masked entries BY VALUE, not by finiteness:
        # finfo.min is finite, so an isfinite() test would let the edit resurrect a masked
        # column (s -> s*(1-alpha) for s < 0, which lands on 0 at alpha == 1).
        live = sc > (torch.finfo(sc.dtype).min / 2.0)
        safe = live & vcol
        sc = torch.where(safe, sc + ST["alpha"] * sc.abs(), sc)
        if ST["check_mask"]:
            still = sc > (torch.finfo(sc.dtype).min / 2.0)
            ST["mask_leak"] += int((still & (~live)).sum())
            ST["check_mask"] = False
        if lidx == LAYER_LO:
            ST["applied"] += 1
            if ST["n_edited"] == 0:
                ST["n_edited"] = int(safe.sum())

    w = torch.softmax(sc, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(w, val_s)

    # G4 monitor: decode-time image-token attention mass on the conditional stream. Free --
    # one reduction per layer per step, no extra generation.
    if ST["active"] and ST["cond"] and decode and lidx is not None and ST["mass"] is not None:
        K = w.shape[-1]
        vm = ST["vis"]
        if vm.shape[1] < K:
            vm = F.pad(vm, (0, K - vm.shape[1]))
        m = w.float().mul(vm[:, None, None, :].to(torch.float32)).sum(-1).mean(dim=(1, 2))
        ST["mass"][:, lidx] += m
        if lidx == 0:
            ST["cnt"] += 1
    return out.transpose(1, 2).contiguous(), w


def load_model_pa():
    from transformers import AutoProcessor, LlavaForConditionalGeneration, AttentionInterface
    AttentionInterface.register("pa_eager", pa_attention)
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"
    if proc.tokenizer.pad_token is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True).eval()
    lm = model.model.language_model
    lm.config._attn_implementation = "pa_eager"
    try:
        model.config.text_config._attn_implementation = "pa_eager"
    except Exception:
        pass
    return model, proc, proc.tokenizer


def pos_ids(am):
    """Left-padding-correct position ids, identical to what HF's
    `prepare_inputs_for_generation` builds. Calling `model()` directly bypasses that helper,
    and Llama would otherwise fall back to `cache_position`, which counts PAD tokens as real
    positions and shifts RoPE for every left-padded row. Without this the padded rows of a
    batch decode differently from the same image run alone -- a silent, batch-shape-dependent
    corruption of exactly the kind this project has already been bitten by."""
    p = am.long().cumsum(-1) - 1
    return p.masked_fill(am == 0, 1)


def combine(logits_c, logits_u, gamma):
    """PAI Eq. 4 exactly as CFG.py implements it, including the gamma == 1 short-circuit
    that returns the conditional log-probs untouched and skips the plausibility cutoff."""
    scores = F.log_softmax(logits_c.float(), dim=-1)
    if gamma == 1.0:
        return scores
    uncond = F.log_softmax(logits_u.float(), dim=-1)
    cutoff = float(np.log(CFG_PLAUS)) + scores.max(dim=-1, keepdim=True).values
    out = gamma * (scores - uncond) + uncond
    return out.masked_fill(scores < cutoff, -float("inf"))


@torch.no_grad()
def greedy_two_stream(model, tok, enc_c, pix, ids_u, am_u, alpha, gamma, max_new,
                      eos_id, pad_id, track_mass=True):
    """Lockstep greedy decode of the conditional (image) and unconditional (no-image)
    streams. The SAME chosen token is appended to both, which is what makes log p_uncond the
    text-only continuation probability PAI's Eq. 4 subtracts."""
    ids_c, am_c = enc_c["input_ids"], enc_c["attention_mask"]
    B = ids_c.shape[0]
    ST["alpha"] = alpha
    ST["vis"] = (ids_c == ST["img_tok"])
    assert int(ST["vis"].sum(1).min()) == 576, "FATAL_IMAGE_BLOCK_NOT_576"   # G5
    ST["mass"] = torch.zeros((B, N_LAYERS), device=ids_c.device) if track_mass else None
    ST["cnt"] = 0
    ST["active"] = True
    ST["step"] = -1

    # ---- prefill, both streams
    ST["cond"] = True
    out_c = model(input_ids=ids_c, attention_mask=am_c, position_ids=pos_ids(am_c),
                  pixel_values=pix, use_cache=True)
    cache_c = out_c.past_key_values
    ST["cond"] = False
    out_u = model(input_ids=ids_u, attention_mask=am_u, position_ids=pos_ids(am_u),
                  use_cache=True)
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


def build_uncond(ids_c, am_c, img_tok, pad_id):
    """PAI's negative prompt: the same prompt with the image span deleted
    (model_loader.py `neg_promt = torch.cat([bos, token_before, token_after], dim=1)`),
    re-left-padded so the batch is rectangular."""
    rows = []
    for i in range(ids_c.shape[0]):
        keep = [int(t) for t, a, v in zip(ids_c[i].tolist(), am_c[i].tolist(),
                                          (ids_c[i] == img_tok).tolist()) if a == 1 and not v]
        rows.append(keep)
    L = max(len(r) for r in rows)
    ids = torch.tensor([[pad_id] * (L - len(r)) + r for r in rows], device=ids_c.device)
    am = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in rows], device=ids_c.device)
    return ids, am


def main():
    from PIL import Image
    print(f"[pa_gen] host={socket.gethostname()} arm={ARM} alpha={ALPHA} gamma={GAMMA} "
          f"layers=[{LAYER_LO},{LAYER_HI}) max_new={MAX_NEW} tag={TAG}", flush=True)
    chair = ChairScorer(SYN)
    data = load_coco_sample(chair, n=N_IMAGES, seed=SAMPLE_SEED)
    # ONE deterministic order in every arm, so batch composition is byte-identical across
    # arms. bf16 kernels are batch-shape sensitive; a different batch shape has already
    # broken a replay criterion in this project.
    data.sort(key=lambda d: d["image_id"])
    if LIMIT:
        data = data[:LIMIT]
    assert data, "FATAL_NO_ITEMS"
    print(f"[pa_gen] {len(data)} images, sample_seed={SAMPLE_SEED}", flush=True)

    model, proc, tok = load_model_pa()
    ST["img_tok"] = image_token_id(model)
    pad = tok.pad_token_id
    eos = tok.eos_token_id
    w = JsonlWriter(f"{OUT}/{TAG}.jsonl")
    t0 = time.time()
    n_trunc = 0
    for b0 in range(0, len(data), BS):
        chunk = data[b0:b0 + BS]
        ims = [Image.open(d["file"]).convert("RGB") for d in chunk]
        enc = proc(images=ims, text=[prompt_text()] * len(chunk), return_tensors="pt",
                   padding=True)
        enc = {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda"))
               for k, v in enc.items()}
        pix = enc["pixel_values"]
        ids_u, am_u = build_uncond(enc["input_ids"], enc["attention_mask"], ST["img_tok"], pad)
        gen, mass = greedy_two_stream(model, tok, enc, pix, ids_u, am_u, ALPHA, GAMMA,
                                      MAX_NEW, eos, pad)
        for i, d in enumerate(chunk):
            ids = [int(x) for x in gen[i].tolist()]
            hit_eos = eos in ids
            ids = ids[:ids.index(eos)] if hit_eos else ids
            ids = [x for x in ids if x != pad]
            text = tok.decode(ids, skip_special_tokens=True)
            trunc = (not hit_eos) and len(ids) >= MAX_NEW
            n_trunc += int(trunc)
            w.write({"image_id": d["image_id"], "arm": ARM, "alpha": ALPHA, "gamma": GAMMA,
                     "layers": [LAYER_LO, LAYER_HI], "text": text, "gen_ids": ids,
                     "n_tokens": len(ids), "hit_eos": bool(hit_eos), "truncated": bool(trunc),
                     "gt": d["gt"], "img_mass": [float(x) for x in mass[i]]})
        done = min(b0 + BS, len(data))
        print(f"[pa_gen] {done}/{len(data)} imgs  {time.time()-t0:.0f}s  trunc={n_trunc}",
              flush=True)
    w.close()
    meta = {"arm": ARM, "alpha": ALPHA, "gamma": GAMMA, "layers": [LAYER_LO, LAYER_HI],
            "model": MODEL, "n": len(data), "max_new": MAX_NEW, "sample_seed": SAMPLE_SEED,
            "n_truncated": n_trunc, "prompt": prompt_text(), "bs": BS,
            "attn_calls": ST["calls"], "pai_applied_steps": ST["applied"],
            "n_edited_first": ST["n_edited"], "mask_present": ST["mask_present"],
            "causal_repairs": ST["causal_repairs"], "nomask_decode": ST["nomask_decode"],
            "mask_leak": ST["mask_leak"], "host": socket.gethostname(),
            "job_id": os.environ.get("SLURM_JOB_ID", ""), "secs": time.time() - t0,
            "operator_is_prior_work": "arXiv 2407.21771v1 Eq.3 + Eq.4",
            "prereg": "CAMPAIGN/PAI_ALPHA_PREREG.md @ 1f47296"}
    assert ST["mask_leak"] == 0, f"FATAL_MASK_LEAK {ST['mask_leak']}"
    assert ST["mask_present"] + ST["causal_repairs"] > 0, "FATAL_NO_PREFILL_SEEN"
    if ALPHA > 0:
        assert ST["applied"] > 0, "FATAL_PAI_NEVER_APPLIED"
    json.dump(meta, open(f"{OUT}/{TAG}_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1), flush=True)
    print("PA_SENTINEL_GEN_OK", flush=True)


if __name__ == "__main__":
    main()
