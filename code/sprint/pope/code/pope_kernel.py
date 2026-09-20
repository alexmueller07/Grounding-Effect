"""POPE lane -- the PAI kernel and two-stream decode loop.

THIS IS A FROZEN COPY of /data/alexmueller/paialpha/code/pa_gen.py
(sha256 594c7103462fd58bbdf0afa2d59f74ad7737239b41dccf7623b1b5690ff8e677, frozen under
code/_frozen_refs/).  It is copied rather than imported because on this cluster a queued job
runs whatever is on disk when it STARTS, so importing another lane's live module would let an
edit there silently change a job queued here.  Gate G_COPY asserts token-identity against the
original on a probe before any endpoint arm runs.

THE INTERVENTION IS PRIOR WORK AND IS NOT CLAIMED.  Both stages are PAI's (arXiv:2407.21771,
ECCV 2024), verified against LALBJ/PAI@master:

  Eq. 3, attention amplification, pre-softmax, image key columns, LAST QUERY ROW
  (attention.py L90-93):
      if use_attn and not use_cfg:
          attn_weights[:, :, -1, img_start:img_end] += alpha * attn_weights[...].abs()

  Eq. 4, image-centric logit refine (CFG.py L28-52):
      out = gamma * (log p_cond - log p_uncond) + log p_uncond
      out = out.masked_fill(log p_cond < log(0.1) + max(log p_cond), -inf)
  with `if self.guidance_scale == 1: return scores`, PAI's own exact no-op at gamma == 1.

THE ONE ADDITION over pa_gen.py is ST["rowmode"]:

  "decode"  reproduces the released `--use-attn --use-cfg` path exactly.  llama_modify sets
            self_attn.use_cfg = True; the guard is `use_attn and not use_cfg`; CFGLogits only
            clears use_cfg inside its first __call__, which runs on the PREFILL logits.  So
            the conditional prefill pass is never amplified and THE FIRST GENERATED TOKEN IS
            NEVER AMPLIFIED.  pa_gen.py implements exactly this and says so.
  "lastrow" reproduces the released `--use-attn` path alone (use_cfg False from the start):
            the last query row of EVERY conditional forward is amplified, prefill included.

CAUSALITY.  `transformers` hands a registered AttentionInterface `attention_mask=None` and
expects the kernel to enforce causality itself.  A kernel honouring only the mask it is handed
attends BIDIRECTIONALLY during prefill; this project retracted eight claims to that bug.  The
repair below is mandatory and counted.
"""
import numpy as np
import torch
import torch.nn.functional as F

from pope_common import MODEL, CFG_PLAUS

N_LAYERS = 32

ST = {"active": False, "cond": True, "step": -1, "vis": None, "alpha": 0.0,
      "rowmode": "decode", "img_tok": None,
      "calls": 0, "applied": 0, "applied_prefill": 0, "n_edited": 0,
      "mask_present": 0, "causal_repairs": 0, "nomask_decode": 0, "mask_leak": 0,
      "check_mask": True, "mass": None, "cnt": 0}


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
    from pope_common import LAYER_LO, LAYER_HI

    # PAI Eq. 3.  Conditional stream only, layers [LAYER_LO, LAYER_HI), last query row.
    row_ok = decode or (ST["rowmode"] == "lastrow")
    do_pai = (ST["active"] and ST["cond"] and row_ok and ST["alpha"] > 0.0
              and lidx is not None and LAYER_LO <= lidx < LAYER_HI)
    if do_pai:
        K = sc.shape[-1]
        vm = ST["vis"]
        if vm.shape[1] < K:
            vm = F.pad(vm, (0, K - vm.shape[1]))
        vcol = vm[:, None, None, :].bool()                       # (B,1,1,K) image key columns
        Q = sc.shape[-2]
        qrow = torch.zeros((1, 1, Q, 1), dtype=torch.bool, device=sc.device)
        qrow[..., -1, :] = True                                  # LAST QUERY ROW only
        # MASKED-COLUMN GUARD.  Exclude masked entries BY VALUE, not by finiteness:
        # finfo.min is finite, so an isfinite() test would let the edit resurrect a masked
        # column (s -> s*(1-alpha) for s < 0, which lands on 0 at alpha == 1).
        live = sc > (torch.finfo(sc.dtype).min / 2.0)
        safe = live & vcol & qrow
        sc = torch.where(safe, sc + ST["alpha"] * sc.abs(), sc)
        if ST["check_mask"]:
            still = sc > (torch.finfo(sc.dtype).min / 2.0)
            ST["mask_leak"] += int((still & (~live)).sum())
            ST["check_mask"] = False
        if lidx == LAYER_LO:
            ST["applied"] += 1
            if not decode:
                ST["applied_prefill"] += 1
            if ST["n_edited"] == 0:
                ST["n_edited"] = int(safe.sum())

    w = torch.softmax(sc, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(w, val_s)

    # Monitor: image-token attention mass on the conditional stream, decode steps.
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
    AttentionInterface.register("pope_eager", pa_attention)
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"
    if proc.tokenizer.pad_token is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True).eval()
    lm = model.model.language_model
    lm.config._attn_implementation = "pope_eager"
    try:
        model.config.text_config._attn_implementation = "pope_eager"
    except Exception:
        pass
    return model, proc, proc.tokenizer


def pos_ids(am):
    """Left-padding-correct position ids, identical to what HF's
    `prepare_inputs_for_generation` builds.  Calling `model()` directly bypasses that helper,
    and Llama would otherwise fall back to `cache_position`, which counts PAD tokens as real
    positions and shifts RoPE for every left-padded row."""
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


@torch.no_grad()
def two_stream(model, enc_c, pix, ids_u, am_u, alpha, gamma, rowmode, max_new,
               eos_id, pad_id, track_mass=True):
    """Lockstep greedy decode of the conditional (image) and unconditional (no-image) streams.
    LINE-FOR-LINE mirror of pa_gen.greedy_two_stream; the only addition is `rowmode`, which is
    passed to the kernel through ST.  The SAME chosen token is appended to both streams, which
    is what makes log p_uncond the text-only continuation probability Eq. 4 subtracts."""
    ids_c, am_c = enc_c["input_ids"], enc_c["attention_mask"]
    B = ids_c.shape[0]
    ST["alpha"] = alpha
    ST["rowmode"] = rowmode
    ST["vis"] = (ids_c == ST["img_tok"])
    assert int(ST["vis"].sum(1).min()) == 576, "FATAL_IMAGE_BLOCK_NOT_576"
    ST["mass"] = torch.zeros((B, N_LAYERS), device=ids_c.device) if track_mass else None
    ST["cnt"] = 0
    ST["active"] = True
    ST["step"] = -1

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
