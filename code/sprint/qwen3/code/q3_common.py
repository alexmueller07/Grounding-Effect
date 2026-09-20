"""SPRINT lane `qwen3` shared plumbing -- Qwen/Qwen3-VL-8B-Instruct (family 5).

Lane isolation (SPRINT_CONVENTIONS.md "Working areas"): this directory holds its OWN copies of
`j5_common.py`, `gen_common.py`, `wia_redun_common.py` and `synonyms.txt`, with the hard-coded
`/data/alexmueller/j5_gates/code` path in the latter two rewritten to this directory. A pending
job runs whatever is on disk when it STARTS, so nothing outside this lane can change a queued
job here. md5s of both originals and copies are recorded in SPRINT/QWEN3_PREREG.md section 8.0.

`f3_generate_spliced` (WIA-FAM3) is ported here verbatim in behaviour and renamed
`q3_generate_spliced`. Its reason for existing is unchanged and applies to Qwen3-VL as well:
transformers 5.x's Qwen-VL processors emit length-tied companion tensors (e.g.
`mm_token_type_ids`) that `get_rope_index` indexes with the attention mask to build M-RoPE
positions. Forwarding them untouched while input_ids are re-padded raises; DROPPING them does
not raise and silently degrades M-RoPE to 1-D text positions -- fluent text on wrong visual
positions. So every length-tied tensor is REBUILT in lockstep with input_ids, and an
unrecognised one is FATAL.
"""
import os, sys

LANE = "/data/alexmueller/sprint/qwen3"
sys.path.insert(0, f"{LANE}/code")

from gen_common import (generate_spliced, prompt_for, MAX_NEW, MAX_NEW_CONT, QUESTION,  # noqa: F401
                        cut_of, load_coco_sample, ChairScorer, SYN, MIN_LEN)            # noqa: F401

M5 = "Qwen/Qwen3-VL-8B-Instruct"
M5_SHA = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
WHICH = "m5"                       # anything but "m1" -> gen_common.prompt_for uses the model's
                                   # OWN chat template via proc.apply_chat_template
GRAY = (127, 127, 127)
ROOT = LANE
OUT = f"{ROOT}/out"

# Batch shape is NOT inert (env_replay_gate_batch_shape). Fixed here, identical for every arm
# of every stage, and asserted into every meta file.
BS_FREE = int(os.environ.get("Q3_BS_FREE", "4"))
BS_CONT = int(os.environ.get("Q3_BS_CONT", "4"))

MAX_NEW_FREE = 256                 # free-running captions, as families 1-4
MAX_NEW_384 = 384                  # continuations, as families 1-4
MAX_NEW_192 = 192                  # the reconstruction regime


# ---------------------------------------------------------------------------------------------
def q3_generate_spliced(model, proc, tok, images, prompts, prefix_ids=None, suffix_texts=None,
                        max_new=MAX_NEW, sample=False, seed=None, strict=True):
    """Port of f3_common.f3_generate_spliced (WIA-FAM3), unchanged in behaviour.

    Splices `prefix_ids` immediately after the processor's own prompt ids -- i.e. immediately
    after the assistant-turn opener produced by `add_generation_prompt=True` -- so the prefix is
    the START OF THE ASSISTANT'S OWN ANSWER and the model continues it. HG-2 in q3_gate.py
    asserts exactly that, on the realised token ids, rather than assuming it.
    """
    import torch
    from gen_common import _sample_kw, _clean_ids
    n = len(images)
    enc = proc(images=images, text=prompts, return_tensors="pt", padding=True)
    ids, am = enc["input_ids"], enc["attention_mask"]
    pad, eos = tok.pad_token_id, tok.eos_token_id
    W = ids.shape[1]

    rows, keep = [], []
    for i in range(n):
        k = am[i].bool()
        row = ids[i][k].tolist()          # strip processor left-pad
        n_app = 0
        if prefix_ids is not None and prefix_ids[i]:
            row += list(prefix_ids[i]); n_app += len(prefix_ids[i])
        if suffix_texts is not None and suffix_texts[i]:
            sfx = tok(suffix_texts[i], add_special_tokens=False)["input_ids"]
            row += sfx; n_app += len(sfx)
        rows.append(row); keep.append((k, n_app))

    maxlen = max(len(r) for r in rows)
    inp, msk = [], []
    for r in rows:
        need = maxlen - len(r)
        inp.append([pad] * need + r)
        msk.append([0] * need + [1] * len(r))
    inp_t = torch.tensor(inp, device="cuda")
    msk_t = torch.tensor(msk, device="cuda")

    attended_pads = int(((inp_t == pad) & (msk_t == 1)).sum().item())
    if strict and attended_pads:
        raise AssertionError(f"ATTENDED_PAD_POSITIONS={attended_pads} -- mask reconstruction bug")

    extra, rebuilt, forwarded = {}, [], []
    for name, v in enc.items():
        if name in ("input_ids", "attention_mask"):
            continue
        tied = (torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == n and v.shape[1] == W)
        if tied:
            out = []
            for i in range(n):
                k, n_app = keep[i]
                real = v[i][k]
                need = maxlen - (int(k.sum().item()) + n_app)
                out.append(torch.cat([torch.zeros(need, dtype=v.dtype),
                                      real,
                                      torch.zeros(n_app, dtype=v.dtype)]))
            extra[name] = torch.stack(out).to("cuda")
            rebuilt.append(name)
        elif torch.is_tensor(v):
            extra[name] = (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point
                           else v.to("cuda"))
            forwarded.append(name)
        else:
            extra[name] = v
            forwarded.append(name)

    for name in forwarded:
        v = enc[name]
        if torch.is_tensor(v) and v.shape and v.shape[-1] == W:
            raise AssertionError(
                f"FATAL_UNHANDLED_LENGTH_TIED_TENSOR {name} shape={tuple(v.shape)} W={W} -- "
                "forwarded untouched while input_ids were re-padded; rebuild it or fail loudly")

    with torch.no_grad():
        gen = model.generate(input_ids=inp_t, attention_mask=msk_t, **extra,
                             max_new_tokens=max_new, pad_token_id=pad,
                             **_sample_kw(sample, seed))
    new = gen[:, maxlen:]
    outs = [_clean_ids(row, pad, eos) for row in new]
    return outs, {"attended_pads": attended_pads, "maxlen": maxlen, "proc_width": W,
                  "rebuilt": rebuilt, "forwarded": forwarded,
                  "row_lens": [len(r) for r in rows],
                  "spliced_rows": [list(r) for r in rows]}


_CACHE = {}


def load_m5():
    """Returns (model, proc, tok, meta)."""
    import torch
    if "m5" in _CACHE:
        return _CACHE["m5"]
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration as Cls
    proc = AutoProcessor.from_pretrained(M5, revision=M5_SHA)
    tok = proc.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Host RAM, not GPUs, is the rationed resource on vgi1 (env_vgi_cluster_traps). The default
    # `.from_pretrained(...).to("cuda")` materialises all 16.34 GiB on the CPU first, so the job
    # must reserve >=24 GB of a 122 GB node that four other lanes are sharing. Streaming the
    # shards straight to the card keeps host RSS near one shard. The checkpoint is natively
    # bfloat16, so `dtype` is a no-op cast on both paths and this is placement only -- but that
    # is an argument, not evidence, so q3_loadeq.py asserts the two paths are BYTE-IDENTICAL on
    # real generations before any stage relies on it. Q3_LOAD_TO_CUDA=1 forces the old path.
    legacy = os.environ.get("Q3_LOAD_TO_CUDA", "0") == "1"
    if legacy:
        model = Cls.from_pretrained(M5, revision=M5_SHA, dtype=torch.bfloat16).to("cuda").eval()
    else:
        model = Cls.from_pretrained(M5, revision=M5_SHA, dtype=torch.bfloat16,
                                    device_map={"": "cuda:0"}).eval()
    off = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
           if t.device.type != "cuda"]
    assert not off, f"FATAL_TENSORS_NOT_ON_CUDA {off[:8]}"
    nbf = [n for n, t in model.named_parameters() if t.dtype != torch.bfloat16]
    assert not nbf, f"FATAL_PARAM_DTYPE {nbf[:8]}"
    tc = getattr(model.config, "text_config", None)
    meta = {"model": M5, "revision": M5_SHA, "pad_id": tok.pad_token_id,
            "eos_id": tok.eos_token_id, "pad_is_eos": tok.pad_token_id == tok.eos_token_id,
            "tok_class": type(tok).__name__, "vocab": len(tok),
            "cls": type(model).__name__, "dtype": str(model.dtype),
            "proc_class": type(proc).__name__,
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "load_path": "to_cuda" if legacy else "device_map_cuda0",
            "rope_scaling": getattr(tc, "rope_scaling", None) if tc is not None else None,
            "gen_eos": getattr(model.generation_config, "eos_token_id", None)}
    _CACHE["m5"] = (model, proc, tok, meta)
    return _CACHE["m5"]


def free_m5(model, proc, tok, recs):
    """Model's own greedy free-running caption; `gen_common.gen_free` shape, gray option."""
    return _free(model, proc, tok, recs, gray=False)


def _img(rec, gray):
    from PIL import Image
    im = Image.open(rec["file"]).convert("RGB")
    if not gray:
        return im
    g = Image.new("RGB", im.size, GRAY)
    assert g.size == im.size, f"FATAL_GRAY_IMAGE size {g.size} != {im.size}"
    assert g.getextrema() == ((127, 127), (127, 127), (127, 127)), "FATAL_GRAY_IMAGE extrema"
    return g


def _free(model, proc, tok, recs, gray=False, q=QUESTION, max_new=MAX_NEW_FREE, bs=None):
    bs = bs or BS_FREE
    p = prompt_for(WHICH, proc, q)
    out = []
    for s in range(0, len(recs), bs):
        b = recs[s:s + bs]
        ims = [_img(r, gray) for r in b]
        ids_l, _ = q3_generate_spliced(model, proc, tok, ims, [p] * len(b), max_new=max_new)
        for r, g in zip(b, ids_l):
            out.append({"image_id": r["image_id"], "gen_ids": g,
                        "text": tok.decode(g, skip_special_tokens=True), "gen_len": len(g),
                        "gray": gray})
    return out


q3_free = _free


def q3_cont(model, proc, tok, items, gray=False, q=QUESTION, max_new=MAX_NEW_384, bs=None,
            keep_diag=False):
    """Continuation from a spliced assistant-turn prefill. Batching, sort key and passthrough
    contract are family 3's, so batch composition is shared between the two arms of a contrast
    and between the sighted and blind runs (prefixes are bit-identical across them)."""
    bs = bs or BS_CONT
    p = prompt_for(WHICH, proc, q)
    items = sorted(items, key=lambda it: len(it["prefix_ids"]))
    out = []
    for s in range(0, len(items), bs):
        b = items[s:s + bs]
        ims = [_img(it, gray) for it in b]
        ids_l, diag = q3_generate_spliced(model, proc, tok, ims, [p] * len(b),
                                          prefix_ids=[it["prefix_ids"] for it in b],
                                          max_new=max_new)
        for j, (it, g) in enumerate(zip(b, ids_l)):
            r = {k: v for k, v in it.items() if k not in ("prefix_ids", "file")}
            r["cont_ids"] = g
            r["cont_text"] = tok.decode(g, skip_special_tokens=True)
            r["cont_len"] = len(g)
            r["prefix_text"] = tok.decode(it["prefix_ids"], skip_special_tokens=True)
            if keep_diag:
                r["_spliced_ids"] = diag["spliced_rows"][j]
            out.append(r)
    return out


def jdump(path, rows):
    import json
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def jload(path):
    import json
    return [json.loads(l) for l in open(path) if l.strip()]


def sentinel(name):
    print(f"Q3_SENTINEL_{name}_OK", flush=True)
