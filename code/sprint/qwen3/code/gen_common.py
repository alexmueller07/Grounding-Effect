"""SC1-GEN harness (lane SC1-GEN): generality axes for SC1.

Prereg: SC1GEN_PREREG.md (committed before any GPU cell). The J5 code at
/data/alexmueller/sprint/qwen3/code is imported IN PLACE -- identical CHAIR scorer, prefix
builder, position mapping, bootstrap -- so the scorer's first-order bias is identical
across every arm and cancels in every contrast. Sentinels are printed by Python; exit
codes are NEVER the success signal.

Does NOT read or write /data/alexmueller/sc1_week1 (sibling lane J11 owns it).
J5 outputs are read-only cached artifacts.
"""
import os, sys, json, math, random

J5CODE = "/data/alexmueller/sprint/qwen3/code"
sys.path.insert(0, J5CODE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from j5_common import *  # noqa: F401,F403 -- np/json/math/random, helpers, ChairScorer, ...
from j5_common import (load_coco_sample, ChairScorer, build_prefix_ids, glue,
                       mention_positions, JsonlWriter)

import numpy as np

ROOT = "/data/alexmueller/sc1_gen"
OUT = f"{ROOT}/out"
J5OUT = "/data/alexmueller/j5_gates/out"
SYN = f"{J5CODE}/synonyms.txt"

M1 = "llava-hf/llava-1.5-7b-hf"                          # primary model (J5/J11)
M2 = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"         # second model, cross-family
M2_SHA = "74dd0bf867a4cda7950c17663794267c60cf4b40"

SMOKE = os.environ.get("SMOKE", "0") == "1"
N = 8 if SMOKE else int(os.environ.get("SC1GEN_N", "500"))
PFX = "smoke_" if SMOKE else ""

MIN_LEN = 32            # inherited J5 eligibility floor (in the model's OWN tokens)
MAX_NEW = 256           # free-running
MAX_NEW_CONT = 192      # continuations
PRIMARY_F = 0.5         # primary cut (fraction of own length -> tokenizer-invariant)
QTR_F = 0.25
TEMP = 0.7
SEEDS = (1,) if SMOKE else (1, 2, 3)
NBOOT = 200 if SMOKE else 10000
REL_MIN = 0.15                      # inherited Gate A support line
M1_SHARE_CI = (0.913, 0.976)        # LLaVA-1.5 Gate A relative-share CI (for ATTENUATED test)
BS = int(os.environ.get("SC1GEN_BS", "4"))   # anyres: ~3k tokens/row, keep batches small
TOKRATIO_QWEN_LLAMA = 0.8805        # measured this session, for LLaMA-equivalent reporting

QUESTION = "Please describe this image in detail."
POPE_TEMPLATE = "Is there a {obj} in the image?"


def sentinel(name):
    print(f"SC1GEN_SENTINEL_{name}_OK", flush=True)


def dump(stage, obj):
    obj["sentinel"] = True
    os.makedirs(OUT, exist_ok=True)
    json.dump(obj, open(f"{OUT}/{PFX}{stage}.json", "w"), indent=1, default=_jd)


def _jd(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(str(type(o)))


# --------------------------------------------------------------------------------------
# Model loading. Model IDENTITY is isolated here; everything downstream takes (model, proc, tok).
# --------------------------------------------------------------------------------------
_CACHE = {}


def load_any(which):
    """which in {'m1','m2'}. Returns (model, proc, tok, meta)."""
    import torch
    if which in _CACHE:
        return _CACHE[which]
    from transformers import AutoProcessor
    if which == "m1":
        from transformers import LlavaForConditionalGeneration as Cls
        name = M1
    else:
        from transformers import LlavaOnevisionForConditionalGeneration as Cls
        name = M2
    proc = AutoProcessor.from_pretrained(name)
    tok = proc.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = Cls.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    meta = {"model": name, "pad_id": tok.pad_token_id, "eos_id": tok.eos_token_id,
            "pad_is_eos": tok.pad_token_id == tok.eos_token_id}
    _CACHE[which] = (model, proc, tok, meta)
    return _CACHE[which]


def prompt_for(which, proc, question=QUESTION):
    """Prompt ending at the assistant turn. Model 2 uses its OWN chat template, because a
    cross-RECIPE test must use the model's own recipe (prereg 0.4)."""
    if which == "m1":
        return f"{SYS_STR}USER: <image>\n{question} ASSISTANT:"  # noqa: F405 (J5 SYS_STR)
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return proc.apply_chat_template(conv, add_generation_prompt=True)


def followup_suffix(which, proc, question):
    """Text that turns an assistant-turn prefix into a second user question. Used by POPE."""
    if which == "m1":
        return f" USER: {question} ASSISTANT:"
    return f"<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"


def _clean_ids(row, pad, eos):
    """Cut at first EOS; drop pads. (pad != eos on both models -- asserted in smoke.)"""
    out = []
    for t in row:
        t = int(t)
        if t == eos:
            break
        if t == pad:
            continue
        out.append(t)
    return out


def _sample_kw(sample, seed=None):
    import torch
    if sample:
        if seed is not None:
            torch.manual_seed(30011 * seed)
        return dict(do_sample=True, temperature=TEMP, top_p=1.0, top_k=0)
    return dict(do_sample=False)


# --------------------------------------------------------------------------------------
# ★ The anyres-safe generation core.
#
# J5's prefix path assumed `base_ids = enc["input_ids"]  # identical prompt -> same length,
# no pad`. Under OneVision anyres that is FALSE (measured: input_ids 2088..3714 across 12
# images), so the processor LEFT-PADS rows and J5's hand-rebuilt mask would mark those pads
# as attended. Here every row's real tokens are recovered via the processor's OWN
# attention_mask before splicing, and all non-(input_ids, attention_mask) processor tensors
# (pixel_values, image_sizes, batch_num_images, ...) are forwarded untouched.
# --------------------------------------------------------------------------------------
def generate_spliced(model, proc, tok, images, prompts, prefix_ids=None, suffix_texts=None,
                     max_new=MAX_NEW, sample=False, seed=None, strict=True):
    """Returns (list[list[int]] new-token ids, dict diagnostics).

    prefix_ids[i]   : token ids forced into the assistant turn (or None)
    suffix_texts[i] : text appended AFTER the prefix (e.g. a follow-up question)
    """
    import torch
    n = len(images)
    enc = proc(images=images, text=prompts, return_tensors="pt", padding=True)
    ids, am = enc["input_ids"], enc["attention_mask"]
    pad, eos = tok.pad_token_id, tok.eos_token_id

    rows = []
    for i in range(n):
        real = ids[i][am[i].bool()].tolist()          # strip processor left-pad
        row = list(real)
        if prefix_ids is not None and prefix_ids[i]:
            row += list(prefix_ids[i])
        if suffix_texts is not None and suffix_texts[i]:
            row += tok(suffix_texts[i], add_special_tokens=False)["input_ids"]
        rows.append(row)

    maxlen = max(len(r) for r in rows)
    inp, msk = [], []
    for r in rows:
        need = maxlen - len(r)
        inp.append([pad] * need + r)
        msk.append([0] * need + [1] * len(r))
    inp_t = torch.tensor(inp, device="cuda")
    msk_t = torch.tensor(msk, device="cuda")

    # hard correctness gate (prereg §6): no attended position may hold a pad id
    attended_pads = int(((inp_t == pad) & (msk_t == 1)).sum().item())
    if strict and attended_pads:
        raise AssertionError(f"ATTENDED_PAD_POSITIONS={attended_pads} -- mask reconstruction bug")

    extra = {}
    for k, v in enc.items():
        if k in ("input_ids", "attention_mask"):
            continue
        if torch.is_tensor(v):
            extra[k] = v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda")
        else:
            extra[k] = v

    with torch.no_grad():
        gen = model.generate(input_ids=inp_t, attention_mask=msk_t, **extra,
                             max_new_tokens=max_new, pad_token_id=pad,
                             **_sample_kw(sample, seed))
    new = gen[:, maxlen:]
    outs = [_clean_ids(row, pad, eos) for row in new]
    return outs, {"attended_pads": attended_pads, "maxlen": maxlen,
                  "row_lens": [len(r) for r in rows]}


def gen_free(which, model, proc, tok, recs, q=QUESTION, sample=False, seed=None,
             max_new=MAX_NEW, bs=None):
    """Free-running captions. Under greedy this IS the self-prefix continuation at every cut."""
    from PIL import Image
    bs = bs or BS
    p = prompt_for(which, proc, q)
    out = []
    for s in range(0, len(recs), bs):
        b = recs[s:s + bs]
        ims = [Image.open(r["file"]).convert("RGB") for r in b]
        ids_l, _ = generate_spliced(model, proc, tok, ims, [p] * len(b), max_new=max_new,
                                    sample=sample, seed=seed)
        for r, g in zip(b, ids_l):
            out.append({"image_id": r["image_id"], "gen_ids": g,
                        "text": tok.decode(g, skip_special_tokens=True), "gen_len": len(g)})
    return out


def gen_cont(which, model, proc, tok, items, q=QUESTION, sample=False, seed=None,
             max_new=MAX_NEW_CONT, bs=None):
    """Prefix-forced continuations. items: dicts with file, prefix_ids, plus passthrough keys."""
    from PIL import Image
    bs = bs or max(2, BS // 2)
    p = prompt_for(which, proc, q)
    items = sorted(items, key=lambda it: len(it["prefix_ids"]))   # minimize padding
    out = []
    for s in range(0, len(items), bs):
        b = items[s:s + bs]
        ims = [Image.open(it["file"]).convert("RGB") for it in b]
        ids_l, _ = generate_spliced(model, proc, tok, ims, [p] * len(b),
                                    prefix_ids=[it["prefix_ids"] for it in b],
                                    max_new=max_new, sample=sample, seed=seed)
        for it, g in zip(b, ids_l):
            r = {k: v for k, v in it.items() if k not in ("prefix_ids", "file")}
            r["cont_ids"] = g
            r["cont_text"] = tok.decode(g, skip_special_tokens=True)
            r["cont_len"] = len(g)
            r["prefix_text"] = tok.decode(it["prefix_ids"], skip_special_tokens=True)
            out.append(r)
    return out


# --------------------------------------------------------------------------------------
# Cuts / prefixes (inherited J5 conventions)
# --------------------------------------------------------------------------------------
def wrong_map(recs):
    sids = sorted(r["image_id"] for r in recs)
    return {sids[i]: sids[(i + 1) % len(sids)] for i in range(len(sids))}


def cut_of(L, f):
    return int(math.floor(f * L))


def repetition_frac(texts):
    """Fraction of texts with a 4-gram repeated >=3x consecutively (J5 guard)."""
    bad = 0
    for t in texts:
        w = t.lower().split()
        hit = False
        for i in range(len(w) - 11):
            g = w[i:i + 4]
            if w[i + 4:i + 8] == g and w[i + 8:i + 12] == g:
                hit = True
                break
        bad += hit
    return bad / max(1, len(texts))


# --------------------------------------------------------------------------------------
# POPE-style construction from LOCAL COCO val2017 (prereg §3.3; zero download)
# --------------------------------------------------------------------------------------
def add_present(chair, recs):
    """Attach instance-annotation-only node sets ('present') to records.

    load_coco_sample's 'gt' is instances UNION caption-mined; POPE's 'present' is the
    segmentation set only, so it is recomputed here with the SAME node mapping.
    """
    inst = json.load(open(ANN_INST))  # noqa: F405 (J5 constant)
    cat2name = {c["id"]: c["name"] for c in inst["categories"]}
    by_img = {}
    for a in inst["annotations"]:
        by_img.setdefault(a["image_id"], set()).add(a["category_id"])
    for r in recs:
        pres = set()
        for cid in by_img.get(r["image_id"], ()):
            name = cat2name[cid]
            node = chair.node_for(name) or chair.node_for(name.replace(" ", ""))
            if node is None:
                ms = chair.mentions(name)
                node = ms[0][0] if ms else None
            if node:
                pres.add(node)
        r["present"] = sorted(pres)
    return recs


def build_pope(chair, recs, seed=20260817):
    """Faithful port of the published POPE recipe onto val2017.

    present = instance-annotation categories; absent = 80 COCO categories minus the RICHER
    J5 gt node set (instances UNION caption-mined) -- a deliberate conservative deviation
    that reduces false-'absent' label noise. Splits: adversarial (primary), random.
    """
    cats = sorted({c for line in open(SYN) for c in [line.strip().split(",")[0].strip()] if c})
    # co-occurrence over the sampled population, for the adversarial split
    co = {}
    for r in recs:
        pres = sorted(set(r["present"]))
        for a in pres:
            for b in pres:
                if a != b:
                    co[(a, b)] = co.get((a, b), 0) + 1
    qs = []
    for r in recs:
        rng = random.Random(seed + r["image_id"])
        pres = sorted(set(r["present"]))
        absent = [c for c in cats if c not in set(r["gt"])]     # richer gt -> confident absence
        if not pres or len(absent) < 3:
            continue
        yes = rng.sample(pres, min(3, len(pres)))
        score = {c: sum(co.get((p, c), 0) for p in pres) for c in absent}
        adver = sorted(absent, key=lambda c: (-score[c], c))[:3]
        rnd = rng.sample(absent, 3)
        for o in yes:
            qs.append({"image_id": r["image_id"], "obj": o, "label": "yes", "split": "both"})
        for o in adver:
            qs.append({"image_id": r["image_id"], "obj": o, "label": "no", "split": "adversarial"})
        for o in rnd:
            qs.append({"image_id": r["image_id"], "obj": o, "label": "no", "split": "random"})
    return qs


def parse_yesno(text):
    t = text.strip().lower().lstrip(" \t\n.,:;!?\"'")
    for tokw in t.replace(".", " ").replace(",", " ").split():
        w = tokw.strip("\"'.,:;!?")
        if w in ("yes", "yeah", "yep"):
            return "yes"
        if w in ("no", "nope"):
            return "no"
        break
    return None
