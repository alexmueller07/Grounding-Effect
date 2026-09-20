"""WIA-BASELINE shared plumbing (WIA_BASELINE_PREREG.md sections 1.2, 1.3, 1.6).

Design rule, from the pre-registration section 1.3: NOTHING existing is modified. `wia_lngen.py`
and `gen_common.py` are shared with live lanes, and on this cluster a queued job runs whatever is
on disk when it STARTS -- editing a shared harness silently changes queued jobs. So the blind arms
get line-for-line mirrors here rather than an edit there.

Three mirrors, one change each -- the pixels of the image:

    gen_batch_gray   mirrors wia_lngen.gen_batch        (family 1 continuations)
    gen_cont_gray    mirrors gen_common.gen_cont        (family 2 continuations)
    free_gray_f1     mirrors j5_gen_af.main's inner loop (family 1 free-running captions)
    free_gray_m2     mirrors gen_common.gen_free         (family 2 free-running captions)

GATE_MIRROR_IDENTICAL (pre-reg 1.3): with gray=False, gen_batch_gray / gen_cont_gray must be
BYTE-IDENTICAL to their references on real images. `mirror_gate_*` below run that check and
return the counts; the caller asserts.

GATE_GRAY_IMAGE (pre-reg 1.3): every gray image is a single-colour (127,127,127) RGB of the
TARGET FILE's own size, asserted per row by getextrema() and size equality -- never assumed.
"""
import os

GRAY = (127, 127, 127)                       # WIA_FAM3_PREREG.md section 2.2 / f3_common.GRAY
_GRAY_EXTREMA = ((127, 127), (127, 127), (127, 127))


def gray_image(path):
    """A flat mid-gray RGB image of the TARGET image's own pixel dimensions, gated."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    g = Image.new("RGB", im.size, GRAY)
    assert g.size == im.size, f"FATAL_GRAY_IMAGE size {g.size} != {im.size} ({path})"
    ex = g.getextrema()
    assert ex == _GRAY_EXTREMA, f"FATAL_GRAY_IMAGE extrema {ex} ({path})"
    return g


def _img(path, gray):
    from PIL import Image
    return gray_image(path) if gray else Image.open(path).convert("RGB")


# ---------------------------------------------------------------- family 1 continuations
def gen_batch_gray(model, proc, tok, items, gray, max_new):
    """LINE-FOR-LINE mirror of wia_lngen.gen_batch. The ONLY change is `_img(..., gray)` in
    place of `Image.open(it["file"]).convert("RGB")`, and `max_new` passed explicitly instead of
    read from the module global (so the 384 override cannot drift between the two paths)."""
    import torch
    from j5_common import prompt_text
    with torch.no_grad():
        ims = [_img(it["file"], gray) for it in items]
        enc = proc(images=ims, text=[prompt_text()] * len(items), return_tensors="pt",
                   padding=True)
        base_ids = enc["input_ids"]
        pix = enc["pixel_values"].to("cuda", torch.bfloat16)
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


def mirror_gate_f1(model, proc, tok, items, bs, max_new):
    """GATE_MIRROR_IDENTICAL, family 1: gen_batch_gray(gray=False) vs wia_lngen.gen_batch on the
    SAME chunks and the SAME batch size, real images. Returns (n_checked, n_identical)."""
    import wia_lngen
    assert wia_lngen.MAX_NEW_CONT == max_new, \
        f"FATAL_MIRROR_CAP_DRIFT {wia_lngen.MAX_NEW_CONT} != {max_new}"
    nchk = nid = 0
    for s in range(0, len(items), bs):
        chunk = items[s:s + bs]
        a = gen_batch_gray(model, proc, tok, chunk, gray=False, max_new=max_new)
        b = wia_lngen.gen_batch(model, proc, tok, chunk)
        for x, y in zip(a, b):
            nchk += 1
            nid += (list(x) == list(y))
    return nchk, nid


def free_gray_f1(model, proc, tok, recs, gray, bs, max_new):
    """Mirror of j5_gen_af.main's inner loop (encode_batch + greedy + pad/eos strip)."""
    import torch
    from j5_common import encode_batch, prompt_text
    out = []
    with torch.no_grad():
        for b0 in range(0, len(recs), bs):
            batch = recs[b0:b0 + bs]
            ims = [_img(d["file"], gray) for d in batch]
            enc = encode_batch(proc, ims, [prompt_text()] * len(batch))
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new,
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for d, row in zip(batch, new):
                ids = [int(t) for t in row if int(t) != tok.pad_token_id]
                if ids and ids[-1] == tok.eos_token_id:
                    ids = ids[:-1]
                out.append({"image_id": d["image_id"], "gen_ids": ids,
                            "text": tok.decode(ids, skip_special_tokens=True),
                            "gen_len": len(ids), "gray": gray})
    return out


# ---------------------------------------------------------------- family 2
def gen_cont_gray(which, model, proc, tok, items, gray, max_new, bs, q=None):
    """LINE-FOR-LINE mirror of gen_common.gen_cont. Only change: `_img(..., gray)`."""
    from gen_common import generate_spliced, prompt_for, QUESTION
    p = prompt_for(which, proc, q or QUESTION)
    items = sorted(items, key=lambda it: len(it["prefix_ids"]))
    out = []
    for s in range(0, len(items), bs):
        b = items[s:s + bs]
        ims = [_img(it["file"], gray) for it in b]
        ids_l, _ = generate_spliced(model, proc, tok, ims, [p] * len(b),
                                    prefix_ids=[it["prefix_ids"] for it in b],
                                    max_new=max_new, sample=False, seed=None)
        for it, g in zip(b, ids_l):
            r = {k: v for k, v in it.items() if k not in ("prefix_ids", "file")}
            r["cont_ids"] = g
            r["cont_text"] = tok.decode(g, skip_special_tokens=True)
            r["cont_len"] = len(g)
            r["prefix_text"] = tok.decode(it["prefix_ids"], skip_special_tokens=True)
            out.append(r)
    return out


def mirror_gate_m2(which, model, proc, tok, items, bs, max_new):
    """GATE_MIRROR_IDENTICAL, family 2: gen_cont_gray(gray=False) vs gen_common.gen_cont."""
    from gen_common import gen_cont
    a = gen_cont_gray(which, model, proc, tok, items, gray=False, max_new=max_new, bs=bs)
    b = gen_cont(which, model, proc, tok, items, max_new=max_new, bs=bs)
    ka = {(r["image_id"], r["k"], r["arm"]): r["cont_ids"] for r in a}
    kb = {(r["image_id"], r["k"], r["arm"]): r["cont_ids"] for r in b}
    assert set(ka) == set(kb), "FATAL_MIRROR_KEYSET"
    nid = sum(1 for k in ka if list(ka[k]) == list(kb[k]))
    return len(ka), nid


def free_gray_m2(which, model, proc, tok, recs, gray, bs, max_new):
    """Mirror of gen_common.gen_free."""
    from gen_common import generate_spliced, prompt_for, QUESTION
    p = prompt_for(which, proc, QUESTION)
    out = []
    for s in range(0, len(recs), bs):
        b = recs[s:s + bs]
        ims = [_img(r["file"], gray) for r in b]
        ids_l, _ = generate_spliced(model, proc, tok, ims, [p] * len(b), max_new=max_new,
                                    sample=False, seed=None)
        for r, g in zip(b, ids_l):
            out.append({"image_id": r["image_id"], "gen_ids": g,
                        "text": tok.decode(g, skip_special_tokens=True),
                        "gen_len": len(g), "gray": gray})
    return out


# ---------------------------------------------------------------- shared helpers
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
