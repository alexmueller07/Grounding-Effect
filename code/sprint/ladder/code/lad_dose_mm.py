"""Severity of the `mismatch` arm, which lad_dose.py cannot compute.

`mismatch` is not a pixel transform, so `apply_rung` refuses it by design and the pixel-rung
dose script skips it. Its severity is nonetheless well defined and is exactly the anchor the
registered L2 severity floor uses (LADDER_PREREG.md Amendment 3D): the cosine between the
vision tower's embedding of image A and of the image substituted for it, B = derangement(A) --
both CLEAN images, so no transform is involved.

Writes out/lad_dose_mm.json. Sentinel printed by Python.
"""
import os, sys, json, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")

import numpy as np
import torch
from PIL import Image
from lad_common import OUT, canon, derangement, sentinel, jdump, MISMATCH_SEED
from lad_dose import feats
from j5_common import load_coco_sample, ChairScorer, N_IMAGES
from wia_lngen import load_model_lowmem

BS = 8


def main():
    assert socket.gethostname().startswith("vgi1"), f"FATAL_WRONG_NODE {socket.gethostname()}"
    chair = ChairScorer("/data/alexmueller/j5_gates/code/synonyms.txt")
    data = load_coco_sample(chair, n=N_IMAGES)
    data.sort(key=lambda d: d["image_id"])
    ids = [int(d["image_id"]) for d in data]
    sub = derangement(ids)
    assert all(sub[i] != i for i in ids), "FATAL_MISMATCH_SELF_PAIRED"
    assert len(set(sub.values())) == len(ids), "FATAL_MISMATCH_NOT_BIJECTIVE"

    model, proc, tok = load_model_lowmem()
    ip = proc.image_processor
    pooled, per_tok = {}, {}
    with torch.no_grad():
        for b0 in range(0, len(data), BS):
            chunk = data[b0:b0 + BS]
            cl = [canon(proc, Image.open(d["file"]).convert("RGB")) for d in chunk]
            pv = ip(images=cl, return_tensors="pt")["pixel_values"].to("cuda", torch.bfloat16)
            f = feats(model, pv).float()
            for d, v in zip(chunk, f):
                pooled[int(d["image_id"])] = v.mean(0).cpu()
                per_tok[int(d["image_id"])] = v.cpu()
            if (b0 // BS) % 20 == 0:
                print(f"[lad_dose_mm] {b0 + len(chunk)}/{len(data)}", flush=True)

    cos, cost = [], []
    for a in ids:
        b = sub[a]
        cos.append(float(torch.nn.functional.cosine_similarity(
            pooled[a][None], pooled[b][None], dim=-1)[0]))
        cost.append(float(torch.nn.functional.cosine_similarity(
            per_tok[a], per_tok[b], dim=-1).mean()))
    # A reference distribution: the same statistic over ALL ordered pairs would be O(n^2); a
    # second independent derangement gives a cheap check that the number is a property of
    # "a different image", not of this particular pairing.
    rng = np.random.default_rng(MISMATCH_SEED + 1)
    perm = rng.permutation(len(ids))
    alt = [float(torch.nn.functional.cosine_similarity(
        pooled[ids[i]][None], pooled[ids[int(perm[i])]][None], dim=-1)[0])
        for i in range(len(ids)) if ids[int(perm[i])] != ids[i]]

    out = {"sentinel": True, "n": len(ids), "host": socket.gethostname(),
           "emb_cos": {"mismatch": float(np.mean(cos))},
           "emb_cos_tok": {"mismatch": float(np.mean(cost))},
           "emb_cos_sd": {"mismatch": float(np.std(cos))},
           "alt_permutation_emb_cos_mean": float(np.mean(alt)),
           "alt_permutation_n": len(alt),
           "note": "cosine between the clean embeddings of image A and its substitute B; no "
                   "pixel transform is involved, which is why lad_dose.py cannot produce it"}
    jdump(f"{OUT}/lad_dose_mm.json", out)
    print(json.dumps(out, indent=1), flush=True)
    sentinel("DOSE_MM")


if __name__ == "__main__":
    main()
