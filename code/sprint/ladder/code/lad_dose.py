"""Realised severity of each rung, measured in the vision tower (LADDER_PREREG.md 2.3).

A nominal knob is not a dose. For each of the 500 frozen images this computes the LLaVA-1.5-7B
vision tower's patch embeddings for the clean canonical image and for each rung, and reports:

  emb_cos        cosine between the MEAN-POOLED 576-patch embeddings  <- the registered axis
  emb_cos_tok    mean over patches of the per-patch cosine
  pix_rmse       RMSE in the processor's normalised pixel_values space

The pixel measure demonstrably mis-orders the ladder (shuffle24 has the largest pixel distance
of all rungs while preserving every patch's content), which is why the registered axis is the
embedding one. Writes out/lad_dose.json.
"""
import os, sys, json, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")

import numpy as np
import torch
from PIL import Image
from lad_common import ALL_RUNGS, OUT, canon, apply_rung, sentinel, jdump
from j5_common import load_coco_sample, ChairScorer, N_IMAGES
from wia_lngen import load_model_lowmem

BS = 8
FEAT_PATH = None      # which embedding path was used; recorded in the output


def _unwrap(v):
    """transformers returns tensors, tuples, lists or ModelOutput objects depending on
    version and entry point. Reduce any of them to the patch-embedding tensor."""
    for _ in range(5):
        if isinstance(v, (list, tuple)):
            assert v, "FATAL_EMPTY_FEATURE_SEQUENCE"
            v = v[0]
            continue
        if torch.is_tensor(v):
            assert v.dim() == 3, f"FATAL_FEATURE_SHAPE {tuple(v.shape)}"
            return v
        for attr in ("image_features", "last_hidden_state"):
            if hasattr(v, attr) and getattr(v, attr) is not None:
                v = getattr(v, attr)
                break
        else:
            if isinstance(v, dict) and v:
                v = next(iter(v.values()))
            else:
                v = None
        if v is not None:
            continue
        break
    raise AssertionError(f"FATAL_UNEXPECTED_FEATURE_TYPE {type(v)}")


def feats(model, pix):
    global FEAT_PATH
    f = (getattr(model, "get_image_features", None)
         or getattr(getattr(model, "model", None), "get_image_features", None))
    if f is not None:
        try:
            t = _unwrap(f(pixel_values=pix))
            FEAT_PATH = "get_image_features"
            return t
        except Exception as e:                      # noqa: BLE001 - any failure falls back
            if FEAT_PATH is None:
                print(f"[lad_dose] get_image_features unusable ({type(e).__name__}: {e}); "
                      f"falling back to the vision tower", flush=True)
    vt = (getattr(model, "vision_tower", None)
          or getattr(getattr(model, "model", None), "vision_tower", None))
    assert vt is not None, "FATAL_NO_VISION_TOWER"
    t = _unwrap(vt(pix, output_hidden_states=True).hidden_states[-2])[:, 1:]
    FEAT_PATH = "vision_tower.hidden_states[-2][:,1:]"
    return t


def main():
    assert socket.gethostname().startswith("vgi1"), f"FATAL_WRONG_NODE {socket.gethostname()}"
    chair = ChairScorer("/data/alexmueller/j5_gates/code/synonyms.txt")
    data = load_coco_sample(chair, n=N_IMAGES)
    data.sort(key=lambda d: d["image_id"])
    model, proc, tok = load_model_lowmem()
    ip = proc.image_processor
    # Coordinator fix 2026-09-20: "mismatch" substitutes a DIFFERENT image and is not a pixel
    # transform -- apply_rung refuses it (FATAL_MISMATCH_IS_NOT_A_PIXEL_TRANSFORM), which killed
    # this job after the black rung finished. Its severity needs the substitute-image path and is
    # a registered quantity (the Amendment 3 L2 floor anchors on it), so it is left to the lane
    # rather than improvised here; it is reported as null, never as 0 or 1.
    DOSE_RUNGS = [r for r in ALL_RUNGS if r != "mismatch"]
    acc = {r: {"cos": [], "cos_tok": [], "rmse": []} for r in DOSE_RUNGS}
    with torch.no_grad():
        for b0 in range(0, len(data), BS):
            chunk = data[b0:b0 + BS]
            cl = [canon(proc, Image.open(d["file"]).convert("RGB")) for d in chunk]
            pc = ip(images=cl, return_tensors="pt")["pixel_values"]
            fc = feats(model, pc.to("cuda", torch.bfloat16)).float()
            for r in DOSE_RUNGS:
                dg = [apply_rung(r, c, d["image_id"]) for c, d in zip(cl, chunk)]
                pd_ = ip(images=dg, return_tensors="pt")["pixel_values"]
                fd = feats(model, pd_.to("cuda", torch.bfloat16)).float()
                a, b = fc.mean(1), fd.mean(1)
                cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
                cost = torch.nn.functional.cosine_similarity(fc, fd, dim=-1).mean(1)
                acc[r]["cos"] += [float(x) for x in cos]
                acc[r]["cos_tok"] += [float(x) for x in cost]
                acc[r]["rmse"] += [float(x) for x in
                                   torch.sqrt(((pd_ - pc) ** 2).mean(dim=(1, 2, 3)))]
            if (b0 // BS) % 10 == 0:
                print(f"[lad_dose] {b0+len(chunk)}/{len(data)}", flush=True)
    out = {"sentinel": True, "n": len(data), "host": socket.gethostname(),
           "dose_rungs": DOSE_RUNGS,
           "mismatch_severity": None,
           "mismatch_note": "not a pixel transform; needs the substitute-image embedding path",
           "emb_cos": {r: float(np.mean(acc[r]["cos"])) for r in DOSE_RUNGS},
           "emb_cos_tok": {r: float(np.mean(acc[r]["cos_tok"])) for r in DOSE_RUNGS},
           "pix_rmse": {r: float(np.mean(acc[r]["rmse"])) for r in DOSE_RUNGS},
           "emb_cos_sd": {r: float(np.std(acc[r]["cos"])) for r in DOSE_RUNGS},
           "feature_path": FEAT_PATH}
    jdump(f"{OUT}/lad_dose.json", out)
    print(json.dumps({k: out[k] for k in ("emb_cos", "emb_cos_tok", "pix_rmse")}, indent=1),
          flush=True)
    sentinel("DOSE")


if __name__ == "__main__":
    main()
