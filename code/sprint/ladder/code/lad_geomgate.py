"""GATE_GEOM_IDENTITY + GATE_GREY_EQUIVALENCE + GATE_RUNG_DETERMINISTIC (pre-reg section 3).

Runs before the pre-registration is finalised and before ANY generation. No model weights, no
endpoint: this only decides whether the ladder's transform space is admissible.

  GATE_GEOM_IDENTITY     proc(canon(im)).pixel_values must be BIT-IDENTICAL to
                         proc(im).pixel_values on N real COCO images. If it fails, applying
                         rungs in canonical space would itself be an intervention, and the
                         registration aborts the lane rather than quietly changing space.
  GATE_GREY_EQUIVALENCE  apply_rung("grey", canon(im)) must reach the SAME pixel_values as
                         wia_blind_common.gray_image(path) -- the published blind arm's own
                         constructor, which builds grey at the image's native size. Without
                         this the grey positive control would not be the paper's grey.
  GATE_IDENTITY_ROUNDTRIP  apply_rung("identity", canon(im)) must equal canon(im).
  GATE_RUNG_DETERMINISTIC  every rung recomputed twice must be byte-equal, and two different
                         images must not collide.
  GATE_RUNG_DISTINCT     every rung must actually change the tensor (a silently inert rung
                         would read as a perfect null).

Writes out/lad_geomgate.json. Sentinel printed by Python; the exit code is never the signal.
"""
import os, sys, json, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")
os.environ.setdefault("HF_HOME", "/data/alexmueller/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image
from lad_common import (canon, apply_rung, ALL_RUNGS, RUNGS, CANON, OUT, sentinel, jdump)
from j5_common import load_coco_sample, ChairScorer, MODEL, SEED
from wia_blind_common import gray_image

N_CHECK = int(os.environ.get("LAD_NGEOM", "32"))


def pv(proc, im):
    return np.asarray(proc.image_processor(images=im, return_tensors="np")["pixel_values"][0])


def main():
    from transformers import AutoProcessor
    print(f"[geomgate] host={socket.gethostname()} model={MODEL} n={N_CHECK}", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    ip = proc.image_processor
    cfg = {"size": dict(ip.size), "crop_size": dict(ip.crop_size),
           "resample": int(ip.resample),
           "do_center_crop": bool(ip.do_center_crop), "do_resize": bool(ip.do_resize)}
    print(f"[geomgate] image_processor {json.dumps(cfg)}", flush=True)

    chair = ChairScorer("/data/alexmueller/j5_gates/code/synonyms.txt")
    data = load_coco_sample(chair, n=500, seed=SEED)
    data.sort(key=lambda d: d["image_id"])
    sub = data[:N_CHECK]

    res = {"sentinel": True, "host": socket.gethostname(), "model": MODEL, "n": len(sub),
           "processor": cfg, "native_sizes": {}, "gates": {}, "per_rung": {}}

    n_geom = n_grey = n_ident = 0
    maxabs_geom = 0.0
    sizes = {}
    for d in sub:
        im = Image.open(d["file"]).convert("RGB")
        sizes[str(im.size)] = sizes.get(str(im.size), 0) + 1
        a = pv(proc, im)
        c = canon(proc, im)
        b = pv(proc, c)
        n_geom += int(np.array_equal(a, b))
        maxabs_geom = max(maxabs_geom, float(np.abs(a.astype(np.float64) - b).max()))
        g_new = pv(proc, apply_rung("grey", c, d["image_id"]))
        g_old = pv(proc, gray_image(d["file"]))
        n_grey += int(np.array_equal(g_new, g_old))
        n_ident += int(np.array_equal(np.asarray(apply_rung("identity", c, d["image_id"])),
                                      np.asarray(c)))
    res["native_sizes"] = sizes
    res["gates"]["GATE_GEOM_IDENTITY"] = {"identical": n_geom, "n": len(sub),
                                          "max_abs_diff": maxabs_geom}
    res["gates"]["GATE_GREY_EQUIVALENCE"] = {"identical": n_grey, "n": len(sub)}
    res["gates"]["GATE_IDENTITY_ROUNDTRIP"] = {"identical": n_ident, "n": len(sub)}

    # per-rung determinism, distinctness, cross-image non-collision, realised pixel dose
    for r in ALL_RUNGS:
        det = dist = coll = 0
        l2, linf = [], []
        prev = None
        for d in sub:
            c = canon(proc, Image.open(d["file"]).convert("RGB"))
            x1 = np.asarray(apply_rung(r, c, d["image_id"]))
            x2 = np.asarray(apply_rung(r, c, d["image_id"]))
            det += int(np.array_equal(x1, x2))
            base = pv(proc, c)
            got = pv(proc, Image.fromarray(x1, "RGB"))
            dist += int(not np.array_equal(base, got))
            l2.append(float(np.sqrt(((got - base) ** 2).mean())))
            linf.append(float(np.abs(got - base).max()))
            if prev is not None and r != "grey":
                coll += int(np.array_equal(x1, prev))
            prev = x1
        res["per_rung"][r] = {"determinism_equal": det, "n": len(sub),
                              "changes_tensor": dist,
                              "cross_image_collisions": coll,
                              "pix_rmse_mean": float(np.mean(l2)),
                              "pix_linf_mean": float(np.mean(linf))}
        print(f"[geomgate] {r:10s} det {det}/{len(sub)} changed {dist}/{len(sub)} "
              f"coll {coll} rmse {np.mean(l2):.4f}", flush=True)

    ok = (n_geom == len(sub) and n_grey == len(sub) and n_ident == len(sub)
          and all(v["determinism_equal"] == v["n"] for v in res["per_rung"].values())
          and all(res["per_rung"][r]["changes_tensor"] == len(sub) for r in RUNGS)
          and all(res["per_rung"][r]["cross_image_collisions"] == 0 for r in RUNGS))
    res["ALL_GATES_PASS"] = bool(ok)
    os.makedirs(OUT, exist_ok=True)
    jdump(f"{OUT}/lad_geomgate.json", res)
    print(json.dumps({k: res[k] for k in ("gates", "ALL_GATES_PASS")}, indent=1), flush=True)
    assert n_geom == len(sub), f"FATAL_GEOM_IDENTITY {n_geom}/{len(sub)} maxabs={maxabs_geom}"
    assert n_grey == len(sub), f"FATAL_GREY_EQUIVALENCE {n_grey}/{len(sub)}"
    assert n_ident == len(sub), f"FATAL_IDENTITY_ROUNDTRIP {n_ident}/{len(sub)}"
    assert ok, "FATAL_GEOMGATE"
    sentinel("GEOMGATE")


if __name__ == "__main__":
    main()
