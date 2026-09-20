"""Exercise the mismatch cache path exactly as pope_gen.run_arm builds it, with no model.
The running jobs reach this arm as their THIRD cell; a crash there would kill the afterok
chain, so it is checked now."""
import inspect, os, sys
sys.path.insert(0, "/data/alexmueller/sprint/pope/code")
from PIL import Image
import pope_gen, pope_pixels
from pope_items import load_items

src = inspect.getsource(pope_gen.run_arm)
assert 'if pixarm == "mismatch":' in src, "FATAL_MISMATCH_BRANCH_MISSING"
assert 'FATAL_MISMATCH_SELF_PAIR' in src, "FATAL_SELFPAIR_ASSERT_MISSING"

items = load_items()
pixarm = "mismatch"
pil_cache, deran = {}, None
# --- the exact lines from run_arm ---
by_id = {it["image_id"]: it for it in items}
deran = pope_pixels.derangement(by_id.keys())
for it in items:
    if it["image"] not in pil_cache:
        s = by_id[deran[it["image_id"]]]
        assert s["image_id"] != it["image_id"], "FATAL_MISMATCH_SELF_PAIR"
        with Image.open(s["file"]) as im:
            pil_cache[it["image"]] = im.convert("RGB")
assert len(pil_cache) == len({it["image"] for it in items}), "FATAL_PIL_CACHE"
# --- checks ---
print("items:", len(items), "cached:", len(pil_cache))
print("all RGB:", all(v.mode == "RGB" for v in pil_cache.values()))
orig = {it["image"]: it["file"] for it in items}
diff = sum(1 for k, v in pil_cache.items()
           if Image.open(orig[k]).size != v.size)
print("cached image differs in SIZE from its own original for", diff, "of", len(pil_cache))
import numpy as np
same_pixels = 0
for k in list(pil_cache)[:25]:
    a = np.asarray(Image.open(orig[k]).convert("RGB"))
    b = np.asarray(pil_cache[k])
    if a.shape == b.shape and np.array_equal(a, b):
        same_pixels += 1
print("of 25 probed, cached pixels identical to own original:", same_pixels, "(must be 0)")
mb = sum(v.size[0] * v.size[1] * 3 for v in pil_cache.values()) / 1e6
print("cache footprint: %.0f MB" % mb)
print("MISMATCH_PATH_OK")
