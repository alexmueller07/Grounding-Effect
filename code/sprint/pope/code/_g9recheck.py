import os, sys, json
os.environ.setdefault("HF_HOME", "/data/alexmueller/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, "/data/alexmueller/sprint/pope/code")
import numpy as np
from PIL import Image
from transformers import AutoProcessor
import pope_pixels as P
from pope_common import MODEL, prompt_text, DATA, IMGDIR

proc = AutoProcessor.from_pretrained(MODEL)
rows = [json.loads(l) for l in open(f"{DATA}/coco_pope_random.json")]
seen, out = set(), []
for r in rows:
    if r["image"] in seen:
        continue
    seen.add(r["image"])
    with Image.open(os.path.join(IMGDIR, r["image"])) as im:
        im = im.convert("RGB")
        a = proc(images=[im], text=[prompt_text(r["text"])],
                 return_tensors="pt")["pixel_values"].float().numpy()
        b = proc(images=[P._to336(im)], text=[prompt_text(r["text"])],
                 return_tensors="pt")["pixel_values"].float().numpy()
    out.append((r["image"], im.size, float(np.abs(a - b).max()), float(np.abs(a - b).mean())))
    if len(out) >= 40:
        break
mx = max(o[2] for o in out)
mn = max(o[3] for o in out)
print("n probed:", len(out))
print("max  abs pixel diff over images:", round(mx, 6))
print("max mean abs pixel diff over images:", round(mn, 6))
print("worst 5:", sorted(out, key=lambda o: -o[2])[:5])
