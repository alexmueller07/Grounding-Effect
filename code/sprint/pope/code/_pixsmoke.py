import sys
sys.path.insert(0, "/data/alexmueller/sprint/pope/code")
import numpy as np
from PIL import Image
import pope_pixels as P

f = "/data/alexmueller/sprint/pope/data/val2014_images/COCO_val2014_000000310196.jpg"
im = Image.open(f).convert("RGB")
print("orig", im.size)
for a in P.PIXEL_ARMS:
    o = P.make(a, im, 310196)
    arr = np.asarray(o)
    print("%-9s size=%s mean=%7.2f std=%7.2f" % (a, o.size, arr.mean(), arr.std()))
x1 = np.asarray(P.make("noise50", im, 310196))
x2 = np.asarray(P.make("noise50", im, 310196))
print("noise deterministic:", np.array_equal(x1, x2))
y1 = np.asarray(P.make("pshuffle", im, 310196))
y2 = np.asarray(P.make("pshuffle", im, 310196))
print("pshuffle deterministic:", np.array_equal(y1, y2))
b = np.asarray(P._to336(im))
print("pshuffle preserves pixel multiset:",
      np.array_equal(np.sort(y1.ravel()), np.sort(b.ravel())))
print("different images get different perms:",
      not np.array_equal(np.asarray(P.make("pshuffle", im, 1)), y1))
