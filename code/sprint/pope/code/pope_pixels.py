"""POPE lane -- the pixel arms. Every arm changes ONLY the pixels.

Pre-registered in SPRINT/POPE_PREREG.md. Constants here must not change after the first
endpoint row exists.

FRAME. Arms marked `own` act in the image's own pixel frame; arms marked `m336` act in the
model's own input frame (shortest-edge resize to 336 + center crop 336, i.e. exactly the
geometry the LLaVA-1.5 processor itself applies), so that a dose is a controlled dose and a
patch is a real ViT patch. Gate G_FRAME asserts that pushing an already-336x336 image through
the processor gives the same pixel_values as pushing the original, so the frame choice cannot
by itself move a number.

GREY is verbatim the constant this paper uses everywhere: a flat (127,127,127) RGB rectangle
at the target image's own pixel dimensions (wia_blind_common.GRAY, WIA_FAM3_PREREG 2.2).
"""
import numpy as np
from PIL import Image, ImageFilter

GRAY = (127, 127, 127)
_GRAY_EXTREMA = ((127, 127), (127, 127), (127, 127))
PIX_SEED = 20260919          # per-image rng seed base; fixed before any generation
CROP = 336                   # LLaVA-1.5 CLIP ViT-L/14-336 input
PATCH = 14                   # -> 24x24 patch grid


# ------------------------------------------------------------------ helpers
def _to336(im):
    """The processor's own geometry: resize shortest edge to 336 (bicubic), center crop 336."""
    # HF's get_resize_output_image_size FLOORS the long side (int(), not round()).  Using
    # round() here shifted the centre crop by one pixel on some aspect ratios, which gate G9
    # caught as a 2.22 max-abs pixel difference on 2 of 8 probe images.  Matching the floor
    # makes this function what the registration says it is -- "exactly the processor's
    # geometry" -- rather than an approximation of it.
    w, h = im.size
    s = CROP / min(w, h)
    nw, nh = max(CROP, int(w * s)), max(CROP, int(h * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    l, t = (nw - CROP) // 2, (nh - CROP) // 2
    return im.crop((l, t, l + CROP, t + CROP))


def _rng(image_id):
    return np.random.default_rng(PIX_SEED + int(image_id))


# ------------------------------------------------------------------ the arms
def a_sighted(im, image_id):
    return im


def a_grey(im, image_id):
    """Flat mid-grey at the image's own size -- the paper's blinding constant, gated."""
    g = Image.new("RGB", im.size, GRAY)
    assert g.size == im.size, "FATAL_GREY_SIZE %s != %s" % (g.size, im.size)
    assert g.getextrema() == _GRAY_EXTREMA, "FATAL_GREY_EXTREMA %s" % (g.getextrema(),)
    return g


def _noise(im, image_id, sigma):
    base = _to336(im)
    a = np.asarray(base, dtype=np.float32)
    n = _rng(image_id).normal(0.0, sigma, size=a.shape).astype(np.float32)
    return Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8), "RGB")


def a_noise25(im, i):
    return _noise(im, i, 25.0)


def a_noise50(im, i):
    return _noise(im, i, 50.0)


def a_noise100(im, i):
    return _noise(im, i, 100.0)


def _blur(im, image_id, radius):
    return _to336(im).filter(ImageFilter.GaussianBlur(radius=radius))


def a_blur4(im, i):
    return _blur(im, i, 4.0)


def a_blur16(im, i):
    return _blur(im, i, 16.0)


def a_lowres(im, image_id):
    """21x linear downsample (336 -> 16 px) and back, bicubic both ways."""
    b = _to336(im)
    small = b.resize((16, 16), Image.BICUBIC)
    return small.resize((CROP, CROP), Image.BICUBIC)


def a_pshuffle(im, image_id):
    """Permute the 24x24 grid of 14x14 ViT patches with a fixed per-image permutation.
    Local statistics are preserved exactly; global layout is destroyed."""
    b = np.asarray(_to336(im), dtype=np.uint8)
    g = CROP // PATCH
    blocks = (b.reshape(g, PATCH, g, PATCH, 3)
               .transpose(0, 2, 1, 3, 4).reshape(g * g, PATCH, PATCH, 3))
    perm = _rng(image_id).permutation(g * g)
    assert perm.size == g * g and len(set(perm.tolist())) == g * g, "FATAL_PSHUFFLE_PERM"
    out = (blocks[perm].reshape(g, g, PATCH, PATCH, 3)
                       .transpose(0, 2, 1, 3, 4).reshape(CROP, CROP, 3))
    return Image.fromarray(out, "RGB")


def derangement(image_ids, seed=PIX_SEED):
    """A fixed, seeded derangement of the POPE image set: every image is paired with a
    DIFFERENT real image, none with itself, and the map is a bijection.

    Why this arm exists (prereg Amendment 1).  The standing objection to a flat grey field is
    that it is out of distribution, so "blind" might mean "degenerate" rather than
    "image-free".  A real-but-mismatched COCO image is in distribution and still carries no
    information about the queried image, so it is the cleanest version of the same control.
    Liao et al. require several ablations to agree; grey and mismatch are that pair here.
    """
    ids = sorted(int(i) for i in image_ids)
    n = len(ids)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            break
    else:
        raise AssertionError("FATAL_NO_DERANGEMENT")
    m = {ids[i]: ids[int(perm[i])] for i in range(n)}
    assert len(set(m.values())) == n, "FATAL_DERANGEMENT_NOT_BIJECTIVE"
    assert all(k != v for k, v in m.items()), "FATAL_DERANGEMENT_HAS_FIXED_POINT"
    return m


def a_mismatch(im, image_id):
    """Handled by the caller, which substitutes the partner image's pixels wholesale.
    Present here only so the arm has an entry in PIXEL_ARMS."""
    raise AssertionError("FATAL_MISMATCH_MUST_BE_BUILT_BY_CALLER")


PIXEL_ARMS = {
    "sighted":  (a_sighted,  "own",  "unmodified image"),
    "grey":     (a_grey,     "own",  "flat (127,127,127) at the image's own pixel size"),
    "mismatch": (a_mismatch, "own",  "a DIFFERENT real COCO image, fixed seeded derangement "
                                     "over the 500 POPE images, no image paired with itself"),
    "noise25":  (a_noise25,  "m336", "additive Gaussian sigma=25/255, clipped"),
    "noise50":  (a_noise50,  "m336", "additive Gaussian sigma=50/255, clipped"),
    "noise100": (a_noise100, "m336", "additive Gaussian sigma=100/255, clipped"),
    "blur4":    (a_blur4,    "m336", "Gaussian blur radius 4 px at 336"),
    "blur16":   (a_blur16,   "m336", "Gaussian blur radius 16 px at 336"),
    "lowres":   (a_lowres,   "m336", "bicubic 336->16->336"),
    "pshuffle": (a_pshuffle, "m336", "permute the 24x24 grid of 14x14 ViT patches"),
}

TIER_A = ("sighted", "grey", "mismatch")
ABLATIONS = ("grey", "mismatch")
TIER_B = ("noise25", "noise50", "noise100", "blur4", "blur16", "lowres", "pshuffle")


def make(arm, im, image_id):
    fn = PIXEL_ARMS[arm][0]
    out = fn(im, image_id)
    assert out.mode == "RGB", "FATAL_MODE %s" % out.mode
    return out
