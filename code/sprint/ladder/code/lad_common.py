"""Ablation-ladder lane (SPRINT/LADDER_PREREG.md) -- shared definitions.

WHAT THIS LANE TESTS. The paper's blind arm replaces the image with a flat mid-grey field.
Section 4.2 concedes "We did not run a graded image degradation, which would add a designed
dose-response control", and Appendix G.7 concedes "Swapping to a different ablation after
seeing the registered one degenerate was available and was not done". This lane runs the
graded degradation, on both endpoints.

NOTHING OUTSIDE /data/alexmueller/sprint/ladder IS WRITTEN OR EDITED. A queued job runs
whatever is on disk when it STARTS, so the shared harnesses under sc1_interv/code,
mitdecomp/code, paialpha/code and j5_gates/code are imported read-only and never modified.

TRANSFORM SPACE. Every rung is applied to the image in the encoder's own canonical geometry:
the CLIP resize-shortest-edge-to-336 (bicubic) plus 336x336 centre crop that
`CLIPImageProcessor` performs anyway. `canon()` reproduces that geometry with the processor
itself, so severities mean the same thing for every image regardless of its native size and
`shuffle24`'s blocks align with the ViT-L/14-336 patch grid (24x24 patches of 14 px).
GATE_GEOM_IDENTITY (lad_geomgate.py) asserts that feeding canon(im) back through the full
processor gives a pixel_values tensor BIT-IDENTICAL to feeding the raw image, i.e. that the
pre-pass is a no-op and the identity rung is the sighted arm. The pre-registration makes the
whole lane conditional on that gate: it aborts rather than silently changing the estimand.

RANDOMNESS. Every stochastic rung is seeded from (LAD_SEED, rung index, image_id) only -- not
from the arm, the derivation index k or the row order -- so the two arms of every contrast
(cap1 vs cap5) see the BYTE-IDENTICAL degraded image. The contrast is within-image; a rung
that re-drew its noise per arm would put noise into the contrast itself.
"""
import os
import numpy as np

LAD_SEED = 20260919
CANON = 336                       # CLIPImageProcessor crop_size for llava-1.5-7b
PATCH = 14                        # ViT-L/14-336
GRID = CANON // PATCH             # 24 -> 576 patches, matching the 576 image tokens
GREY = (127, 127, 127)            # wia_blind_common.GRAY, the paper's blinding constant
BLACK = (0, 0, 0)                 # Lan et al. arXiv 2605.22903's blinding constant

ROOT = "/data/alexmueller/sprint/ladder"
OUT = f"{ROOT}/out"
CODE = f"{ROOT}/code"

# ---- the ladder ---------------------------------------------------------------------------
# Fixed in SPRINT/LADDER_PREREG.md section 2 before any generation. Ordered by intended
# severity; the REALISED severity is measured separately (lad_dose.py) and reported, because
# a nominal knob is not a dose.
RUNGS = (
    "noise16",      # additive i.i.d. Gaussian, sigma = 16/255
    "noise32",      # sigma = 32/255
    "noise64",      # sigma = 64/255
    "blur4",        # PIL GaussianBlur radius 4.0 px (at 336)
    "blur16",       # PIL GaussianBlur radius 16.0 px
    "lowres16",     # BOX downsample to 16x16, BOX upsample to 336x336
    "shuffle24",    # fixed permutation of the 24x24 grid of 14x14 patches
)
# `identity` and `grey` are not ladder rungs: identity IS the sighted arm (frozen) and grey IS
# the published blind arm (frozen). Both are reproduced as positive controls, never re-read as
# new endpoints.
CONTROL_RUNGS = ("identity", "grey")
# Amendment 2 (2026-09-19): the in-distribution ablation. Not a degradation rung -- it
# substitutes a DIFFERENT REAL COCO image from the same 500-image sample, so the encoder
# receives a fully valid, in-distribution input that carries no information about THIS image.
# Appended at the END of ALL_RUNGS on purpose: RUNG_IX feeds the per-rung random seeds, so
# appending leaves every already-generated rung's seed untouched.
MISMATCH_RUNGS = ("mismatch",)
# Amendment 3 (2026-09-19): flat black, a second uniform-field control. Lan et al.
# (arXiv 2605.22903), whom the paper cites, blind with flat BLACK and report blinded
# LLaVA-1.5-7B still naming objects on AMBER (CHAIR 48.3 / Cover 6.4), whereas our flat GREY
# arm names none of the 80 categories in any of 500 captions. If black keeps CHAIR's
# denominator non-empty, the emptying is specific to grey rather than to uniform fields.
# Appended at the END so every already-generated rung's seed index is untouched.
BLACK_RUNGS = ("black",)
# Uniform fields: constant across images BY CONSTRUCTION. The "rung must differ between
# images" gate must not be applied to them; the correct gate is the opposite one.
UNIFORM_RUNGS = ("grey", "black")
ALL_RUNGS = RUNGS + CONTROL_RUNGS + MISMATCH_RUNGS + BLACK_RUNGS
RUNG_IX = {r: i for i, r in enumerate(ALL_RUNGS)}
MISMATCH_SEED = 20260919


def _rng(rung, image_id):
    return np.random.default_rng([LAD_SEED, RUNG_IX[rung], int(image_id)])


def derangement(ids):
    """THE fixed, seeded, fixed-point-free permutation of the frozen image sample.

    Deterministic in the sorted id list alone, so L1 and L2 get the identical mapping and it
    can be reproduced from the ids without the artefacts. Rejection-sampled: a permutation is
    accepted only if it has no fixed point, i.e. no image is ever paired with itself."""
    ids = sorted(int(i) for i in ids)
    n = len(ids)
    rng = np.random.default_rng([MISMATCH_SEED, n])
    for _ in range(10000):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            m = {ids[i]: ids[int(p[i])] for i in range(n)}
            assert len(set(m.values())) == n, "FATAL_DERANGEMENT_NOT_BIJECTIVE"
            assert all(k != v for k, v in m.items()), "FATAL_DERANGEMENT_HAS_FIXED_POINT"
            return m
    raise AssertionError("FATAL_NO_DERANGEMENT")


def canon(proc, im):
    """The processor's OWN resize+centre-crop geometry, returned as a uint8 336x336 RGB PIL
    image. Uses the processor rather than a re-implementation so it cannot drift from it."""
    from PIL import Image
    a = proc.image_processor(images=im, do_rescale=False, do_normalize=False,
                             return_tensors="np")["pixel_values"][0]
    a = np.asarray(a)
    if a.shape[0] == 3:                     # channels-first -> HWC
        a = np.transpose(a, (1, 2, 0))
    assert a.shape == (CANON, CANON, 3), f"FATAL_CANON_SHAPE {a.shape}"
    lo, hi = float(a.min()), float(a.max())
    assert -0.01 <= lo and hi <= 255.01, f"FATAL_CANON_RANGE {lo} {hi}"
    return Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8), "RGB")


def apply_rung(rung, im, image_id):
    """Apply one rung to a uint8 336x336 RGB PIL image. Deterministic in (rung, image_id)."""
    from PIL import Image, ImageFilter
    assert im.size == (CANON, CANON) and im.mode == "RGB", f"FATAL_RUNG_INPUT {im.size} {im.mode}"
    assert rung != "mismatch", "FATAL_MISMATCH_IS_NOT_A_PIXEL_TRANSFORM"
    if rung == "identity":
        out = im.copy()
    elif rung == "grey":
        out = Image.new("RGB", im.size, GREY)
    elif rung == "black":
        out = Image.new("RGB", im.size, BLACK)
    elif rung.startswith("noise"):
        sigma = float(rung[5:])
        x = np.asarray(im, dtype=np.float64)
        n = _rng(rung, image_id).normal(0.0, sigma, size=x.shape)
        out = Image.fromarray(np.clip(np.rint(x + n), 0, 255).astype(np.uint8), "RGB")
    elif rung.startswith("blur"):
        out = im.filter(ImageFilter.GaussianBlur(radius=float(rung[4:])))
    elif rung.startswith("lowres"):
        s = int(rung[6:])
        out = im.resize((s, s), Image.BOX).resize((CANON, CANON), Image.BOX)
    elif rung == "shuffle24":
        x = np.asarray(im)
        b = x.reshape(GRID, PATCH, GRID, PATCH, 3).transpose(0, 2, 1, 3, 4)   # (24,24,14,14,3)
        b = b.reshape(GRID * GRID, PATCH, PATCH, 3)
        p = _rng(rung, image_id).permutation(GRID * GRID)
        b = b[p].reshape(GRID, GRID, PATCH, PATCH, 3).transpose(0, 2, 1, 3, 4)
        out = Image.fromarray(np.ascontiguousarray(b.reshape(CANON, CANON, 3)), "RGB")
    else:
        raise AssertionError(f"FATAL_UNKNOWN_RUNG {rung}")
    assert out.size == (CANON, CANON) and out.mode == "RGB", f"FATAL_RUNG_OUTPUT {rung}"
    return out


class RungImages:
    """Per-(path, rung) cache with a determinism assertion. `canon` is called once per path."""

    def __init__(self, proc, rung, check_every=97, sub=None):
        self.proc, self.rung, self.check_every = proc, rung, check_every
        self.sub = sub                       # {image_id: (sub_image_id, sub_file)}
        assert (rung != "mismatch") or sub, "FATAL_MISMATCH_WITHOUT_SUBSTITUTION_MAP"
        self._canon, self._out = {}, {}
        self.n_calls = self.n_checked = self.n_recomputed_equal = 0

    def canon_of(self, path):
        from PIL import Image
        if path not in self._canon:
            self._canon[path] = canon(self.proc, Image.open(path).convert("RGB"))
        return self._canon[path]

    def _make(self, path, image_id):
        if self.rung == "mismatch":
            sid, sfile = self.sub[int(image_id)]
            assert int(sid) != int(image_id), f"FATAL_MISMATCH_SELF_PAIRED {image_id}"
            return self.canon_of(sfile)
        return apply_rung(self.rung, self.canon_of(path), image_id)

    def get(self, path, image_id):
        self.n_calls += 1
        key = (path, image_id)
        if key not in self._out:
            self._out[key] = self._make(path, image_id)
            return self._out[key]
        if self.n_calls % self.check_every == 0:      # GATE_RUNG_DETERMINISTIC, sampled
            again = self._make(path, image_id)
            self.n_checked += 1
            same = np.array_equal(np.asarray(again), np.asarray(self._out[key]))
            assert same, f"FATAL_RUNG_NONDETERMINISTIC {self.rung} {image_id}"
            self.n_recomputed_equal += 1
        return self._out[key]

    def stats(self):
        return {"rung": self.rung, "n_calls": self.n_calls, "n_distinct": len(self._out),
                "determinism_rechecks": self.n_checked,
                "determinism_equal": self.n_recomputed_equal}


def sentinel(name):
    print(f"LAD_SENTINEL_{name}_OK", flush=True)


def jdump(path, obj):
    import json
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
