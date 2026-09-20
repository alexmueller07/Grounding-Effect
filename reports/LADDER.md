# LADDER — the ablation ladder on both endpoints

Lane `ladder`. Pre-registration: `SPRINT/LADDER_PREREG.md`, committed at `f186bcf` **before any
generation**, with Amendments 1 (`09ddcf6`), 2 (`463fe01`) and 3 (`122bbd3`), each dated and each
recorded before the data it governs existed, plus a provenance note (`8e2aa7a`).

Code: `/data/alexmueller/sprint/ladder/code`. Output: `.../out`. Nothing under `sc1_interv/`,
`mitdecomp/`, `paialpha/` or `j5_gates/` was modified; the paper build directory was not touched.

---

## 1. The headline, stated plainly

**The paper's central null does not survive an in-distribution ablation.**

`Δ_image` — the part of the one-scene-to-five-scene rise attributable to the image — is null under
flat grey and under **every one of seven graded degradations**, and is **+0.16 with an interval
excluding both zero and the ±0.10 reference margin** when the image is replaced by a *different
real COCO image*.

| ablation | what the encoder receives | Δ_image | verdict |
|---|---|---|---|
| flat grey (the paper's blind arm) | a uniform field, out of distribution | **−0.0160** [−0.0726, +0.0410] | null |
| flat black (a second uniform field) | a uniform field, out of distribution | **+0.0085** [−0.0494, +0.0671] | null |
| seven graded degradations | a corrupted version of the right image | **−0.0125 … +0.0012**, all null | null |
| **real but wrong image** | **a valid, in-distribution scene** | **+0.1598** [+0.1052, +0.2129] | **image contributes** |

The grey and mismatch intervals do not overlap. Paired on the same bootstrap draw, the difference
is **+0.1758 [+0.1193, +0.2302]**.

This is the adverse outcome pre-registered at §8.2 and Amendment 2, and the registered headline
verdict is **`IN-DISTRIBUTION-ABLATION-DISAGREES`**, which Amendment 2 fixed in advance as
overriding a `LADDER-NULL` on the degradation rungs.

### Why the two ablations disagree, mechanically

The image-attributable part of the mismatch effect is **entirely in the hit rate**: dH = +0.1494
[+0.0968, +0.2014], dF = −0.0104 [−0.0336, +0.0130]. In levels:

| condition | H (one scene) | H (five scenes) | change |
|---|---|---|---|
| true image | 0.676 | 0.780 | **+0.104** |
| flat grey | 0.468 | 0.708 | +0.241 |
| real but wrong image | 0.644 | 0.599 | **−0.045** |

With the true image the model gets *better* at carrying forward objects that are really present as
the passage covers more scenes. With a valid wrong image it gets *worse*. Grey hides this because
grey collapses the one-scene baseline (H = 0.468) so far that the *difference* is preserved while
the *level* is destroyed. That is the "degenerate regime" objection made quantitative: the grey
arm does not measure an image-free condition, it measures a condition in which the model has
stopped using the visual pathway at all, and a contrast of two such conditions can look normal.

A qualitative check confirms the mechanism rather than inferring it. With the prefix token ids
bit-identical, the mismatch arm fluently describes the substituted scene:

> prefix: *"Half a dozen donuts are sitting in a box. Vari…"*
> sighted: *"…ous donuts are placed in the box, with some closer to the front…"*
> mismatch: *"…ous benches are placed around the area… A person is walking down the snowy path…"*

The model is not collapsing. It is competently describing a different image, and that competes
with carrying the prefix's objects forward.

### What this does and does not license

* It **does** show the estimand `Δ_image` is not robust to the choice of ablation, and that the
  grey-blank default under-states the image's contribution on this endpoint.
* It **does not** show the mismatch arm is the "correct" counterfactual. A wrong image supplies
  *conflicting* evidence, not *absent* evidence; grey supplies absent evidence but pushes the
  encoder out of distribution. Neither is a clean estimate of "the image's contribution", and the
  honest summary is that the two disagree by 0.18 and the paper cannot claim the null without
  saying which ablation it holds for.
* The seven degradations show the null is **not** an artifact of flat grey *in particular* — it
  holds all along the "destroy the image" axis. What breaks it is stepping off that axis.

---

## 2. Method, and the two controls that license reading anything

Seven degradation rungs applied **in the encoder's own canonical geometry** (the CLIP
resize-shortest-edge-336 plus 336×336 centre crop the processor performs anyway), plus three
control rungs. Stochastic rungs are seeded from `(LAD_SEED, rung index, image_id)` only — never
from the arm or the derivation index — so the two arms of every contrast see the byte-identical
degraded image.

| rung | definition |
|---|---|
| `noise16` / `noise32` / `noise64` | additive i.i.d. Gaussian, σ = 16/32/64 of 255, clipped |
| `blur4` / `blur16` | PIL `GaussianBlur` radius 4.0 / 16.0 px at 336 |
| `lowres16` | BOX downsample to 16×16, BOX upsample to 336×336 |
| `shuffle24` | fixed permutation of the 576 14×14 patches (the ViT-L/14-336 token grid) |
| `grey` / `black` | flat (127,127,127) / (0,0,0) — uniform-field controls |
| `mismatch` | a different real COCO image, one fixed fixed-point-free derangement of the 500 |

`GATE_GEOM_IDENTITY` (job 21024, no model, no endpoint, run **before** the registration was
finalised): `proc(canon(im))` is **bit-identical** to `proc(im)` on 32 real images, max abs diff
**0.0**. So the canonical pre-pass is a measured no-op, the identity rung *is* the sighted arm
bit-for-bit, and `apply_rung("grey", …)` reaches the same tensor as the paper's own
`wia_blind_common.gray_image` (32/32). Every rung was also asserted deterministic, distinct from
clean, and — for non-uniform rungs — distinct between images.

### Positive controls

Both passed before any new endpoint was read.

| control | measured | reference | abs dev | tol | pass |
|---|---|---|---|---|---|
| PC1 Delta_primary | 0.48587082 | 0.4859 | 2.92e-05 | 0.005 | PASS |
| PC1 Delta_blind | 0.50187704 | 0.5019 | 2.30e-05 | 0.005 | PASS |
| PC1 Delta_image | -0.01600622 | -0.016 | 6.22e-06 | 0.005 | PASS |
| PC2a token identity | 3000/3000 (1.0000) | >= 0.98 | - | - | PASS |
| PC2b Delta_image(grey re-gen) | -0.016006 | -0.0160 | 6.22e-06 | 0.01 | PASS |

PC1 reproduces the published numbers to 3×10⁻⁵ — and to the digit against Appendix B's
independently written second scorer, which reports `0.48587082`. PC2 is the stronger control: the
grey arm **re-generated end to end through this lane's own generator** is token-identical to
`at_blind1.jsonl` on **3000/3000** continuations, and returns Δ_image = −0.016006, deviation
0.000000. The harness is the paper's harness.


---

## 3. The collapse screen was wrong twice, and the second fix decided the lane's headline

This is the most consequential methodological finding here, and it is reported first because it
determines whether the adverse result above survives.

The pre-registration's §5 rule said a rung could be read as IMAGE-CONTRIBUTES only if it was "no
more degenerate than the grey arm" on median continuation length, empty rate and two diversity
ratios. Both amendments to that rule were forced by errors **running in the direction that
protects the paper**.

**Amendment 1 (before any rung endpoint existed).** In the one-scene condition the grey arm is
*more* lexically diverse than the sighted arm — distinct-8-gram 0.780 against 0.575 — because its
continuations are short and varied while the sighted ones are long and repetitive. So "no more
degenerate than grey" is a bar **the paper's own sighted arm fails**. Re-set to the weaker of
sighted and grey.

**Amendment 3 (before any ladder verdict was read).** Amendment 1 was not enough, and the residual
error was worse. In the five-scene condition **both** reference arms sit at the 384-token
generation budget:

| condition | sighted median | grey median | cap-hit rate (grey) | resulting bar |
|---|---|---|---|---|
| `cap1` | 64 | 16 | 0.231 | 16 |
| `cap5` | 384 | 384 | **0.715** | **384 — the ceiling itself** |

The minimum of two ceiling values is the ceiling, so every rung that **terminates naturally**
before the budget was flagged "more degenerate" when it is *less* degenerate. The diversity
criteria have the same defect from the other side: those ratios fall mechanically as output
length rises, so a rung that writes *more* is flagged as writing *worse*.

The corrected rule tests collapse **absolutely, at the bottom of the length distribution** —
median < 8 tokens, empty rate > 0.25, a new ≤3-token rate > 0.25, distinct-8-gram < 0.20, fewer
than 10 distinct texts, or fewer than 100 true/false units — thresholds calibrated so the
published grey arm passes and the arms the paper declines to interpret (Qwen2.5-VL-7B at medians
of 4–5 tokens) fail. The cap-hit rate is reported for every arm and is **never** a criterion in
either direction. The comparative comparison survives only as a diagnostic.

### What the fix changed

| rung | corrected rule | ORIGINAL rule | flags that would have fired |
|---|---|---|---|
| `mismatch` | **IMAGE-CONTRIBUTES** | CONFOUNDED-BY-COLLAPSE | cap1: ngram8 0.567 < weaker-ref 0.575; cap1: tokid 0.485 < weaker-ref 0.493 |

**It changed exactly one verdict, and it was the one that matters.** Under the pre-amendment rule
the mismatch arm — the single most damaging result in the lane — would have been labelled
`CONFOUNDED-BY-COLLAPSE` and discarded, on the strength of an 8-gram ratio of 0.567 against the
sighted arm's 0.575 and a token ratio of 0.485 against 0.493. Differences of 0.008, in statistics
that fall mechanically with output length, against an arm that is otherwise **healthier than the
sighted arm** (median 70 vs 64, 1376 vs 1386 distinct texts, identical empty rate).

How badly mis-set the original bar was is visible in the diagnostic column below: measured against
grey, **every single rung is flagged**, including `noise16`, which is almost indistinguishable from
the sighted arm. A screen that rejects everything is not a screen.

### Diagnostic: the original comparison against grey, for every rung

| rung | more degenerate than grey | more degenerate than weaker reference |
|---|---|---|
| `noise16` | cap1: ngram8 0.580 < grey 0.780; cap1: tokid 0.496 < grey 0.699 | no |
| `noise32` | cap1: ngram8 0.577 < grey 0.780; cap1: tokid 0.494 < grey 0.699 | no |
| `noise64` | cap1: ngram8 0.579 < grey 0.780; cap1: tokid 0.494 < grey 0.699 | no |
| `blur4` | cap1: ngram8 0.608 < grey 0.780; cap1: tokid 0.522 < grey 0.699; cap5: empty 0.007 > grey 0.005 | cap5: empty 0.007 > weaker-ref 0.005 |
| `blur16` | cap1: ngram8 0.735 < grey 0.780; cap1: tokid 0.652 < grey 0.699; cap5: empty 0.007 > grey 0.005 | cap5: empty 0.007 > weaker-ref 0.005 |
| `lowres16` | cap1: ngram8 0.714 < grey 0.780; cap1: tokid 0.629 < grey 0.699; cap5: ngram8 0.322 < grey 0.327; cap5: tokid 0.252 < grey 0.260; cap5: empty 0.007 > grey 0.005 | cap5: ngram8 0.322 < weaker-ref 0.327; cap5: tokid 0.252 < weaker-ref 0.260; cap5: empty 0.007 > weaker-ref 0.005 |
| `shuffle24` | cap1: ngram8 0.501 < grey 0.780; cap1: tokid 0.424 < grey 0.699; cap5: ngram8 0.238 < grey 0.327; cap5: tokid 0.168 < grey 0.260 | cap1: ngram8 0.501 < weaker-ref 0.575; cap1: tokid 0.424 < weaker-ref 0.493; cap5: ngram8 0.238 < weaker-ref 0.327; cap5: tokid 0.168 < weaker-ref 0.260 |
| `mismatch` | cap1: ngram8 0.567 < grey 0.780; cap1: tokid 0.485 < grey 0.699 | cap1: ngram8 0.567 < weaker-ref 0.575; cap1: tokid 0.485 < weaker-ref 0.493 |
| `black` | cap1: ngram8 0.752 < grey 0.780; cap1: tokid 0.679 < grey 0.699; cap1: empty 0.047 > grey 0.045; cap5: empty 0.006 > grey 0.005 | cap1: empty 0.047 > weaker-ref 0.045; cap5: empty 0.006 > weaker-ref 0.005 |

---

## 4. L1 — the ladder on the prefix endpoint (LLaVA-1.5-7B, 1,500 cells, 500 images)

Estimand `Δ_image(r) = Δ_primary − Δ_r`, with `Δ_primary` read from the frozen sighted rows and
every contrast computed **inside one bootstrap loop on one shared image-clustered draw**
(B = 4000, seed 20260919, resample size asserted per replicate, no `set()` anywhere on the path).
The seven degradation rungs form the Bonferroni family (m = 7); `grey`, `black` and `mismatch`
are single pre-specified comparisons at the nominal level, fixed in advance.

Delta_primary (frozen sighted) = +0.4859 [+0.4321, +0.5410]

| rung | Delta_rung | Delta_image (95%) | width | Bonferroni 99.29% | verdict |
|---|---|---|---|---|---|
| `grey` (published, frozen) | +0.5019 [+0.4458, +0.5624] | -0.0160 [-0.0726, +0.0410] | 0.1137 | - | positive control |
| `grey` (this lane, re-generated) | +0.5019 [+0.4458, +0.5624] | -0.0160 [-0.0726, +0.0410] | 0.1137 | - | positive control |
| `noise16` | +0.4934 [+0.4394, +0.5488] | -0.0076 [-0.0379, +0.0216] | 0.0596 | -0.0076 [-0.0488, +0.0325] | **NULL-WITHIN-MARGIN** |
| `noise32` | +0.4924 [+0.4399, +0.5480] | -0.0065 [-0.0447, +0.0298] | 0.0745 | -0.0065 [-0.0571, +0.0425] | **NULL-WITHIN-MARGIN** |
| `noise64` | +0.4846 [+0.4314, +0.5405] | +0.0012 [-0.0364, +0.0390] | 0.0754 | +0.0012 [-0.0483, +0.0537] | **NULL-WITHIN-MARGIN** |
| `blur4` | +0.4984 [+0.4426, +0.5567] | -0.0125 [-0.0536, +0.0307] | 0.0843 | -0.0125 [-0.0701, +0.0447] | **NULL-WITHIN-MARGIN** |
| `blur16` | +0.4850 [+0.4269, +0.5439] | +0.0008 [-0.0547, +0.0548] | 0.1095 | +0.0008 [-0.0722, +0.0767] | **NULL-WITHIN-MARGIN** |
| `lowres16` | +0.4894 [+0.4299, +0.5478] | -0.0035 [-0.0546, +0.0480] | 0.1026 | -0.0035 [-0.0709, +0.0676] | **NULL-WITHIN-MARGIN** |
| `shuffle24` | +0.4983 [+0.4464, +0.5537] | -0.0124 [-0.0632, +0.0405] | 0.1037 | -0.0124 [-0.0790, +0.0579] | **NULL-WITHIN-MARGIN** |
| `black` | +0.4773 [+0.4163, +0.5394] | +0.0085 [-0.0494, +0.0671] | 0.1165 | +0.0085 [-0.0494, +0.0671] | **NULL-WITHIN-MARGIN** |
| `mismatch` (real but wrong image) | +0.3261 [+0.2699, +0.3837] | +0.1598 [+0.1052, +0.2129] | 0.1077 | +0.1598 [+0.1052, +0.2129] | **IMAGE-CONTRIBUTES** |

Pooled ladder mean Delta_image (seven degradation rungs only) = -0.0058 [-0.0365, +0.0253]

**Post hoc** — each ablation's Delta_image minus grey's, paired on the same bootstrap draw:

| rung | Delta_image(rung) - Delta_image(grey) |
|---|---|
| `pubgrey` | +0.0000 [+0.0000, +0.0000] |
| `grey` | +0.0000 [+0.0000, +0.0000] |
| `noise16` | +0.0084 [-0.0519, +0.0656] |
| `noise32` | +0.0095 [-0.0499, +0.0659] |
| `noise64` | +0.0172 [-0.0408, +0.0763] |
| `blur4` | +0.0035 [-0.0568, +0.0639] |
| `blur16` | +0.0168 [-0.0235, +0.0577] |
| `lowres16` | +0.0125 [-0.0317, +0.0571] |
| `shuffle24` | +0.0036 [-0.0533, +0.0593] |
| `mismatch` | +0.1758 [+0.1193, +0.2302] |
| `black` | +0.0245 [-0.0092, +0.0593] |

**LADDER_VERDICT (seven degradation rungs) = LADDER-NULL**

**MISMATCH_VERDICT = IMAGE-CONTRIBUTES**

**OVERALL_VERDICT = IN-DISTRIBUTION-ABLATION-DISAGREES**


The two `grey` rows are identical because PC2 found the re-generation token-identical on all
3,000 rows — they are the same continuations, not a duplicated row.

### No arm is degenerate

Every arm of every rung passes the absolute screen, including both uniform-field controls. This matters: a collapsed arm shrinks `Δ_r`
and so inflates `Δ_image` in the adverse direction, which means collapse and a genuine image
contribution push the primary the same way. None of these arms collapsed.

| arm | n | median len | mean len | cap-hit | empty | <=3 tok | distinct-8gram | distinct-tokid | distinct texts | top text share | true/false units | screen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `cap1` | 1500 | 64 | 185.5 | 0.448 | 0.031 | 0.143 | 0.575 | 0.493 | 1386 | 0.031 | 528/2332 | ok |
| `cap5` | 1500 | 384 | 272.2 | 0.677 | 0.005 | 0.029 | 0.377 | 0.279 | 1477 | 0.009 | 892/4988 | ok |
| `cap1g` | 1500 | 16 | 102.1 | 0.231 | 0.045 | 0.215 | 0.780 | 0.699 | 1298 | 0.057 | 528/2332 | ok |
| `cap5g` | 1500 | 384 | 282.8 | 0.715 | 0.005 | 0.033 | 0.327 | 0.260 | 1409 | 0.013 | 892/4988 | ok |
| `cap1_grey` | 1500 | 16 | 102.1 | 0.231 | 0.045 | 0.215 | 0.780 | 0.699 | 1298 | 0.057 | 528/2332 | ok |
| `cap5_grey` | 1500 | 384 | 282.8 | 0.715 | 0.005 | 0.033 | 0.327 | 0.260 | 1409 | 0.013 | 892/4988 | ok |
| `cap1_noise16` | 1500 | 61 | 183.5 | 0.443 | 0.031 | 0.139 | 0.580 | 0.496 | 1382 | 0.031 | 528/2332 | ok |
| `cap5_noise16` | 1500 | 384 | 274.3 | 0.683 | 0.005 | 0.029 | 0.369 | 0.273 | 1475 | 0.008 | 892/4988 | ok |
| `cap1_noise32` | 1500 | 63 | 184.1 | 0.445 | 0.031 | 0.142 | 0.577 | 0.494 | 1382 | 0.031 | 528/2332 | ok |
| `cap5_noise32` | 1500 | 384 | 274.6 | 0.685 | 0.004 | 0.025 | 0.366 | 0.274 | 1480 | 0.008 | 892/4988 | ok |
| `cap1_noise64` | 1500 | 61 | 183.3 | 0.442 | 0.031 | 0.137 | 0.579 | 0.494 | 1380 | 0.031 | 528/2332 | ok |
| `cap5_noise64` | 1500 | 384 | 276.0 | 0.690 | 0.005 | 0.025 | 0.359 | 0.268 | 1480 | 0.007 | 892/4988 | ok |
| `cap1_blur4` | 1500 | 54 | 171.7 | 0.411 | 0.035 | 0.151 | 0.608 | 0.522 | 1376 | 0.035 | 528/2332 | ok |
| `cap5_blur4` | 1500 | 384 | 273.6 | 0.684 | 0.007 | 0.030 | 0.367 | 0.276 | 1475 | 0.009 | 892/4988 | ok |
| `cap1_blur16` | 1500 | 25 | 119.7 | 0.276 | 0.042 | 0.197 | 0.735 | 0.652 | 1319 | 0.052 | 528/2332 | ok |
| `cap5_blur16` | 1500 | 384 | 274.9 | 0.693 | 0.007 | 0.033 | 0.345 | 0.275 | 1458 | 0.009 | 892/4988 | ok |
| `cap1_lowres16` | 1500 | 29 | 127.9 | 0.297 | 0.039 | 0.175 | 0.714 | 0.629 | 1342 | 0.043 | 528/2332 | ok |
| `cap5_lowres16` | 1500 | 384 | 285.6 | 0.722 | 0.007 | 0.029 | 0.322 | 0.252 | 1453 | 0.007 | 892/4988 | ok |
| `cap1_shuffle24` | 1500 | 384 | 213.5 | 0.525 | 0.027 | 0.114 | 0.501 | 0.424 | 1402 | 0.027 | 528/2332 | ok |
| `cap5_shuffle24` | 1500 | 384 | 322.8 | 0.820 | 0.005 | 0.017 | 0.238 | 0.168 | 1479 | 0.005 | 892/4988 | ok |
| `cap1_mismatch` | 1500 | 70 | 188.5 | 0.457 | 0.031 | 0.139 | 0.567 | 0.485 | 1376 | 0.035 | 528/2332 | ok |
| `cap5_mismatch` | 1500 | 384 | 274.6 | 0.683 | 0.003 | 0.026 | 0.370 | 0.272 | 1483 | 0.008 | 892/4988 | ok |
| `cap1_black` | 1500 | 17 | 111.8 | 0.260 | 0.047 | 0.220 | 0.752 | 0.679 | 1289 | 0.057 | 528/2332 | ok |
| `cap5_black` | 1500 | 384 | 284.0 | 0.721 | 0.006 | 0.035 | 0.330 | 0.265 | 1402 | 0.011 | 892/4988 | ok |


The degradation rungs do move the model monotonically toward the grey regime — one-scene median
continuation length runs sighted 64 → `noise16` 61 → `noise32` 63 → `noise64` 61 → `blur4` 54 →
`lowres16` 29 → `blur16` 25 → black 17 → grey 16 — and `Δ_image` stays flat at zero the whole
way. The
mismatch arm sits at 70, *above* the sighted arm.

`shuffle24` is the one rung worth flagging: it drives the model to the token cap in both
conditions (cap-hit 0.525 and 0.820) and has the lowest diversity of any arm (`cap5` 8-gram
0.238). It clears the absolute floor of 0.20 but not by much, so its null is the thinnest of the
seven.

### The rate split, and where the two rates cancel

| rung | image-attributable dH | image-attributable dF |
|---|---|---|
| `noise16` | -0.0022 [-0.0304, +0.0244] | +0.0053 [-0.0079, +0.0180] |
| `noise32` | +0.0057 [-0.0290, +0.0393] | +0.0122 [-0.0018, +0.0259] |
| `noise64` | +0.0207 [-0.0155, +0.0564] | +0.0194 [+0.0027, +0.0365] |
| `blur4` | -0.0179 [-0.0573, +0.0216] | -0.0054 [-0.0221, +0.0110] |
| `blur16` | -0.0606 [-0.1125, -0.0091] | -0.0615 [-0.0871, -0.0362] |
| `lowres16` | -0.0538 [-0.1010, -0.0062] | -0.0503 [-0.0736, -0.0268] |
| `shuffle24` | +0.0472 [-0.0002, +0.0968] | +0.0596 [+0.0377, +0.0816] |
| `mismatch` | +0.1494 [+0.0968, +0.2014] | -0.0104 [-0.0336, +0.0130] |
| `black` | -0.0956 [-0.1476, -0.0410] | -0.1041 [-0.1298, -0.0778] |


This reproduces, rung by rung, the pattern of the paper's Table 4: on several rungs the image's
contributions to `H` and to `F` both move and cancel in `J`. On `blur16` and `lowres16` both are
materially negative with intervals excluding zero (dH −0.061 and −0.054; dF −0.062 and −0.050) and
`J` is null anyway. On `shuffle24` dF is positive and excludes zero. Only on `mismatch` is the
effect carried by `H` alone. **A single score hides exactly this**, which is the paper's own
argument, now demonstrated across eight ablations rather than one.

### Length matching does not explain the mismatch result

| rung | within-condition | cross-condition |
|---|---|---|
| `noise16` | -0.0094 [-0.0368, +0.0178] | +0.0063 [-0.0171, +0.0306] |
| `noise32` | -0.0093 [-0.0442, +0.0256] | +0.0128 [-0.0166, +0.0428] |
| `noise64` | -0.0170 [-0.0541, +0.0188] | +0.0272 [-0.0026, +0.0558] |
| `blur4` | -0.0054 [-0.0425, +0.0327] | +0.0330 [+0.0033, +0.0638] |
| `blur16` | -0.0137 [-0.0671, +0.0397] | -0.0088 [-0.0530, +0.0345] |
| `lowres16` | -0.0148 [-0.0618, +0.0357] | -0.0024 [-0.0421, +0.0371] |
| `shuffle24` | -0.0524 [-0.0995, -0.0049] | -0.0177 [-0.0542, +0.0196] |
| `mismatch` | +0.1311 [+0.0805, +0.1802] | +0.1379 [+0.0960, +0.1808] |
| `black` | +0.0076 [-0.0471, +0.0641] | +0.0267 [-0.0168, +0.0710] |
| `grey` | -0.0124 [-0.0662, +0.0431] | +0.0223 [-0.0216, +0.0660] |


The cross-condition rule holds the compared continuations at equal length by construction, so a
length difference cannot produce its estimate. It returns **+0.1379 [+0.0960, +0.1808]** for
mismatch — larger than the raw estimate — while every degradation rung stays inside ±0.04.
`blur4`'s cross-condition interval excludes zero at +0.0330 [+0.0033, +0.0638], well inside the
±0.10 margin and one of eight secondaries, so we do not read it as an effect.


### The ladder reaches and exceeds grey's severity — this is not a "too mild" null

Severity is measured, not assumed: the cosine between the vision tower's embedding of the clean
canonical image and of the ablated one, over the same 500 images. Lower is more severe.



`mismatch` has no pixel transform, so its severity is the cosine between the clean embeddings of
image A and of its substitute B (`lad_dose_mm.json`). An independent second permutation returns
0.9213 against this derangement's 0.9201, so the number is a property of "a different image", not
of this particular pairing.

**Four ablations are strictly more severe than `mismatch`, and all four return a null:**

| ablation | emb_cos (lower = more severe) | Δ_image |
|---|---|---|
| `blur16` | 0.8740 | +0.0008 [−0.0547, +0.0548] |
| `grey` | 0.8979 | −0.0160 [−0.0726, +0.0410] |
| `black` | 0.9063 | +0.0085 [−0.0494, +0.0671] |
| `lowres16` | 0.9106 | −0.0035 [−0.0546, +0.0480] |
| **`mismatch`** | **0.9201** | **+0.1598 [+0.1052, +0.2129]** |
| `shuffle24` | 0.9214 | −0.0124 [−0.0632, +0.0405] |

`shuffle24` is within 0.0013 of `mismatch` on this axis and returns −0.0124. So the mismatch
result **cannot** be explained by it removing more image information — it removes less than four
ablations that show nothing. Spearman between measured severity and `Δ_image` across the nine
pixel ablations is **+0.10** (n = 9, descriptive, not a gate): the ladder is flat at zero along
the whole severity axis.

What distinguishes `mismatch` is not how much it destroys but that what it supplies is a
**coherent alternative scene**. That is the finding.

---

## 5. L2 — the ladder on standard CHAIR

Appendix G reports that the grey control could not be run at all: under a grey image
LLaVA-1.5-7B named none of the 80 COCO categories in any of 500 captions, so CHAIR was 0/0. The
ladder makes the control runnable.

### Positive control (PC3)

| quantity | measured | reference | abs dev |
|---|---|---|---|
| CHAIR_s_vanilla | 45.000000 | 45.0 | 0.00e+00 |
| CHAIR_i_vanilla | 12.777633 | 12.78 | 2.37e-03 |
| CHAIR_i_pai05 | 7.213396 | 7.2 | 1.34e-02 |
| H_vanilla | 0.781577 | 0.78 | 1.58e-03 |
| F_vanilla | 0.010018 | 0.01 | 1.85e-05 |

PC3 PASS = True. Sighted PAI gain on CHAIR_i = +5.564 [+3.881, +7.176], dJ = -0.0683 [-0.0881, -0.0485]

### The denominator screen

| rung | emb_cos | material? | mention occurrences | captions with >=1 mention | CHAIR_i | CHAIR_s | median tokens | distinct texts | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `noise16` | 0.9925 | no | 3755 | 500/500 | 13.05 | 45.60 | 114 | 500 | **USABLE** |
| `noise32` | 0.9844 | no | 3593 | 500/500 | 11.69 | 37.60 | 112 | 500 | **USABLE** |
| `noise64` | 0.9660 | no | 3346 | 498/500 | 13.36 | 39.40 | 101 | 500 | **USABLE** |
| `blur4` | 0.9558 | yes | 3310 | 489/500 | 14.74 | 43.00 | 103 | 500 | **USABLE** |
| `blur16` | 0.8740 | yes | 1465 | 378/500 | 39.18 | 40.20 | 77 | 500 | **USABLE** |
| `lowres16` | 0.9106 | yes | 1441 | 405/500 | 43.79 | 42.60 | 78 | 500 | **USABLE** |
| `shuffle24` | 0.9214 | yes | 2439 | 450/500 | 25.83 | 40.40 | 101 | 499 | **USABLE** |
| `mismatch` | 0.9201 | yes | 3827 | 500/500 | 81.79 | 98.80 | 115 | 500 | **USABLE** |
| `black` | 0.9063 | yes | 0 | 0/500 | n/a (0/0) | 0.00 | 66 | 1 | **UNDEFINED (0/0)** |
| `grey_published` | - | - | 0 | 0/500 | n/a (0/0) | - | 66 | 1 | **UNDEFINED (0/0)** |

**Every one of the seven degradations leaves CHAIR's denominator non-empty**, against 0/500 for
grey. This closes the open item in Appendix G.7: the blinding control the paper says it could not
run *can* be run, at any rung of the ladder.

**Flat black behaves exactly like grey: 0 mention occurrences, 0/500 captions, one distinct
caption, median 66 tokens.** Amendment 3C registered both readings in advance, and this is the
second one: the emptying is **a property of uniform fields, not of the grey constant**, so
Appendix G.7's claim stands as written and its scope paragraph does not need rewriting. Lan et
al.'s contrasting AMBER result (blinded LLaVA-1.5-7B still naming objects under flat black) must
therefore be attributable to their benchmark, prompt or scorer rather than to the blinding colour
— we do not have their harness and do not adjudicate it further. *(That comparison rests on a
second-hand citation; see the provenance note.)*

### Does PAI's CHAIR gain survive ablation?

| rung | CHAIR_i gain at rung | image-attributable part of the gain | dJ at rung | image-attributable dJ | verdict |
|---|---|---|---|---|---|
| `noise16` | +4.489 [+3.091, +5.867] | +1.075 [-0.773, +2.886] | -0.0855 [-0.1031, -0.0678] | +0.0172 [-0.0048, +0.0388] | **GAIN-SURVIVES-BUT-ABLATION-NOT-MATERIAL** |
| `noise32` | +3.327 [+1.919, +4.808] | +2.237 [+0.356, +4.058] | -0.0873 [-0.1063, -0.0680] | +0.0190 [-0.0056, +0.0427] | **GAIN-SURVIVES-BUT-ABLATION-NOT-MATERIAL** |
| `noise64` | +5.002 [+3.540, +6.434] | +0.562 [-1.415, +2.544] | -0.0651 [-0.0847, -0.0450] | -0.0032 [-0.0297, +0.0227] | **GAIN-SURVIVES-BUT-ABLATION-NOT-MATERIAL** |
| `blur4` | +4.942 [+3.176, +6.685] | +0.622 [-1.814, +2.939] | -0.0652 [-0.0834, -0.0471] | -0.0032 [-0.0266, +0.0206] | **GAIN-SURVIVES-ABLATION** |
| `blur16` | -3.136 [-9.076, +2.842] | +8.700 [+2.554, +14.960] | -0.0398 [-0.0528, -0.0266] | -0.0285 [-0.0521, -0.0056] | **GAIN-IS-IMAGE-ATTRIBUTABLE** |
| `lowres16` | -3.974 [-7.683, -0.318] | +9.538 [+5.508, +13.489] | -0.0131 [-0.0259, -0.0007] | -0.0553 [-0.0792, -0.0321] | **CANNOT-RESOLVE** |
| `shuffle24` | +2.326 [-0.719, +5.331] | +3.238 [-0.197, +6.670] | -0.0652 [-0.0856, -0.0454] | -0.0031 [-0.0293, +0.0228] | **CANNOT-RESOLVE** |
| `mismatch` | +0.400 [-1.167, +1.962] | +5.164 [+2.885, +7.421] | +0.0010 [-0.0089, +0.0111] | -0.0694 [-0.0914, -0.0489] | **GAIN-IS-IMAGE-ATTRIBUTABLE** |

| rung | arm | CHAIR_i | CHAIR_s | H | F | J | d' | c |
|---|---|---|---|---|---|---|---|---|
| `noise16` | vanilla | 13.05 | 45.60 | 0.7736 | 0.0098 | 0.7637 | 3.083 | 0.791 |
| `noise16` | pai05 | 8.56 | 29.60 | 0.6836 | 0.0054 | 0.6782 | 3.028 | 1.036 |
| `noise32` | vanilla | 11.69 | 37.60 | 0.7452 | 0.0078 | 0.7374 | 3.076 | 0.879 |
| `noise32` | pai05 | 8.36 | 27.00 | 0.6547 | 0.0046 | 0.6501 | 3.003 | 1.103 |
| `noise64` | vanilla | 13.36 | 39.40 | 0.6793 | 0.0081 | 0.6712 | 2.869 | 0.969 |
| `noise64` | pai05 | 8.36 | 26.40 | 0.6103 | 0.0042 | 0.6061 | 2.913 | 1.176 |
| `blur4` | vanilla | 14.74 | 43.00 | 0.6553 | 0.0085 | 0.6468 | 2.787 | 0.994 |
| `blur4` | pai05 | 9.80 | 27.60 | 0.5863 | 0.0046 | 0.5816 | 2.821 | 1.192 |
| `blur16` | vanilla | 39.18 | 40.20 | 0.2116 | 0.0070 | 0.2046 | 1.657 | 1.629 |
| `blur16` | pai05 | 42.32 | 37.20 | 0.1704 | 0.0056 | 0.1648 | 1.586 | 1.746 |
| `lowres16` | vanilla | 43.79 | 42.60 | 0.1882 | 0.0070 | 0.1812 | 1.572 | 1.670 |
| `lowres16` | pai05 | 47.76 | 43.00 | 0.1747 | 0.0066 | 0.1681 | 1.545 | 1.708 |
| `shuffle24` | vanilla | 25.83 | 40.40 | 0.3971 | 0.0093 | 0.3878 | 2.094 | 1.308 |
| `shuffle24` | pai05 | 23.50 | 31.00 | 0.3287 | 0.0061 | 0.3226 | 2.064 | 1.475 |
| `mismatch` | vanilla | 81.79 | 98.80 | 0.1445 | 0.0370 | 0.1075 | 0.727 | 1.424 |
| `mismatch` | pai05 | 81.39 | 97.40 | 0.1377 | 0.0292 | 0.1085 | 0.802 | 1.492 |

**L2_VERDICT = noise16:GAIN-SURVIVES-BUT-ABLATION-NOT-MATERIAL; noise32:GAIN-SURVIVES-BUT-ABLATION-NOT-MATERIAL; noise64:GAIN-SURVIVES-BUT-ABLATION-NOT-MATERIAL; blur4:GAIN-SURVIVES-ABLATION; blur16:GAIN-IS-IMAGE-ATTRIBUTABLE; lowres16:CANNOT-RESOLVE; shuffle24:CANNOT-RESOLVE; mismatch:GAIN-IS-IMAGE-ATTRIBUTABLE**

Read against the registered severity floor (material iff `emb_cos` ≤ the midpoint between
identity and `mismatch`, i.e. ≤ 0.9601), the pattern is coherent: PAI's CHAIR gain **survives the
ablations that barely touch the image and disappears once the ablation is material**. The three
noise rungs keep the gain but fail the floor, so they are reported as
`GAIN-SURVIVES-BUT-ABLATION-NOT-MATERIAL` and are not quoted as evidence. At `blur16` the gain is
−3.14 [−9.08, +2.84] with an image-attributable part of +8.70 [+2.55, +14.96]; at `mismatch` the
gain is +0.40 [−1.17, +1.96] with an image-attributable part of +5.16 [+2.89, +7.42] against a
sighted gain of +5.56. **PAI's CHAIR improvement is substantially image-attributable.**

At `lowres16` PAI makes CHAIR *worse* (−3.97 [−7.68, −0.32]), which the registered rule returns as
`CANNOT-RESOLVE` because the interval excludes zero in the wrong direction.

### The L2 mismatch arm is algebraically the permutation secondary, and we verified it

CHAIR generation is deterministic and its prompt carries no image-specific text, so generating a
caption from image B and scoring it against image A's ground truth is the **same computation** as
generating from B normally and permuting the ground truth. We checked rather than assumed:
**500/500** `vanilla_mismatch` captions are bit-identical to `md_vanilla`'s caption of the
substituted image, and **500/500** for `pai05_mismatch` against `md_pai05`. So the L2 mismatch arm
reproduces the permuted-ground-truth secondary already in Appendix G.7 (+0.22 [−0.94, +1.33]
there, +0.40 [−1.17, +1.96] here, the difference being that theirs averages 20 derangements inside
each replicate and ours fixes one) and adds **no new generation-side information at L2**.

That equivalence does **not** hold at L1, and we checked that too: only **6/3000** L1 mismatch
continuations coincide with the sighted continuation of the substituted image, because the prefix
belongs to the target cell while the image is the substitute. The L1 mismatch arm is genuinely new
evidence; the L2 one is a re-derivation. Worth recording as a general point: on a prompt-only
endpoint, "mismatched image" and "permuted ground truth" are the same experiment.

---

## 6. Limitations, stated as limitations

* **One model on the primary.** LLaVA-1.5-7B, one prompt, one decoding mode, one 500-image draw.
  LLaVA-OV-0.5B was registered as a conditional secondary and **was not run**. The reason is
  specific, not a time excuse: its processor uses `image_grid_pinpoints` AnyRes multi-crop, so the
  336×336 canonical geometry that `GATE_GEOM_IDENTITY` licenses does not transfer, and the gate
  would have to be redesigned for a variable-size crop grid before any rung could be trusted there.
* **Eight ablations are not the space of ablations.** A null across seven degradations does not
  show that no degradation moves `Δ_image`; it shows these seven do not. The mismatch arm is
  evidence that the space matters more than the paper assumed.
* **Neither ablation is a clean counterfactual.** Grey removes information but leaves the encoder
  out of distribution; a mismatched image is in distribution but supplies *conflicting* rather
  than *absent* information. The defensible claim is the disagreement between them, not that
  either one is the true `Δ_image`.
* **`emb_cos` is measured on the vision tower we are also measuring through.** It orders the
  rungs; it is not a physical dose.
* **The severity floor and `shuffle24`.** `shuffle24` is the thinnest null of the seven: it drives
  the model to the token cap in both conditions and has the lowest diversity of any arm, clearing
  the absolute floor but not comfortably.
* **Two citations are second-hand.** arXiv 2509.23499v2 (the out-of-distribution objection, said
  to be ICLR 2026) and arXiv 2605.22903 (Lan et al., the flat-black precedent and its AMBER
  numbers) reached this lane through the sprint coordinator and **have not been verified at
  primary source here**. They motivate two arms; no verdict depends on them. Both must be checked
  at source — including the acceptance claim — before either enters the paper.
* **Post-hoc analyses are labelled.** The paired "each ablation's `Δ_image` minus grey's" contrast
  was added after the mismatch endpoint was read and is marked post hoc in `lad_l1.json`. It
  introduces no new estimator (it equals `Δ_grey − Δ_rung` on the same draw).

## 7. What this lane asks the paper to change

1. **Section 4.2 cannot claim the rise does not need the image without naming the ablation.** The
   image-attributable component is null under image-destroying ablation and +0.16 [+0.11, +0.21]
   under an in-distribution one.
2. **Section 6, recommendation 2 needs correcting.** It currently tells other researchers to adopt
   a grey-blank default for blinding grounding scores. On this endpoint that default returns a
   null that a real-but-wrong image does not. The recommendation should require an in-distribution
   mismatched-image arm alongside the blank, and should say the two can disagree by more than the
   effect being tested.
3. **The concession "We did not run a graded image degradation" is now false** and the ladder
   replaces it — with the result that the graded degradation *supports* the paper and the
   in-distribution control does not.
4. **The Limitations section should state that the paper's own blinding instrument has a measured
   blind spot**, and quote its size.

---

## 8. What ran, and how to check it

All generation on `vgi1`, `--gres=gpu:rtx_4090:1`, explicit `--time`, every payload ending
`|| exit 9`, never more than two concurrent GPU jobs, success verified by an output file plus a
sentinel string and never by SLURM state.

| job | what | outcome |
|---|---|---|
| 21024 | `GATE_GEOM_IDENTITY` (no model, no endpoint) | passed 32/32, max abs diff 0.0 |
| 21026 / 21027 | L1 and L2 smoke | gates passed; L2 grey 64/64 token-identical to `md_vanilla_blind` |
| 21028 | L1 `grey` (PC2) | 3,000 rows |
| 21029 | L2 vanilla × 7 degradations | denominator screen |
| 21040 | L1 `noise64` | 3,000 rows |
| 21061 | L1 `mismatch`, `blur16`, `lowres16`, `noise16` | 4 × 3,000 rows |
| 21062 | L1 `shuffle24`, `noise32`, `blur4` | 3 × 3,000 rows |
| 21063 / 21064 / 21088 / 21089 | superseded or failed (see below) | — |
| 21123 | L1 `black` (3,000 rows) then dose | black OK; dose crashed on `mismatch` |
| 21124 | L2 `black` + PAI × 9 | all arms |
| 21211 | `mismatch` severity anchor | `lad_dose_mm.json` |
| 21212 / 21213 | final L1 and L2 scoring | `lad_l1.json`, `lad_l2.json` |

### Failures, and what caused them

* **21028 and 21040 each ran only their first rung.** `sbatch --export=ALL,LAD_RUNGS=a,b,c`
  splits the value on commas and treats `b` and `c` as further variable names, so `LAD_RUNGS`
  arrived as `a`. Both jobs completed cleanly with a sentinel, having done a quarter of the work
  — the failure is silent by construction. Fixed by passing the variable inside the payload
  string instead. Recorded in project memory.
* **21088, 21089 (and 21064) aborted on the `black` rung**, `FATAL_RUNG_CONSTANT_ACROSS_IMAGES`.
  The "a rung must differ between images" gate exempted only `grey`; flat black is a uniform
  field too, so it is *correctly* constant. Fixed by naming `UNIFORM_RUNGS = ("grey", "black")`
  and, for those rungs, asserting the opposite (all rows bit-identical) rather than skipping the
  check. Verified behaviourally: `black` and `grey` return byte-identical tensors across images
  with ranges (0,0) and (127,127), while `noise64` and `shuffle24` stay image-dependent and keep
  the original gate.
* **21064 and the dose step of 21123 crashed on `mismatch`**, which `apply_rung` refuses by
  design because it is not a pixel transform. The pixel-rung doses were re-run without it and
  `mismatch`'s severity is computed separately by `lad_dose_mm.py` from the clean embeddings of
  A and B.

### Reproducing

`lad_l1.json` and `lad_l2.json` carry every number in this report, the md5 of every shared module
imported read-only, and the per-rung `verdict` alongside `verdict_under_original_collapse_rule`.
Every table here was generated by `lad_report.py` directly from those files — no number in this
document was typed by hand — and all 129 table rows were checked for column-count alignment
against their headers before the file was written.
