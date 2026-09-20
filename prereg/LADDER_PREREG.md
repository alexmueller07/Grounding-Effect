# LADDER — pre-registration of the ablation-ladder experiment

Lane `ladder`. Written and committed **before any generation job for this lane was submitted**.
Conventions: `SPRINT/SPRINT_CONVENTIONS.md`.

Code: `/data/alexmueller/sprint/ladder/code` (own directory; nothing under `sc1_interv/code`,
`mitdecomp/code`, `paialpha/code` or `j5_gates/code` is edited — they are imported read-only).
Output: `/data/alexmueller/sprint/ladder/out`. Logs: `.../logs`.

---

## 1. The question, and why it can go against us

The paper's blind arm replaces the image with a flat mid-grey rectangle. Two objections, from
opposite sides, both concede the same gap:

1. A uniform field is out of distribution for the image encoder, so the blind arm may be a
   *degenerate* regime rather than an *image-free* one. Under a milder ablation the
   image-attributable component `Δ_image` might be larger — the image would then contribute to the
   `cap1 → cap5` rise whenever it carries partial information, and the paper's central null would
   be an artifact of the particular ablation it chose.
2. On standard CHAIR the grey arm is so destructive that LLaVA-1.5-7B names none of the 80
   categories in any of 500 captions, so `CHAIR_i` is 0/0 and the control cannot be run at all
   (Appendix G.7). A milder ablation might leave the denominator non-empty.

The paper already concedes both gaps in writing — Section 4.2: "We did not run a graded image
degradation, which would add a designed dose-response control"; Appendix G.7: "Swapping to a
different ablation after seeing the registered one degenerate was available and was not done".
This lane closes them. **It is the experiment in this sprint most likely to weaken the paper**,
and Section 8 fixes in advance what we write for each way it can come out.

---

## 2. The ladder: exact transform definitions

### 2.1 Transform space (fixed by a gate that has already run)

Every rung is applied to the image **in the encoder's own canonical geometry**: the
`CLIPImageProcessor` resize-shortest-edge-to-336 (bicubic, `resample=3`) plus 336×336 centre crop
that the processor performs anyway for `llava-hf/llava-1.5-7b-hf`. `lad_common.canon()` performs
that geometry *with the processor itself*, so it cannot drift from it.

This choice was made so that (a) a severity parameter means the same thing for every image
regardless of native size, and (b) `shuffle24`'s blocks coincide with the ViT-L/14-336 patch grid
(24×24 blocks of 14 px = the model's 576 image tokens).

**GATE_GEOM_IDENTITY, run before this registration was finalised, on 32 real COCO images
(job 21024, `out/lad_geomgate.json`):**

| gate | result |
|---|---|
| `proc(canon(im)).pixel_values` bit-identical to `proc(im).pixel_values` | **32/32, max abs diff 0.0** |
| `apply_rung("grey", canon(im))` reaches the same tensor as the paper's own `wia_blind_common.gray_image(path)` | **32/32** |
| `apply_rung("identity", canon(im))` equals `canon(im)` | **32/32** |
| every rung deterministic on re-computation | **32/32 each** |
| every rung changes the tensor; no cross-image collisions | **32/32 each, 0 collisions** |

The pre-pass is therefore a measured no-op: the **identity rung is the sighted arm bit-for-bit**,
and **our grey is the paper's grey bit-for-bit**. That gate involves no model and no endpoint. Had
it failed, this registration aborts the lane rather than quietly changing space.

### 2.2 The seven rungs

Applied to the uint8 336×336 RGB canonical image. `sigma` is in 0–255 units.

| rung | definition |
|---|---|
| `noise16` | `clip(round(x + N(0, 16)), 0, 255)`, i.i.d. per pixel per channel |
| `noise32` | as above, `sigma = 32` |
| `noise64` | as above, `sigma = 64` |
| `blur4` | `PIL.ImageFilter.GaussianBlur(radius=4.0)` |
| `blur16` | `PIL.ImageFilter.GaussianBlur(radius=16.0)` |
| `lowres16` | `resize((16,16), Image.BOX)` then `resize((336,336), Image.BOX)` |
| `shuffle24` | fixed permutation of the 576 14×14 patches of the 24×24 grid |

Two **control rungs**, which are not ladder rungs and are never read as new endpoints:

| control | definition | role |
|---|---|---|
| `identity` | unchanged canonical image | equals the frozen sighted arm |
| `grey` | flat `(127,127,127)` | equals the paper's published blind arm |

**Randomness.** Stochastic rungs (`noise*`, `shuffle24`) are seeded from
`np.random.default_rng([20260919, rung_index, image_id])` — from the image only, never from the
arm, the derivation index `k` or the row order. The two arms of every contrast therefore see the
**byte-identical** degraded image. A rung that re-drew noise per arm would put noise into the
contrast itself.

### 2.3 Realised dose is measured, not assumed

Nominal severity is not a dose. Two realised measures are computed for all nine rungs on the same
500 images and reported as the ladder's x-axis:

* `pix_rmse` — RMSE in the processor's normalised `pixel_values` space (already measured at the
  gate: `noise16` 0.22, `blur4` 0.33, `noise32` 0.43, `lowres16` 0.49, `blur16` 0.52, `noise64`
  0.79, `grey` 1.10, `shuffle24` 1.28).
* `emb_cos` — cosine similarity between the vision tower's 576 patch embeddings (mean-pooled, and
  also per-token mean cosine) for the degraded and the clean image, on the frozen 500-image draw.
  **This is the pre-registered severity axis**, because `pix_rmse` already demonstrably
  mis-orders the ladder: `shuffle24` has the largest pixel distance of all nine rungs while
  preserving every patch's content exactly.

Severity is `s(r) = 1 − emb_cos(r)`, and rungs are ordered by measured `s`, not by name.

---

## 3. L1 — the ladder on the paper's prefix endpoint

### 3.1 Design

Model: `llava-hf/llava-1.5-7b-hf`. Cells: the frozen 1,500 cells of
`sc1_interv/out/corrob_cells.json` (500 COCO val2017 images × `k` ∈ {1,2,3}), the same cells the
published sighted and grey arms use.

For each rung `r`, two arms `cap1_r` and `cap5_r`, generated by
`lad_gen.py`, which reuses `wia_blind_common.gen_batch_gray` / the `wia_blindgen.py` build order
unchanged and replaces only the pixels. Greedy, 384-token budget, `bs = 8` — identical to the
published blind run.

The sighted arms are **not regenerated**: `Δ_primary` is read from the frozen
`sc1_interv/out/at_corrob.jsonl`, exactly as the published analysis does.

### 3.2 Pre-generation gates (each aborts its rung)

| gate | condition |
|---|---|
| `GATE_PREFIX_IDENTICAL` | prefix token ids and prefix text re-derived from `j5_common.build_prefix_ids` / `wia_redun_common.assemble_nodouble` must equal the frozen sighted row's, for all 3,000 rows, 0 drift |
| `GATE_BATCH_ORDER` | the sorted `(image_id, k, arm)` sequence must equal `at_corrob.jsonl`'s own write order, row for row |
| `GATE_MIRROR_IDENTICAL` | with the rung disabled, the generator must be token-identical to `wia_lngen.gen_batch` on 32 real-image rows |
| `GATE_RUNG_DETERMINISTIC` | sampled re-computation of the degraded image must be byte-equal |
| `GATE_RUNG_APPLIED` | the degraded `pixel_values` must differ from the clean one on every row, and (for non-`grey` rungs) must differ **between** images — a silently inert rung would read as a perfect null |
| `GATE_NODE` | `hostname` is `vgi1` |

### 3.3 Positive controls (both must pass before any new endpoint is read)

* **PC1 — scoring.** Run the ladder scorer on the *frozen* sighted + *frozen* published grey rows.
  It must return `Δ_primary = +0.4859`, `Δ_blind = +0.5019`, `Δ_image = −0.0160`.
  **Tolerance: 5×10⁻³ absolute on each** (the parent lane's own `CTL_TOL`).
* **PC2 — generation, end to end.** `grey` is re-generated as a full eighth rung, all 3,000 rows,
  through *this lane's* generator and scored by *this lane's* scorer.
  * PC2a: ≥ 2,940/3,000 (98%) of continuations token-identical to `at_blind1.jsonl`.
    bf16 greedy near-ties under an identical batch shape are the only admitted source of
    disagreement; any mismatch is reported with its first divergent token index.
  * PC2b: the re-generated grey rung must return `Δ_image` within **0.010** of −0.0160.
* If PC1 fails, the lane is reported as no-data. If PC2 fails, the seven new rungs are reported as
  no-data pending diagnosis, and the failure is reported.

### 3.4 Primary estimand, per rung

    Δ_image(r)  =  Δ_primary  −  Δ_r
    Δ_primary   =  gap(cap5)   − gap(cap1)      [frozen sighted rows]
    Δ_r         =  gap(cap5_r) − gap(cap1_r)    [rung-r rows]
    gap(arm)    =  carry-forward rate on true units − carry-forward rate on false units

Units, `gap_raw`, `Units.gather`, `trunc_text` and the bootstrap are **ported unmodified** from
`wia_blindscore.py`, which itself ports them from `wia_corrobscore.py`.

Intervals: paired bootstrap **clustered on the image**, `B = 4000`, seed 20260919. `Δ_primary`,
every `Δ_r` and every `Δ_image(r)` are computed **inside one bootstrap loop on one shared image
draw**, so they cannot drift apart. Resample size is asserted on every replicate and `set()` is
never applied to a resample.

Reference margin: **±0.10**, the paper's own reporting choice (not a registered bound), stated as
such wherever it is used.

### 3.5 Per-rung decision rule, in this order

1. **Degeneracy screen first** (Section 6). If the rung's arm pair is DEGENERATE, the verdict is
   **NO-DATA** for that rung. *A collapsed arm is not a null, and a collapsed arm is not an image
   contribution either.*
2. **Power.** If the 95% CI width of `Δ_image(r)` exceeds **0.14** (the parent lane's
   `WIDTH_CEIL`), or either arm has fewer than 100 true or 100 false units, the verdict is
   **UNDERPOWERED**.
3. Otherwise, on the 95% CI:
   * CI entirely inside (−0.10, +0.10) → **NULL-WITHIN-MARGIN**
   * CI excludes 0 and point ≥ +0.10 → **IMAGE-CONTRIBUTES** (the adverse direction)
   * CI excludes 0 and point ≤ −0.10 → **REVERSE** (ablation enlarges the rise more than grey does)
   * CI excludes 0 but |point| < 0.10 → **SMALL-BUT-NONZERO**
   * CI contains 0 and extends beyond ±0.10 → **INCONCLUSIVE**

### 3.6 Ladder-level verdict

Seven rungs are tested, so the per-rung 95% intervals are also reported at a Bonferroni-adjusted
99.29% (`1 − 0.05/7`) level, and one pooled statistic is registered:

    Δ_image_ladder  =  mean over non-degenerate rungs of Δ_image(r)

computed on the same joint bootstrap draws (the rungs share the sighted arm and the image draw, so
they are correlated; a joint bootstrap is the correct pooling).

* **LADDER-NULL** — every non-degenerate rung is NULL-WITHIN-MARGIN **and** `Δ_image_ladder`'s CI
  lies inside ±0.10.
* **NARROW-TO-STRONG-ABLATION** — `Δ_image_ladder` ≥ +0.10 with CI excluding 0, **or** any single
  rung reaches IMAGE-CONTRIBUTES with its *Bonferroni-adjusted* interval excluding 0.
* **LADDER-UNINFORMATIVE** — anything else: mixed verdicts, non-monotone pattern, or degeneracy at
  the milder rungs.

Monotonicity is reported descriptively (Spearman ρ between measured severity `s(r)` and
`Δ_image(r)` across all non-degenerate rungs plus the grey control) and is **not** a gate; with
eight points it cannot carry a verdict.

### 3.7 Secondaries, fixed now

* `Δ_image(r)` under the paper's **cross-condition** length-matching rule (one `L` per cell applied
  to both halves) and under its **within-condition** rule.
* CP-R: both arms of the contrast uncapped.
* The `H`/`F` split per rung (`r_true`, `r_false` and their image-attributable parts), because the
  paper's own Table 4 shows the two rates move in opposite directions and cancel in `J`. A rung at
  which `J` is null while the two rates move *differently from grey* is a real finding and will be
  reported as one.
* `LLaVA-OneVision-0.5B` on the same ladder — **conditional secondary**, run only if L1 and L2 on
  LLaVA-1.5-7B are complete with time to spare. Its published blind arms are already thin
  (medians 5 and 11 tokens), so it is registered as expected-to-be-degenerate; if it is, that is
  reported as NO-DATA and not as a replication failure.

---

## 4. L2 — the ladder on standard CHAIR

### 4.1 Design

The Appendix-G setup, unchanged: LLaVA-1.5-7B, 500 COCO val2017 images, PAI's own prompt, greedy,
512-token budget, `α = 0.5, γ = 1.1`, layers [2,32). `lad_cgen.py` reuses `md_gen.md_two_stream`
and `pa_gen`'s gated kernel and changes only the pixels.

**Stage 1 — denominator screen (vanilla only, all seven rungs).**
**Stage 2 — PAI** at the rungs that pass, plus their already-frozen sighted counterparts.

### 4.2 Registered denominator thresholds

At rung `r`, from the vanilla arm's 500 captions:

* `CHAIR` is **DEFINED** iff pooled category-mention occurrences ≥ 1.
* `CHAIR` is **USABLE** iff **≥ 250/500 captions contain at least one of the 80 categories** *and*
  **pooled mention occurrences ≥ 1000** (≥ 26% of the sighted vanilla arm's 3,827).

PAI is generated only at USABLE rungs. A rung that is DEFINED but not USABLE is reported as
`DENOMINATOR-TOO-THIN` — the same verdict Appendix G.7 gives grey — and no gain is read from it.

### 4.3 Primary and secondary at a usable rung

The **primary is denominator-free**, because CHAIR's denominator is endogenous to the arm
(Appendix G.7's own argument: "even a blind arm that still named objects would give a `CHAIR_i`
computed over a *different* item set"). On the **fixed** 500 × 80 = 40,000-unit set:

    ΔJ(r)        =  J(pai05_r) − J(vanilla_r)
    ΔJ_image(r)  =  ΔJ(sighted) − ΔJ(r)

with `J = H − F`, `H` the mention rate over present units, `F` over absent units, exactly as
`md_score.py` forms them. `d'` and `c` are reported alongside, with the log-linear correction and
the same materiality flag.

The **secondary** is the quantity the objection actually asks about:

    gain_i(r)        =  CHAIR_i(vanilla_r) − CHAIR_i(pai05_r)
    gain_i_image(r)  =  gain_i(sighted) − gain_i(r)

reported *with* the endogenous-denominator caveat attached, never promoted over the primary.

Intervals: paired image-clustered bootstrap, `B = 4000`, seed 20260919, paired across arms.

### 4.4 L2 decision rule at a usable rung

* `gain_i(r)` CI excludes 0 and `gain_i(r) ≥ 0.5 × gain_i(sighted)` → **GAIN-SURVIVES-ABLATION**:
  at least half of PAI's CHAIR improvement is reproduced with the image degraded, so the
  improvement is substantially not image-attributable. This is a **new and strong result for the
  paper** and closes Appendix G.7's open item.
* `gain_i(r)` CI contains 0 while `gain_i(sighted)`'s CI excludes 0, and `gain_i_image(r)`'s CI
  excludes 0 → **GAIN-IS-IMAGE-ATTRIBUTABLE**: PAI's CHAIR gain needs the image. The paper must
  then say so, must retire "we did not test whether a weaker ablation … would leave something to
  score", and must weaken the insinuation in Section 5 that the improvement's nature is untestable.
* Neither → **CANNOT-RESOLVE** at that rung.

### 4.5 L2 positive controls

* **PC3 — scoring.** Re-scoring the frozen `mitdecomp` arms must return
  `CHAIR_s(vanilla) = 45.00`, `CHAIR_i(vanilla) = 12.78`, `CHAIR_i(pai05) = 7.2`,
  `H(vanilla) = 0.78`, `F(vanilla) = 0.010`. **Tolerance: 0.05 on the CHAIR percentages, 5×10⁻³
  on the rates.**
* **PC4 — generation.** `vanilla_grey` re-generated through this lane's generator on the first 64
  images must be token-identical to `md_vanilla_blind.jsonl` for ≥ 62/64, and must reproduce the
  zero-category-mention degeneracy Appendix G.7 reports.

---

## 5. Distinguishing "the image contributes" from "the ablated arm collapsed"

This is the failure mode that would make an adverse result spurious, and it is stated before the
data exist.

A collapsed ablated arm produces **little carry-forward differentiation at all**, so `Δ_r` shrinks
toward 0 and `Δ_image = Δ_primary − Δ_r` is inflated **in the positive, adverse direction**.
Collapse and a genuine image contribution therefore push the primary the same way and cannot be
separated by the primary alone.

Registered separation, applied *before* any rung is labelled IMAGE-CONTRIBUTES. A rung may be read
as IMAGE-CONTRIBUTES only if, in **both** conditions, it is **no more degenerate than the grey arm**
— whose `Δ_blind` already reproduces `Δ_primary` at 1.03×, and which is therefore the calibration
point that matters:

* median continuation length ≥ the grey arm's in that condition (grey: `cap1g` 16, `cap5g` 384);
* distinct-8-gram ratio (mean over rows) ≥ the grey arm's in that condition;
* empty-continuation rate ≤ the grey arm's in that condition (grey: 0.045, 0.005);
* distinct-token-id ratio (mean over rows) ≥ the grey arm's in that condition.

A rung that shows positive `Δ_image` **while being more degenerate than grey on any of the four**
is reported as **CONFOUNDED-BY-COLLAPSE**, not as an image contribution, and its number is still
printed.

Conversely, a rung that shows a **null** `Δ_image` while being more degenerate than grey is *also*
not evidence for the paper: it is reported as NO-DATA under Section 6. The screen cuts both ways
by construction.

---

## 6. Degeneracy screens (every arm of every rung, reported whatever the verdict)

Reported for each arm: mean and **median continuation length**, **truncation rate** at the 384
budget, **empty rate**, **distinct-8-gram ratio**, **distinct-token-id ratio**, number of distinct
continuation texts, and the true/false unit counts.

An arm is **DEGENERATE** if any of:

| screen | threshold | grey arm's value (calibration) |
|---|---|---|
| median continuation length | < 8 tokens | 16 (`cap1g`), 384 (`cap5g`) |
| empty-continuation rate | > 0.25 | 0.045, 0.005 |
| mean distinct-8-gram ratio | < 0.20 | measured at PC2 |
| distinct continuation texts | < 10 of 1,500 | measured at PC2 |
| true units or false units | < 100 | 528/2,332 (`cap1`), 892/4,988 (`cap5`) |

These thresholds are set so that **the published grey arm passes them** — a screen that rejected
our own headline arm could not test whether the headline reproduces — and so that the arms the
paper itself declines to interpret (Qwen2.5-VL-7B, medians 4–5 tokens) would fail.

For L2, an arm is additionally DEGENERATE if it fails the USABLE threshold of Section 4.2.

---

## 7. Execution rules

* Own code directory, own sbatch wrappers, every payload ends `|| exit 9`.
* `--gres=gpu:rtx_4090:1 --nodelist=vgi1`, explicit `--time=` on every submission.
* **At most 2 concurrent GPU jobs.**
* Success is a **sentinel string in the output file**, never SLURM state.
* Resume conditions state the real condition (row count *and* meta agreement), never a marker file.
* `df -h /` before and after; stop and report rather than proceed if free space drops below 20 GB.
  Expected footprint: ~62 MB (L1, 8 rungs × 3,000 rows) + ~20 MB (L2).

---

## 8. Adverse outcomes: what we write for each

Fixed before any generation. **No outcome here is un-reportable.**

### 8.1 `Δ_image` stays null across the whole ladder → LADDER-NULL

The strongest outcome for the paper. We write: the null on `Δ_image` is **not** an artifact of the
flat-grey ablation; it survives seven graded degradations spanning noise, blur, resolution loss and
spatial scrambling, over a measured severity range from `emb_cos` ≈ 1 down to the grey field. The
concession in Section 4.2 ("We did not run a graded image degradation") is replaced by the ladder,
and Figure 4's robustness panel gains a dose axis. **The claim itself does not widen**: the ladder
shows the estimand is insensitive to *which* ablation, not that the image is unused — Appendix D
already shows directly that it is used.

### 8.2 `Δ_image` grows as the ablation gets milder → NARROW-TO-STRONG-ABLATION

The outcome that weakens the paper, and the most likely one to be right if the reviewers are right.
We write: the image-attributable component of the `cap1 → cap5` rise is **null only under strong
ablation**; under partial-information ablation it is +X [CI]. The central claim must then be stated
as *"a contrast that survives removal of the image"* and **not** as *"a contrast the image does not
explain"*. Specifically:

* **the abstract changes** — the sentence reporting `Δ_image` acquires "under ablation strong enough
  to remove image information entirely", and the ladder's largest rung-level estimate is quoted in
  the abstract, not buried;
* Section 4.2's "Both intervals exclude an image contribution … of 0.10 in either direction" is
  narrowed to the grey condition and immediately followed by the ladder;
* Section 6's recommendation 2 ("Blind every endpoint…") gains an explicit warning that a
  grey-blank default can under-state the image's contribution, which is a **correction to the
  advice the paper gives other people**;
* the Limitations section states that the paper's own blinding instrument has a measured blind
  spot.

### 8.3 Non-monotone ladder, or the milder rungs degenerate → LADDER-UNINFORMATIVE

We write: the ladder cannot separate the two readings. The rungs and their degeneracy statistics
are reported in full, the verdict is **no-data**, and Section 4.2's concession **stays in the
paper unchanged**. A no-data ladder is not converted into support by quoting whichever rungs
happened to come out null.

### 8.4 L2: CHAIR becomes usable at some rung

Either L2 verdict is a substantive addition and both are written:

* **GAIN-SURVIVES-ABLATION** — Section 5 gains a measured blind control where it currently has a
  hole, and the paper can say a published method's CHAIR improvement is at least half reproduced
  with the image degraded. This *strengthens* the paper considerably.
* **GAIN-IS-IMAGE-ATTRIBUTABLE** — the paper must report that PAI's CHAIR gain **does** need the
  image once a runnable ablation exists, retire the Appendix G.7 sentence "we did not test whether
  a weaker ablation or another model would leave something to score", and soften Section 5's
  framing accordingly. This runs against the lane's own rhetorical interest and is reported first,
  not last.

### 8.5 L2: no rung is usable

We write: CHAIR's denominator empties under every ablation we tried, not only under grey. This
*supports* Appendix G.7's structural claim — the exposure is a property of the endpoint, not of the
grey constant — and the appendix's scope paragraph ("a weaker ablation … would *likely* leave the
support non-empty") is corrected, since we tried and it did not.

---

## 9. Stop conditions

1. `GATE_GEOM_IDENTITY` fails → abort lane. *(Ran first; passed 32/32.)*
2. PC1 fails → the whole lane is no-data.
3. PC2 fails → the seven new L1 rungs are no-data pending diagnosis.
4. Any per-rung gate in Section 3.2 fails → that rung is aborted and reported as aborted.
5. `/` free space < 20 GB → stop and report.
6. More than 2 concurrent GPU jobs would be needed → queue instead.
7. No endpoint is read before PC1 and PC2 have both passed and been printed.

---

## 10. What this lane cannot decide

* It is one model on the primary (LLaVA-1.5-7B), one prompt, one decoding mode, one 500-image draw.
* Seven rungs are not the space of ablations. A ladder-null does not show that *no* ablation would
  move `Δ_image`; it shows these seven do not.
* `emb_cos` is a severity axis measured on the vision tower we are also measuring through. It
  orders the rungs; it does not make them a physical dose.
* L2's `CHAIR_i` contrast between a sighted and an ablated arm conflates a rate change with a
  composition change, because the denominator is endogenous. That is why the L2 primary is the
  fixed-unit `J`, and the caveat travels with every `CHAIR_i` number we print.

---

## Amendment 1 — the collapse screen's reference arm

**Recorded 2026-09-19, after PC1 and before any ladder-rung endpoint existed.** At the time of
writing, no rung file had a completed meta and the scorer had read no rung endpoint; the only
numbers in hand were the frozen sighted and frozen grey arms' own degeneracy statistics, which
are published data, not results of this lane.

Section 5 as registered says a rung may be read as IMAGE-CONTRIBUTES only if it is "no more
degenerate than the grey arm" on median continuation length, distinct-8-gram ratio, empty rate
and distinct-token-id ratio. PC1's screens show that bar is mis-set:

| arm | median len | empty | distinct-8-gram | distinct-token-id | distinct texts |
|---|---|---|---|---|---|
| `cap1` (sighted) | 64 | 0.031 | **0.575** | **0.493** | 1386 |
| `cap1g` (grey)   | 16 | 0.045 | **0.780** | **0.699** | 1298 |
| `cap5` (sighted) | 384 | 0.005 | 0.377 | 0.279 | 1477 |
| `cap5g` (grey)   | 384 | 0.005 | 0.327 | 0.260 | 1409 |

In the one-scene condition the **grey arm is lexically more diverse than the sighted arm** — its
continuations are short and varied, the sighted ones long and more repetitive. "No more degenerate
than grey" is therefore a bar **the paper's own sighted arm fails**, and any rung close to sighted
would fail it too. As written, the rule would convert almost every IMAGE-CONTRIBUTES verdict into
CONFOUNDED-BY-COLLAPSE — i.e. it would systematically suppress the adverse outcome. A screen whose
error runs in the direction that protects the claim is worse than no screen.

**Amended rule.** A rung may be read as IMAGE-CONTRIBUTES only if, in both conditions, it is no
more degenerate than the **weaker of the two arms this paper already reads endpoints from** —
the sighted arm and the grey arm. Concretely, per condition: median length and the two diversity
ratios must be ≥ `min(sighted, grey)`, and the empty rate ≤ `max(sighted, grey)`.

Both flags are computed and both are reported: `more_degenerate_than_grey` (the originally
registered comparison) and `more_degenerate_than_weaker_reference` (the amended one, which sets
the verdict). The absolute degeneracy screens of Section 6 are unchanged and still run first.

---

## Amendment 2 — the mismatched-real-image arm (`mismatch`)

**Recorded 2026-09-19. No endpoint, and no generated row, existed for this arm when this was
written.** At the time of writing the lane had generated only the `grey` and `noise64` rungs of
L1 and the seven vanilla arms of the L2 denominator screen; the `mismatch` arm did not exist in
any form.

### What is added

An eighth ablation on the prefix endpoint (L1), and the same arm on the CHAIR endpoint (L2):
**the target image is replaced by a different real COCO image from the same frozen 500-image
sample.** Everything else is identical — same prefix token ids, same prompt, same greedy
decoding, same cells, same batch order.

The substitution is `lad_common.derangement()`: one fixed, seeded (`MISMATCH_SEED = 20260919`),
**fixed-point-free** permutation of the 500 sorted image ids, rejection-sampled so that no image
is ever paired with itself, asserted bijective at generation time (`GATE_MISMATCH_DERANGED`), and
recorded per row as `sub_image_id`. It is a deterministic function of the sorted id list alone,
so L1 and L2 receive the identical mapping and it is reproducible without the artefacts.

### Why it may be the most informative arm in the lane

The standing objection to the grey arm is that a flat field is out of distribution for the
encoder, so "blind" may be a degenerate regime rather than an image-free one. That objection is
now in print: arXiv 2509.23499v2 ("Multi-modal Data Spectrum", stated as accepted at ICLR 2026)
rejects "zeroing out (e.g., using a blank image or an empty string)" on the grounds that it
"creates unnatural, out-of-distribution inputs that elicit unpredictable model behavior", and
prefers shuffling modalities so the model still receives a valid input. Liao et al.
(arXiv 2606.31257), whom the paper already cites, require five ablations to agree and include a
real-but-mismatched in-distribution image among them, reporting that it changes the picture on
one of their axes.

A mismatched real image is **in distribution and carries full visual information**, but carries
no information about *this* image. The graded rungs cannot answer the OOD objection on its own
terms, because every one of them still hands the encoder a corrupted input. This arm does.

*(We have not verified arXiv 2509.23499v2 or its ICLR 2026 acceptance at primary source; it is
recorded here as the stated provenance of the objection, not as a citation.)*

### Registration

* **Estimand, interval and per-rung decision rule: identical to Section 3.4–3.5.**
  `Δ_image(mismatch) = Δ_primary − Δ_mismatch`, same joint bootstrap draw, same ±0.10 reference
  margin, same degeneracy screen first, same Amendment-1 collapse rule.
* **It is a separate registered primary, not a ladder rung.** It is therefore **excluded** from
  the pooled seven-rung `Δ_image_ladder` mean and from the ladder's Bonferroni family (which
  stays at m = 7, fixed before any rung endpoint), and it carries a nominal 95% interval as a
  single pre-specified comparison. This is fixed now, before its data exist, so that it can
  neither be slipped into the pooled mean if it is null nor excluded from it if it is not.
* **It is reported beside `grey` in the results table**, with the same estimand and intervals.
* **Diagnostic, reported whatever it shows:** the number of cells whose substituted image
  happens to be one of that cell's five prefix source images (expected ≈ 1% by chance). If that
  count exceeds 2% of cells, the endpoint is additionally reported with those cells excluded.

### The adverse outcome for this arm, named in advance

**`Δ_image(mismatch)` is material and positive while `Δ_image(grey)` is null.** That would mean
the `cap1 → cap5` rise *shrinks* when the model is given a valid but wrong image, and survives
only when the image is destroyed — i.e. **our headline null is partly an artefact of a degenerate
input, exactly as the objection claims.** If that happens:

* the lane's headline verdict is **`IN-DISTRIBUTION-ABLATION-DISAGREES`**, and it **overrides**
  whatever the degradation ladder returns, including a LADDER-NULL;
* the paper's central claim must be restated as holding *under ablation that removes the image
  entirely*, and must say in the abstract that the in-distribution control disagrees;
* Section 6's recommendation 2 — which currently tells other researchers to adopt a grey-blank
  default — must be **corrected** to require an in-distribution mismatched-image arm alongside it.

The converse outcome, `Δ_image(mismatch)` null, answers the OOD objection on the objection's own
terms and is the strongest single result this lane can produce.

### L2

The same arm is added to the CHAIR ladder as `vanilla_mismatch` and `pai05_mismatch`. A
mismatched real image should keep the model naming objects, so CHAIR's denominator should stay
non-empty. The registered USABLE thresholds of Section 4.2 and the decision rule of Section 4.4
apply unchanged.

### Answer-distribution reporting

If any arm of either endpoint produces a degenerate constant-answer regime, the **distribution of
what it emits** is reported, not only the score: the number of distinct continuations, the most
frequent continuation and its share, alongside the existing degeneracy screens. (A substantial
slice of above-chance blind accuracy on multiple-choice readouts can be answer-position or
answer-yes bias rather than content leakage; this endpoint is free generation rather than
multiple choice, but the same discipline applies to a collapsed arm.)

---

## Amendment 3 — the collapse screen tested the wrong tail; plus a black control and an L2 severity floor

**Recorded 2026-09-19, before any ladder verdict had been read.** At the time of writing, the
scorer had been run only on the frozen sighted and frozen grey arms (PC1, job 21036) and on the
L2 denominator screen; no `Δ_image` for any new rung had been computed, and the two completed
L1 rungs (`grey`, `noise64`) had been scored for nothing beyond the positive control.

### 3A. The bug

Section 5, as amended by Amendment 1, requires a rung's **median continuation length** to be at
or above `min(sighted, grey)` before it may be read as IMAGE-CONTRIBUTES. In the five-scene
condition **both reference arms sit at the 384-token generation budget**:

| condition | sighted median | grey median | resulting bar |
|---|---|---|---|
| `cap1` | 64 | 16 | 16 |
| `cap5` | 384 (at the cap) | 384 (at the cap) | **384 — the ceiling itself** |

`cap5`'s cap-hit rates are 0.715 (grey) and comparable sighted, so the references are truncated,
not finished. A rung that **terminates naturally** before the budget therefore has a lower
median than both references and is flagged "more degenerate" — when it is *less* degenerate. The
`min()` of Amendment 1 does not help, because the minimum of two ceiling values is the ceiling.

A second error of the same kind sits in the diversity criteria: the distinct-8-gram and
distinct-token-id ratios fall mechanically as output length rises, so a rung that writes **more**
than the references is flagged as writing **worse**.

Both errors run in the direction that **suppresses the adverse verdict**, which is the direction
that protects the paper. That is the same fault Amendment 1 caught, and catching it twice on the
same screen means the screen's *form* was wrong, not its thresholds.

### 3B. The corrected criterion — fix the test, not the threshold

Collapse means an arm **fails to emit scoreable content**. That is a property of the bottom of
the length distribution, not of falling below a truncated reference. So:

1. **Collapse is now tested absolutely**, by the Section 6 screen, which gains one criterion:
   * rate of continuations of **≤ 3 tokens > 0.25** → DEGENERATE.
   (Retained unchanged: median length < 8, empty rate > 0.25, mean distinct-8-gram < 0.20,
   fewer than 10 distinct texts of 1,500, or fewer than 100 true or false units. These were
   already calibrated so the published grey arm passes and the arms the paper itself declines to
   interpret — Qwen2.5-VL-7B at medians of 4–5 tokens — fail.)
2. **The cap-hit rate is reported for every arm and is never a collapse criterion in either
   direction.** An arm that stops early is not thereby collapsed; an arm that runs to the budget
   is not thereby healthy.
3. **The comparative collapse gate is retired as a gate.** Both comparisons — against grey (as
   originally registered) and against the weaker of sighted and grey (Amendment 1) — are still
   computed and still reported, as diagnostics. Neither can downgrade a verdict any more, and
   the verdict category **CONFOUNDED-BY-COLLAPSE no longer exists**. A rung either passes the
   absolute screen, in which case its `Δ_image` is read at face value, or fails it and is
   NO-DATA.
4. **Auditability.** For every rung the scorer records `verdict`, `verdict_under_original_
   collapse_rule`, and `collapse_rule_changed_verdict`, and `lad_l1.json` carries a top-level
   `AMENDMENT3_verdicts_changed` block naming **exactly which rungs' labels differ under the
   corrected rule and which comparative flags would have fired**. That table is reproduced in
   `SPRINT/LADDER.md` whichever way it comes out. This amendment can only ever *loosen* the path
   to an adverse verdict, never tighten it.
5. **What replaces the lost safeguard.** Two already-registered quantities become mandatory
   reading for any IMAGE-CONTRIBUTES rung, and are reported for every rung regardless:
   (a) the rung's own `Δ_rung` with its `H`/`F` split — a collapse-driven `Δ_image` appears as
   `Δ_rung → 0` with *both* carry-forward rates depressed, which the split makes visible;
   (b) the **cross-condition length-matched** `Δ_image`, which holds the compared continuations
   at equal length by construction and so cannot be produced by a length difference at all.

### 3C. A flat-black control rung (`black`)

Lan et al. (arXiv 2605.22903), whom the paper cites, blind with **flat black** and report blinded
LLaVA-1.5-7B still naming objects on AMBER (CHAIR 48.3 / Cover 6.4). Our flat **grey** arm names
none of the 80 COCO categories in any of 500 captions. Those two facts cannot both be a general
property of uniform fields.

`black` = flat `(0,0,0)` at the canonical geometry, otherwise identical to `grey`. It is a
**control rung, not a degradation rung**: like `mismatch`, it is excluded from the pooled
seven-rung ladder mean and from the m = 7 Bonferroni family, carries a nominal 95% interval, and
is reported beside `grey`. It is run on both endpoints. The registered readings:

* **L2, black keeps the denominator non-empty while grey does not** → the emptying reported in
  Appendix G.7 is **specific to the grey constant**, not a property of uniform fields or of
  blinding as such. Appendix G.7's scope paragraph must then be rewritten: it currently
  generalises from one constant, and the paper must instead state which ablation empties CHAIR
  and which do not.
* **L2, black also empties the denominator** → the emptying is a property of uniform fields, our
  Appendix G.7 claim stands as written, and Lan et al.'s contrasting AMBER result is attributable
  to their benchmark, prompt or scorer rather than to the blinding constant. We do not have their
  harness and will say so rather than adjudicate it.
* **L1**: `Δ_image(black)` is read under the same rule as every other rung.

### 3D. An L2 severity floor

So that "PAI's CHAIR gain survives at rung X" cannot be read as surviving a trivial degradation,
each L2 verdict is reported against the rung's **measured** severity. A rung is **MATERIAL** iff
its mean `emb_cos` lies at or below the midpoint between the identity rung (1.0) and the
**mismatched-real-image** rung — that is, it has travelled at least half the distance to "a
different real image entirely".

The anchor is the mismatch arm rather than an absolute cosine because pooled patch embeddings are
anisotropic: an absolute threshold such as 0.90 is not scale-free and could be met or missed for
reasons having nothing to do with the ablation. **This floor is registered before `emb_cos` has
been measured for any rung** — the dose job has not run. A rung that returns
GAIN-SURVIVES-ABLATION while failing the floor is reported as
**GAIN-SURVIVES-BUT-ABLATION-NOT-MATERIAL** and is not quoted as evidence.

### 3E. Answer-distribution reporting (implementing Amendment 2's last clause)

Every arm of both endpoints now reports, alongside the existing screens, the **rate of ≤ 3-token
outputs**, the **number of distinct outputs**, the **most frequent output** and **its share**. A
constant-answer regime is then visible as a distribution, not inferred from a score.

---

## Provenance note on the two citations in Amendments 2 and 3

Both external references that motivate these amendments reached this lane **second-hand, through
the sprint coordinator, and neither has been verified at primary source by this lane**:

* **arXiv 2509.23499v2**, "Multi-modal Data Spectrum", said to be accepted at ICLR 2026, quoted as
  rejecting "zeroing out (e.g., using a blank image or an empty string)" because it "creates
  unnatural, out-of-distribution inputs that elicit unpredictable model behavior".
* **arXiv 2605.22903** (Lan et al.), said to blind with flat black and to report blinded
  \textsc{LLaVA-1.5-7B} on AMBER at CHAIR 48.3 / Cover 6.4.

They are recorded here as the **stated provenance of the objection and of the black control**, not
as citations. Neither the venue claims nor the numbers have been checked against the arXiv source
or a proceedings index. **Before either appears in the paper**, both must be verified at primary
source under this project's standing rule — including the ICLR 2026 acceptance claim, since a
blank `journal-ref` is not evidence either way and "submitted" is not "accepted".

Nothing in this lane's design or verdicts depends on those claims being true. The `black` rung and
the `mismatch` arm are motivated by them but are evaluated entirely on this lane's own measurements
against this lane's own registered rules.
