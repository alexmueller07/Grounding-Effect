# QWEN3 lane resolution — a 2025-generation model, and the registered adverse outcome

Lane `qwen3`. `Qwen/Qwen3-VL-8B-Instruct`, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`,
bfloat16, one RTX 4090. Registered in `SPRINT/QWEN3_PREREG.md` (commit `7a30268`, written before
the checkpoint existed locally; §1.1 and §8 filled by commit `cec3d34` before the first endpoint
arm ran).

**Registered cell: `IMAGE-ATTRIBUTABLE`** — §6 outcome 3, the outcome that bounds the paper's
claim. It is reported here as a bound, which is what the registration committed us to.

---

## 1. The headline

| quantity | estimate | 95% CI | status |
|---|---|---|---|
| `Δ_primary` (sighted) | **+0.5828** | [+0.5337, +0.6342] | POSITIVE |
| `Δ_blind` (grey) | **+0.5231** | [+0.4708, +0.5764] | POSITIVE |
| **`Δ_image` = primary − blind** | **+0.0598** | **[+0.0235, +0.0963]** | **POSITIVE** |
| blind-to-sighted ratio | **0.90** | | |

Paired bootstrap clustered on the image, B = 4000, seed 20260919, resample size asserted per
replicate and duplicate-presence asserted on the first 20; `set()` never applied to a resample.
`Δ_image` uses **one joint image resample driving all four arms**, so it is paired across the two
generation jobs rather than a difference of two independent intervals.

**What this says, precisely.** On a 2025-generation model the one-scene-to-five-scene rise is
large and reproduces without the image — `Δ_blind` = +0.52 of a +0.58 sighted rise, 90% of it —
but the image-attributable component **excludes zero**. So the paper's "the rise does not need the
image" does not hold unqualified here.

**And the size of the bound matters as much as its sign.** The entire interval
[+0.0235, +0.0963] lies **inside** the paper's own ±0.10 reference margin. The honest reading is
therefore not "the image drives the rise on current models" — it is *there is a real but small
image contribution, below the margin the paper reads `Δ_image` against, accounting for 10.3% of
the sighted rise.* Ninety per cent of the rise still survives blinding.

### Where the image contribution sits

| condition | image contribution to `J` (sighted − blind) | 95% CI |
|---|---|---|
| one scene | **+0.0014** | [−0.0262, +0.0285] |
| five scenes | **+0.0612** | [+0.0392, +0.0841] |

This is the shape of the effect, not just its sign: the image contributes **nothing measurable in
the one-scene condition** and essentially all of `Δ_image` in the five-scene condition. `Δ_image`
is the difference of these two rows by construction.

### Per-condition rates

| arm | `H` | `F` | `J` | n_true | n_false |
|---|---|---|---|---|---|
| `cap1` sighted | 0.8273 | 0.7780 | +0.0493 | 556 | 2482 |
| `cap5` sighted | 0.7911 | 0.1590 | +0.6322 | 1039 | 6536 |
| `cap1g` blind | 0.8022 | 0.7542 | +0.0479 | 556 | 2482 |
| `cap5g` blind | 0.7218 | 0.1509 | +0.5710 | 1039 | 6536 |

The mechanism of `Δ_image` is legible in these four rows. In the five-scene condition the image
raises the **hit rate** by +0.069 (0.7218 → 0.7911) while raising the false-alarm rate by only
+0.008 (0.1509 → 0.1590), so `J` rises. In the one-scene condition it raises both by about +0.025
and they cancel. That is a grounding-shaped contribution — the image selectively helps the model
name objects that are actually present — and it is only visible when the passage supplies five
competing scenes.

### Supporting contrasts

| contrast | estimate | 95% CI | status |
|---|---|---|---|
| `PRIMARY_klen` (length-matched) | +0.5761 | [+0.5273, +0.6270] | POSITIVE |
| `BLIND_klen` | +0.5107 | [+0.4591, +0.5642] | POSITIVE |
| `CP-C2` object-type matched | +0.5650 | [+0.5191, +0.6227] | POSITIVE |
| `empty_removed` | +0.5839 | [+0.5345, +0.6348] | POSITIVE |
| `PRIMARY_seed_alt` | +0.5828 | [+0.5331, +0.6311] | POSITIVE |
| `PRIMARY_recon192` (192-token regime) | +0.5926 | [+0.5433, +0.6433] | POSITIVE |
| `CP-R` both-uncapped | +0.5637 | [+0.3237, +0.7828] | **UNDERPOWERED** |

`CP-R` keeps only cells where **neither** arm reached the 384-token budget. Because truncation is
near-universal on this model (below), that leaves **14 cells of 1500**, and its interval is 4.6×
wider than the primary's. It is reported for completeness and carries no weight; it is marked
UNDERPOWERED by the registered floor, not by inspection.

---

## 2. The degeneracy screen — and the caveat the registered screen did not catch

This lane exists because `WIA_FAM3_PREREG.md` ran this design on Qwen2.5-VL-7B and got a median
continuation of **four tokens**, leaving the endpoint nothing to register. **That is emphatically
not what happened here**, and a reader who assumes otherwise will misread the row.

| arm | median | mean | truncation at 384 | empty | ≤3 tokens | unique-token ratio (mean / median) |
|---|---|---|---|---|---|---|
| `cap1` | **384** | 364.78 | 0.9393 | 0.0007 | 0.0100 | 0.0903 / 0.0365 |
| `cap5` | **384** | 357.93 | 0.9013 | 0.0000 | 0.0013 | 0.1173 / 0.0495 |
| `cap1g` | **384** | 348.36 | 0.8960 | 0.0033 | 0.0200 | 0.1183 / 0.0339 |
| `cap5g` | **384** | 374.18 | 0.9693 | 0.0000 | 0.0053 | 0.0736 / 0.0443 |

Registered screen: **ALL_PASS, `failing_arms` empty.** Median 384 in every arm against a threshold
of 12; ≤3-token rates of 0.001–0.02 against a threshold of 0.30.

**But the registered screen was not sufficient, and I am recording that against myself.** The
failure mode here is the opposite of FAM3's and my thresholds — median length and ≤3-token rate —
cannot see it. The unique-token ratio, which §5 registered as *reported* rather than *gated*, is
0.03–0.12: a median of 0.0365 means roughly 14 distinct token ids in 384. Inspecting the text
confirms it: **the continuations are repetition loops that run to the budget.**

> `cap1`, median-repetition cell: *"food on the plate. The birds are standing on the plate of
> food. The birds are pecking at the food on the plate. The birds are standing on the plate of
> food. …"* — repeating to 384 tokens.

Measured as the distinct-8-gram ratio of the **continuation** (the appendix's published
`mean_distinct_8gram_ratio` is computed on the *prefix*, so continuation repetition is an
unreported quantity across the whole paper), the fraction of cells below 0.5 is **0.897** in
`cap1` and **0.827** in `cap5`.

### This is a property of the design, not of this model alone

I measured the same statistic on the frozen rows of every family before drawing any conclusion:

| model | arm | median distinct-8-gram | fraction < 0.5 |
|---|---|---|---|
| LLaVA-1.5-7B | `cap1` | 1.0000 | 0.448 |
| LLaVA-1.5-7B | `cap5` | 0.0924 | **0.675** |
| Qwen2.5-VL-7B | `cap1` / `cap5` | 1.0000 / 1.0000 | 0.006 / 0.001 |
| Kosmos-2 | `cap1` / `cap5` | 1.0000 / 1.0000 | 0.214 / 0.228 |
| **Qwen3-VL-8B** | `cap1` / `cap5` | 0.0572 / 0.0776 | **0.897 / 0.827** |

Three things follow. First, **the paper's lead model already loops in 67.5% of its five-scene
cells**, so this is a pre-existing property of the stimulus, not a Qwen3-VL pathology that
invalidates the new row. Second, Qwen3-VL is nonetheless the most affected. Third, Qwen2.5-VL's
1.0000 is an artefact, not a clean bill of health: its continuations are four tokens, too short to
contain an 8-gram, so the statistic returns 1.0 by construction — a non-discriminating check.

### Does looping drive `Δ_image`?

It is the obvious confound for a sighted-minus-blind contrast, so I checked rather than asserted.
Looping rates (fraction of cells below 0.5) and their sighted-minus-blind differences:

| model | `cap1` | `cap1g` | one-scene diff | `cap5` | `cap5g` | five-scene diff |
|---|---|---|---|---|---|---|
| Qwen3-VL-8B | 0.897 | 0.866 | +0.031 | 0.827 | 0.911 | **−0.085** |
| LLaVA-1.5-7B | 0.448 | 0.231 | **+0.217** | 0.675 | 0.713 | −0.038 |

In the five-scene condition the sighted arm loops *less* than the blind arm (0.827 vs 0.911), and
that is the condition carrying all of `Δ_image`. So the confound is real and I am not dismissing
it: a less-looped continuation has more room to name distinct objects. Two things argue it is not
the whole story. A pure diversity effect would raise `H` and `F` together, whereas the image
raises `H` by +0.069 and `F` by only +0.008. And the paper's lead model shows a **larger**
sighted-minus-blind looping asymmetry (+0.217, one-scene) while still returning `Δ_sight` ≈ −0.016
there, so this asymmetry does not mechanically produce a positive `Δ_image`. **A looping-matched
contrast was not run and would be the right next control.** By the unrun-experiment rule this is
a live limitation on the row, not a footnote.

---

## 3. The harness gate — why this row is not another "possibly our prompt"

The reviewer's objection to the Qwen2.5-VL row was that a four-token continuation suggests the
prompt or chat template is wrong for that model. FAM3 could not answer it because its only
continuation-path check was that a prefix *changes* the output — which a broken template passes
trivially. **Verdict: `RUN-ENDPOINT`, all gates pass.**

| gate | result |
|---|---|
| HG-1 TEMPLATE | PASS — processor's own `apply_chat_template`, frozen at fetch and byte-compared by every later stage; generation-prompt suffix `'<\|im_start\|>assistant\n'` |
| HG-2 PREFILL_LANDS | PASS — of 48 rows: 0 with a special token in the prefill, **0 whose prefill was not immediately adjacent to the generation-prompt ids**, 0 with altered turn-marker counts (string-form diagnostic also 0) |
| **HG-3 SELF_CONT** | **PASS** — median continuation 376.5 vs remainder 204.0; median first-16 agreement **1.000**, mean 0.958; **91.7% of rows ≥ 0.75**; empty 0.000 |
| **HG-3 calibration, LLaVA-1.5-7B** | **PASS** — median agreement **1.000**, mean 0.945, **91.7% of rows ≥ 0.75**, exact full match 16/24 |
| HG-4 FREE_LEN | PASS — median 256, mean 255.82, min 218 |
| HG-5 ASSEMBLED | PASS both arms — median 384, ≤3-token 0.042 / 0.000 |
| HG-6 PREFIX_BINDS | PASS — 48/48 |

HG-3 is the decisive test and it works only because decoding is greedy: the model's own caption
*is* its own continuation from its own prefix, so a correct harness must re-emit the remainder,
while a merely terse model still can. **The identical test was run on LLaVA-1.5-7B in the same
job**, and the two models score 1.000/0.958/92% against 1.000/0.945/92% — so the bar was
calibrated against the model carrying the paper's lead result, not invented. Had LLaVA-1.5-7B
failed it, the registered verdict was `GATE-INVALID` and this lane would have reported nothing.

Qwen3-VL's exact-full-match is 0/24 against LLaVA's 16/24, and that is expected rather than
worrying: its own free-running caption is itself truncated at 256 tokens, so the "remainder" ends
at the cap while the model keeps generating past it. Agreement over the first 16 tokens is the
threshold quantity; exact match is a diagnostic.

**Estimator positive control, run before any endpoint was read** (job 21070):

| family | this scorer | published | deviation |
|---|---|---|---|
| LLaVA-1.5-7B | **+0.48587** | +0.4859 | 2.9e-5 |
| Qwen2.5-VL-7B | **+0.15981** | +0.1598 | 1.2e-5 |

Registered tolerance 5e-3; both clear it by ~170×. Integrity gates recomputed from the written
JSONL: `ALL_HARD_PASS` true, 1500 cells / 6000 rows, blind prefixes bit-identical to sighted in
1500/1500 cells in both arms, one-scene doubling 0.046 (< 0.05). `GATE_SCENE` +0.19162
[+0.18665, +0.19685]; `GATE_ATTENDS` +0.73828 [+0.71588, +0.76077].

---

## 4. Cut, stimulus and scope

The cut was chosen by the **census rule** used for Qwen2.5-VL and Kosmos-2, imported unchanged,
not copied from either: largest `f` on a 0.005 grid over [0.050, 0.500] with cap1 doubling < 5%,
no cap5 shortfall, min budget ≥ 8, and ≥ 100 true and false units per arm.

**`CUT = 0.205`**, mean per-cell budget **51.96 tokens** (range 44–52), 0 images excluded,
1500 cells / 500 images. Exact verification with the real builders at both grid points:
`f = 0.205` gives doubling 69/1500 = 0.046 and 0 shortfall; `f = 0.210` gives 94/1500 = 0.0627
and **fails** condition (a). Composition: distinct object types 2.03 / 5.05 (arm ratio 2.49),
total mentions 6.35 / 6.39 conserved across arms, mentions per type 3.14 / 1.26 — the reciprocal
fall that drives the dilution account is present at full strength.

**Two deviations, recorded as scope rather than apology.**

1. **Interpreter.** The brief specified `miniconda3/bin/python`. That interpreter has **neither
   PIL nor torchvision**, so `AutoProcessor` raises before an image is opened — job 21033 proved
   it, dying on exactly that *after* the 16.341 GiB download had succeeded. This lane runs
   `envs/reason` (torch 2.11.0+cu130, **transformers 5.14.1, the same version as base**), the
   interpreter families 3 and 4 were run under.
2. **`L_i` is censored.** Qwen3-VL's free-running captions hit the 256-token cap on **99.2%** of
   the 500 images (median 256, min 218). `L_i` therefore measures the cap, not the model's natural
   caption length, and the per-cell budget is near-constant (44–52) instead of image-keyed. This
   does **not** make the contrast a length contrast — the budget is identical across all four arms
   of a cell by construction, re-asserted at scoring from the written JSONL — but it does mean
   **`0.205` is not comparable to Qwen2.5-VL's `0.230` or Kosmos-2's `0.460`**, and the write-up
   must not present it as their analogue.

A third methodological change is recorded in the registration: the lane streams checkpoint shards
straight to the GPU rather than materialising 16.34 GiB on the CPU, cutting its host reservation
from 32 GB to 14 GB on a node where four sibling lanes held 116 of 122 GB. That this is
placement-only is an argument, not evidence, so job 21073 asserted it — the plumbing smoke's 8
free-running captions, generated by the old `.to("cuda")` path at the same batch size, came back
**byte-identical 8/8**.

---

## 5. What the row does to the paper's claim

**It bounds it, and the bound belongs in the main text.** Three architectures show no
image-attributable component; this one does. The registered text for outcome 3 is that the claim
"the rise does not need the image" must be stated with that bound in Section 4.2 rather than
buried in an appendix, and that the abstract must not claim the null holds generally.

The main-text table as filed gives sighted, blind, `Δ_image` and the blind-to-sighted ratio for
LLaVA-1.5-7B (1.03), LLaVA-OV-0.5B (1.09), Kosmos-2 (1.48) and Qwen3-VL-8B (**0.90**), with
Qwen2.5-VL-7B remaining in the appendix. That table is right; the sentence filed alongside it is
not.

> ⚠️ **The accompanying text says "three architectures show no image contribution while the
> fourth bounds the claim." That is a miscount and must be fixed before submission.** Of the four
> rows in that table, **two** have a `Δ_image` interval containing zero — LLaVA-1.5-7B −0.0160
> [−0.0701, +0.0399] and LLaVA-OV-0.5B −0.0348 [−0.0906, +0.0179] — and **two exclude zero**:
> Kosmos-2 −0.1028 [−0.1677, −0.0360] and now Qwen3-VL-8B +0.0598 [+0.0235, +0.0963]. Kosmos-2
> cannot be counted among the architectures showing no image contribution; §4.2 of the paper
> already says of it that "removing the image makes the rise larger", and its ratio of 1.48 is in
> the table for exactly that reason. The correct sentence is **two of the four show no
> image-attributable component, and two bound the claim from opposite directions.**

Four points the write-up should keep:

1. **The bound is small.** The whole interval lies inside the paper's own ±0.10 reference margin;
   90% of the rise still survives blinding on this model. "The phenomenon is present on a
   current-generation model, with a small but non-zero image contribution" is the defensible
   sentence. "The image drives the rise on current models" is not.
2. **The reviewer's generality objection is answered.** A 2025-generation model, a different
   post-training recipe, verbose rather than terse, shows the same large blind rise (+0.52). The
   phenomenon is not an artefact of 2023–2024 checkpoints.
3. **Kosmos-2 and Qwen3-VL now bound the claim from opposite directions** — ratio 1.48 (blinding
   *increases* the rise) and 0.90 (blinding decreases it). Two of the four models in the table
   (two of the five ever run) have a non-null image-attributable component, and they have
   **opposite signs**. That is worth saying plainly, and it is a stronger statement than either
   row alone: the estimand is not pinned at zero by construction, it moves in both directions
   across architectures, and no single model's null can carry the general claim.
4. **The repetition caveat must travel with the row.** ~90% of continuations are 8-gram loops, the
   sighted and blind arms loop at different rates in the condition carrying the effect, and no
   looping-matched contrast was run. A reviewer who computes continuation repetition will find
   this; better that we report it. It also touches the existing rows, since the lead model loops
   in 67.5% of its five-scene cells and the paper currently reports repetition only for the
   prefix.

**On the Qwen2.5-VL footnote.** This lane now settles what the paper could not say. Under a
harness verified by HG-1/HG-2 and calibrated on LLaVA-1.5-7B by HG-3, a Qwen-family
instruction-tuned VLM of the next generation produces 384-token continuations on the identical
stimulus and prompt construction. That is evidence the four-token result was specific to that
checkpoint rather than a generic defect of our prompt for Qwen models — **but it is not proof**:
Qwen2.5-VL was never itself re-run through HG-3, and only re-running it would close the question.
The footnote should be narrowed to that, not deleted.

---

## 6. Jobs and artefacts

All jobs `COMPLETED 0:0`, each verified by output file **and** sentinel string, never by SLURM
state. Every payload ended `|| exit 9`; all pinned `--gres=gpu:rtx_4090:1 --nodelist=vgi1` with an
explicit `--time`; never more than 2 concurrent GPU jobs from this lane.

| stage | job | elapsed | sentinel |
|---|---|---|---|
| fetch (failed: inherited `HF_HUB_OFFLINE`) | 21030 | — | FAILED 9:0 |
| fetch (failed: base interpreter lacks PIL/torchvision) | 21033 | — | FAILED 9:0 |
| fetch | 21037 | 9s | `Q3_SENTINEL_FETCH_OK` |
| plumbing smoke, 8 images, no endpoint | 21038 | 38s | `Q3_SENTINEL_AF_OK` |
| estimator positive control | 21070 | 7s | `Q3_SENTINEL_CTL_ONLY_OK` |
| load-path equivalence | 21073 | 35s | `Q3_SENTINEL_LOADEQ_OK` |
| free-running captions | 21074 | 20m29s | `Q3_SENTINEL_AF_OK` |
| census | 21075 | 9s | `Q3_SENTINEL_CENSUS_OK` |
| harness gate | 21076 | 4m08s | `Q3_SENTINEL_GATE_OK` |
| endpoint, sighted | 21085 | 2h03m45s | `Q3_SENTINEL_GEN_SIGHTED_OK` |
| endpoint, blind | 21086 | 2h05m43s | `Q3_SENTINEL_GEN_BLIND_OK` |
| score | 21087 | 22s | `Q3_SENTINEL_SCORE_OK` |

**Disk.** `df -h /` before the download **81.61 GB free**; repo **16.341 GiB** in 4 shards; after
**65.27 GB free**. The fetch refuses to start if the download would leave `/` below 25 GB and
re-asserts the floor afterwards. GPU peak 16.84 GiB of 23.52 GiB — it fits one 4090 with room.

Artefacts in `/data/alexmueller/sprint/qwen3/out/`: `q3_score.json`, `q3_gate.json`,
`q3_census.json`, `q3_cells.json`, `q3_attends.json`, `q3_template.json`, `q3_loadeq.json`,
`q3_ctl_only.json`, `q3_sighted.jsonl`, `q3_blind.jsonl`, `q3_af.jsonl`, `q3_af_gray.jsonl`, and
the per-stage `*_meta.json`. Code in `/data/alexmueller/sprint/qwen3/code/`, which holds its own
copies of `j5_common.py`, `gen_common.py`, `wia_redun_common.py` and `synonyms.txt` with md5s
recorded in `QWEN3_PREREG.md` §8.2, so no edit outside the lane could change a queued job.
