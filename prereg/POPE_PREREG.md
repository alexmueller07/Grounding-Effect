# POPE pre-registration — blinding a standard benchmark, and decomposing a published method's gain on it

**Status: PRE-REGISTERED. Everything above §11 was written BEFORE any generation row of this
lane existed** — before the gates job, before any model was loaded on POPE data, and before any
answer string was produced. Nothing above §11 is edited after the pre-registration commit.
§11 onward is appended as the lane proceeds.

Date: 2026-09-19. Lane: **pope**. Node **vgi1**, `--gres=gpu:rtx_4090:1`, explicit `--time`,
every payload ends `|| exit 9`, verification by output file + sentinel.
Lane code: `/data/alexmueller/sprint/pope/code` (its own directory; nothing under
`sc1_interv/code`, `paialpha/code` or `mitdecomp/code` is edited by this lane).

---

## 0. The question, and why it is the paper's exposed flank

`paper.tex` §5 (`sec:mitigation`) and `appendix.tex` §G (`app:mitig`) apply the paper's
decomposition to PAI on standard CHAIR, and then concede the decisive control could not be run:

> "The grey-image control could not be run here. Under a grey image LLaVA-1.5-7B named none of
> the 80 categories in any of 500 captions, with or without PAI, so CHAIR_i was 0/0."
> (`paper.tex` §5)

Appendix G (`app:mitig-blind`) generalises the reason — CHAIR's denominator is the model's own
mention set, so an ablation can empty it — and states the remedy in the abstract:

> "An endpoint whose candidate set is supplied externally ... does not carry that exposure,
> which is why the blind control of Section 4.2 was runnable there."

and concedes the ladder that was available and skipped:

> "a weaker ablation --- noise, blur, shuffled patches --- would likely leave the support
> non-empty. ... Swapping to a different ablation after seeing the registered one degenerate
> was available and was not done."

`sec:reco` item 2 then *recommends* to the field exactly what this paper has never done on a
standard benchmark:

> "Blind every endpoint whose candidates are supplied in the input. A question naming each
> object, as in POPE ... POPE-style existence questions allow it directly."

**So the paper's own recommendation is, at submission, untested on the benchmark it names.**
Its only blinding demonstration lives on an endpoint the authors invented (the five-caption
prefix of §3), which a reviewer can call an artifact of that construction, and its only
standard-benchmark decomposition is the one where the control degenerated.

This lane runs the recommendation on POPE: a benchmark people actually report, whose candidate
objects are supplied by the question, on the official splits.

### 0.1 What this lane cannot say, binding whatever the result

- It cannot show any published POPE number is wrong. It measures a reimplementation of one
  method in this harness, on the official 500-image POPE draw, under one checkpoint port.
- It cannot say what any released implementation does at runtime. The liveness gate (§7)
  establishes that the intervention is live *here*.
- One operating point per arm is not an ROC. `d'` from a single (H, F) pair assumes
  equal-variance Gaussian evidence; that assumption is not tested by this lane.
- A grey field is out of distribution for the image encoder. The grey arm shows what the score
  does when the image carries no information, not what an unchanged visual representation
  would do. The degradation ladder (§4) exists precisely to bound how much of any grey result
  is "no information" versus "out of distribution", and it is registered, not optional.
- Nothing here generalises to every training-free method. One method, two released
  configurations and one constructed one, one model, one prompt, greedy decoding.

---

## 1. What is prior work and is not claimed

POPE and its three splits (Li et al., arXiv:2305.10355). PAI Eq. 3 and Eq. 4 (Liu et al.,
arXiv:2407.21771, ECCV 2024). LLaVA-1.5 (arXiv:2310.03744). COCO (arXiv:1405.0312). The
signal-detection decomposition (Snodgrass & Corwin 1988; Hautus 1995). The attention-hook
implementation, its causality repair and its two-stream decode loop are this project's existing
gate-verified code (`/data/alexmueller/paialpha/code/pa_gen.py`, frozen copy under
`code/_frozen_refs/`, sha256 `594c7103462fd58bbdf0afa2d59f74ad7737239b41dccf7623b1b5690ff8e677`).

**New here is only the measurement.**

---

## 2. Sample and data provenance — fixed before any generation

**Questions.** The official POPE files, fetched 2026-09-19 from the POPE authors' repository:

| split | URL | sha256 | rows |
|---|---|---|---|
| random | `https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json` | `ac25245170b975a5bdf9080b23fd431dfe6be458bc038259c1f4f09a6bef7994` | 3000 |
| popular | `.../coco_pope_popular.json` | `72c1a8ad45d0c13514f5f22598261df41d3b533854d29682e924db50ed8aa753` | 3000 |
| adversarial | `.../coco_pope_adversarial.json` | `420b3407db1fa9f1187a805dca41cb7b97fd91504e6c2179706188c107fb8ef8` | 3000 |

Verified on the cluster before registration: JSON-lines, 3000 rows each, 3000 distinct
`question_id`, **500 distinct images, the same 500 in all three splits**, and each split exactly
1500 `yes` / 1500 `no`. The checksums are re-asserted at load time (`pope_items.load_items`);
a changed file aborts.

**Images.** POPE references `COCO_val2014_*.jpg`. `/data/datasets/coco/val2014` is a symlink to
`images_val`, which holds **val2017**-named files; only 74 of the 500 POPE images are present
there under 2017 naming and none under the val2014 name. All 500 were therefore fetched from
the `coco_url` field of `/data/datasets/coco/annotations_2014/instances_val2014.json`
(79 MB, `data/val2014_images/`). Every image is verified openable, its `(width, height)`
asserted equal to the annotation's, and a per-file sha256 manifest written
(`data/image_manifest.json`). **Gate G1 fails loudly if any of the 500 is missing or
mismatched**, before any generation.

**No sub-sampling.** All 500 images and all 9000 questions are used in every arm.

---

## 3. Model, prompt and decoding — fixed

- `llava-hf/llava-1.5-7b-hf`, bfloat16, single RTX 4090, `device_map="cuda"`.
- Prompt: PAI's own LLaVA-1.5 POPE template, reproduced byte-for-byte from
  `LALBJ/PAI@master` `constants.py` (`SYSTEM_MESSAGE` + `INSTRUCTION_TEMPLATE["llava-1.5"]`,
  concatenated with **no** separator, as `pope_eval.py` L88–90 does), with the POPE question
  verbatim:

  `A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.USER: <image> {question} ASSISTANT:`

  **PAI's POPE protocol appends no "answer with a single word" suffix.** That suffix belongs to
  the LLaVA-1.5 repo's own POPE protocol. Using it would make this our approximation of PAI
  rather than PAI, so it is not used. This is a registered deviation from the *LLaVA* POPE
  protocol and is expected to cost accuracy relative to LLaVA-1.5's published POPE numbers;
  the positive control in §8 is set against PAI's numbers for that reason.
- Greedy: `do_sample=False`, `num_beams=1`, no temperature (PAI `pope_eval.py` defaults).
- Budget: `max_new_tokens = 64`. PAI's default is 512; POPE answers are a word or a short
  sentence, and the per-arm truncation rate is reported and gated (§7 G_TRUNC) so the
  generation-budget confound is excluded by measurement. **Registered contingency:** any arm
  with truncation rate > 0.02 is re-run at 512 and the 512 run is the one reported for that arm.
- Batching: items are grouped by **exact prompt token length**, so every batch is rectangular
  with **zero padding**, and the plan depends only on the text — never on the pixels, the
  method or the arm. Every arm therefore runs byte-identical token tensors in byte-identical
  batches (asserted: G2, and `FATAL_BATCH_HAS_PADDING` inside the encoder).

---

## 4. The arms

### 4.1 Pixel arms — only the pixels change

| arm | frame | definition |
|---|---|---|
| `sighted` | own | the unmodified image |
| `grey` | own | flat **(127,127,127)** RGB at the image's own pixel size — the identical blinding constant used in §4.2 and Appendix G of the paper |
| `noise25` | m336 | additive Gaussian, sigma = 25/255, clipped |
| `noise50` | m336 | additive Gaussian, sigma = 50/255, clipped |
| `noise100` | m336 | additive Gaussian, sigma = 100/255, clipped |
| `blur4` | m336 | Gaussian blur, radius 4 px |
| `blur16` | m336 | Gaussian blur, radius 16 px |
| `lowres` | m336 | bicubic 336 -> 16 -> 336 |
| `pshuffle` | m336 | permute the 24x24 grid of 14x14 ViT patches, fixed per-image permutation |

`own` = the image's own pixel frame. `m336` = the model's own input frame (shortest-edge resize
to 336 then centre crop 336, exactly the processor's geometry), so that a dose is a controlled
dose and a patch is a real ViT patch. Gate G9 measures how far the two frames can differ by
themselves and is REPORTED, not blocking. All randomness is `default_rng(20260919 + image_id)`,
fixed before any generation and asserted deterministic.

### 4.2 Method arms

All four share one execution path: `alpha = 0` is `sc + 0*|sc|`, exact in IEEE, and
`gamma = 1` takes PAI's own short-circuit in `CFG.py`. **`vanilla` is kernel-matched and
loop-matched to the intervened arms, not a different code path** — this project has had a
baseline diverge at token 6 purely from running a different path.

| method | alpha | gamma | Eq.3 rows | what it is |
|---|---|---|---|---|
| `vanilla` | 0.0 | 1.0 | — | no intervention |
| `pai_full` | 0.5 | 1.1 | conditional **decode** steps only | the released `--use-attn --use-cfg` path, at the alpha PAI's paper prescribes for LLaVA-1.5-7B |
| `pai_attn` | 0.5 | 1.0 | last query row of **every** conditional forward, prefill included | the released `--use-attn` path alone |
| `pai_both` | 0.5 | 1.1 | last query row incl. prefill | **CONSTRUCTED, not a released configuration** |

**A mechanism fact, verified at primary source before registration, that makes this axis
necessary rather than decorative.** In `LALBJ/PAI@master`, `attention.py` L90 guards the
amplification with `if use_attn and not use_cfg`, and `llama_modify` sets
`self_attn.use_cfg = True`; `CFG.py` L31–33 only clears `use_cfg` *inside* `CFGLogits.__call__`,
which first runs on the **prefill** logits. Therefore, under the README's
`--use-attn --use-cfg` command, **PAI's attention stage never touches the first generated
token.** On POPE the answer *is* the first generated token. So:

- `pai_full` minus `vanilla` measures **Eq. 4 alone** on POPE.
- `pai_attn` minus `vanilla` measures **Eq. 3 alone** on POPE.
- `pai_both` exists so that a null on the two released arms cannot be answered with "you
  crippled the method". It is labelled CONSTRUCTED wherever it is reported and is never
  promoted over a released arm.

### 4.3 Tiers and the cut order

- **Tier A (mandatory):** `{sighted, grey}` x `{vanilla, pai_full, pai_attn, pai_both}`, all
  three splits. This carries the primary.
- **Tier B (registered secondary):** the seven ladder arms x `{vanilla, pai_full, pai_attn}`.
- **Tier C (exploratory, not part of any primary):** LLaVA-OneVision-0.5B, Tier A only.

If wall-clock forces a cut the order is fixed now: Tier C first, then `pai_attn` in Tier B,
then `noise25` and `blur4` (the mildest rungs). **Whatever is cut is named in the report with
the reason.** Tier A is never cut; if Tier A cannot complete, the lane reports NO-DATA.

---

## 5. Endpoint and primary estimand

For each cell (method x pixel arm x split) the scored unit is a POPE question.

- `H` = yes-rate on `label == yes` (present) items — POPE's recall.
- `F` = yes-rate on `label == no` (absent) items — the false-alarm rate.
- `J = H - F`; `d' = Z(H) - Z(F)`; `c = -0.5[Z(H) + Z(F)]`, with the **log-linear
  (add-one-half) correction** applied to every arm unconditionally, as in `app:mitig-rates`.
- POPE's own reported statistics — accuracy, precision, recall, F1, yes-ratio — are computed by
  the **official** `POPE/evaluate.py` mapping, reproduced verbatim (§6).

**Because each split is exactly 1500/1500 and the pooled set is exactly 4500/4500,
`accuracy = (1 + J)/2` identically.** The paper's decomposition coordinate and the benchmark's
headline number are therefore the same statistic in two coordinates, not two endpoints; a
decision rule on `J` is a decision rule on accuracy, and `Delta(acc) = Delta(J)/2` exactly.

**PRIMARY.** Pooled over the three splits (9000 items, 500 images), for `pai_full`:

```
  Delta_s    = J(pai_full, sighted) - J(vanilla, sighted)
  Delta_a    = J(pai_full, grey)    - J(vanilla, grey)
  Delta_image = Delta_s - Delta_a            <-- THE PRIMARY ESTIMAND
  rho        = Delta_a / Delta_s             (fraction of the gain surviving blinding)
```

**SECONDARIES, all registered here:**

- **S1** the same four quantities for `pai_attn` (released) and for `pai_both` (constructed).
- **S2** the same, per split. Three splits x three methods; reported in full, never selected.
- **S3** the same decomposition on `d'` and on `c`, and on `H` and `F` separately — a null on
  `J` can hide offsetting movements in the two rates (`sec:reco` item 1).
- **S4** the dose-response ladder: levels and `Delta_image` across the seven degradation arms.
  **Registered reading:** if `J(vanilla, .)` falls monotonically along
  `sighted -> noise25 -> blur4 -> noise50 -> lowres -> blur16 -> pshuffle -> noise100 -> grey`
  toward the grey arm's value, the grey arm is the limit of a degradation continuum rather than
  an isolated out-of-distribution point, which is the control Appendix G says was available and
  not run. Non-monotonicity is reported as non-monotonicity; no rung is dropped for being
  inconvenient, and the ordering above is fixed now and is not re-sorted after seeing the data.
- **S5 blindability:** `accuracy(vanilla, grey)` against chance 0.50.

---

## 6. Answer parsing — two parsers, both fixed now

- **`lenient` (PRIMARY).** The official POPE mapping, reproduced verbatim from
  `RUCAIBox/POPE@main/evaluate.py` (sha256 `f0371dcec5f1bdbde017d1a02f5a9a04c53cb561a2874f3fb8f0718ce3d87036`,
  identical to `haotian-liu/LLaVA` `llava/eval/eval_pope.py`): keep only the text before the
  first `.`, delete `,`, split on spaces; if `No`, `not` or `no` is among the words the answer
  is `no`, **otherwise it is `yes`**. This is used for the primary because it is what every
  published POPE number uses.
- **`strict` (SECONDARY and the degeneracy screen).** The answer, lowercased and stripped of
  leading whitespace and punctuation, must *begin* with the word `yes` or the word `no`.
  Anything else is **UNPARSEABLE** and is counted, never silently mapped.

**Why both.** The lenient mapping sends anything without a negation to `yes`. An arm that
babbles therefore scores `H = F = 1`, `J = 0`, accuracy 0.50 — numerically "chance" for a
reason that has nothing to do with chance. The strict parser exists to catch exactly that, and
§7 makes it blocking.

---

## 7. Gates — all blocking unless marked REPORTED. None is read after the fact.

| gate | condition |
|---|---|
| **G1 data integrity** | all 500 images present and openable, dims equal to `instances_val2014.json`, POPE file sha256 match, 3000 rows and 1500/1500 per split |
| **G2 token identity** | the per-item `sha256(input_ids)` is identical across every pixel arm and every method; zero padding in every batch |
| **G3 image block** | exactly 576 image tokens in every row |
| **G4 copy drift** | this lane's frozen kernel at `rowmode="decode"` reproduces `pa_gen.greedy_two_stream` **token-for-token**; `pa_gen.py`'s sha256 still equals the frozen value |
| **G5 decode loop** | the two-stream loop at `alpha=0, gamma=1` matches HF `generate(do_sample=False)` token-for-token **through the same kernel** |
| **G6 causality** | post-softmax mass on strictly-future keys is exactly 0 when the kernel is handed no mask; `mask_leak == 0` |
| **G7a liveness — edits** | `vanilla`: 0 edits. `pai_full`: edits > 0 **and prefill edits == 0**. `pai_attn`/`pai_both`: edits > 0 **and prefill edits > 0**. A violation means the arm is not the configuration it claims to be |
| **G7b liveness — mass** | mean image-token attention mass strictly higher under each PAI arm than under `vanilla` |
| **G7c answers move** | at least some answers differ between `vanilla` and the PAI arms. REPORTED magnitudes: on a binary endpoint most answers coincide even under a live intervention, so only "some move" is blocking |
| **G8 grey constancy** | the grey pixel tensor is bit-identical across every row and every image |
| **G9 frame equivalence** | REPORTED. max abs pixel difference between processing the original and processing the pre-cropped 336 image; bounds how much the `m336` ladder frame can move a number by itself |
| **G10 batch invariance** | REPORTED. answers batched vs singly. Batch composition is identical across arms by construction, so any batch-shape effect is common to all arms and cancels in the paired contrasts |
| **G11 determinism** | `vanilla` run twice gives identical answers on every item |
| **G_DEGEN** | see §9. Blocking for the reading of any cell it fires on |
| **G_TRUNC** | per-arm truncation rate <= 0.02, else the §3 contingency fires |
| **G_PC** | the positive control of §8 |
| **G_BOOT** | resample size asserted equal to n on every replicate; no `set()` anywhere on the resampling path; multiplicity preserved |

**No endpoint is computed until every blocking gate has passed.** A blocking failure aborts the
lane and is reported as a failure, not worked around.

---

## 8. Positive control

Registered before any generation. The port is not trusted until it reproduces an already-
published number for **vanilla LLaVA-1.5-7B on COCO POPE**, greedy decoding, per split.

### PC-A (published level, blocking)

**The target is PAI's own published number, not LLaVA-1.5's.** Verified at primary source
(arXiv:2407.21771v1 -- arXiv has only v1 -- cross-checked cell-by-cell against the ECVA
open-access camera-ready `ecva.net/papers/eccv_2024/papers_ECCV/papers/10933.pdf` and its
supplementary, extracted and grepped with positive controls):

- **PAI Table 2, single-turn, greedy, "Vanilla", LLAVA (= LLaVA-1.5-7B): Accuracy 84.76,
  F1 85.51.** PAI supplementary Table S3 gives the same accuracy 84.76 and F1 85.59 for the
  same row -- an internal inconsistency of 0.08 F1, present in the proceedings version too.
- **PAI reports POPE only as an average over the three splits.** Its Table S2/S3 say so
  verbatim; its Table 2 columns are single-turn / multi-turn, not random / popular /
  adversarial. There is therefore **no published per-split target for this protocol**, and we
  do not invent one.
- PAI's setup text: "500 images, with each image having 6 questions for each split of POPE" --
  the same 500 x 6 x 3 design this lane runs.

**Why not LLaVA-1.5's per-split numbers (87.3 / 86.1 / 84.2, Table 4).** Three reasons, all
verified: (i) those are **F1, not accuracy** -- the paper reports no POPE accuracy; (ii) they
use the LLaVA repo's own POPE protocol, whose prompt appends "Answer the question using a
single word or phrase.", which PAI's protocol does not -- and this lane runs PAI's protocol
because the method under test is PAI; (iii) they were measured on the **pinned
`AoiDragon/POPE@e3e3926` snapshot**, whose random split has **2910 questions, 1500 yes /
1410 no**, not 3000 balanced. On that snapshot an all-"yes" model scores 51.55% rather than
50.00%, so it is not exchangeable with the file this lane uses, and the balanced identity
`accuracy = (1+J)/2` does not hold on it. This lane uses current `RUCAIBox/POPE@main`, which
is exactly 1500/1500 in all three splits (verified, §2); it differs from PAI's own files only
by a spelling fix ("imange" -> "image") on 44 popular and 58 adversarial questions, which is
content-neutral, and PAI's random file is already the 3000-question version.

**PC-A band, fixed now: the 3-split-average vanilla sighted accuracy must fall in
[82.26, 87.26] and the 3-split-average F1 in [83.01, 88.01]** -- PAI Table 2's 84.76 / 85.51
plus or minus **2.5** points.

Why 2.5. The two same-family protocols that can be compared at all (LLaVA-1.5's mean of its
three F1 values, 85.87, and PAI's 85.51/85.59) agree to within 0.36 F1, and PAI's internal
inconsistency is 0.08, so 2.5 is about seven times the observed same-protocol spread -- wide
enough to absorb the deviations this port really has (the `llava-hf` checkpoint rather than
the original release, a hand-written two-stream decode loop, a registered attention kernel,
bfloat16, a 64-token budget instead of 512, and the spelling fix) and still narrow enough to
**fail** the one protocol error that matters: OPERA (arXiv:2311.17911v3 Table 4) reports 82.2
average F1 for the same model and the same decoding, and its `pope_eval.py` uses the same
`USER: <ImageHere> <question> ASSISTANT:` template **with the Vicuna system message omitted**.
82.2 lies outside [83.01, 88.01], so a port that silently dropped the system message fails this
gate. A band wide enough to contain it would not be a control.

### PC-A2 (structural, blocking)

`accuracy(vanilla, sighted)` must satisfy **random >= popular >= adversarial**. This ordering
holds in every published POPE table we verified (POPE's own Table 3 for all five of its models,
VCD Table 1, SID Table 1) and is the qualitative signature the popular and adversarial splits
are constructed to produce. A port that scrambles it is scoring the wrong questions against the
wrong labels.

### PC-A3 (reported, NOT blocking)

Our `Delta_s` against PAI's published gain: Table 2 single-turn greedy gives PAI 85.82 / 85.97
against vanilla 84.76 / 85.51, i.e. **+1.06 accuracy points and +0.46 F1**; Table S3 gives
86.13 / 86.42 against 84.76 / 85.59, i.e. **+1.37 and +0.83**. This is reported, never gated.
**Requiring the gain to reproduce would calibrate the gate against the answer** and would make
`NO-GAIN-TO-DECOMPOSE` -- a registered outcome of §10 -- unreportable by construction.


Secondary internal control, registered unconditionally and independent of any published value:

- **PC-B (internal, blocking).** `accuracy(vanilla, sighted)` must exceed
  `accuracy(vanilla, grey)` by at least 0.10 on every split, with the paired 95% interval
  excluding zero. A port in which the real image buys less than ten accuracy points over a flat
  grey field is not measuring POPE, whatever its absolute level.
- **PC-C (internal, blocking).** `d'(vanilla, sighted)` exceeds `d'(vanilla, grey)` with the
  interval excluding zero, on every split. This is the analogue of the paper's own attention
  gate (`app:gates`), which established that the model uses the image at all.

**If PC-A fails, the lane reports NO-DATA and the endpoint is not read.** A port that does not
reproduce a published level cannot decompose a published gain.

---

## 9. Degeneracy screens — a collapsed arm is not a null

Thresholds fixed now, per cell (method x pixel arm x split, and pooled):

| screen | threshold | label |
|---|---|---|
| constant response | `yes_ratio >= 0.95` or `<= 0.05` | `DEGENERATE-CONSTANT` |
| unparseable | strict-parser unparseable fraction `> 0.05` | `DEGENERATE-UNPARSEABLE` |
| empty | empty-answer fraction `> 0.01` | `DEGENERATE-EMPTY` |
| truncation | `> 0.02` at the budget | `BUDGET-SUSPECT`, §3 contingency |

Also reported for every arm regardless: mean and median answer length, empty rate, number of
distinct answer strings, and mean image-attention mass.

**THE TRAP THIS EXISTS TO STOP, NAMED IN ADVANCE.** If both grey arms answer "yes" to
everything, then `H = F = 1` and `J = 0` in both, so `Delta_a` is **mechanically** zero and
`Delta_image = Delta_s` — which would read as IMAGE-ATTRIBUTABLE while measuring nothing at all.
**That reading is forbidden here, before the data exist.** If either grey arm fires
`DEGENERATE-CONSTANT` or `DEGENERATE-UNPARSEABLE`, the primary returns
**`BLIND-ARM-DEGENERATE`** and `Delta_image` is reported as **UNDEFINED, not as a null and not
as an attribution** — exactly as `app:mitig-blind` reports the blind CHAIR arm. The ladder (S4)
is then the only ablation evidence this lane has, and is reported as such.

---

## 10. Decision rules — every outcome, including the ones that hurt

`MDE_J = 0.01` in `J` units (= **0.5 accuracy points**). Fixed now, and fixed from a *published*
quantity rather than from anything this lane produces: PAI's published POPE gain for this model
is **+1.06 accuracy points** (Table 2 single-turn greedy) or **+1.37** (Table S3), so the floor
for "there is a gain worth decomposing" is set at half the smaller of the two.

**The effect being decomposed is small, and the rules below are written knowing that.** A gain
of ~1 accuracy point on 9000 items clustered in 500 images may not be separable from zero after
subtracting a second noisy contrast. Rather than discover that afterwards, the power condition
is a registered step that runs *before* any outcome label is assigned, and `UNDERPOWERED` is a
first-class registered outcome, not a hedge added later.

**P0 -- precondition.** If `Delta_s < MDE_J` or its 95% interval contains 0, the registered
outcome is **`NO-GAIN-TO-DECOMPOSE`**. There is then no gain whose attribution could be tested,
and we say so plainly rather than decomposing noise.

**P1 -- blind-arm validity.** Section 9. If it fires: **`BLIND-ARM-DEGENERATE`**, primary
UNDEFINED.

**P2 -- power.** Let `h` be the half-width of the 95% interval of `Delta_image`. If
`h > Delta_s` (point estimate), the data cannot distinguish "all of the gain is
image-attributable" from "none of it is", and the registered outcome is **`UNDERPOWERED`**.
We report `h`, `Delta_s`, `Delta_a` and `rho` with their intervals, state the sample at which
the question would be decidable, and make **no attribution claim in either direction**.

**If P0, P1 and P2 all clear, the primary returns exactly one of the four below.** The dividing
line is **half the sighted gain**, `0.5 * Delta_s` (point estimate), fixed now because an
absolute margin comparable to the whole effect would be meaningless at this effect size.

| outcome | condition | what we write |
|---|---|---|
| **`NOT-IMAGE-ATTRIBUTABLE`** | 95% CI of `Delta_image` contains 0 **and** its upper bound `< 0.5 * Delta_s` | The gain survives blinding, and more than half of it being image-attributable is excluded. **Supports the paper.** Reported in §5 as the standard-benchmark demonstration the CHAIR lane could not produce. |
| **`IMAGE-ATTRIBUTABLE`** | 95% CI of `Delta_image` excludes 0 (positive) **and** its lower bound `> 0.5 * Delta_s` | Most or all of the gain is gone under blinding. **This CUTS AGAINST the paper and becomes the headline of that section.** See below. |
| **`PARTIAL`** | CI of `Delta_image` excludes 0 (positive) but its lower bound `<= 0.5 * Delta_s` | Part of the gain survives blinding, part does not. Report `rho` with its interval and both halves; neither reading is promoted. |
| **`CANNOT-RESOLVE`** | anything else | Reported as unresolved, never as a null. |

**What we write if the result is `IMAGE-ATTRIBUTABLE` — fixed now, before the data exist.**
We report it first and plainly, as the headline of that section, in these terms: on a standard
benchmark with an externally supplied candidate set, the paper's own control was runnable, was
run, and the published method *passed* it — the gain disappears when the image is removed, so it
is image-attributable. The paper's §5 claim is then narrowed to what it actually establishes:
that CHAIR's endogenous denominator makes *CHAIR* unable to say what kind of improvement it is,
and that this is a property of the metric, not a suspicion about the method. `sec:reco` item 2
is *strengthened*, not weakened — blinding is shown to be informative and cheap on a benchmark
people report. Any sentence in the paper that reads as "published gains may not be
image-attributable" must be cut or restricted to endpoints of the paper's own prefix type, and
the abstract must not survive unchanged. We do not report this outcome as a limitation
paragraph.

**What we write if it is `NOT-IMAGE-ATTRIBUTABLE`.** Reported as the paper's strongest single
result: a published method's improvement on a benchmark the field reports is reproduced with no
contribution from the image, measured on the official splits with an interval. It replaces the
invented endpoint as the lead demonstration, and the invented endpoint becomes the mechanism
study.

**What we write if it is `NO-GAIN-TO-DECOMPOSE`.** That we could not test attribution because
the method did not improve POPE in this port, reported with the per-stage decomposition of §4.2
— specifically whether Eq. 3, Eq. 4 or neither moves POPE — and with the primary-source fact
that the released `--use-attn --use-cfg` path cannot reach a one-token answer with Eq. 3 at all.
That fact is itself reportable and is not a workaround.

**What we write if it is `BLIND-ARM-DEGENERATE`.** That POPE, like CHAIR, resisted this
particular ablation on this model — which would *weaken* `sec:reco` item 2's claim that
"POPE-style existence questions allow it directly" and must be written as such, with the
recommendation softened to name the failure mode.

**S5, blindability — registered separately.** `BLINDABLE` if `accuracy(vanilla, grey)` is at or
below 0.55 with its interval, and the arm passes §9. `NOT-BLINDABLE` if it stays materially
above chance — which would mean POPE is partly answerable from the question text alone, a
finding about the benchmark that we report whether or not it suits us, and which would also
mean the "candidates supplied in the input" route the paper recommends carries its own
text-only shortcut.

**Multiplicity.** One primary. Everything else is labelled secondary and is reported in full,
never selected. No outcome is read off the most favourable split, the most favourable method,
or the most favourable rung of the ladder. The registered primary is `pai_full`, pooled.

---

## 10.1 Intervals

Paired image-level cluster bootstrap, `B = 4000`, `default_rng(20260919)`. A replicate resamples
the **500 image ids** with replacement and carries **all** of that image's rows — all three
splits, all arms — into the replicate together, so every contrast is paired on the same
resampled image multiset. Multiplicity is preserved: an image drawn three times counts three
times. **The resample size is asserted equal to 500 on every replicate, and `set()` never
touches a resample** (it silently makes it a 63.2% subsample; that bug cost this project nine
intervals). Intervals are 2.5/97.5 percentile ranges. An alternate-seed replication
(`default_rng(20260920)`) is run for the primary and reported.

---

## 10.2 Amendment 1 — 2026-09-19, before any endpoint generation

**Provenance, stated precisely.** This amendment was written after the integrity-gate job
(SLURM 21044, COMPLETED 0:0, sentinel `POPE_SENTINEL_GATES_OK`, all blocking gates passed) and
**before any endpoint generation job was submitted** — the Tier A chain is jobs 21052–21055,
submitted after this text was fixed. The gate job did generate answers, on 16 images, for
`sighted` and `grey` under three methods; `pope_gates.py` computes **no POPE statistic of any
kind** from them — it compares answer strings for equality, counts attention edits and checks
token identity. **No accuracy, F1, H, F, J, d' or c has been computed on any arm at the time
of this amendment, and the arm added below has no data at all.**

### A1.1 A mismatched-real-image ablation is added, as a CO-PRIMARY

New pixel arm **`mismatch`**: the queried image's pixels are replaced by **a different real
COCO image** drawn from the same 500-image POPE set, under a fixed seeded **derangement**
(`default_rng(20260919)`, re-drawn until no fixed point; asserted bijective, asserted no image
paired with itself, verified: 500 images, 0 fixed points). The mapping is recorded in the
arm's meta file. The question text and every token id are unchanged, as in every other arm.

**Why.** The standing objection to a flat grey field is that it is out of distribution, so a
collapse under it may mean the model is behaving unpredictably rather than that the endpoint is
image-free. That objection is now in print: arXiv:2509.23499v2 rejects "zeroing out (e.g.,
using a blank image...)" because it "creates unnatural, out-of-distribution inputs that elicit
unpredictable model behavior". Liao et al. (2606.31257) answer it by requiring several
ablations to agree, one of which is a real-but-mismatched in-distribution image. A mismatched
real COCO image is in distribution *and* carries no information about the queried image, so it
is the cleanest form of the same control. This lane's own §0.1 already conceded the exposure;
the amendment closes it instead of conceding it.

**Registered expectation, fixed now:** accuracy falls toward the model's own answer prior while
the model keeps answering normally — i.e. the arm should *not* trip the §9 degeneracy screens.

**Decision rule — CO-PRIMARY, and it can downgrade the headline.** The primary estimand of §5
is computed under both ablations. Then:

| condition | registered outcome |
|---|---|
| grey and mismatch return the same label | that label, reported as confirmed under two ablations |
| both return an attribution label but they differ | **`ABLATION-DISAGREEMENT`** — both reported in full, **neither promoted**, and the disagreement is itself the finding |
| grey fires `BLIND-ARM-DEGENERATE` but mismatch returns an attribution label | the **mismatch** label carries the primary, and we say that the in-distribution ablation rescued a control the out-of-distribution one could not run — which is the POPE analogue of the CHAIR failure in `app:mitig-blind` |
| mismatch fires a degeneracy screen | reported as such; grey alone carries the primary and the co-primary is recorded as unmet |

`S5` (blindability) is likewise computed for both ablations. Nothing else in §5, §9 or §10
changes: the same `MDE_J`, the same `0.5 * Delta_s` dividing line, the same power condition,
the same four outcomes.

### A1.2 The answer distribution is reported for every arm, not only the score

For every arm **and every split** we now report the **yes-ratio** together with accuracy,
precision, recall and F1 and the `H / F / J / d' / c` decomposition, plus the **raw answer
distribution** (the 20 most frequent answer strings with counts, pooled, and the 10 most
frequent per split), the number of distinct answer strings, mean and median answer length,
empty rate, truncation rate, and the strict-vs-lenient parser agreement rate.

**Why.** POPE's own paper reports a yes-ratio, so this is native to the benchmark, and
above-chance accuracy under ablation can be answer bias rather than content leakage — a recent
audit attributes 20–31% of above-chance blind accuracy on multiple-choice video benchmarks to
answer-position bias alone. Without the distribution, a blinded arm that answers "yes" to
nearly everything reports as accuracy 0.50, "chance", when it is a degenerate regime.

The §9 degeneracy screen is **unchanged in threshold** (`yes_ratio >= 0.95` or `<= 0.05`; it
was fixed in the original registration, before any data existed) but now runs **per split as
well as pooled**, because an arm can be constant on the adversarial split and not on the
random one, in which case that split's contrast is undefined while the pooled one is not.

---

## 10.3 Amendment 2 — 2026-09-19, during Tier A generation, before any endpoint was read

**Provenance.** Written while jobs 21052/21053 were generating and **before `pope_score.py`
had ever been run on any cell** — no accuracy, F1, H, F, J, d' or c existed for any arm. None
of the three items below changes an arm, an estimand, a threshold or a decision rule. Two are
additional checks; one conforms an implementation to what §4.1 already said it was.

### A2.1 The constant-responder regime is named explicitly, and why it is the motivating case

The §9 screen already fired on `yes_ratio >= 0.95` **or** `<= 0.05`, both fixed before any data
existed. It now emits the two regimes under separate labels, `DEGENERATE-CONSTANT-YES` and
`DEGENERATE-CONSTANT-NO`, because they are different failures and the write-up must name which
one occurred. **Thresholds are unchanged.**

The reason this matters is a published instance. Lan et al. (arXiv:2605.22903), whom this paper
cites, report blinded LLaVA-1.5-7B on POPE at **Accuracy 0.50, Precision 0.00, Recall 0.00** —
an always-"No" responder — and read the 0.50 in their main text as evidence that models "are
not independent of visual input". On a 1:1 balanced split a constant answer scores 0.50 by
arithmetic, and `J = H - F = 0` whichever constant it is; it is not evidence of anything about
visual dependence. **That is precisely the misreading this paper exists to prevent, and it is
the clearest available motivating example for why the decomposition is needed.** Verified
against our own scorer: the always-"No" corner returns Acc 50.00 / Rec 0.00 / yes-ratio 0.00 /
H 0.000 / F 0.000 / **J 0.000**, and the always-"Yes" corner returns Acc 50.00 / Prec 50.00 /
Rec 100.00 / yes-ratio 1.00 / H 1.000 / F 1.000 / **J 0.000** — the same accuracy and the same
J from opposite behaviour, which only the answer distribution distinguishes. (Our scorer
returns Precision `nan` rather than `0.00` at that corner, since it is 0/0; the difference is a
convention, not a disagreement.)

**Their ablation is not ours.** Lan et al. blind with a flat **black** image; this lane uses
the flat **grey** (127,127,127) the paper uses throughout, plus the in-distribution mismatched
real image of Amendment 1. Their number is therefore a motivating example, **not** a target
this lane is trying to reproduce, and it is not used as a positive control.

### A2.2 A free algebraic cross-check between the two coordinate systems

On a 1:1 balanced split with `m` positives and `m` negatives:

```
  H = Recall          Accuracy = (H + 1 - F)/2      =>  F = Recall + 1 - 2*Accuracy
  J = 2*Accuracy - 1  Precision = H/(H+F)           F1 = 2H/(1 + H + F)
```

so the POPE-standard table (accuracy, precision, recall, F1, yes-ratio) and our decomposition
(H, F, J, d', c) are **two coordinate systems on the same two rates**, and any disagreement
between them is a bug in one of the two code paths rather than a finding. The scorer now
asserts all five identities on **every full-sample cell**, at a tolerance of `1e-9`, and
records the realised worst deviation. Calibration before the assertion went live: **604
synthetic balanced cells, including all four degenerate corners, worst absolute deviation
4.44e-16.**

The identity holds only where the split is balanced. Bootstrap replicates resample *images*,
and an image does not contribute equal numbers of yes and no questions, so replicates are not
balanced; the check is therefore asserted on full-sample point estimates only and is **not**
claimed on replicates.

Corollary worth recording for the paper: because the map is exact, **any published POPE table
can be recoded into (H, F, J, d', c) without rerunning anything**, provided the split is
balanced and the yes-ratio or two of the four rates are given.

### A2.3 The decoding rule is matched across every arm, and is stated

"The Mirage of Performance Gains" (NeurIPS 2025) reports that contrastive-decoding papers
commonly compare a **sampled** baseline against a near-**greedy** intervened arm, so a `d'`
computed from their published tables is confounded by the decoding rule rather than the method.
This lane is immune by construction and now says so in its output: every arm uses
`do_sample=False, num_beams=1`, the same registered attention kernel and the same two-stream
loop; `vanilla` is `alpha = 0` (`sc + 0*|sc|`, exact in IEEE) and `gamma = 1.0` (PAI's own
`CFG.py` short-circuit), so it is kernel-matched and loop-matched rather than a different code
path. **No contrast anywhere in this lane compares across decoding modes.** The statement is
emitted into `pope_results.json` so it cannot drift out of the write-up.

### A2.4 `_to336` conformed to the registered definition

§4.1 defines the `m336` frame as "exactly the processor's geometry". Gate G9 (REPORTED, run
before any endpoint arm) measured a max absolute pixel difference of **2.221** on 2 of 8 probe
images: our helper rounded the long side where HF's `get_resize_output_image_size` floors it,
shifting the centre crop by one pixel on some aspect ratios. The helper now floors, which makes
it what §4.1 already said it was. Re-measured over 40 images: **max absolute difference 0.030,
max mean absolute difference 0.000113** — float32 resample noise. This touches only the seven
Tier B ladder arms (`m336`); `sighted`, `grey` and `mismatch` are all `own`-frame and were never
affected, so **no primary or co-primary quantity is touched**. The fix was on disk before any
Tier B job started.

---

## 11. Execution log and results

*(appended after the pre-registration commit)*
