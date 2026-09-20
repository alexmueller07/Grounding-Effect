# POPE lane — report

Lane `pope`. Pre-registration: `SPRINT/POPE_PREREG.md`, commit `90e428d`, with Amendment 1
(`e18eb52`) and Amendment 2 (`f8a52e0`). Node vgi1, `--gres=gpu:rtx_4090:1`, explicit
`--time`, every payload ends `|| exit 9`, verification by output file + sentinel.
Lane code: `/data/alexmueller/sprint/pope/code`. Nothing in the paper build directory was
touched, and no shared lane's code was edited.

**What this lane is for.** `paper.tex` §5 concedes that the grey-image control *could not be
run* on CHAIR, because CHAIR's denominator is the model's own mention set and a grey image
empties it (`app:mitig-blind`). `sec:reco` item 2 then recommends to the field exactly the
thing the paper has never done on a standard benchmark: blind every endpoint whose candidates
are supplied in the input, and it names POPE. This lane runs that recommendation on POPE —
official splits, official question files, 500 COCO val2014 images — and applies the paper's
`Delta_image = Delta_sighted - Delta_ablated` decomposition to a published method's POPE gain.

---

## 0. Headline — and it cuts against the paper

**PAI's POPE gain is image-attributable.** On the arm that reproduces PAI's published POPE
number, the improvement **disappears when the image is replaced by a real image of another
scene**. Pooled over the three official splits, 9,000 questions, 500 images, paired
image-clustered bootstrap B = 4000:

| | `pai_attn` (released `--use-attn`, α = 0.5) |
|---|---|
| `Delta_sighted(J)` | **+0.0220** [+0.0107, +0.0336] |
| `Delta_ablated(J)`, mismatched real image | **+0.0013** [−0.0102, +0.0124] |
| **`Delta_image(J)`** | **+0.0207** [+0.0042, +0.0371] — **excludes zero** |
| fraction surviving blinding, `rho` | **+0.06** [−0.52, +0.68] |
| in accuracy points | `Delta_sighted` +1.100 [+0.53, +1.68]; `Delta_ablated` +0.067 [−0.51, +0.62]; **`Delta_image` +1.033 [+0.21, +1.86]** |

The registered label is **`PARTIAL`**, not `IMAGE-ATTRIBUTABLE`, and we report the label the
rule produced: `Delta_image`'s interval excludes zero but its lower bound (+0.0042) sits just
below the pre-registered half-gain line (0.0110), so the rule cannot certify that *more than
half* is image-attributable. What the numbers say beyond the label: the point estimate of
`Delta_image` is **94% of the whole sighted gain**, and `Delta_ablated` is tight around zero.
**The control worked and the method survived it.** `pai_both` (constructed) agrees:
`Delta_image(J)` +0.0238 [+0.0078, +0.0398], `rho` −0.03.

**What this does to the paper.** Section 5's argument must narrow to what it actually
establishes. It shows that *CHAIR* cannot say what kind of improvement PAI's is, because
CHAIR's denominator is the model's own mention set. It does **not** license any suggestion that
published grounding gains are generally not image-attributable: on a standard benchmark with an
externally supplied candidate set, the paper's own control was runnable, was run, and **the
published method passed it.** Any sentence reading as "published gains may not be
image-attributable" must be cut or restricted to endpoints of the paper's own prefix type, and
the abstract cannot stand unchanged. This is not a limitations-paragraph result.

**Two things that survive intact, and one that is new.** The decomposition machinery is
vindicated, not undermined — see §7.8: PAI's criterion shift is **largely not**
image-attributable (`Delta c` +0.154 total, only +0.047 of it image-attributable) while its
accuracy gain **is**, and the decomposition is what separates them. The registered degeneracy
trap fired on the grey arm and prevented a spurious attribution (§7.1). And the whole POPE gain
turns out to live in a single query row of the attention matrix (§7.7).

---

## 1. What ran

| job | name | what | state (verified by sentinel, not by SLURM) |
|---|---|---|---|
| 21044 | `pope_gates` | integrity gates, 16 images | COMPLETED 0:0, 5m15s, `POPE_SENTINEL_GATES_OK` |
| 21052 | `pope_A_vanilla` | vanilla x {sighted, grey, mismatch} | COMPLETED 0:0, 1h34m, `POPE_SENTINEL_GEN_OK` |
| 21053 | `pope_A_paifull` | pai_full x {sighted, grey, mismatch} | COMPLETED 0:0, 1h18m, `POPE_SENTINEL_GEN_OK` |
| 21054 | `pope_A_paiattn` | pai_attn x {sighted, grey, mismatch} | COMPLETED 0:0, 1h11m, `POPE_SENTINEL_GEN_OK` |
| 21055 | `pope_A_paiboth` | pai_both (CONSTRUCTED) x {sighted, grey, mismatch} | COMPLETED 0:0, 1h12m, `POPE_SENTINEL_GEN_OK` |
| 21056 | `pope_scoreA` | scoring + verdict, Tier A | COMPLETED 0:0, `POPE_SENTINEL_SCORE_OK` |
| 21057 | `pope_B_vanilla` | vanilla x 7 ladder arms | RUNNING |
| 21058 | `pope_B_paifull` | pai_full x 7 ladder arms | RUNNING |
| 21059 | `pope_B_paiattn` | pai_attn x 7 ladder arms | PENDING (dependency) |
| 21060 | `pope_scoreB` | scoring + verdict, Tier A + B | PENDING (dependency) |

**Tier A is complete**: 12 cells x 9,000 questions = **108,000 generations**, all 500 images,
all three official splits, four method arms x three pixel arms.

Earlier jobs, recorded for completeness: 21039 cancelled (requested 20G on a node with ~15G
free — memory, not GPUs, is what is rationed here); 21042 FAILED 9:0 on
`FATAL_NO_IMAGE_MANIFEST`, which is the `|| exit 9` convention working: the manifest was
missing because the image fetch had died on an absent Pillow in the *base* interpreter, and
the wrapper's preflight refused to run rather than proceeding on incomplete data.

Never more than 2 concurrent GPU jobs; the whole lane was submitted as one dependency chain.

## 2. Data provenance

**Questions.** Official POPE files from `RUCAIBox/POPE@main/output/coco/`, fetched 2026-09-19,
checksums re-asserted at every load:

| split | sha256 | rows |
|---|---|---|
| random | `ac25245170b975a5bdf9080b23fd431dfe6be458bc038259c1f4f09a6bef7994` | 3000 |
| popular | `72c1a8ad45d0c13514f5f22598261df41d3b533854d29682e924db50ed8aa753` | 3000 |
| adversarial | `420b3407db1fa9f1187a805dca41cb7b97fd91504e6c2179706188c107fb8ef8` | 3000 |

Verified: JSON-lines, 3000 distinct `question_id` each, **500 distinct images, the same 500 in
all three splits**, exactly **1500 yes / 1500 no** per split.

**Images.** POPE references `COCO_val2014_*.jpg`. `/data/datasets/coco/val2014` is a symlink to
`images_val`, which holds **val2017**-named files; **only 74 of the 500 POPE images are there**
under 2017 naming and none under the val2014 name. All 500 were fetched from the `coco_url`
field of `annotations_2014/instances_val2014.json` (79 MB). Every one is verified openable with
its `(width, height)` equal to the annotation's; per-file sha256 manifest written; aggregate
sha256 of the 500 images **`5e3e9de97139411a39e8057a9c3c06890321dfd6855b4ff087faac7695ed7330`**.
Gate G1 re-checks all of this before any generation and aborts on any miss.

**A snapshot trap, recorded because it silently breaks cross-paper comparison.** LLaVA-1.5's
own evaluation docs point at a *different* repo at a *pinned commit*
(`AoiDragon/POPE@e3e3926`). That snapshot's **random** split has **2910 questions, 1500 yes /
1410 no**, not 3000 balanced, and 144 of its questions contain the typo "imange". An all-"yes"
model scores **51.55%** there against **50.00%** on the current file — 1.55 free accuracy points
for a yes-biased model, and LLaVA-1.5 is yes-biased on POPE. LLaVA-1.5's published 87.3/86.1/84.2
were measured on that snapshot. This lane uses current `RUCAIBox/POPE@main`, which is exactly
balanced, so the identity `accuracy = (1+J)/2` holds here and does not hold there.

## 3. Method under test, and a mechanism fact verified at source

The method is PAI (arXiv:2407.21771, ECCV 2024). Its kernel, causality repair and two-stream
decode loop are this project's existing gate-verified code, copied and frozen into this lane
(`pa_gen.py` sha256 `594c7103…`, re-asserted by gate G4 against the original at run time).

**Under the configuration PAI's own README prescribes, PAI's attention stage cannot reach a
one-token answer.** Verified in `LALBJ/PAI@master`: `attention.py` L90 guards the amplification
with `if use_attn and not use_cfg`; `llama_modify` sets `self_attn.use_cfg = True`; and
`CFG.py` L31–33 only clears `use_cfg` *inside* `CFGLogits.__call__`, which first runs on the
**prefill** logits. So with `--use-attn --use-cfg`, the conditional prefill pass is never
amplified, and the first generated token is never amplified. On POPE the answer *is* the first
generated token.

The method axis is therefore not decorative — it separates the two stages:

| arm | alpha | gamma | Eq.3 rows | what it is | on POPE it measures |
|---|---|---|---|---|---|
| `vanilla` | 0.0 | 1.0 | — | no intervention | — |
| `pai_full` | 0.5 | 1.1 | conditional decode only | released `--use-attn --use-cfg` | **Eq. 4 alone** |
| `pai_attn` | 0.5 | 1.0 | last row incl. prefill | released `--use-attn` alone | **Eq. 3 alone** |
| `pai_both` | 0.5 | 1.1 | last row incl. prefill | **CONSTRUCTED**, not a released configuration | both |

`pai_both` exists so that a null on the released arms cannot be answered with "you crippled the
method". It is labelled CONSTRUCTED wherever it appears and is never promoted over a released arm.

`vanilla` is kernel-matched and loop-matched, not a separate code path: `alpha = 0` is
`sc + 0*|sc|` (exact in IEEE) and `gamma = 1.0` takes PAI's own `CFG.py` short-circuit. Every
arm is greedy (`do_sample=False, num_beams=1`); **no contrast in this lane compares across
decoding modes**, which is the confound "The Mirage of Performance Gains" documents in
contrastive-decoding tables.

## 4. Gates — job 21044, all blocking gates passed

| gate | result |
|---|---|
| G1 data integrity | **Pass.** 500/500 images, 0 missing, aggregate sha256 as above |
| G2 token identity | **Pass.** per-item `sha256(input_ids)` identical across `sighted/vanilla`, `sighted/pai_full`, `sighted/pai_attn`, `grey/vanilla`; zero padding in every batch (items are grouped by exact token length: 4614 / 2662 / 1613 / 111 items at lengths 49 / 50 / 51 / 52) |
| G3 image block | **Pass.** exactly 576 image tokens per row (asserted every batch) |
| G4 copy drift | **Pass.** frozen kernel at `rowmode="decode"` reproduces `pa_gen.greedy_two_stream` token-for-token; `pa_gen.py` sha256 unchanged |
| G5 decode loop | **Pass.** 0/4 mismatches against HF `generate(do_sample=False)` through the same kernel |
| G6 causality | **Pass.** post-softmax mass on strictly-future keys exactly **0.0** with no mask handed in |
| G7a liveness (edits) | **Pass.** vanilla (0, 0); `pai_full` (518 applied, **0 prefill**); `pai_attn` (410 applied, **20 prefill**) — each arm is the configuration it claims to be |
| G7b liveness (mass) | **Pass.** mean image-attention mass 0.1643 vanilla < 0.2912 `pai_attn` < 0.3036 `pai_full` |
| G7c answers move | **Pass.** 93.4% of answers differ vanilla vs `pai_full`, 90.6% vs `pai_attn`, 100% sighted vs grey |
| G8 grey constancy | **Pass.** grey pixel tensor bit-identical across every row and image |
| G_masks | **Pass.** `mask_leak = 0`; 6,401 causal repairs fired and 175,680 unmasked decode calls, so the repair is load-bearing |
| G11 determinism | **Pass.** vanilla run twice, 0/288 answers differ |
| G9 frame equivalence | REPORTED. Initially **2.221** max abs pixel difference on 2 of 8 probes — our helper rounded the long side where HF floors it, shifting the centre crop by a pixel. Fixed to floor (Amendment 2.4) and re-measured over 40 images: **max abs 0.030, max mean abs 0.000113**. Touches only the `m336` ladder arms; `sighted`, `grey` and `mismatch` are `own`-frame and were never affected |
| G10 batch invariance | REPORTED. 1 of 4 probe answers differs batched vs singly. Batch composition is identical across arms by construction (the plan depends only on the text, and all jobs report the same plan sha `84ef6b5f…`), so any batch-shape effect is common to every arm and cancels in the paired contrasts |

## 5. A zero-compute result: published POPE tables recoded into the paper's coordinates

On POPE's balanced 1:1 splits the two coordinate systems are exactly interconvertible:

```
  H = Recall      Accuracy = (H + 1 - F)/2   =>   F = Recall + 1 - 2*Accuracy
  J = 2*Accuracy - 1     Precision = H/(H+F)      F1 = 2H/(1 + H + F)
```

so **any published POPE table can be recoded into (H, F, J, d', c) with no compute at all.**
The scorer asserts these identities on every full-sample cell at `1e-9`; calibrated on 604
synthetic cells including all four degenerate corners, worst deviation **4.44e-16**.

**Positive control on the inversion itself.** Where a paper publishes recall *and* F1, the
`(Acc, F1) -> H` inversion must return the published recall. Across six rows from two
independent papers (VCD Table 1 x3, POPE Table 3 x3) the worst deviation is **0.047 accuracy
points**, which is their 2-decimal rounding.

### 5.1 PAI's own published POPE gain, decomposed

From PAI Table 2 (single-turn, greedy, LLaVA-1.5-7B, 3-split average):

| arm | Acc | F1 | H | F | J | d' | c |
|---|---|---|---|---|---|---|---|
| vanilla | 84.76 | 85.51 | 0.8994 | 0.2042 | +0.6952 | +2.105 | −0.226 |
| PAI | 85.82 | 85.97 | 0.8689 | 0.1525 | +0.7164 | +2.147 | −0.048 |
| **change** | **+1.06** | **+0.46** | **−0.0305** | **−0.0517** | **+0.0212** | **+0.042** | **+0.178** |

From PAI supplementary Table S3: change = H −0.0225, F −0.0499, J +0.0274, **d' +0.064,
c +0.156**.

For comparison, the paper's own CHAIR decomposition of the same method (`tab:pai`):
**d' +0.02, c +0.24**.

**The signature is the same on both benchmarks, and on POPE it comes from PAI's own published
table.** The gain is dominated by criterion movement, with a small separation component;
PAI names fewer present objects *and* fewer absent ones. This is a derived recoding, not a
measurement: it carries no interval, it inherits the equal-variance Gaussian assumption for
`d'`/`c`, and because F1 is non-linear the inversion from a **3-split average** is approximate
(accuracy averages exactly over equal-size splits; F1 does not). It is reported as a
cross-check on the lane's own measurement, not in place of it.

### 5.2 A published instance of the degenerate regime, in POPE's founding paper

Recoding POPE's own Table 3 (MSCOCO, exact — recall is published) shows three of its five
models sitting in or beside the constant-"yes" regime:

| model | split | Acc | H | F | J |
|---|---|---|---|---|---|
| MultiModal-GPT | random | 50.03 | 1.0000 | 0.9994 | **+0.0006** |
| mPLUG-Owl | popular | 50.63 | 0.9927 | 0.9801 | +0.0126 |
| LLaVA (v0) | adversarial | 50.77 | 0.9987 | 0.9833 | +0.0154 |
| InstructBLIP | random | 88.73 | 0.9393 | 0.1647 | +0.7746 |
| MiniGPT-4 | random | 77.83 | 0.8267 | 0.2701 | +0.5566 |

MultiModal-GPT answers "yes" to essentially every question in every split; its accuracy of
50.03 is arithmetic, not behaviour. The same regime is reported for a *blinded* model by Lan
et al. (arXiv:2605.22903) — LLaVA-1.5-7B blinded with a flat black image at Accuracy 0.50,
Precision 0.00, Recall 0.00, an always-"No" responder — whose main text reads the 0.50 as
evidence that models "are not independent of visual input". Both corners give
**J = 0 and accuracy 0.50 from opposite behaviour**, and only the answer distribution
distinguishes them. That is the motivating case for reporting `H` and `F` rather than a single
score, and it is why this lane's degeneracy screen (`yes_ratio >= 0.95` or `<= 0.05`, fixed
before any data existed) is blocking rather than advisory.

Note their ablation is flat **black** and ours is flat **grey** plus an in-distribution
mismatched real image, so their number is a motivating example, not a control target.

---

## 6. The positive control (gate G_PC) — PASS

Read as soon as the `vanilla / sighted` cell landed and before any contrast was computed, as
the registration requires. All 9000 items, lenient (official POPE) parser.

| split | acc | prec | rec | F1 | yes-ratio | H | F | J | d' | c |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 89.27 | 90.07 | 88.27 | 89.16 | 0.490 | 0.8827 | 0.0973 | +0.7853 | +2.482 | +0.054 |
| popular | 86.10 | 84.60 | 88.27 | 86.39 | 0.522 | 0.8827 | 0.1607 | +0.7220 | +2.178 | −0.098 |
| adversarial | 79.37 | 74.93 | 88.27 | 81.05 | 0.589 | 0.8827 | 0.2953 | +0.5873 | +1.725 | −0.325 |
| **pooled** | **84.91** | 82.72 | 88.27 | **85.40** | 0.534 | 0.8827 | 0.1844 | +0.6982 | +2.086 | −0.145 |

- **PC-A PASS.** Target, PAI Table 2 single-turn greedy vanilla LLaVA-1.5-7B, 3-split average:
  **accuracy 84.76, F1 85.51**. Measured: **84.91 / 85.40** — **+0.15 accuracy points and
  −0.11 F1**, inside a registered band of ±2.5 and tighter than the 0.36-point spread between
  the two same-protocol published sources. The `llava-hf` checkpoint port, the hand-written
  two-stream loop, the registered attention kernel, bf16 and a 64-token budget together move
  the published number by about a tenth of a point.
- **PC-A2 PASS.** random 89.27 ≥ popular 86.10 ≥ adversarial 79.37, the ordering the splits are
  constructed to produce. Note `H` is identical (0.8827) across all three splits, as it must be
  — the splits differ only in which *absent* objects are sampled — and the whole split effect
  lands in `F`: 0.097 → 0.161 → 0.295. That is the frequency-prior effect `sec:blind` describes,
  visible directly in the decomposition and invisible in accuracy alone.
- **Identity cross-check:** worst deviation between the POPE table and the decomposition,
  **2.8e-17**.
- **Degeneracy screens, vanilla/sighted:** yes-ratio 0.534, unparseable 0.000, truncation
  0.0000, 4594 distinct answer strings. Not degenerate.
- Corroboration of §5.1: recoding PAI's *published* vanilla row gave H 0.8994, F 0.2042,
  J +0.6952, d' +2.105, c −0.226; we *measure* H 0.8827, F 0.1844, J +0.6982, d' +2.086,
  c −0.145. The recoding and the measurement agree to within the approximation §5.1 declares.

## 7. Results — Tier A (complete)

### 7.1 Headline: grey makes the ground truth false, and the registered trap caught what that does to the score

**Under a flat grey image, LLaVA-1.5-7B answers "No" to all 9,000 POPE questions.**

| arm | acc | F1 | yes-ratio | distinct answers | median tokens | empty | H | F | J | d' | c |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `vanilla / sighted` | 84.91 | 85.40 | 0.534 | 4594 | 23 | 0 | 0.8827 | 0.1844 | +0.6982 | +2.086 | −0.145 |
| `pai_full / sighted` | 84.96 | 85.50 | 0.538 | 1656 | 12 | 0 | 0.8873 | 0.1882 | +0.6991 | +2.096 | −0.164 |
| `vanilla / grey` | **50.00** | n/a (0/0) | **0.000** | 79 | 22 | 0 | 0.0000 | 0.0000 | **+0.0000** | — | — |
| `pai_full / grey` | **50.00** | n/a (0/0) | **0.000** | 79 | 10 | 0 | 0.0000 | 0.0000 | **+0.0000** | — | — |

**This is not a collapsed model, and it is not a parsing artifact.** Verified directly from the
raw `answer` field, independently of the scorer: yes-rate 0.0000 over 9,000 grey rows against
0.5336 sighted, **0 empty answers**, median 22 tokens. The blinded model is fluent, and
**100.0% of its 9,000 answers explicitly describe the grey field itself**. Its modal replies:

> "No, there is no person in the image. It is a grayscale photo of a sky background." (1340x)
> "No, there is no car in the image. The image features a gray sky and a gray background." (902x)

**The model is right.** Under a flat grey field there is no person and no car. The label says
`yes` only because the *original photograph* contained one. **Grey does not merely remove
information from an existence question — it makes the ground truth false**, because the labels
describe a stimulus the model was not shown. `H = F = 0` then follows by construction, on all
4,500 `label = yes` items as well as the 4,500 `label = no` ones.

*(Guard added on the strength of this: these records have no `text` key at all, so any
downstream `row.get("text", "")` would return `""` for every row in **both** arms and drive the
yes-rate to 0 everywhere — indistinguishable from the result above. The scorer now asserts the
presence of `answer`, `label`, `method`, `pixarm`, `split`, `question_id`, `image_id` and
`pid_sha` and asserts no answer is empty, rather than defaulting any of them.)*

**The registered trap fired.** §9 named this exact failure before any data existed: with both
grey arms constant, `J = 0` in both, so `Delta_ablated` is **mechanically** zero and
`Delta_image` collapses onto `Delta_sighted` — which would have read as **IMAGE-ATTRIBUTABLE**
while measuring nothing whatsoever. The scorer returns what the registration requires:

> **PRIMARY (grey) = `BLIND-ARM-DEGENERATE`. `Delta_image` is UNDEFINED — not a null, not an
> attribution, and no number for it is reported here.**

Had the trap not been named in advance, this data set would have produced a confident,
clean-looking and entirely spurious headline.

**The same regime, published twice over.** Lan et al. (arXiv:2605.22903) report blinded
LLaVA-1.5-7B at Accuracy 0.50 / Precision 0.00 / Recall 0.00 with a flat **black** image, and
read the 0.50 in their main text as evidence that models "are not independent of visual input".
We reproduce the identical always-"No" signature with flat **grey**. And POPE's own Table 3 has
the mirror-image corner: MultiModal-GPT answers "yes" to essentially everything (H 1.0000,
F 0.9994, accuracy 50.03). **Both corners give accuracy 0.50 and J = 0 from opposite
behaviour**, and only the answer distribution tells them apart.

**What this does to the paper — and what it does not.** It is *not* "POPE cannot be blinded".
Whether a blind control is informative is a property of the **ablation**, not of the benchmark:
the same flat grey that empties CHAIR's model-chosen denominator (`app:mitig-blind`) also
falsifies POPE's externally-supplied labels, while the seven milder degradations of the ladder
do neither. That is one principle covering both endpoints, and it is the empirical backing for
the two corrections now in the paper — that recommendation 2 must say to *check the answers are
not degenerate before reading the result*, and that supplied candidates make the check
**runnable, not automatically informative**. `sec:reco` item 2 as originally written ("POPE-style
existence questions allow it directly") was too strong, and this lane is why.

**The principled repair is the in-distribution ablation**, which is exactly why Amendment 1
added it: a real image from another scene keeps objects genuinely present and genuinely absent,
so the responder can stay non-degenerate while the label–stimulus correspondence is broken in a
controlled way. Under the registered Amendment 1 contingency the mismatch arm now **carries the
primary**. §7.5 reports whether it escapes the degeneracy screen.

**One thing POPE's balance does buy.** In `app:mitig-blind` the log-linear correction
manufactured a spurious `d' = +0.781` on the blind CHAIR arms, because the present and absent
populations had very different sizes. Here they are exactly equal (4500/4500), so the same
correction returns exactly `d' = 0.000` and `c = +3.692`. The balanced design protects `d'` from
the artefact that bit CHAIR. Those values remain correction arithmetic, not measurement.

### 7.2 The mismatched-real-image ablation escapes degeneracy and rescues the control

Amendment 1's in-distribution ablation works. Pooled over 9,000 questions, 500 images,
seeded derangement with no image paired with itself:

| arm | alpha | gamma | Eq.3 rows | acc | F1 | yes-ratio | distinct ans | med tok | H | F | J | d' | c | screen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vanilla / sighted` | 0.0 | 1.0 | — | 84.91 | 85.40 | 0.5336 | 4594 | 23 | 0.8827 | 0.1844 | +0.6982 | +2.086 | −0.145 | clean |
| `vanilla / mismatch` | 0.0 | 1.0 | — | 52.02 | 38.05 | 0.2744 | 4812 | 26 | 0.2947 | 0.2542 | **+0.0404** | +0.121 | +0.600 | **clean** |
| `vanilla / grey` | 0.0 | 1.0 | — | 50.00 | n/a | 0.0000 | 79 | 22 | 0.0000 | 0.0000 | +0.0000 | — | — | **DEGENERATE-CONSTANT-NO** |
| `pai_full / sighted` | 0.5 | 1.1 | decode | 84.96 | 85.50 | 0.5378 | 1656 | 12 | 0.8873 | 0.1882 | +0.6991 | +2.096 | −0.164 | clean |
| `pai_full / mismatch` | 0.5 | 1.1 | decode | 52.06 | 38.44 | 0.2788 | 1532 | 11 | 0.2993 | 0.2582 | **+0.0411** | +0.122 | +0.587 | **clean** |
| `pai_full / grey` | 0.5 | 1.1 | decode | 50.00 | n/a | 0.0000 | 79 | 10 | 0.0000 | 0.0000 | +0.0000 | — | — | **DEGENERATE-CONSTANT-NO** |

Every mismatch cell passes the screen: yes-ratio 0.27–0.28, well inside the registered
[0.05, 0.95] band, 0.000 unparseable, 4812 and 1532 distinct answer strings, median 26 and 11
tokens. **`Delta_image` is therefore defined under the mismatch ablation even though it is
UNDEFINED under grey**, which is the registered Amendment 1 contingency doing its job.

Three things follow, and the first two are positive findings, not damage control.

1. **POPE is strongly image-dependent, and the instrument discriminates.** `J` falls from
   **+0.6982 sighted to +0.0404** under a real image of another scene — the endpoint retains
   about **5.8%** of its score. A control that flagged everything would be worthless; this one
   separates an informative arm from an uninformative one by a factor of seventeen.
2. **The blind control is feasible on a standard benchmark, once the ablation stops falsifying
   the labels.** This is the constructive half of §7.1: grey was the wrong ablation for an
   existence question, not POPE the wrong benchmark.
3. **The residual blind signal is a frequency-prior effect, and POPE's own splits expose it.**
   Registered secondary S5 returns **`NOT-BLINDABLE`** for mismatch, and the reason is entirely
   in one split:

   | split | `vanilla / mismatch` accuracy | `d'` |
   |---|---|---|
   | random | **57.77** | **+0.543** |
   | popular | 48.47 | −0.087 |
   | adversarial | 49.83 | −0.010 |

   On **popular** and **adversarial** the blinded model is *at chance* and `d'` is zero. On
   **random** it keeps 7.8 accuracy points and `d' = +0.54` with no information about the
   queried image. Random draws absent objects uniformly, so a model answering from a general
   object-frequency prior gets traction; popular and adversarial draw absent objects that are
   frequent or co-occurring, which is exactly what they were built to defeat (`pope`). We
   report `NOT-BLINDABLE` as registered, and note that the correct reading is narrower and more
   useful: **POPE-random is partly answerable without the image, by prior alone; POPE-popular
   and POPE-adversarial are not.** Anyone blinding POPE should use popular or adversarial.

### 7.3 The released `--use-attn --use-cfg` configuration produces no POPE gain — and the three caveats, now all settled

| quantity | `pai_full` − `vanilla`, sighted, pooled | PAI published (Table 2, recoded) |
|---|---|---|
| accuracy | **+0.05 pts** | +1.06 pts |
| `J` | **+0.00089** [−0.0031, +0.0053] | +0.0212 |
| `H` | +0.0046 | −0.0305 |
| `F` | +0.0038 | −0.0517 |
| `d'` | +0.010 | +0.042 |
| `c` | −0.019 | +0.178 |

`Delta_sighted = +0.00089` in `J`, interval **[−0.0031, +0.0053]**, containing zero and an
order of magnitude below the registered `MDE_J = 0.01`. The registered precondition P0 returns
**`NO-GAIN-TO-DECOMPOSE`** for `pai_full` under both ablations. The positive control rules out a
broken port — vanilla reproduces PAI's published *level* to 0.15 accuracy points. What did not
reproduce is the *gain*. Per §8 (PC-A3) the gain was deliberately never gated, so that this
outcome stayed reportable.

**This is *not* "PAI's POPE gain does not reproduce" — and §7.7 shows why.** Three things had
to be settled before that framing could even be considered. All three are now settled, and the
third settled in the affirmative: the gain *does* reproduce, in a different released
configuration.

- **(i) Hyperparameters — SETTLED, our configuration is the paper's.** Verified by me in the
  arXiv HTML of 2407.21771v1, §5.1 Implementation Details, quoted: *"we set α = 0.5 for LLAVA,
  α = 0.6 for Shikra … and α = 0.2 for resampler models … we continuously use γ = 1.1"*, and
  *"in the beam search tests, the beam number is set to 5"* — so greedy is the basic baseline.
  α = 0.5 and γ = 1.1 are stated globally, not per task, so they are the POPE values too. The
  README's POPE command shows `--alpha 0.2`, which is the **resampler** value carried by a
  `MODEL_NAME` placeholder, not a LLaVA-specific setting; `pope_eval.py`'s own argparse
  defaults are `alpha 0.2, gamma 2`, which neither the paper nor the README uses for LLaVA.
  Layer range [2, 32) matches the repo default and the README. **Our arm is their arm.**
- **(ii) Prompt and parsing — SETTLED, byte-for-byte.** PAI's `constants.py` gives
  `SYSTEM_MESSAGE + "USER: <ImageHere> <question> ASSISTANT:"` concatenated with no separator
  (`pope_eval.py` L88–90), and **appends no format instruction**. Ours is identical. Parsing is
  the official POPE `evaluate.py` mapping verbatim, which is also PAI's own logic.
- **(iii) The discriminating test — SETTLED, and it PASSED.** `pai_attn` (α = 0.5, γ = 1.0,
  Eq. 3 amplified at the last query row of *every* conditional forward, prefill included — the
  released `--use-attn` path) gives **+1.100 accuracy points against PAI's published +1.06**,
  and reproduces the published decomposition in all four coordinates (§7.7). The method axis is
  live and the harness does apply the method. So the `pai_full` null is a property of that
  configuration, not of this port.

**A correction to the mechanism story itself.** The source-verified fact is that under
`--use-attn --use-cfg` Eq. 3 never touches the **first** generated token (G7a: `pai_full` made
14,194 edits, **0** at prefill). It does **not** follow that Eq. 3 cannot reach the answer,
because under PAI's POPE protocol the answer is **not** one token: with no format instruction
the model replies in whole sentences, measured median **22–23 tokens**, and the official parser
scans the entire first sentence. Eq. 3 can therefore influence the parsed answer through later
tokens. §7.5 records this as a pre-registration gloss falsified by the data. The mechanism
reading is consequently *weaker* than "Eq. 3 cannot reach the answer": it is that Eq. 3, applied
from the second token onward, changes the wording heavily and the decision barely.

**The intervention is live and is doing a great deal — to the text, not to the decision.**
`pai_full` changes **93.4%** of answer strings, cuts mean answer length from 23.3 to 13.8
tokens, collapses distinct answer strings from 4594 to 1656, and raises image-attention mass
from 0.164 to 0.306, while moving the yes-ratio by 0.004.

### 7.4 The registered verdict, reported as the rule produced it

| readout | arm | outcome |
|---|---|---|
| PRIMARY, grey ablation | `pai_full` | **`BLIND-ARM-DEGENERATE`** — `Delta_image` UNDEFINED |
| CO-PRIMARY, mismatch ablation | `pai_full` | **`NO-GAIN-TO-DECOMPOSE`** |
| combined (Amendment 1 rule) | `pai_full` | **`ABLATION-DISAGREEMENT`** |
| S1, mismatch ablation | **`pai_attn`** (released, reproduces the published gain) | **`PARTIAL`** — `Delta_image(J)` +0.0207 [+0.0042, +0.0371] |
| S1, grey ablation | `pai_attn` | `BLIND-ARM-DEGENERATE` |
| S1, mismatch ablation | `pai_both` (CONSTRUCTED) | **`PARTIAL`** — `Delta_image(J)` +0.0238 [+0.0078, +0.0398] |
| S2 per split, `pai_attn` / mismatch | — | random `NO-GAIN-TO-DECOMPOSE`, popular `CANNOT-RESOLVE`, adversarial **`PARTIAL`** |
| S5 blindability, grey | `vanilla` | `BLIND-ARM-DEGENERATE` |
| S5 blindability, mismatch | `vanilla` | `NOT-BLINDABLE` (driven entirely by the random split; see §7.2) |

**The combined label is a defect in the combiner, and is reported rather than repaired.**
Amendment 1's table routes `grey = BLIND-ARM-DEGENERATE` to the mismatch arm *only when the
mismatch arm returns an attribution label*. It returned `NO-GAIN-TO-DECOMPOSE`, which is not
one, so the rule falls through to `ABLATION-DISAGREEMENT`. But the two arms are not
disagreeing about attribution — **they both decline to attribute**, for different and
compatible reasons: grey because the ablation falsifies the labels, mismatch because there is
no gain to attribute. The substantive conclusion is `NO-GAIN-TO-DECOMPOSE`, with the grey
control additionally undefined. The registered label is stated first because that is what the
pre-registered rule produced; the under-specification is logged here as a second registration
defect caught by the data, alongside §7.5.

**So: is PAI's POPE gain image-attributable?** **Yes — see §0 and §7.8.** Under the released
configuration that reproduces the published number (`pai_attn`), `Delta_image(J)` is
**+0.0207 [+0.0042, +0.0371]**, excluding zero, with only ~6% of the gain surviving the
mismatched-real-image ablation. The registered primary arm `pai_full` returns
`NO-GAIN-TO-DECOMPOSE` because that configuration produces no POPE gain at all in this port;
the attribution question is answered by the registered secondary S1, whose decision rule was
fixed in advance and identical to the primary's.

### 7.5 A correction to a pre-registration gloss, forced by the data

§4.2 of the registration states: "On POPE the answer *is* the first generated token. So
`pai_full` minus `vanilla` measures **Eq. 4 alone**." **The first clause is false under PAI's
own POPE protocol, and therefore so is the second.** Measured median answer length is 22–23
tokens, because PAI's protocol appends no one-word instruction, and the official parser reads
the whole first sentence.

- The **arms are unaffected.** They are defined by `(alpha, gamma, rowmode)`, all verified live
  and correctly configured by G7a. Nothing generated is wrong.
- The **primary estimand and every decision rule are unaffected.** They never referred to stages.
- What is wrong is the *interpretive label*: `pai_full` − `vanilla` is **Eq. 3 on conditional
  decode steps plus Eq. 4**, not Eq. 4 alone, and **no registered arm isolates Eq. 4**. Recorded,
  not silently repaired.

### 7.6 Gate and control status

| check | result |
|---|---|
| PC-A published level | **PASS** — 84.91 / 85.40 against PAI's 84.76 / 85.51, band ±2.5 |
| PC-A2 split ordering | **PASS** — 89.27 ≥ 86.10 ≥ 79.37 |
| PC-B sighted − grey accuracy ≥ 10 pts | **PASS** — pooled +34.91, all intervals excluding 0. **Uninformative**: the comparison arm is a constant responder, so it passes trivially. PC-A2 and the mismatch contrast are the informative controls |
| PC-C sighted − grey `d'` | **PASS** — pooled +2.086, but grey's `d'` is correction arithmetic, not measurement |
| sighted − mismatch `J` (informative image-use check) | **+0.658**, from +0.6982 to +0.0404 |
| identity cross-check | **PASS** — worst deviation 2.2e-16 over 24 balanced cells |
| field-name guard | added after the grey result: `answer`, `label`, `method`, `pixarm`, `split`, `question_id`, `image_id`, `pid_sha` are asserted present and answers asserted non-empty, because these records have **no `text` key** and a `.get("text", "")` read would drive the yes-rate to 0 in every arm — indistinguishable from the §7.1 result |

### 7.7 The discriminating test PASSED: the gain reproduces, and it lives entirely in the prefill row

`pai_attn` — §7.3(iii)'s positive control for the whole method axis — **moves POPE, and by
almost exactly the published amount.** All pooled, 9,000 items, sighted.

| arm | alpha | gamma | Eq.3 rows | acc | ΔAcc vs vanilla | ΔH | ΔF | Δd' | Δc | yes-ratio | med tok | distinct ans | img-attn mass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vanilla` | 0.0 | 1.0 | — | 84.91 | — | — | — | — | — | 0.5336 | 23 | 4594 | 0.171 |
| `pai_full` | 0.5 | 1.1 | decode only | 84.96 | **+0.044** | +0.0047 | +0.0038 | +0.010 | −0.019 | 0.5378 | 12 | 1656 | 0.306 |
| `pai_attn` | 0.5 | 1.0 | **incl. prefill** | 86.01 | **+1.100** | −0.0247 | −0.0467 | +0.075 | +0.154 | 0.4979 | 11 | 842 | 0.296 |
| `pai_both` (CONSTRUCTED) | 0.5 | 1.1 | **incl. prefill** | 86.07 | **+1.156** | −0.0213 | −0.0444 | +0.080 | +0.142 | 0.5007 | 11 | 1116 | 0.302 |
| *PAI published, recoded* | 0.5 | 1.1 | — | 85.82 | *+1.06* | *−0.0305* | *−0.0517* | *+0.042* | *+0.178* | — | — | — | — |

**`pai_attn` reproduces PAI's published POPE gain to 0.04 accuracy points (+1.100 against
+1.06), and reproduces its decomposition in all four coordinates** — `H` and `F` both fall,
`d'` barely moves, `c` rises. That is the same signature the paper finds for PAI on CHAIR
(`tab:pai`: Δd' +0.02, Δc +0.24) and the same signature we recovered in §5.1 from PAI's own
published table with no compute at all.

**So the harness is not the problem, and §7.3's caveat (iii) is discharged.** The method axis
is live and capable of producing the published effect. What does *not* produce it is the
configuration the README's POPE command specifies.

**The gain lives entirely in one row of the attention matrix.** `pai_both` and `pai_full` have
**identical** α = 0.5 and γ = 1.1 and differ *only* in whether Eq. 3 is applied to the last
query row of the prefill pass. That one difference is worth **+1.11 accuracy points**
(86.07 vs 84.96) — the whole effect:

| comparison | isolates | ΔAcc |
|---|---|---|
| `pai_full` − `vanilla` | Eq. 3 on decode steps + Eq. 4 | +0.044 |
| `pai_both` − `pai_full` | **the prefill last row alone** (α, γ held fixed) | **+1.112** |
| `pai_both` − `pai_attn` | Eq. 4 added on top | +0.056 |

Eq. 4 (the CFG logit refine at γ = 1.1) contributes **+0.06 accuracy points**. Eq. 3 restricted
to decode steps contributes essentially nothing. **Amplifying image attention on the single
query row that generates the first answer token contributes all of it.**

**What we may and may not conclude.** §0.1 binds us: this lane cannot say what any released
implementation does at runtime. What we can say, and do:

1. In this port, PAI's POPE gain is produced entirely by Eq. 3 at the prefill last row.
2. Reading the released source, `attention.py` L90 guards that amplification with
   `use_attn and not use_cfg`, `llama_modify` sets `use_cfg = True`, and `CFG.py` clears it
   only inside `CFGLogits.__call__`, which first runs *after* prefill — so the README's POPE
   command, `--use-attn --use-cfg`, suppresses at prefill exactly the operation that produces
   the gain here. The configuration that reproduces their number here is `--use-attn` alone.
3. We do not claim to know which of these produced their published table. We report which
   configuration reproduces it in this harness, and leave the discrepancy on the record.

**None of the three intervened arms is degenerate**: yes-ratios 0.498–0.538, unparseable
0.000, 842–1656 distinct answer strings. The screens are doing their job and are not
suppressing a real effect.

### 7.8 The decisive decomposition — and what it says about the paper's own machinery

All pooled, 9,000 items, 500 image clusters, paired bootstrap B = 4000, seed 20260919.
`Delta_s` = sighted, `Delta_a` = mismatched real image, `Delta_img = Delta_s − Delta_a`.

**`pai_attn` (α = 0.5, γ = 1.0, Eq. 3 incl. prefill — the released `--use-attn` path)**

| quantity | `Delta_s` | `Delta_a` (mismatch) | `Delta_image` |
|---|---|---|---|
| accuracy | +1.100 [+0.533, +1.678] | +0.067 [−0.511, +0.622] | **+1.033 [+0.211, +1.856]** |
| `J` | +0.0220 [+0.0107, +0.0336] | +0.0013 [−0.0102, +0.0124] | **+0.0207 [+0.0042, +0.0371]** |
| `d'` | +0.0747 [+0.0245, +0.1257] | +0.0131 [−0.0218, +0.0477] | +0.0616 [−0.0015, +0.1256] |
| `c` | +0.1543 [+0.1271, +0.1827] | **+0.1073 [+0.0897, +0.1260]** | +0.0470 [+0.0151, +0.0802] |
| `H` | −0.0247 [−0.0340, −0.0160] | −0.0338 [−0.0433, −0.0247] | +0.0091 [−0.0038, +0.0218] |
| `F` | −0.0467 [−0.0549, −0.0384] | −0.0351 [−0.0422, −0.0284] | −0.0116 [−0.0220, −0.0009] |

**`pai_both` (CONSTRUCTED, α = 0.5, γ = 1.1, incl. prefill)** — `Delta_image(J)` +0.0238
[+0.0078, +0.0398]; accuracy +1.189 [+0.389, +1.989]; `rho` −0.029 [−0.584, +0.528];
registered outcome **`PARTIAL`**.

**`pai_full` (released `--use-attn --use-cfg`)** — `Delta_s(J)` +0.0009 [−0.0031, +0.0053],
below `MDE_J`: registered outcome **`NO-GAIN-TO-DECOMPOSE`** under both ablations. Alternate
seed (20260920) reproduces it: `Delta_image` +0.0002 [−0.0056, +0.0060] mismatch, +0.0009
[−0.0029, +0.0051] grey.

#### The decomposition earns its keep here, and this is the part worth putting in the paper

Look at `c` against `J`. PAI's criterion shift is **+0.154**, and **+0.107 of it — about 70% —
happens even when the model is shown the wrong image.** Shifting the criterion is something the
intervention does largely *regardless of image content*. Yet `Delta_image(J)` excludes zero.

The two facts are consistent, and the reason is exactly the paper's Equation 3 argument. Under
the mismatched image the criterion shift drives `H` and `F` down **together** (−0.0338 and
−0.0351), so `J` barely moves: `Delta_a(J)` = +0.0013. Under the real image the same shift
drives `F` down **twice as far as** `H` (−0.0467 against −0.0247), and `J` rises. **A criterion
shift on its own does not change `J`; it changes `J` only when the two rates respond
differently, and responding differently is what requires the image.**

So the honest reading of PAI on POPE is neither "a real grounding gain" nor "a threshold
artefact": it is **a largely image-independent criterion shift whose *effect on the score* is
image-dependent**. Reporting accuracy alone shows only the +1.10. Reporting `J` alone shows
only the +0.022. Only `H`, `F`, `d'` and `c` together, each with its image-attributable part,
show what happened — which is precisely what `sec:reco` item 1 asks for. **The paper's
recommended reporting is validated by the case that cuts against its suspicion.**

`d'` also rises, `Delta_s(d')` = +0.0747 with its interval excluding zero — unlike the CHAIR
result (`tab:pai`: +0.02 [−0.06, +0.09]) where separation did not move. Its image-attributable
part, +0.0616 [−0.0015, +0.1256], sits marginally on zero, so we do not claim a resolved
separation gain.

### 7.9 Per split: the gain is an adversarial-split effect

`pai_attn`, mismatch ablation, `J`:

| split | `Delta_s` | `Delta_a` | `Delta_image` | registered outcome |
|---|---|---|---|---|
| random | +0.0053 [−0.0067, +0.0173] | −0.0133 [−0.0253, −0.0013] | +0.0187 [+0.0007, +0.0360] | **`NO-GAIN-TO-DECOMPOSE`** (no sighted gain to attribute) |
| popular | +0.0233 [+0.0093, +0.0373] | +0.0113 [−0.0033, +0.0260] | +0.0120 [−0.0073, +0.0307] | **`CANNOT-RESOLVE`** |
| adversarial | +0.0373 [+0.0226, +0.0527] | +0.0060 [−0.0080, +0.0200] | **+0.0313 [+0.0106, +0.0520]** | **`PARTIAL`** |

PAI buys **nothing** on the random split and **+3.7 points of `J`** on adversarial, where
almost all of it is image-attributable. That is coherent with §7.2: random is the split a
frequency prior can already answer, so there is little for an image-attention intervention to
add; adversarial is built to defeat the prior, so attending to the image is what helps. It also
means a POPE number averaged over the three splits — which is how PAI reports it — hides a
3-to-1 difference in where the method works.

### 7.10 One principle across both endpoints

Putting this beside the ladder lane's result on the paper's own prefix endpoint
(`Delta_image` +0.16 [+0.11, +0.21] under its in-distribution ablation, ratio 0.67 against
grey's 1.03):

| endpoint | what a flat grey field does | consequence for the blind control |
|---|---|---|
| CHAIR (`app:mitig-blind`) | empties the model-chosen denominator — 0 of 80 categories named in 500 captions | `CHAIR_i` is 0/0, undefined; `CHAIR_s` floored at 0 |
| **POPE (this lane)** | **falsifies the labels** — the labels describe the original photograph, and a grey field genuinely contains none of the queried objects, so constant "no" is correct | constant responder, `H = F = 0`, `J = 0` by construction: `BLIND-ARM-DEGENERATE` |
| the paper's prefix endpoint (ladder lane) | depresses the baseline enough to manufacture a null | grey ratio 1.03 against 0.67 in distribution |

**Three different failure modes of one ablation, on three endpoints — and in every case the
milder or in-distribution ablation avoids it.** Whether a blind control is informative is a
property of the **ablation**, not of the benchmark. This lane is not evidence that POPE cannot
be blinded; under a mismatched real image it blinds cleanly, `J` falling from +0.698 to +0.040
with every degeneracy screen passing.

### 7.11 Limitations, binding whatever the numbers say

- One model, one prompt, one decoding mode, one method at one strength. `pai_both` is a
  configuration PAI does not ship and is labelled CONSTRUCTED throughout.
- `PARTIAL` rather than `IMAGE-ATTRIBUTABLE` is a power statement: at 500 image clusters,
  `Delta_image`'s interval is wide relative to a ~1-point accuracy gain. A larger image sample
  would be needed to certify the fraction; the direction and the exclusion of zero are secure.
- `d'` and `c` assume equal-variance Gaussian evidence at one operating point. One point is not
  an ROC.
- The mismatched real image is in distribution but not information-free in the strict sense: it
  is a COCO image, so it shares COCO's object prior with the target. §7.2 measures what that
  buys — `J` +0.040, and +0.058 accuracy above chance only on the random split.
- We cannot say what any released implementation of PAI does at runtime (§0.1). We report which
  configuration reproduces its published number in this harness.

### 7.12 The degradation ladder (S4), complete for `vanilla` and `pai_full`

Registered secondary S4. Nine pixel arms x 9,000 questions, `vanilla` (21057) and `pai_full`
(21058) both COMPLETED with `POPE_SENTINEL_GEN_OK`. Shown for `vanilla`; `pai_full` tracks it
to within 0.4 accuracy points on every rung.

| rung | acc | yes-ratio | H | F | J | d' | distinct ans | screen |
|---|---|---|---|---|---|---|---|---|
| `sighted` | 84.91 | 0.534 | 0.8827 | 0.1844 | +0.6982 | +2.086 | 4594 | clean |
| `noise25` | 84.24 | 0.510 | 0.8520 | 0.1671 | +0.6849 | +2.010 | 4633 | clean |
| `noise50` | 83.74 | 0.487 | 0.8247 | 0.1498 | +0.6749 | +1.970 | 4657 | clean |
| `blur4` | 81.72 | 0.456 | 0.7733 | 0.1389 | +0.6344 | +1.835 | 4594 | clean |
| `noise100` | 78.14 | 0.433 | 0.7140 | 0.1511 | +0.5629 | +1.596 | 4581 | clean |
| `pshuffle` | 76.39 | **0.593** | **0.8573** | **0.3296** | +0.5278 | +1.509 | 4037 | clean |
| `lowres` | 65.82 | 0.287 | 0.4453 | 0.1289 | +0.3164 | +0.994 | 4586 | clean |
| `blur16` | 59.60 | 0.188 | 0.2836 | 0.0916 | +0.1920 | +0.759 | 4553 | clean |
| **`mismatch`** | **52.02** | **0.274** | 0.2947 | 0.2542 | **+0.0404** | +0.121 | 4812 | **clean** |
| **`grey`** | **50.00** | **0.000** | 0.0000 | 0.0000 | **+0.0000** | +0.000 | **79** | **DEGENERATE-CONSTANT-NO** |

Truncation at the 64-token budget is 0.0000 on every rung except `vanilla/pshuffle`
(2 of 9,000 = 0.0002), far inside the registered 0.02 threshold, so the generation-budget
confound is excluded by measurement on all of them.

**1. Every graded degradation keeps the model answering; grey alone does not.** Yes-ratios run
0.188 to 0.593 and distinct answer strings 4,037 to 4,657 across all eight degraded rungs.
Blur at radius 16 costs 25 accuracy points and two thirds of `d'` and still leaves a
responder. **Grey alone goes to a yes-ratio of 0.000 and 79 distinct strings.** Grey is a
qualitative discontinuity, not the far end of a severity continuum — §7.1 gives the reason: a
corrupted photograph still plausibly *contains* a person, so the label stays arguable; a
uniform field contains nothing, so "no" is correct and the labels stop describing the stimulus.

**2. Severity is not informativeness, and this rung pair proves it.** `blur16` retains
`J = +0.192`; the mismatched **sharp, natural, unmodified** photograph retains `J = +0.040`.
By any pixel metric `blur16` is the far more violent operation, yet it leaves four times more
task-relevant signal. **What matters for a blind control is how much information about *this
image* survives, not how far the pixels moved.**

**3. Patch shuffle dissociates the two rates, which is the decomposition's whole point.**
`pshuffle` permutes the 24x24 grid of ViT patches, destroying global layout while preserving
local texture exactly. `H` barely moves (0.8827 -> **0.8573**) while `F` nearly doubles
(0.1844 -> **0.3296**), and the yes-ratio rises *above* sighted (0.534 -> 0.593). Accuracy
falls 8.5 points and `J` falls 0.17, and a single-score report would say only "the model got
worse". The decomposition says what actually happened: **local features are enough to keep
recognising objects that are present, and global layout is what suppresses claims about
objects that are absent.** No other rung behaves this way — every other degradation lowers `H`
and `F` together.

**4. The registered rung ordering is violated, and is reported rather than re-sorted.** §5's
S4 fixed the sequence `sighted -> noise25 -> blur4 -> noise50 -> lowres -> blur16 -> pshuffle
-> noise100 -> grey` before any data existed. Observed `J` along that sequence is
0.698, 0.685, 0.634, 0.675, 0.316, 0.192, 0.528, 0.563, 0.000 — **non-monotone at three
pairs** (`blur4 -> noise50`, `blur16 -> pshuffle`, `pshuffle -> noise100`). Our a-priori guess
at relative severity was simply wrong, most of all about patch shuffle. The registration
forbids re-sorting after the fact and we do not re-sort. None of findings 1-3 depends on the
ordering.

### 7.13 Still running

`pai_attn`'s ladder (21059) and the final scoring pass (21060), queued on `afterok`. That arm's
ladder is the **first item in the registered cut order** and cannot change the primary, the
co-primary, or anything in §0; 21060 rescores every cell when it lands. Everything above rests
on Tier A (12 cells, 108,000 generations, COMPLETE) plus the 14 completed ladder cells.
