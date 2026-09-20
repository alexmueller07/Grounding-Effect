# QWEN3 pre-registration: does the phenomenon exist in a 2025-generation model, and can our harness measure it there?

Lane `qwen3`. Written 2026-09-19.

**Nothing of this lane exists when this file is first committed.** No checkpoint has been
fetched, nothing has been generated, no `gap` has been computed on any family-5 row, and the
harness gate of §3 has never been run. The `git log` for this path is the audit trail.

**Three scoped commits, in this order:**

1. **This commit** — the whole design: cells, arms, prompt rule, the harness gate and *every one
   of its thresholds*, the census rule, the estimand, the decision rule for every outcome, and
   the stop conditions. §1.1 and §8 are open and are marked as open. Committed before the fetch
   job is submitted.
2. **Before the first endpoint arm** — §1.1 (the realised cut, filled by a census that loads no
   model) and §8 (what ran: fetch, `af`, census, harness gate — job ids, sentinels, disk, and
   the gate evidence) are appended and committed. No family-5 endpoint exists at that point.
3. **After the endpoint** — §9 (Resolution) is appended, alongside `SPRINT/QWEN3.md`.

Sections 0–8 are not edited after the first row of the sighted arm is generated. Commits are
scoped (`git commit -- <path>`), never `-A`, never pushed. The paper build directory
(`scratchpad/iclrbuild2`) is not opened or edited by this lane.

---

## 0. The objection this lane exists to answer

### 0.1 The reviewer's objection

The central table rests on `llava-hf/llava-1.5-7b-hf` (2023),
`llava-hf/llava-onevision-qwen2-0.5b-ov-hf` (2024) and `microsoft/kosmos-2-patch14-224` (2023).
A reviewer can say: *the phenomenon may be an artefact of a model generation that is now two
years old.* If a 2025-generation model shows the same pattern, the result stops being a quirk of
old checkpoints and becomes a general evaluation principle. That is the entire purpose of this
lane.

**Fifth model: `Qwen/Qwen3-VL-8B-Instruct`**, revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, bfloat16, one RTX 4090, 17.546 GiB on disk.

### 0.2 The cautionary precedent, and why this lane's first job is the harness

`WIA_FAM3_PREREG.md` ran this same design on `Qwen/Qwen2.5-VL-7B-Instruct` and got a **median
continuation of four tokens in every arm** (`f3_sighted_meta.json`: mean 8.87 / 6.80 tokens,
empty rate 0.097 / 0.107; blind 10.34 / 5.52, empty 0.081 / 0.088), against free-running captions
of 112–256 tokens on the same model and the same prompt. The carry-forward endpoint had almost
nothing to register. The paper reports the row as uninterpretable and says so in
Appendix D (`tab:blind`, dagger footnote).

A reviewer replied, correctly, that a four-token continuation *suggests the prompt or chat
template is wrong for that model — fix the harness rather than exclude the result.* FAM3 cannot
answer that, because the only check it ran on the continuation path was `GATE_PREFIX_BINDS`: that
a prefix **changes** the continuation. A harness that lands the prefill in the wrong turn passes
that check trivially.

**So this lane's first job is to get the harness right for this model and to prove it, before
running any endpoint.** §3 is that proof, and its thresholds are fixed below, before the
checkpoint has been fetched.

### 0.3 What this lane does NOT cross

Qwen3-VL-8B-Instruct shares the dense-visual-token interface of families 1–3 (families 1 and 2:
one token per patch; family 3: one per merged 2×2 block). The Q-Former / learned-query axis was
crossed by family 4 and is not crossed again here. **What is crossed is the model generation and
the post-training recipe, not the connector.** The write-up must say "a 2025-generation model",
not "a fifth architecture family".

### 0.4 What is inherited and not re-opened

Magnitude framing is inherited from `WIA_FAM3_PREREG.md` §0.4: the four existing rows already
disagree on magnitude (`+0.4859`, `+0.3780`, `+0.1598`, `+0.2148`). Every decision cell below
fires on the **sign, the power and the interval of `Δ_image`**, never on proximity to another
model's point estimate.

---

## 1. The per-cell budget and the cut, fixed by a criterion stated before the census

The cut is **not copied from another model's realised number**. It is chosen by the census rule
that chose `f = 0.230` for Qwen2.5-VL (`WIA_FAM3_PREREG.md` §1) and `f = 0.460` for Kosmos-2
(`WIA_FAM4_PREREG.md` §1), imported unchanged.

Per-cell budget is `b_i = floor(f · L_i)` with `L_i` **this model's own** free-running length
(`q3_af.gen_len`, greedy, `max_new_tokens = 256`).

**Selection rule, fixed here and not changed after the census runs.** The cut is the **largest**
fraction `f` on a **0.005 grid over [0.050, 0.500]** at which **all four** hold:

- **(a)** strictly **fewer than 5%** of `cap1` prefixes have to double (`cycles > 0`);
- **(b)** **no** `cap5` cell suffers inventory shortfall (`assemble_nodouble(...)["shortfall"]`);
- **(c)** the minimum per-cell budget is **≥ 8 tokens**;
- **(d)** the realised prefix-object unit counts clear the §5 floor in **both** arms
  (`n_true ≥ 100` and `n_false ≥ 100` per arm), counted on the census's own prefixes with the
  same `objset()` the scorer uses.

If **no** `f` on the grid satisfies all four, the lane stops at `NO-DATA` and reports which
condition is binding. Ties cannot occur: the grid is totally ordered and "largest" is unique. If
the largest feasible `f` is the grid maximum 0.500, that is the cut and there is no larger point
to falsify; the census says so explicitly rather than asserting against a point that does not
exist.

Screen-then-verify is family 3's, imported unchanged: every grid point is screened with two cheap
per-cell lengths, then the candidate **and the next larger grid point** are checked with the
**real** `build_prefix_ids` and `assemble_nodouble` on all cells. The screen decides nothing; if
it contradicts the exact check the grid is walked downward with the real builders.

The census reports mean budget against family 1's **44.603**, family 2's **50.272**, family 3's
**40.744** and family 4's **36.142**, and the composition census against family 1's, whatever
they are. **Those are reported for context and none of them can move this lane's cut.**

### 1.1 Realised cut — FILLED BY THE CENSUS, BY THE RULE ABOVE, NOT BY CHOICE

**Census job, `vgi1`, CPU, sentinel `Q3_SENTINEL_CENSUS_OK`.** Stimulus-side only: no model
loaded, nothing generated. 500 images, **0 excluded** by the `L_i < 32` rule, leaving
**500 images / 1500 cells**.

**`CUT = 0.205`**, mean per-cell budget **51.960 tokens** (sd 0.484, range [44, 52]).

| `f` | mean budget | `cap1` doubling (exact) | `cap5` shortfall | min budget | verdict |
|---|---|---|---|---|---|
| **0.205** | **51.960** | **69/1500 = 0.04600** | **0** | **44** | **SELECTED** |
| 0.21 | — | 94/1500 = **0.06267** | 0 | 45 | **FAILS**, confirmed with the real builders |

**Section 1(d) expected unit counts** (floor 100): `cap1` n_true **556**, n_false 2482; `cap5`
n_true **1039**, n_false 6536. PASS = True.

**`GATE_SCENE` = PASS.** `S_word(cap1)` 0.20180, `S_word(cap5)` 0.01019, diff
**+0.19162** [+0.1866, +0.1969]. `S_obj` 0.74379 vs 0.09702, diff +0.64678 [+0.6336, +0.6595].

**Census against the prior families, as section 1 requires:**

| per prefix | fam5 one / five | fam1 | fam3 | fam4 |
|---|---|---|---|---|
| mean budget | **51.960** | 44.603 | 40.744 | 36.142 |
| distinct object types \|P\| | **2.025 / 5.050** | 1.907 / 3.920 | 1.906 / 4.111 | 1.850 / 3.691 |
| total object mentions | **6.351 / 6.386** | 4.933 / 4.829 | 4.993 / 5.013 | 4.525 / 4.532 |
| mentions per type | **3.136 / 1.265** | 2.623 / 1.248 | 2.620 / 1.219 | 2.446 / 1.228 |
| arm ratio \|P\|(cap5)/\|P\|(cap1) | **2.493** | 2.056 | 2.157 | 1.995 |

**Scope note, recorded because it is a real difference from families 1-4.** Qwen3-VL's
free-running captions are censored by the 256-token cap on 99.2% of the 500 images
(median 256, mean 255.82). `L_i` therefore measures the cap rather than the model's natural
caption length on most images, and the per-cell budget is close to constant instead of
image-keyed. This does NOT make the contrast a length contrast: the budget is still identical
across all four arms of a cell by construction, and the scorer re-asserts that from the written
JSONL. It does mean the cut is not comparable to the other families' in the same way, and the
write-up must say so rather than presenting 0.205 as the analogue of 0.230 or 0.460.


---

## 2. Cells, arms and construction — families 1–4's construction, imported in place

### 2.1 Cells

500 COCO `val2017` images (`load_coco_sample(chair, n=500)`, `default_rng(20260817)` — the
identical 500 every arm of this paper uses) × derangements `k ∈ {1,2,3}` = **1500 cells**.
Sources `srcs[j] = sids[(pos(i) + k + 100·j) mod 500]`, `j = 0..4`, from
`wia_redun_common.sources`, imported in place, not re-implemented. `i ∉ srcs` and no collisions,
asserted.

**Eligibility.** The inherited J5 floor applies: an image whose own free-running length
`L_i < 32` tokens is excluded from **every** arm, so the four arms always run on an identical
cell set. The excluded count and ids are reported in §1.1 whatever they are.

Per-cell budget `b_i` is **identical across all four arms of that cell**, asserted at generation
(`FATAL_BUDGET_MISMATCH`) and **re-asserted at scoring from the written JSONL**.

### 2.2 The four arms

| arm | image shown | sources | assembly | doubling |
|---|---|---|---|---|
| `cap1` | the target image | 1 other scene (`srcs[0]`) | `j5_common.build_prefix_ids`, seed `i·7 + 1950 + k` | as measured in §1.1 |
| `cap5` | the target image | 5 other scenes (`srcs[0..4]`) | `wia_redun_common.assemble_nodouble`, join seed `i·7 + 3350 + k`, caption permutation seed `i·7 + 3300 + k + 10j` | none, by construction |
| `cap1g` | **flat mid-grey** (127,127,127), the target's own pixel dimensions | *(prefix bit-identical to `cap1`'s)* | | |
| `cap5g` | **flat mid-grey**, the target's own pixel dimensions | *(prefix bit-identical to `cap5`'s)* | | |

The blind set **re-derives** its prefixes from the same builders and seeds rather than copying
the sighted rows, so prefix identity across the pair is an independent check and not a tautology;
the scorer re-asserts it from the written JSONL (`blind_prefix_identical`, must equal the cell
count in both arms).

### 2.3 Prompt, decoding, batch shape

- The prompt is the **processor's own** `apply_chat_template([{user: [image, text]}],
  add_generation_prompt=True)` with the identical question every other model in the paper is
  asked, `"Please describe this image in detail."` **It is never hand-rolled.** The rendered
  string is frozen by the fetch job into `out/q3_template.json` and every later stage
  byte-compares against that frozen string (`FATAL_TEMPLATE_DRIFT`).
- The prefill is spliced as token ids immediately after the processor's prompt ids — i.e.
  immediately after the generation-prompt suffix — so it is the **start of the assistant's own
  answer**. HG-2 asserts that on the realised ids rather than assuming it.
- Decoding is greedy throughout (`do_sample=False`), no sampling parameters of any kind.
  Free-running captions `max_new_tokens = 256`; continuations `384`, with a 192-token
  reconstruction reported alongside.
- **Batch shape is not inert.** `BS_FREE = 4` and `BS_CONT = 4`, fixed here, identical for every
  arm of every stage including the harness gate, and written into every meta file. Continuations
  are sorted by prefix length before batching, so batch composition is shared between the two
  arms of a contrast and between the sighted and blind runs.
- Every length-tied companion tensor the processor emits (`mm_token_type_ids` and any other) is
  **rebuilt** in lockstep with the re-padded `input_ids`; an unrecognised length-tied tensor is
  FATAL. Dropping such a tensor does not raise and silently degrades M-RoPE to 1-D text
  positions — fluent text on wrong visual positions. `attended_pads` must be 0 on every batch.

### 2.4 Lane isolation

`/data/alexmueller/sprint/qwen3/code` holds its **own copies** of `j5_common.py`,
`gen_common.py`, `wia_redun_common.py` and `synonyms.txt`, with the hard-coded
`/data/alexmueller/j5_gates/code` path in the latter two rewritten to the lane directory, and is
first on `PYTHONPATH`. A queued job runs whatever is on disk when it STARTS, so no edit outside
this lane can change a queued job here. md5s of originals and copies go in §8.0.

---

## 3. THE HARNESS GATE — runs before any endpoint arm, and decides whether one is run

All thresholds in this section are fixed by this commit, before the checkpoint exists locally.

### 3.1 Why HG-3 is the decisive test

Decoding is greedy, so **the model's own free-running caption *is* its own continuation from its
own prefix at any cut.** The paper leans on this already (Appendix A.1: "because decoding is
deterministic, the model's own free-running caption *is* its own continuation from its own prefix
at any cut"). Therefore: cut the model's own caption at the census cut, splice the first part
back in as an assistant-turn prefill, and **the model must re-emit the rest of it**.

A harness that lands the prefill outside the assistant turn, that closes the turn, that corrupts
M-RoPE, or that attends a pad **cannot** pass this test. A model that is merely terse on an
out-of-distribution stimulus **can**. That is exactly the discrimination FAM3 lacked.

It is *not* required to be bit-exact. Splicing a prefix computes its KV cache in one forward pass
rather than incrementally, and bf16 greedy near-ties resolve differently under a different batch
shape (`env_replay_gate_batch_shape`). Exact-match count is reported as a diagnostic; the gate is
on length and agreement.

### 3.2 The gates

| gate | what it checks | threshold |
|---|---|---|
| **HG-1 TEMPLATE** | the prompt is byte-identical to the frozen processor-rendered template, and a non-empty generation-prompt suffix exists (`with` minus `without add_generation_prompt`) | exact |
| **HG-2 PREFILL_LANDS** | on the realised spliced token ids: (i) the prefill contains **no** special/control token; (ii) the decoded sequence ends with `generation-prompt suffix + prefill text`, so the prefill sits immediately inside the assistant turn; (iii) every non-image special token's count equals the template's, so the prefill opens and closes no turn | 0 violating rows of 48 |
| **HG-3 SELF_CONT** | self-prefill re-emits the model's own caption remainder | (a) median continuation length ≥ **0.50 ×** median remainder length; (b) median first-16-token agreement ≥ **0.75**; (c) empty rate ≤ **0.10** |
| **HG-4 FREE_LEN** | free-running captions are of sane length | median over all 500 ≥ **32** tokens |
| **HG-5 ASSEMBLED** | on the **real** `cap1`/`cap5` stimulus at the census cut, in **both** arms | (a) median continuation length ≥ **12** tokens; (b) ≤3-token rate ≤ **0.30** |
| **HG-6 PREFIX_BINDS** | the assembled prefill changes the continuation relative to free running | ≥ n−1 of n rows |

HG-2, HG-5 and HG-6 run on 24 images × {`cap1`,`cap5`} = 48 rows at `k = 1`; HG-3 on the same 24
images; HG-4 on all 500.

### 3.3 Calibration — the gate is run on the paper's own lead model in the same job

`feedback_calibrate_gates_against_own_corpus`: a gate that rejects our own headline cannot test
whether it reproduces.

- **HG-3 is run unchanged on `llava-hf/llava-1.5-7b-hf`**, at LLaVA-1.5-7B's own registered
  primary cut `f = 0.5`, on the same 24 images, at the same `BS_CONT = 4`, in this same job
  (loaded and freed first, so only one checkpoint is resident on the 24 GB card at a time).
  **If LLaVA-1.5-7B fails HG-3, the verdict is `GATE-INVALID` and this lane reports nothing
  until the thresholds are re-derived.** That is a registered outcome, not an escape hatch: it
  would mean the thresholds are wrong, and it would have to be recorded as such.
- **HG-5's thresholds are calibrated against the published per-arm screens of families 1 and 4.**
  The shortest median continuation in any arm the paper interprets is **16** (LLaVA-1.5-7B
  `cap1g`, `at_blind1_meta.json`) and the highest ≤3-token rate is **0.209** (Kosmos-2 `cap5g`,
  `f4_blind_meta.json`). Thresholds of median ≥ 12 and ≤3-rate ≤ 0.30 therefore admit every arm
  the paper currently interprets, and reject FAM3's Qwen2.5-VL arms (medians 4–5).
- **HG-4's threshold is the inherited `MIN_LEN = 32` eligibility floor**, which every prior
  family clears (family 1's observed minimum free-running length 55, family 4's median 76).

### 3.4 Decision rule on the gate — fixed here

| gate outcome | verdict | what happens |
|---|---|---|
| LLaVA-1.5-7B fails HG-3 | `GATE-INVALID` | **Stop.** The gate is wrong, not the model. Report that and re-derive thresholds. |
| HG-1 or HG-2 fails | `HARNESS-BROKEN` | **Stop. No endpoint.** The prompt or the prefill placement is wrong for this model; fix it and re-run the gate. |
| HG-3 fails (with HG-1, HG-2 passing and the calibration passing) | `HARNESS-OR-MODEL-CANNOT-CONTINUE` | **Stop. No endpoint.** Report that the model does not re-emit its own caption remainder from its own prefix under a harness that is otherwise proven correct and that LLaVA-1.5-7B passes. Report as evidence: the HG-3 table for both models, the HG-2 example spliced tail, and the free-running length distribution. **This is the outcome in which we must NOT call the model image-inert.** |
| HG-3 passes, HG-4/HG-5/HG-6 fails | `RUN-ENDPOINT-DEGENERACY-ATTRIBUTED` | The harness continues a pre-filled answer correctly on this model, so a short continuation on the assembled stimulus is a property of the **stimulus or the model**, not of our prompt. **The endpoint IS run** and the row is reported as degenerate-and-attributable with the gate evidence. |
| all pass | `RUN-ENDPOINT` | Run the endpoint. |

The generator refuses to start on any verdict other than the last two
(`FATAL_GATE_VERDICT`).

---

## 4. The estimand

The scored unit is **(image, object type named in the passage)**. Carry-forward is the event that
the model's continuation names that type. `H` and `F` are carry-forward rates over units whose
type is and is not in the image's COCO ground-truth set. `J = H − F`.

- `Δ_primary = J(cap5) − J(cap1)` — sighted.
- `Δ_blind = J(cap5g) − J(cap1g)` — grey.
- **`Δ_image = Δ_primary − Δ_blind`** — the primary estimand of this lane. (The paper calls this
  `Δ_sight`; the two are the same quantity.)

Intervals are **95% paired bootstraps clustered on the image**, `B = 4000`, seed `20260919`,
resample size asserted per replicate and duplicate-presence asserted on the first 20 replicates.
`set()` is never applied to a resample. `Δ_image` uses **one joint image resample driving all
four arms**, so it is a paired quantity across the two generation jobs and not a difference of
two independent intervals.

Reported alongside, for comparability with the published table: `Δ_primary` and `Δ_blind` raw and
length-matched, the four per-condition `H` and `F`, the image's contribution to `J` within each
condition, an object-type-matched contrast, a both-uncapped restriction, an
empty-continuations-removed restriction, a 192-token reconstruction, and an alternate-seed
bootstrap.

**Power floors** (inherited): a contrast is `UNDERPOWERED` if either arm has fewer than 100 true
or 100 false units. §1(d) makes the census refuse a cut that would not clear this.

**Estimator positive control, before any endpoint is read** (SPRINT_CONVENTIONS "Positive
controls"): the scorer recomputes, on the **frozen rows of families 1 and 3**, their published
`Δ_primary` of **+0.4859** and **+0.1598**. If either deviates by more than `5e-3` the lane
reports `NO-DATA` and stops before touching a family-5 row.

---

## 5. Degeneracy screen — reported for every arm, whatever it says

Per arm: **median and mean continuation length, truncation rate at the 384-token budget, empty
rate, ≤3-token rate, and unique-token ratio** (distinct continuation token ids / continuation
length, mean and median over rows with ≥1 token). A collapsed arm is not a null.

The screen's decision thresholds are HG-5's, reused: an arm **fails** the screen if its median
continuation length is below **12 tokens** or its ≤3-token rate exceeds **0.30**.

---

## 6. Decision rule for every outcome, including the ones that hurt

Read on `Δ_image` and its 95% interval, against the paper's `±0.10` reference margin. The
degeneracy screen is read **first**: a failing arm overrides every cell below.

| # | outcome | cell | what we write |
|---|---|---|---|
| 1 | any arm fails the degeneracy screen | `UNINTERPRETABLE-DEGENERATE` | The row is uninterpretable. **And this time we say whether it is the model or our harness**, from §3: verdict `HARNESS-BROKEN` ⇒ ours, and it is fixed rather than reported; verdict `RUN-ENDPOINT-DEGENERACY-ATTRIBUTED` ⇒ the model's, stated with HG-2's spliced-tail evidence and HG-3's self-continuation table for this model and for LLaVA-1.5-7B. The paper's Qwen2.5-VL footnote is amended to say which of the two FAM3's row was, if this lane's evidence settles it. |
| 2 | `Δ_image` interval inside `±0.10` **and** `Δ_blind` large and positive | `REPLICATES-NULL` | **The outcome we want, and we say so.** The paper's pattern reproduces on a 2025-generation model: the rise does not need the image. The central table gains a current-generation row and the claim becomes a general evaluation principle rather than a property of 2023–2024 checkpoints. |
| 3 | `Δ_image` interval excludes zero and is **positive** | `IMAGE-ATTRIBUTABLE` | **This weakens the paper and is reported as a bound in the main text, exactly as Kosmos-2 is.** On a current model the rise *is* image-attributable, so the claim "the rise does not need the image" is bounded to the architectures where `Δ_image` is null and must be stated with that bound in Section 4.2, not in an appendix. The abstract must not claim the null holds generally. |
| 4 | `Δ_image` interval excludes zero and is **negative** | `NEGATIVE-DELTA-IMAGE` | Removing the image makes the rise *larger*, as on Kosmos-2. Reported as a second instance of the Kosmos-2 direction; it does not rescue the null, and the main text says two of five models have a non-null image-attributable component. |
| 5 | `Δ_image` contains zero but the interval is wider than `±0.10` | `NULL-BUT-IMPRECISE` | The row does not resolve the question either way, and is reported as a non-answer rather than as a replication. It does not enter the central table as support. |
| 6 | census finds no feasible cut, or the estimator positive control fails | `NO-DATA` | Reported with the binding condition; no row. |

A lane whose only reportable outcome is favourable is not registered: cells 1, 3, 4, 5 and 6 all
weaken or fail to help the paper, and each has its text written above.

---

## 7. Stop conditions

- Estimator positive control on families 1 and 3 misses by more than `5e-3` → `NO-DATA`, stop.
- No feasible cut on the grid → `NO-DATA`, stop, naming the binding condition.
- `GATE_SCENE` fails (the two arms do not differ in scene count on the frozen prefixes) → stop.
- Harness gate verdict `GATE-INVALID`, `HARNESS-BROKEN` or
  `HARNESS-OR-MODEL-CANNOT-CONTINUE` → no endpoint (§3.4).
- Disk: the fetch job refuses to start if the download would leave `/` below **25 GB** free, and
  re-asserts the floor afterwards. `df -h /` is reported before and after.
- Cluster etiquette: at most **2** concurrent GPU jobs from this lane; every job pins
  `--gres=gpu:rtx_4090:1 --nodelist=vgi1` and passes an explicit `--time`; every payload ends
  `|| exit 9`; success is verified by output file + sentinel string, never by SLURM state.

---

## 8. What ran

### 8.0 What ran before the first endpoint arm

Environment, and a deviation from the brief. The brief specified the base interpreter
`/data/alexmueller/miniconda3/bin/python` (torch 2.13.0+cu130, transformers 5.14.1). **That
interpreter has neither PIL nor torchvision**, so `AutoProcessor.from_pretrained` for any
Qwen-VL checkpoint raises `ImportError` before a single image is opened; job 21033 died on
exactly that after the download had already succeeded. This lane therefore runs
`/data/alexmueller/miniconda3/envs/reason/bin/python` — torch 2.11.0+cu130, **transformers
5.14.1, the same version as base** — which is the interpreter families 3 and 4 were run under.
Recorded as a deviation rather than silently taken.

| stage | job | state | sentinel |
|---|---|---|---|
| fetch (1st attempt) | 21030 | FAILED 9:0 | none — `HF_HUB_OFFLINE` is inherited into batch jobs and the online wrapper did not unset it |
| fetch (2nd attempt) | 21033 | FAILED 9:0 | none — download of all 16.341 GiB completed, then the base interpreter's missing PIL/torchvision |
| fetch | 21037 | COMPLETED | `Q3_SENTINEL_FETCH_OK` |
| plumbing smoke (8 images, no endpoint) | 21038 | COMPLETED | `Q3_SENTINEL_AF_OK` |
| load-path equivalence | 21073 | COMPLETED | `Q3_SENTINEL_LOADEQ_OK` |
| free-running captions `af` | 21074 | COMPLETED | `Q3_SENTINEL_AF_OK` |
| census | 21075 | COMPLETED | `Q3_SENTINEL_CENSUS_OK` |
| harness gate | 21076 | COMPLETED | `Q3_SENTINEL_GATE_OK` |

**Disk.** `df -h /` before the download: **65.27 GB free**. Repo 16.341 GiB in 4 shards.
After: **65.27 GB free**. The fetch refuses to start if the download would leave `/`
below 25 GB and re-asserts the floor afterwards.

**Frozen template** (the processor's own `apply_chat_template(..., add_generation_prompt=True)`,
never hand-rolled), byte-compared by every later stage:

```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Please describe this image in detail.<|im_end|>
<|im_start|>assistant

```

Note it carries **no system turn**, unlike Qwen2.5-VL's template, which does.

**Model identity.** `Qwen3VLForConditionalGeneration`, 8,767,123,696 parameters, torch.bfloat16,
revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, processor `Qwen3VLProcessor`, tokenizer `Qwen2Tokenizer`.
GPU peak 16.84 GiB of 23.52 GiB — it fits one RTX 4090.

**Load-path change, and the evidence for it.** Host RAM, not GPUs, is the rationed resource on
vgi1 (116 of 122 GB were held by four other lanes while this lane queued). `load_m5` streams the
shards straight to the card instead of materialising 16.34 GiB on the CPU first, which cut the
lane's reservation from 32 GB to 14 GB. The argument that this is placement-only is an argument,
so job 21073 asserted it: the 8 free-running captions of the plumbing smoke, produced by the old
`.to("cuda")` path at the same batch size, were regenerated and came back **byte-identical
8/8**.

**One amendment, made before the gate job started.** HG-2 clause (ii) — "the prefill sits
immediately after the generation-prompt suffix" — is tested in **token-id** space rather than on
the decoded string. Testing it on the string would make it hostage to how the tokenizer
re-inserts whitespace around control tokens, a decode artefact unrelated to where the prefill
landed. The string form is still computed and reported, as a diagnostic, never as the pass
condition.

### 8.1 Harness gate result — `VERDICT = RUN-ENDPOINT`

All harness gates pass.

| gate | result |
|---|---|
| HG-1 TEMPLATE | PASS — generation-prompt suffix `'<|im_start|>assistant\n'` |
| HG-2 PREFILL_LANDS | PASS — of 48 rows: 0 with a special token in the prefill, 0 not adjacent to the generation prompt, 0 with changed turn markers |
| **HG-3 SELF_CONT (Qwen3-VL-8B)** | **PASS** — median continuation 376.5 vs remainder 204.0 (threshold 0.5x), median first-16 agreement 1.000 (threshold 0.75), empty 0.000, exact full match 0/24 |
| **HG-3 calibration (LLaVA-1.5-7B)** | **PASS** — median continuation 60.5 vs remainder 60.5, median agreement 1.000, empty 0.000, exact 16/24 |
| HG-4 FREE_LEN | PASS — median 256, mean 255.82, at cap 0.992 |
| HG-5 ASSEMBLED `cap1` | PASS — median 384.0, <=3-token 0.042, empty 0.000 |
| HG-5 ASSEMBLED `cap5` | PASS — median 384.0, <=3-token 0.000, empty 0.000 |
| HG-6 PREFIX_BINDS | PASS — 48/48 |

**`GATE_ATTENDS` = PASS** on all 500 images: delta-recall
**+0.73828** [+0.7159, +0.7608], real 0.7383 against grey
0.0000, identical-caption rate 0.0000.

### 8.2 Lane isolation md5s

Lane copies (`/data/alexmueller/sprint/qwen3/code`):

```
2aef26cbfc02fc6d00dc2de7aaf41e45  j5_common.py
24b49b0ce6fdc8edb2bb0ab2358c19e0  gen_common.py
1ec512693d9fc723eb6e983614998c10  wia_redun_common.py
fdf03d16beb2507696d1b881c958904d  synonyms.txt
994c8ebc2a27c11f3caee08d5060dceb  q3_common.py
2d163933173a4554f7ab80a9cb2b2756  q3_af.py
7dbdc1f46ac1a0fe5f64fef94ecc2ccc  q3_census.py
74654b7477722c0bb0d52226c08c7a2b  q3_gate.py
8f8b2035c01ec3e1f5bae82de787c51f  q3_gen.py
875b1fbfdd18129e13ae9301beab7e57  q3_score.py
e28f3df0d10b9545c8ed7e23d90af011  q3_loadeq.py
```

Originals the first four were copied from (`gen_common.py` and `wia_redun_common.py` differ from
their originals only in the hard-coded `j5_gates/code` path, rewritten to the lane directory):

```
2aef26cbfc02fc6d00dc2de7aaf41e45  /data/alexmueller/j5_gates/code/j5_common.py
fdf03d16beb2507696d1b881c958904d  /data/alexmueller/j5_gates/code/synonyms.txt
53f8867ae75bb6757413954930f3e50f  /data/alexmueller/sc1_gen/code/gen_common.py
8e9bcb01f4e48c77a5717a9da9225cce  /data/alexmueller/sc1_interv/code/wia_redun_common.py
```


---

## 9. Resolution

*(appended after the endpoint)*
