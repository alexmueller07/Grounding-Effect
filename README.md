# When a Blank Is Not a Control

Code, pre-registrations, and results for *When a Blank Is Not a Control: Ablation Choice, Generation
Budget, and the Attribution of Grounding Gains* (Mueller, Lee, Park; ICLR 2027 submission).

Object-hallucination benchmarks score a vision-language model by checking the objects it names
against those annotated in the image, and a better score is read as better visual grounding. The
natural way to check that reading is to remove the image and ask whether the gain survives. This
repository contains everything behind the paper's central finding: **the answer depends on two
choices evaluations rarely report — what replaces the image, and whether generations are
truncated — and the usual pairing, a blank field on truncated generations, hides an image
contribution of about half.**

## Headline results

| claim | evidence |
|---|---|
| A blank on truncated generations manufactures a null | On the registered 500 images a grey field gives an ablated/sighted ratio of 1.03 [0.92, 1.16]. On one 3,500-image population (post hoc): grey 1.01 [0.96, 1.05], mismatched real image 0.68 [0.64, 0.72], grey on cells whose sighted continuations end before the budget 0.71 [0.56, 0.87], both corrections 0.47 [0.33, 0.62] with Δ_image = +0.182 [+0.122, +0.245] |
| On Qwen3-VL-8B the two ablations agree | Mismatched image Δ_image = +0.065 [+0.031, +0.100] against grey's +0.060 [+0.023, +0.096]; its sighted one-scene J is already ≈ 0.05, so grey has no level to destroy (post hoc; `q3_gen_mm.py`, `q3_mm_score.py`) |
| The mismatch estimate is conservative | The substitute image also contains 46–48% of the present objects the passage names; excluding those units raises Δ_image to +0.241 [+0.211, +0.270] |
| Severity is not informativeness | Over nine pixel ablations, Spearman(severity, Δ_image) = +0.10. Four ablations — grey and flat black among them — destroy *more* of the image than the mismatched scene that is the only one to expose a contribution |
| A blank breaks two standard benchmarks outright | Under grey, LLaVA-1.5-7B names none of the 80 COCO categories in 500 captions (CHAIR's denominator is 0/0) and answers "no" to all 9,000 POPE questions — correctly, since a grey field contains none of them |
| Used properly, the control works and methods can pass it | With a mismatched image, PAI's gain is image-attributable on POPE (Δ_image = +0.021 [+0.004, +0.037]) and on CHAIR (+5.2 [+2.9, +7.4] of a +5.6 gain) |
| The claim is bounded, not universal | Across four architectures the blind/sighted ratio is 1.03, 1.09, 1.48 and 0.90; on Qwen3-VL-8B part of the rise does need the image, Δ_image = +0.060 [+0.024, +0.096] |

## Layout

```
paper/      LaTeX source, figures, and the built PDF. `build_both.sh` produces the
            anonymous and named builds and runs the submission gates.
code/       The three experimental lanes plus every module they import, with the
            directory structure preserved (the scripts use absolute sys.path inserts).
              sprint/ladder/   ablation ladder: ten ablations on both endpoints
              sprint/pope/     POPE lane: official splits, blind + mismatch arms
              sprint/qwen3/    Qwen3-VL-8B endpoint replication
              sc1_interv/      shared helpers, plus the uncapped-regime arms:
                               wia_cpw_gen_blind.py (grey), wia_cpw_gen_mm.py
                               (mismatched image) and their scorers
              j5_gates, mitdecomp, paialpha:  shared helpers
data/       synonyms.txt (CHAIR scorer), the three official POPE question files with
            their image manifest, and the sighted baseline generations.
results/    Every scored output as JSON — the numbers in the paper come from these.
prereg/     Pre-registrations with their dated amendments, committed before the
            corresponding data existed.
reports/    Per-lane write-ups, including the gates that passed and the ones that failed.
```

## Reproducing

Analysis is CPU-only and runs from `results/`; you do not need a GPU or the checkpoints to check
any number in the paper. Generation does need a GPU.

```bash
# scored results -> paper numbers (no GPU)
python code/sprint/ladder/code/lad_score.py     # L1 endpoint, ten ablations
python code/sprint/ladder/code/lad_cscore.py    # L2 CHAIR ladder
python code/sprint/pope/code/pope_score.py      # POPE decomposition
python code/sprint/qwen3/code/q3_score.py       # Qwen3-VL-8B endpoint
```

The uncapped-regime scorers (`code/sc1_interv/code/cpw_blind_score.py`, `cpw_mm_score.py`) read the
raw generations, which are listed with their sha256 in `results/regime_blind/BLIND_ROWS_MANIFEST.txt`;
their outputs are `results/regime_blind/cpw_blind_score.json` and `cpw_mm_score.json`.
`cpw_2x2_extra.py` computes the full 2×2 on one population (all 10,500 cells and the 2,089 uncapped
cells), ratio intervals, cap rates of every arm, the substitute-image overlap analysis and a
derangement-reconstruction check; its output is `results/regime_blind/cpw_2x2_extra.json` and it
asserts the three earlier estimates as positive controls. `code/sprint/ladder/code/lad_score_dprime.py` is the ladder scorer
extended with d′ and c on the same bootstrap draw (output `results/ladder/lad_l1_dprime.json`); it
reproduces the registered Δ_primary, Δ_ablated and Δ_image as its positive control.

Each scorer re-derives previously published per-condition rates before reporting anything new and
aborts if they do not reproduce to 5e-3. The ladder scorer additionally re-generates the published
grey arm and asserts it is token-identical (it is, 3000/3000).

To regenerate from scratch, the `.sbatch` wrappers in each lane's `code/` directory are the exact
ones used. They are SLURM-specific and contain absolute paths under `/data/alexmueller`; adapt
`W`, `PY`, and `HF_HOME` for your system.

## What is not in this repository

Raw generations (265 MB of `.jsonl`) are not committed. `results/RAW_GENERATIONS_MANIFEST.txt`
lists every file with its sha256 and line count so a regeneration can be verified byte-for-byte.
The 500 COCO images the POPE lane uses are public; `data/pope_questions/image_manifest.json`
records exactly which ones and their checksums, and `code/sprint/pope/code/pope_fetch_images.py`
re-downloads them.

## Notes for anyone checking the work

- Pre-registrations were committed before the data they govern existed, and the amendments are
  dated. Amendment 3 of the ladder pre-registration changed exactly one verdict, and the report
  says which and why.
- Where an analysis was chosen after seeing data, the paper marks it post hoc.
- Three bugs found during these runs are documented in the lane reports rather than quietly fixed:
  a comma-splitting `sbatch --export` that silently ran a fraction of the intended work, a
  cross-image-distinctness gate that aborted on uniform fields, and a dose script that crashed on
  an ablation which is not a pixel transform.
- Results that went against the paper are reported: the mismatched-image ablation, Qwen3-VL-8B,
  and PAI passing the control on both benchmarks.

## Citation

Bibliographic details will be added when the paper is public.
