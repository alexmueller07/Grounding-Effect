# A Blank Is Not a Control

Code, pre-registrations, and results for *A Blank Is Not a Control: Attributing Grounding Gains to
the Image* (ICLR 2027 submission).

Object-hallucination benchmarks score a vision-language model by checking the objects it names
against those annotated in the image, and a better score is read as better visual grounding. The
natural way to check that reading is to remove the image and ask whether the gain survives. This
repository contains everything behind the paper's central finding: **the answer depends on which
ablation you use, and the usual choice — a blank field — is often the wrong one.**

## Headline results

| claim | evidence |
|---|---|
| A blank manufactures a null | On the same endpoint and model, a grey field gives a blind/sighted ratio of 1.03 [0.92, 1.16]; a mismatched real image gives 0.67 [0.57, 0.77], with Δ_image = +0.160 [+0.105, +0.213] |
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
              j5_gates, sc1_interv, mitdecomp, paialpha:  shared helpers
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
