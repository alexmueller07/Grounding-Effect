# ICLR sprint conventions — read before touching the cluster

Deadline: full paper **Friday 25 September 2026**. Today is 19 September.

## Access
- `ssh vgi-server` is the ONLY working alias. **`vgi1` does not resolve as a hostname** — it is the
  SLURM node name. Always `ssh -o BatchMode=yes -o ConnectTimeout=20 vgi-server '<command>'`.
- Quote the whole remote command. On this Mac the shell is zsh, which does **not** word-split
  unquoted variables: never build an ssh command in a variable and run `$S ...`; it fails with
  "command not found".
- No `timeout` binary on the Mac. Use ssh's own `-o ConnectTimeout`.

## Working areas
- Cluster project root: `/data/alexmueller/sc1_interv` (`code/`, `out/`, `logs/`).
- **Give your lane its own code directory**: `/data/alexmueller/sprint/<lane>/code`, and copy in the
  modules you need. A pending job runs whatever is on disk when it STARTS, so editing shared code
  under `sc1_interv/code` silently changes other people's queued jobs.
- Local mirror of this repo: `~/Research/new-paper-search/`. Registrations and result records live
  here as markdown, alongside the existing `WIA_*_PREREG.md` and `CAMPAIGN/*.md`.
- `HF_HOME=/data/alexmueller/hf_cache` is set inside the python scripts, not the environment.
- Python: `/data/alexmueller/miniconda3/bin/python` (base) has torch 2.13.0+cu130, transformers
  5.14.1. Env `reason` has torch 2.11.0. Prefer base unless a model needs otherwise.

## SLURM rules (each of these has cost us a run before)
- Submit with the wrappers in `sc1_interv/code`: `sci_gpu.sbatch`, `sci_cpu.sbatch`.
- **Always pin**: `--gres=gpu:rtx_4090:1 --nodelist=vgi1`. vgi2 has **no `/data` mount** — jobs there
  report COMPLETED and write nothing — and it has a Blackwell card the pinned torch cannot use.
- **Always pass an explicit `--time=`.** Without it a job inherits the partition maximum and never
  backfills.
- **Every payload must end in `|| exit 9`.** The wrappers end in an `echo`, so without this every job
  reports COMPLETED 0:0 however the payload died, and `--dependency=afterok` never blocks.
- Verify by **output file + sentinel string**, never by SLURM state.
- `srun`/`ssh`/`sbatch` inside a `while read` loop swallow the loop's stdin: redirect `</dev/null`.
- Never `rm` a file a running job has open; check `squeue`/`lsof` first, prefer `mv` to a trash dir.

## Shared-resource etiquette (important this week)
- The filesystem `/` (which holds `/data`) is at **100%, ~32 GB free**, shared with other users who
  have jobs running. Run `df -h /` before and after anything that downloads. If you need more than
  ~5 GB, say so in your report rather than filling the disk.
- Another user (jiwoong) has jobs on vgi1 and vgi2. Keep our concurrent GPU jobs to **at most 3**.
- Do not delete anything outside `/data/alexmueller`.

## Scientific conventions (non-negotiable — this paper is about measurement discipline)
- **Pre-register before you generate.** Write `SPRINT/<LANE>_PREREG.md` with: the question, arms,
  the primary estimand, the decision rule for every outcome (including the outcome that hurts us),
  gates that must pass before any endpoint is read, and stop conditions. Commit it (scoped
  `git commit -- <paths>`, never `git add -A`, never push) BEFORE the first generation job.
- State the adverse outcome explicitly: for every lane, what result would weaken the paper, and what
  we will write if it happens. A lane whose only reportable outcome is favourable is not registered.
- Intervals: paired bootstrap **clustered on the image**, B = 4000, resample size asserted per
  replicate. Never use `set()` on a resample (it silently makes it a 63% subsample).
- Positive controls: before reading any new endpoint, reproduce an already-published number from
  this project exactly, and include that check in the gates.
- Degeneracy screens: report mean/median continuation length, truncation rate at the budget, empty
  rate, and unique-token ratio for every arm. A collapsed arm is not a null.
- Report numbers as point estimate + 95% CI. Quote the pre-registered endpoint, not the most
  favourable variant.

## Reporting back
When you finish, write `SPRINT/<LANE>.md` with: what ran (job ids), gates and whether they passed,
the registered outcome, every number with its interval, and what the result does to the paper's
claims — including if it weakens them. Then summarise in your final message. Do not edit the paper
build directory (`scratchpad/iclrbuild2`); the writing pass is separate.
