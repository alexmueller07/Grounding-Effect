"""Mitigation-method CHAIR decomposition -- shared definitions.

Every constant here is pre-registered in CAMPAIGN/MITIGATION_DECOMP_PREREG.md and must not
change after the first evaluation row exists.

QUESTION UNDER TEST. Our own paper (When Grounding Metrics Survive Blinding, main.tex:420-425)
asserts that "a method that raises or lowers object-mention propensity can improve a
present-minus-absent score without improving the model's ability to discriminate
image-supported from unsupported objects", and concedes at main.tex:488-492 that this is an
argument from algebra, not a measurement. This lane measures it on published training-free
methods with standard CHAIR on standard captions.

WHAT IS PRIOR WORK AND IS NOT CLAIMED. PAI's Eq.3/Eq.4 (arXiv:2407.21771v1), VCD
(arXiv:2311.16922), CHAIR (Rohrbach et al. 2018), and the SDT decomposition
(Snodgrass & Corwin 1988 / Hautus 1995). The new thing is only the measurement.
"""
import os

MODEL = "llava-hf/llava-1.5-7b-hf"
ROOT = "/data/alexmueller/mitdecomp"
OUT = f"{ROOT}/out"
J5CODE = "/data/alexmueller/j5_gates/code"
PACODE = "/data/alexmueller/paialpha/code"
SYN = f"{J5CODE}/synonyms.txt"

N_IMAGES = 500              # PAI chair_eval.py: `if batch_id == 500: break` at batch size 1
SAMPLE_SEED = 20260817      # this project's frozen 500-image COCO draw (j5_common.SEED)
# PAI chair_eval.py `--max-tokens` default. Env override exists ONLY so a pre-submission
# smoke can run short; the DEFAULT IS BYTE-IDENTICAL to the pre-registered 512, every arm's
# meta records the value actually used, and any smoke run additionally carries MD_LIMIT in
# its tag so its output can never be mistaken for a registered arm.
MAX_NEW = int(os.environ.get("MD_MAX_NEW", "512"))

LAYER_LO, LAYER_HI = 2, 32  # PAI `--start-layer 2 --end-layer 32`, half-open
GAMMA_PAI = 1.1             # PAI paper section 5.1 AND all three README commands
CFG_PLAUS = 0.1             # CFG.py cutoff = log(0.1) + max(log p_cond); also VCD's cd_beta

# ---- VCD: DROPPED, and this is a result, not an omission ----------------------------------
# Pre-registration section 2 says approximating VCD is forbidden and skipping it is the correct
# action if primary-source verification is ambiguous. Verification (arXiv:2311.16922v1 LaTeX
# source -- the arXiv HTML swallows `<` and truncates Eqs. 4-5 -- plus the CVPR camera-ready,
# its supplementary, and DAMO-NLP-SG/VCD at master) returned four blocking facts. Amendment 1
# of the pre-registration records them in full. In short:
#
#   1. VCD HAS NO PUBLISHED CHAIR EVALUATION. "CHAIR" appears in the v1 source exactly once,
#      in a COMMENTED-OUT line of 02_related.tex. Its benchmarks are POPE, MME and LLaVA-Bench.
#      There is therefore no published number to build a positive control against, and this
#      lane's own rule is that an arm without a positive control is not measured, it is
#      guessed.
#   2. VCD's published decoding is pure multinomial sampling (temperature 1.0, no top-p/top-k;
#      04_experiments.tex L40, and the first author twice in the tracker). The released code
#      CANNOT run greedy: evolve_vcd_sampling() patches only GenerationMixin.sample, so
#      do_sample=False silently bypasses VCD entirely. Running it greedy to match our other
#      arms would be our approximation of a method, not the method.
#   3. Paper and repo disagree three ways -- beta 0.1 vs 0.2 (POPE script), T 999 vs 500, and
#      the noise model itself (paper: constant gamma=0.1, Eq. 2; code: a SIGMOID beta schedule
#      2.23e-5 -> 4.99e-3). The paper's stated noise process is not the implemented one.
#   4. The released code was a NO-OP after the first generated token from 2023-11-28 until
#      commit c637c85a on 2024-07-16 (model_kwargs_cd re-copied inside the decode loop),
#      author-admitted. That is exactly the multi-token captioning regime CHAIR scores.
#
# Running VCD here would have produced a number with no published referent, under a decoding
# mode its authors did not use, at hyper-parameters its own sources disagree about. It is not
# run. Nothing below constructs a VCD arm.
VCD_ENABLED = False

# ---- the pre-registered arms --------------------------------------------------------------
# gamma == 1.0 is PAI's OWN no-op: CFG.py short-circuits at `if self.guidance_scale == 1` and
# returns plain log_softmax(scores). It is NOT gamma == 0 (that collapses to the second stream).
# alpha == 0.0 is `sc + 0.0*|sc|`, exact in IEEE. So `vanilla` is kernel- and loop-matched to
# the intervened arms rather than a different code path.
def _arm(alpha, gamma, blind, second):
    return {"alpha": alpha, "gamma": gamma, "blind": blind, "second": second}


ARMS = {
    "vanilla":       _arm(0.0, 1.0,       False, "noimg"),
    "pai02":         _arm(0.2, GAMMA_PAI, False, "noimg"),
    "pai05":         _arm(0.5, GAMMA_PAI, False, "noimg"),
    "vanilla_blind": _arm(0.0, 1.0,       True,  "noimg"),
    "pai02_blind":   _arm(0.2, GAMMA_PAI, True,  "noimg"),
    "pai05_blind":   _arm(0.5, GAMMA_PAI, True,  "noimg"),
}

SIGHTED = ("vanilla", "pai02", "pai05")
BLIND = ("vanilla_blind", "pai02_blind", "pai05_blind")
METHODS = ("pai05", "pai02")            # verdicts are taken per method, against `vanilla`
BLIND_OF = {"vanilla": "vanilla_blind", "pai02": "pai02_blind", "pai05": "pai05_blind"}
ALL_ARMS = SIGHTED + BLIND

# ---- prompt: PAI's LLaVA-1.5 template, reproduced byte-for-byte ----------------------------
# constants.py SYSTEM_MESSAGE + INSTRUCTION_TEMPLATE["llava-1.5"], concatenated with NO
# separator (chair_eval.py L85); question from chair_eval.py L94.
PAI_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
              "The assistant gives helpful, detailed, and polite answers to the user's questions.")
PAI_QUESTION = "Please help me describe the image in detail."


def prompt_text():
    return f"{PAI_SYSTEM}USER: <image> {PAI_QUESTION} ASSISTANT:"


# ---- analysis ------------------------------------------------------------------------------
NBOOT = 8000
BOOT_SEED = 20260915

# Materiality thresholds, fixed a priori -- prereg section 5. None is derived from any number
# this experiment produces.
MDE_CHAIR_I = 1.5       # percentage points, CHAIR_i scale
MDE_CHAIR_S = 3.0       # percentage points, CHAIR_s scale
DELTA_DPRIME = 0.15     # equivalence margin on d'
DELTA_C = 0.15          # criterion-shift materiality on c
DELTA_H = 0.03          # hit-rate scale, truncation-curve test

# Truncation sweep for the PRIMARY-3 curve control.
WINDOWS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)

# Positive control, prereg section 6. Sources: PAI 2407.21771v1 Table 4 (46.2 / 13.8),
# Table 1 (46.6 / 13.4), VISTA 2502.03628 (46.4).
PC_A_S_CENTER, PC_A_S_HALF = 46.2, 10.0     # -> [36.2, 56.2]
PC_A_I_CENTER, PC_A_I_HALF = 13.8, 4.0      # -> [ 9.8, 17.8]
PC_B_MIN = 10.0                             # CHAIR_s(vanilla) - CHAIR_s(pai05)

BS = int(os.environ.get("MD_BS", "8"))


def arm_tag(arm, limit=0):
    return f"md_{arm}" + (f"_lim{limit}" if limit else "")
