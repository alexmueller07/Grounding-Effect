"""PAI alpha-misconfiguration test -- shared definitions.

Every constant here is pre-registered in CAMPAIGN/PAI_ALPHA_PREREG.md and must not change
after the first evaluation row exists.

QUESTION UNDER TEST. PAI (arXiv:2407.21771, ECCV 2024) prescribes alpha=0.5 for LLaVA in its
paper, but its released repository (LALBJ/PAI, branch `master`) defaults `--alpha` to 0.2 at
chair_eval.py line 50 and every README eval command hardcodes `--alpha 0.2` with no per-model
guidance. The hypothesis is that downstream papers running PAI as a baseline inherit 0.2 and
therefore run the baseline at well under its prescribed strength on LLaVA-1.5.

WHAT IS PRIOR WORK. The entire intervention is PAI's and is not claimed here: Eq. 3 attention
amplification and the Eq. 4 CFG-style logit refinement. This lane only measures the effect of
the configuration choice.
"""
import os

# ---- model / data (PAI's own CHAIR protocol wherever this server can match it) ----------
MODEL = "llava-hf/llava-1.5-7b-hf"
ROOT = "/data/alexmueller/paialpha"
OUT = f"{ROOT}/out"
J5CODE = "/data/alexmueller/j5_gates/code"
SYN = f"{J5CODE}/synonyms.txt"

N_IMAGES = 500              # PAI chair_eval.py: `if batch_id == 500: break` at batch size 1
SAMPLE_SEED = 20260817      # this project's frozen 500-image COCO sample (j5_common.SEED)
MAX_NEW = 512               # PAI chair_eval.py `--max-tokens` default = 512
GREEDY = True               # PAI `--sample` is store_true and the README never passes it

# ---- PAI hyper-parameters -----------------------------------------------------------------
LAYER_LO, LAYER_HI = 2, 32  # PAI `--start-layer 2 --end-layer 32`, half-open -> layers 2..31
GAMMA = 1.1                 # README's value in all three eval commands; see prereg section 5
CFG_PLAUS = 0.1             # CFG.py: cutoff = log(0.1) + max(log p_cond); not a CLI flag

# ---- the three pre-registered arms --------------------------------------------------------
# gamma == 1.0 is PAI's OWN no-op: CFG.py short-circuits at `if self.guidance_scale == 1`
# and returns plain log_softmax(scores), i.e. the conditional distribution untouched. It is
# NOT gamma == 0 (that collapses to the unconditional stream).
ARMS = {
    "vanilla": {"alpha": 0.0, "gamma": 1.0},
    "pai02":   {"alpha": 0.2, "gamma": GAMMA},   # what the repo/README actually run
    "pai05":   {"alpha": 0.5, "gamma": GAMMA},   # what the paper prescribes for LLaVA
}
ARM_ORDER = ("vanilla", "pai02", "pai05")

# ---- prompt: PAI's LLaVA-1.5 template, reproduced byte-for-byte ----------------------------
# constants.py SYSTEM_MESSAGE + INSTRUCTION_TEMPLATE["llava-1.5"], concatenated directly with
# NO separator (chair_eval.py line 85), question from chair_eval.py line 94. `<ImageHere>` is
# this checkpoint's `<image>` placeholder.
PAI_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
              "The assistant gives helpful, detailed, and polite answers to the user's questions.")
PAI_QUESTION = "Please help me describe the image in detail."


def prompt_text():
    return f"{PAI_SYSTEM}USER: <image> {PAI_QUESTION} ASSISTANT:"


# ---- analysis ------------------------------------------------------------------------------
NBOOT = 8000
BOOT_SEED = 20260914

# Materiality thresholds, fixed a priori. Justified in prereg section 4: a configuration gap
# that moves CHAIR by less than the smallest effect this literature reports as a real
# improvement cannot explain why a paper reports PAI as ineffective.
MDE_CHAIR_S = 3.0           # percentage points on the 0-100 CHAIR_s scale
MDE_CHAIR_I = 1.5           # percentage points on the 0-100 CHAIR_i scale

BS = int(os.environ.get("PA_BS", "8"))


def arm_tag(arm, smoke=False, limit=0):
    t = ("smoke_" if smoke else "") + f"pa_{arm}"
    return t + (f"_lim{limit}" if limit else "")
