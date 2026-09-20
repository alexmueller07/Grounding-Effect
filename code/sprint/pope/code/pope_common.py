"""POPE lane -- shared, pre-registered definitions.

Every constant here is fixed in SPRINT/POPE_PREREG.md and must not change after the first
endpoint row exists.

WHAT THIS LANE MEASURES.  The paper (sec:reco item 2) recommends blinding every endpoint whose
candidate objects are supplied in the input, and names POPE as such an endpoint.  Appendix G
records that the grey control could NOT be run on CHAIR because CHAIR's denominator is the
model's own mention set, which empties under grey.  POPE's candidate set is supplied by the
question, so the control is runnable.  This lane runs it, and applies the paper's
Delta_image = Delta_sighted - Delta_ablated decomposition to a published method's POPE gain.

WHAT IS PRIOR WORK AND IS NOT CLAIMED.  POPE (arXiv:2305.10355), PAI Eq.3/Eq.4
(arXiv:2407.21771, ECCV 2024), LLaVA-1.5 (arXiv:2310.03744), the SDT decomposition
(Snodgrass & Corwin 1988 / Hautus 1995).  New here is only the measurement.
"""
import os

MODEL = "llava-hf/llava-1.5-7b-hf"
ROOT = "/data/alexmueller/sprint/pope"
DATA = f"{ROOT}/data"
OUT = f"{ROOT}/out"
IMGDIR = f"{DATA}/val2014_images"

SPLITS = ("random", "popular", "adversarial")
N_Q_PER_SPLIT = 3000
N_IMAGES = 500

# Official POPE question files, fetched from the POPE authors' repository on 2026-09-19.
#   https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_<split>.json
POPE_SHA256 = {
    "random":      "ac25245170b975a5bdf9080b23fd431dfe6be458bc038259c1f4f09a6bef7994",
    "popular":     "72c1a8ad45d0c13514f5f22598261df41d3b533854d29682e924db50ed8aa753",
    "adversarial": "420b3407db1fa9f1187a805dca41cb7b97fd91504e6c2179706188c107fb8ef8",
}

# ---- prompt: PAI's own LLaVA-1.5 POPE template, reproduced byte-for-byte ------------------
# PAI constants.py SYSTEM_MESSAGE + INSTRUCTION_TEMPLATE["llava-1.5"], concatenated with NO
# separator (pope_eval.py L88-90: `template = SYSTEM_MESSAGE + template`), with <ImageHere>
# written as llava-hf's <image> placeholder and <question> the POPE question verbatim.
# NOTE: PAI's POPE protocol appends NO "answer with a single word" suffix.  That suffix is the
# LLaVA-1.5 repo's own POPE protocol, not PAI's; using it would be our approximation of PAI.
PAI_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
              "The assistant gives helpful, detailed, and polite answers to the user's questions.")


def prompt_text(question):
    return f"{PAI_SYSTEM}USER: <image> {question} ASSISTANT:"


# ---- decoding -----------------------------------------------------------------------------
# PAI pope_eval.py: do_sample=args.sample (absent -> False), num_beams=args.beam (default 1),
# max_new_tokens=args.max_tokens (default 512).  We cap at 64: POPE answers are a word or a
# short sentence, and the per-arm truncation rate is reported and gated (G_TRUNC) so the
# generation-budget confound is excluded by measurement, not by assumption.
MAX_NEW = int(os.environ.get("POPE_MAX_NEW", "64"))
BS = int(os.environ.get("POPE_BS", "16"))

LAYER_LO, LAYER_HI = 2, 32      # PAI --start-layer 2 --end-layer 32, half-open
GAMMA_PAI = 1.1                 # PAI README: all three commands use --gamma 1.1
ALPHA_PAPER = 0.5               # the value PAI's paper prescribes for LLaVA-1.5-7B
CFG_PLAUS = 0.1                 # CFG.py cutoff = log(0.1) + max(log p_cond)

# ---- the method axis ----------------------------------------------------------------------
# Three configurations, each one a real released configuration of PAI, verified at source
# against LALBJ/PAI@master (attention.py L90, CFG.py L31-33, README L65).
#
#   rowmode "decode"  -- amplify the last query row on CONDITIONAL DECODE steps only.  This is
#     what the released code does under `--use-attn --use-cfg`: llama_modify sets
#     self_attn.use_cfg=True, the guard is `if use_attn and not use_cfg`, and CFGLogits only
#     clears use_cfg inside its first __call__, which runs AFTER the prefill pass.  The FIRST
#     GENERATED TOKEN IS THEREFORE NEVER AMPLIFIED.
#   rowmode "lastrow" -- amplify the last query row of every conditional forward, prefill
#     included.  This is what the released code does under `--use-attn` alone (use_cfg=False
#     from the start), and it is the only configuration in which Eq.3 can move a one-token
#     answer.
#
# On POPE the answer is the first generated token, so the method axis is not cosmetic: under
# the README configuration PAI's attention stage cannot reach the answer at all and any POPE
# effect must come from Eq.4.  `pai_attn` is registered precisely so that this is measured
# rather than assumed.
def _m(alpha, gamma, rowmode, note):
    return {"alpha": alpha, "gamma": gamma, "rowmode": rowmode, "note": note}


METHODS = {
    "vanilla":  _m(0.0, 1.0, "decode",
                   "alpha=0 is sc+0*|sc|, exact in IEEE; gamma=1 takes PAI's own "
                   "short-circuit. Kernel- and loop-matched no-op, not a second code path."),
    "pai_full": _m(ALPHA_PAPER, GAMMA_PAI, "decode",
                   "released `--use-attn --use-cfg`, alpha at the paper's LLaVA-1.5 value"),
    "pai_attn": _m(ALPHA_PAPER, 1.0, "lastrow",
                   "released `--use-attn` alone: Eq.3 only, prefill last row included"),
    # NOT A RELEASED CONFIGURATION. The released code makes the two stages mutually exclusive
    # at prefill (`use_attn and not use_cfg`), so both stages can never reach the first
    # generated token together. This arm constructs that combination anyway, so that a null
    # on pai_full/pai_attn cannot be answered with "you crippled the method". It is labelled
    # CONSTRUCTED everywhere it is reported and is never promoted over a released arm.
    "pai_both": _m(ALPHA_PAPER, GAMMA_PAI, "lastrow",
                   "CONSTRUCTED: Eq.3 at the prefill last row AND Eq.4; not a released "
                   "configuration of PAI"),
}
PAI_METHODS = ("pai_full", "pai_attn")
RELEASED_METHODS = ("vanilla", "pai_full", "pai_attn")
CONSTRUCTED_METHODS = ("pai_both",)

# ---- analysis -------------------------------------------------------------------------------
NBOOT = 4000
BOOT_SEED = 20260919

# Materiality thresholds, fixed a priori. None is derived from any number this lane produces.
# On balanced POPE accuracy = (1+J)/2 exactly, so 0.01 J = 0.5 accuracy points.  Set from
# PAI's PUBLISHED gain for this model (+1.06 acc pts, Table 2 single-turn greedy; +1.37,
# Table S3): the floor is half the smaller of the two.  Not derived from any number this lane
# produces.
MDE_J = 0.01
HALF_GAIN = 0.5         # the registered dividing line on Delta_image is 0.5 * Delta_s
# Positive control: PAI Table 2 single-turn greedy, vanilla LLaVA-1.5-7B, averaged over the
# three POPE splits (PAI publishes no per-split number for this protocol).
PC_ACC_CENTER, PC_ACC_HALF = 84.76, 2.5
PC_F1_CENTER, PC_F1_HALF = 85.51, 2.5
DEGEN_YES_HI = 0.95     # yes-ratio at or above this -> constant-response degenerate
DEGEN_YES_LO = 0.05
DEGEN_UNPARSE = 0.05    # strict-parser unparseable fraction above this -> degenerate
DEGEN_EMPTY = 0.01
TRUNC_MAX = 0.02        # per-arm truncation rate above this fires the budget contingency
CHANCE_ACC_HI = 0.55    # vanilla-grey accuracy at or below this -> POPE is blindable


def tag(method, pixarm):
    return f"pope_{method}_{pixarm}"
