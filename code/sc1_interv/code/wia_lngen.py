"""WIA-LN generator: natural single-author human prose at the HEADLINE budget (cut 0.5).

Closes the open question of SUBMISSION.md sec4.1.1: the 14-token natural arm (wia_natgen.py) was
null, but at a length where the assembled effect was already null. This run puts natural
single-author prose at the published cut-0.5 budget (mean 58.24 tokens) using Localized
Narratives (Pont-Tuset et al., ECCV 2020), COCO val2017 split: one annotator per description,
~40 words each, every one of the 500 sample images covered.

Mirrors wia_natgen.py / j5_gen_at.py / j5_gen_atsw.py in model, prompt, greedy decoding,
MAX_NEW_CONT and batching. The ONLY change from wia_natgen.py is the SOURCE and BUDGET of the
human natural arm (and the model arm now uses the published j5_gen_atsw recipe verbatim).

Three arms, ONE per-row token budget b for every (image, derangement):

    wrongln        HUMAN, NATURAL   : ONE Localized-Narratives description of the deranged source
                                      image -- one annotator, their own word order and punctuation.
                                      Truncated at a token boundary to b ONLY when longer than b.
                                      Never shuffled, never concatenated, never doubled.
    wrongasm       HUMAN, ASSEMBLED : the published build_prefix_ids recipe (shuffle + ". "-join +
                                      double-if-short + truncate) on the source's COCO captions, at b.
    selfwrong_ln   MODEL            : the model's own free-running caption of the source image via
                                      the published j5_gen_atsw.build_prefix_from_text recipe, at b.

    b = min(cut05_i, len(LN_ids))   cut05_i = floor(0.5 * gen_len_i) is the PUBLISHED cut-0.5
                                     budget of the TARGET image i (asserted equal to at.jsonl).

    AUTHOR_LN      = gap(selfwrong_ln) - gap(wrongln)     <- PRIMARY  (WIA_LN_PREREG.md)
    AUTHOR_asm     = gap(selfwrong_ln) - gap(wrongasm)    <- in-run positive control: the published
                                                            construction at exactly these budgets
    NATURALNESS_LN = gap(wrongln)      - gap(wrongasm)    <- secondary

K = 3 derangements per image (shift 1, 2, 3 over the sorted sample ids), exactly as wia_natgen.py.
SMOKE=1 runs the FIRST 16 ids of the published 500-image sample (a true subset, real published
budgets), K = 1, 48 rows.

Read-only w.r.t. every existing artifact. Writes {OUT_LN}/at_ln.jsonl and {OUT_LN}/at_ln_meta.json.
Sentinels are printed by Python; the exit code is NEVER the success signal.
"""
import os, sys, json, time, math, random
sys.path.insert(0, "/data/alexmueller/j5_gates/code")
from j5_common import (load_coco_sample, ChairScorer, build_prefix_ids, read_jsonl,
                       JsonlWriter, load_model, prompt_text, sentinel,
                       MODEL, OUT, N_IMAGES, MAX_NEW_CONT, BS)
import numpy as np
import torch


def load_model_lowmem():
    """j5_common.load_model with the weights streamed straight to the GPU (device_map="cuda",
    low_cpu_mem_usage=True). Same checkpoint, same dtype, same device -> identical forward pass;
    only the host-RAM peak during loading changes. Needed because vgi1 had ~3 GB of allocatable
    RAM on 2026-09-08 and the plain loader was OOM-killed at 3G (job 18937). Equivalence is
    checked empirically by the SMOKE replay check in main()."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"
    if proc.tokenizer.pad_token is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
    model = model.eval()
    return model, proc, proc.tokenizer

SMOKE = os.environ.get("SMOKE", "0") == "1"
N_SMOKE = 16
# LOADER=orig   -> j5_common.load_model (CPU materialise, .to("cuda"); needs --mem=14G): the path
#                  WIA_LN_PREREG.md section 2 calls "unchanged from j5_gen_at.py", and the section 8.1
#                  fallback after the low-RAM loader failed its replay criterion (job 18976, image 3553:
#                  7 agreeing tokens < 20).
# LOADER=lowmem -> load_model_lowmem (default; the section 8.1 deviation, jobs 18976/18978).
# An orig run writes to "<tag>_orig" so the lowmem run's files are never overwritten.
LOADER = os.environ.get("LOADER", "lowmem")
assert LOADER in ("orig", "lowmem"), f"FATAL_LOADER {LOADER}"
SUF = "_orig" if LOADER == "orig" else ""
# WIA_LN_FULLBUDGET_PREREG.md knobs (all three unset -> byte-identical behaviour to the main lane):
#   LN_SUFFIX  extra output-name suffix, composed after the LOADER suffix (e.g. "_b58")
#   LN_MINLEN  keep only cells whose SEEDED LN pick (same seed stream) has >= this many tokens
#   LN_BUDGET  fix b = LN_BUDGET for all three arms in every kept cell (instead of min(cut05_i, LN_len));
#              requires LN_MINLEN >= LN_BUDGET so the natural arm is never doubled to reach it
LN_SUFFIX = os.environ.get("LN_SUFFIX", "")
LN_MINLEN = int(os.environ["LN_MINLEN"]) if os.environ.get("LN_MINLEN") else None
LN_BUDGET = int(os.environ["LN_BUDGET"]) if os.environ.get("LN_BUDGET") else None
assert LN_BUDGET is None or (LN_MINLEN is not None and LN_MINLEN >= LN_BUDGET), "FATAL_LN_BUDGET_EXCEEDS_MINLEN"
TAG = ("smoke_at_ln" if SMOKE else "at_ln") + SUF + LN_SUFFIX
KDER = 1 if SMOKE else 3          # derangement shifts 1..KDER
MIN_LEN = 32                      # inherited J5 target-eligibility floor, unchanged
CUT_F = 0.5                       # the headline cut
CAP_SEED_BASE = 1900              # LN description draw stream (disjoint from wia_natgen's 900)
ASM_SEED_BASE = 1950              # assembled-arm shuffle stream (disjoint from wia_natgen's 950)
LN_PATH = "/data/alexmueller/sc1_interv/data/coco_val_localized_narratives.jsonl"
OUT_LN = "/data/alexmueller/sc1_interv/out"
LN_MIN_COVERAGE = 400             # WIA_LN_PREREG.md: fewer covered images -> NO-DATA, do not run


# ---------------------------------------------------------------------------- prefixes
def norm(c):
    """Terminal-period normalisation byte-identical to build_prefix_ids / wia_natgen.natural_caption,
    so the human arms differ ONLY in assembly, never in punctuation."""
    return c.strip().rstrip(".").strip() + "."


def load_ln():
    """image_id -> list of Localized-Narratives captions (file order), COCO val2017 split only."""
    d = {}
    for l in open(LN_PATH):
        r = json.loads(l)
        assert r["dataset_id"] == "mscoco_val2017", f"FATAL_LN_SPLIT {r['dataset_id']}"
        d.setdefault(int(r["image_id"]), []).append(r["caption"])
    return d


def ln_description(caps, seed):
    """ONE whole LN description of the source image, seeded draw among its annotators."""
    assert caps, "FATAL_NO_LN_CAPTIONS"
    return norm(caps[random.Random(seed).randrange(len(caps))])


def build_prefix_from_text(tok, text, target_len):
    """VERBATIM copy of j5_gen_atsw.build_prefix_from_text (the published `selfwrong` recipe):
    mirror of build_prefix_ids' cycling/truncation, for a single model caption."""
    text = (text or "").strip().rstrip(".").strip() + "."
    ids = tok(text, add_special_tokens=False)["input_ids"]
    cycles = 0
    while len(ids) < target_len:
        cycles += 1
        text = text + " " + text
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if cycles > 6:
            break
    ids = ids[:target_len]
    return ids, tok.decode(ids), cycles


@torch.no_grad()
def gen_batch(model, proc, tok, items):
    """Byte-for-byte j5_gen_at.gen_batch / wia_natgen.gen_batch: same prompt, greedy, flat
    MAX_NEW_CONT, left pad."""
    from PIL import Image
    ims = [Image.open(it["file"]).convert("RGB") for it in items]
    enc = proc(images=ims, text=[prompt_text()] * len(items), return_tensors="pt", padding=True)
    base_ids = enc["input_ids"]
    pix = enc["pixel_values"].to("cuda", torch.bfloat16)
    rows, masks = [], []
    maxlen = max(base_ids.shape[1] + len(it["prefix_ids"]) for it in items)
    pad = tok.pad_token_id
    for i, it in enumerate(items):
        row = base_ids[i].tolist() + list(it["prefix_ids"])
        need = maxlen - len(row)
        rows.append([pad] * need + row)
        masks.append([0] * need + [1] * len(row))
    input_ids = torch.tensor(rows, device="cuda")
    attn = torch.tensor(masks, device="cuda")
    gen = model.generate(input_ids=input_ids, attention_mask=attn, pixel_values=pix,
                         do_sample=False, max_new_tokens=MAX_NEW_CONT, pad_token_id=pad)
    new = gen[:, input_ids.shape[1]:]
    outs = []
    for row in new:
        ids = [int(t) for t in row if int(t) != pad]
        if ids and ids[-1] == tok.eos_token_id:
            ids = ids[:-1]
        outs.append(ids)
    return outs


def build_items(tok, by_id, af, ln, sids, cut05, max_targets=None):
    """All three arms for every (image, derangement). Returns items and construction stats.
    max_targets (LN_MINLEN smoke only): stop once this many target images have emitted >= 1 cell."""
    items = []
    st = {"ln_full": [], "budget": [], "cut05": [], "bound_ln": 0, "bound_cut": 0, "tie": 0,
          "sw_cycles_rows": 0, "asm_cycles_rows": 0, "asm_cycles_total": 0,
          "skipped_short_ln": 0, "n_targets": 0}
    n_skip_target = 0
    pos = {iid: j for j, iid in enumerate(sids)}
    for iid in sids:
        if af[iid]["gen_len"] < MIN_LEN:            # target eligibility, unchanged from J5
            n_skip_target += 1
            continue
        n_before = len(items)
        for k in range(1, KDER + 1):
            src_id = sids[(pos[iid] + k) % len(sids)]
            assert src_id != iid, f"FATAL_DERANGEMENT_FIXPOINT {iid} k={k}"
            src = by_id[src_id]

            # ---- 1. the HUMAN NATURAL arm: one LN description, truncated only if longer ------
            ln_text = ln_description(ln[src_id], seed=iid * 7 + CAP_SEED_BASE + k)
            ln_ids_full = tok(ln_text, add_special_tokens=False)["input_ids"]
            L, c = len(ln_ids_full), cut05[iid]
            if LN_MINLEN is not None and L < LN_MINLEN:   # WIA_LN_FULLBUDGET_PREREG.md subset rule
                st["skipped_short_ln"] += 1
                continue
            b = min(c, L) if LN_BUDGET is None else LN_BUDGET
            assert b > 0, f"FATAL_EMPTY_LN_PREFIX {iid} k={k}"
            ln_ids = ln_ids_full[:b]
            bound = "ln" if L < c else ("cut05" if L > c else "tie")
            st["ln_full"].append(L); st["budget"].append(b); st["cut05"].append(c)
            st[{"ln": "bound_ln", "cut05": "bound_cut", "tie": "tie"}[bound]] += 1

            # ---- 2. the HUMAN ASSEMBLED arm, published recipe, SAME budget -----------------
            asm_ids, asm_text, cyc_a = build_prefix_ids(
                tok, src["human_caps"], b, seed=iid * 7 + ASM_SEED_BASE + k)
            st["asm_cycles_rows"] += (cyc_a > 0); st["asm_cycles_total"] += cyc_a

            # ---- 3. the MODEL arm, published j5_gen_atsw recipe, SAME budget ---------------
            sw_ids, sw_text, cyc_s = build_prefix_from_text(tok, af[src_id]["text"], b)
            st["sw_cycles_rows"] += (cyc_s > 0)

            # ---- per-row token-budget equality, asserted, not assumed ---------------------
            assert len(ln_ids) == len(asm_ids) == len(sw_ids) == b, (
                f"FATAL_BUDGET_MISMATCH iid={iid} k={k} "
                f"ln={len(ln_ids)} asm={len(asm_ids)} sw={len(sw_ids)} b={b}")

            for arm, pids, ptext, cyc in (("wrongln", ln_ids, tok.decode(ln_ids), 0),
                                          ("wrongasm", asm_ids, asm_text, cyc_a),
                                          ("selfwrong_ln", sw_ids, sw_text, cyc_s)):
                items.append({"image_id": iid, "src_id": src_id, "k": k, "arm": arm,
                              "file": by_id[iid]["file"], "prefix_ids": pids,
                              "prefix_text": ptext, "budget": b, "cycles": cyc,
                              "cut05": c, "ln_len_full": L, "ln_truncated": L > b,
                              "budget_bound": bound,
                              "ln_text_raw": ln_text if arm == "wrongln" else None})
        if len(items) > n_before:
            st["n_targets"] += 1
            if max_targets is not None and st["n_targets"] >= max_targets:
                break
    return items, st, n_skip_target


def main():
    chair = ChairScorer(os.path.join("/data/alexmueller/j5_gates/code", "synonyms.txt"))
    data = load_coco_sample(chair, n=N_IMAGES)          # ALWAYS the published 500-image sample
    by_id = {d["image_id"]: d for d in data}
    all_sids = sorted(by_id)
    sids = all_sids[:N_SMOKE] if (SMOKE and LN_MINLEN is None) else all_sids   # smoke = true subset, real budgets
    af = {r["image_id"]: r for r in read_jsonl(f"{OUT}/af.jsonl")}
    assert set(af) == set(by_id), "FATAL_AF_IMAGE_SET_DRIFT"

    # --- LN coverage gate (WIA_LN_PREREG.md NO-DATA cell (a)) -------------------------------
    ln = load_ln()
    covered = [i for i in all_sids if i in ln]
    print(f"[ln] Localized Narratives: {len(ln)} val2017 images; sample coverage "
          f"{len(covered)}/{len(all_sids)}", flush=True)
    assert len(covered) >= LN_MIN_COVERAGE, f"FATAL_LN_COVERAGE {len(covered)} < {LN_MIN_COVERAGE}"
    assert all(i in ln for i in sids), "FATAL_LN_MISSING_IN_WORKING_SET"

    # --- BUDGET GATE: the published cut-0.5 budgets rebuild from af.jsonl and equal at.jsonl --
    cut05 = {i: int(math.floor(CUT_F * af[i]["gen_len"])) for i in all_sids}
    at05 = {r["image_id"]: r["cut_tokens"] for r in read_jsonl(f"{OUT}/at.jsonl")
            if r.get("kind") == "wrong" and r["cut_f"] == CUT_F}
    bad = [i for i in all_sids if at05.get(i) != cut05[i]]
    assert not bad, f"FATAL_BUDGET_DRIFT {len(bad)} images"
    print(f"[ln] BUDGET GATE: floor(0.5*gen_len) == published cut-0.5 cut_tokens on all "
          f"{len(all_sids)} images (mean {np.mean([cut05[i] for i in all_sids]):.2f})", flush=True)

    model, proc, tok = load_model() if LOADER == "orig" else load_model_lowmem()
    print(f"[ln] LOADER={LOADER}  TAG={TAG}", flush=True)
    print(f"[ln] MODEL={MODEL}", flush=True)
    print(f"[ln] tokenizer class = {tok.__class__.__name__}  vocab={tok.vocab_size}", flush=True)
    assert "llava-1.5-7b" in MODEL, f"FATAL_WRONG_MODEL: {MODEL}"

    # --- CONSTRUCTION GATE: the published `wrong` arm must rebuild bit-for-bit ---------------
    # Unchanged from wia_natgen.py (which is j5_gen_at.py:main's recipe): defined over the
    # published 500-image sample and its shift-by-1 derangement, regardless of SMOKE.
    wof = {all_sids[i]: all_sids[(i + 1) % len(all_sids)] for i in range(len(all_sids))}
    at_rows = [r for r in read_jsonl(f"{OUT}/at.jsonl") if r.get("kind") == "wrong"]
    nchk = nbad = 0
    for r in at_rows:
        iid, f = r["image_id"], r["cut_f"]
        pids, ptext, _c = build_prefix_ids(tok, by_id[wof[iid]]["human_caps"],
                                           r["cut_tokens"], seed=iid * 7 + int(f * 100))
        nchk += 1
        nbad += (ptext != r["prefix_text"])
    print(f"[ln] CONSTRUCTION GATE: rebuilt {nchk} published `wrong` prefixes, "
          f"mismatches {nbad}", flush=True)
    assert nchk > 0 and nbad == 0, f"FATAL_CONSTRUCTION_GATE nchk={nchk} nbad={nbad}"

    if SMOKE:
        # REPLAY CHECK: regenerate 4 published cut-0.5 `wrong` rows from their STORED prefix_ids and
        # count leading continuation tokens agreeing with the stored cont_ids. Verifies that the
        # low-RAM loader reproduces the published model/decoding path. bf16 batch-shape effects can
        # cause late divergence, so this is printed, not asserted; expect long agreeing runs.
        want = set(sids)
        rep = [r for r in at_rows if r["cut_f"] == CUT_F and r["image_id"] in want][:4]
        conts = gen_batch(model, proc, tok, [{"file": by_id[r["image_id"]]["file"],
                                              "prefix_ids": r["prefix_ids"]} for r in rep])
        agree = []
        for r, c in zip(rep, conts):
            n = 0
            for a, b in zip(r["cont_ids"], c):
                if a != b:
                    break
                n += 1
            agree.append({"image_id": r["image_id"], "agree": n, "stored": len(r["cont_ids"]),
                          "regen": len(c), "identical": n == len(r["cont_ids"]) == len(c)})
        print(f"[ln] REPLAY CHECK vs published wrong@0.5 (stored prefix_ids): {agree}", flush=True)
        json.dump({"loader": LOADER, "agree": agree,
                   "rows": [{"image_id": r["image_id"], "prefix_ids": r["prefix_ids"],
                             "stored_cont_ids": r["cont_ids"], "regen_cont_ids": list(c)}
                            for r, c in zip(rep, conts)]},
                  open(f"{OUT_LN}/replay_{TAG}.json", "w"))
        if LOADER == "lowmem":
            # WIA_LN_PREREG.md section 8.1 acceptance criterion (>= 20 agreeing leading tokens or
            # identical end-to-end, every row), now ENFORCED. Job 18976 printed agree=7 on image
            # 3553 and the afterok chain proceeded because this block was print-only.
            bad = [a for a in agree if not (a["identical"] or a["agree"] >= 20)]
            assert not bad, f"FATAL_REPLAY_GATE {bad}"

    items, st, skT = build_items(tok, by_id, af, ln, sids, cut05,
                                 max_targets=N_SMOKE if (SMOKE and LN_MINLEN is not None) else None)
    if LN_MINLEN is not None:
        print(f"[ln] LN_MINLEN={LN_MINLEN} LN_BUDGET={LN_BUDGET}: cells dropped (seeded LN pick shorter) "
              f"{st['skipped_short_ln']}; target images with >= 1 kept cell {st['n_targets']}", flush=True)
    Lf, B = np.array(st["ln_full"]), np.array(st["budget"])
    ncell = len(B)
    q = lambda a, p: float(np.percentile(a, p))
    print(f"[ln] LN description length (REAL {tok.__class__.__name__} tokens, seeded pick per cell): "
          f"mean {Lf.mean():.3f} sd {Lf.std():.3f} min {Lf.min()} p5 {q(Lf,5):.0f} "
          f"p50 {q(Lf,50):.0f} p95 {q(Lf,95):.0f} max {Lf.max()}", flush=True)
    print(f"[ln] budget b = min(cut05, LN): mean {B.mean():.3f} sd {B.std():.3f} min {B.min()} "
          f"p5 {q(B,5):.0f} p50 {q(B,50):.0f} p95 {q(B,95):.0f} max {B.max()}  "
          f"(published cut-0.5 mean over these cells {np.mean(st['cut05']):.3f})", flush=True)
    print(f"[ln] cells {ncell}: LN shorter than cut05 (LN defines b) {st['bound_ln']/ncell:.3f}; "
          f"LN longer (truncated) {st['bound_cut']/ncell:.3f}; tie {st['tie']/ncell:.3f}", flush=True)
    print(f"[ln] DOUBLING: wrongasm rows with cycles>0 = {st['asm_cycles_rows']}/{ncell} "
          f"(total cycles {st['asm_cycles_total']}); selfwrong_ln rows with cycles>0 = "
          f"{st['sw_cycles_rows']}/{ncell}; wrongln = 0 by construction", flush=True)
    print(f"[ln] rows {len(items)}  skipped target<MIN_LEN {skT}", flush=True)
    assert len(items) == 3 * ncell, f"FATAL_ROW_COUNT {len(items)} != 3*{ncell}"

    w = JsonlWriter(f"{OUT_LN}/{TAG}.jsonl")
    items.sort(key=lambda it: len(it["prefix_ids"]))
    bs = max(4, BS // 2)
    t0 = time.time()
    for b0 in range(0, len(items), bs):
        chunk = items[b0:b0 + bs]
        conts = gen_batch(model, proc, tok, chunk)
        for it, ids in zip(chunk, conts):
            w.write({"image_id": it["image_id"], "src_id": it["src_id"], "k": it["k"],
                     "arm": it["arm"], "budget": it["budget"], "cycles": it["cycles"],
                     "cut05": it["cut05"], "ln_len_full": it["ln_len_full"],
                     "ln_truncated": it["ln_truncated"], "budget_bound": it["budget_bound"],
                     "prefix_ids": it["prefix_ids"], "prefix_text": it["prefix_text"],
                     "ln_text_raw": it["ln_text_raw"],
                     "cont_ids": ids, "cont_text": tok.decode(ids, skip_special_tokens=True),
                     "cont_len": len(ids)})
        w.flush()
        print(f"[ln] {b0+len(chunk)}/{len(items)} t={time.time()-t0:.0f}s", flush=True)
    w.close()

    rows = read_jsonl(f"{OUT_LN}/{TAG}.jsonl")
    per_arm = {}
    for r in rows:
        per_arm[r["arm"]] = per_arm.get(r["arm"], 0) + 1
    assert len(per_arm) == 3 and all(v > 0 for v in per_arm.values()), f"FATAL_ARMS {per_arm}"
    assert len(set(per_arm.values())) == 1, f"FATAL_ARMS_UNBALANCED {per_arm}"
    assert len(rows) == len(items), f"FATAL_ROWS_WRITTEN {len(rows)} != {len(items)}"
    cl = {a: float(np.mean([r["cont_len"] for r in rows if r["arm"] == a])) for a in per_arm}
    json.dump({"sentinel": True, "model": MODEL, "rows": len(rows), "per_arm": per_arm,
               "k_derangements": KDER, "n_images": len({it["image_id"] for it in items}), "smoke": SMOKE, "loader": LOADER,
               "ln_minlen": LN_MINLEN, "ln_budget": LN_BUDGET, "ln_suffix": LN_SUFFIX,
               "cells_dropped_short_ln": st["skipped_short_ln"],
               "ln_source": LN_PATH, "ln_coverage": len(covered),
               "skipped_target_short": skT,
               "wrongasm_rows_doubled": st["asm_cycles_rows"],
               "selfwrong_ln_rows_doubled": st["sw_cycles_rows"],
               "frac_cells_ln_defines_budget": st["bound_ln"] / ncell,
               "frac_cells_ln_truncated": st["bound_cut"] / ncell,
               "ln_len_full_tokens": {"mean": float(Lf.mean()), "sd": float(Lf.std()),
                                      "min": int(Lf.min()), "max": int(Lf.max()),
                                      "p5": q(Lf, 5), "p50": q(Lf, 50), "p95": q(Lf, 95)},
               "budget_tokens": {"mean": float(B.mean()), "sd": float(B.std()),
                                 "min": int(B.min()), "max": int(B.max()),
                                 "p5": q(B, 5), "p50": q(B, 50), "p95": q(B, 95)},
               "published_cut05_mean_over_cells": float(np.mean(st["cut05"])),
               "mean_cont_len_per_arm": cl},
              open(f"{OUT_LN}/{TAG}_meta.json", "w"), indent=1)
    sentinel("LN_SMOKE" if SMOKE else "LN")


if __name__ == "__main__":
    main()
