"""Lane `qwen3` HARNESS GATE -- runs BEFORE any endpoint arm, and decides whether one is run.

Why this file exists. WIA-FAM3 ran Qwen2.5-VL-7B through this same design and got a MEDIAN
CONTINUATION OF FOUR TOKENS in every arm, which left the carry-forward endpoint nothing to
register. Its plumbing smoke had asserted only that a prefix CHANGES the continuation
(GATE_PREFIX_BINDS) -- a check a broken chat template passes trivially. The reviewer's objection
was that a four-token continuation suggests the prompt or template is wrong for that model. That
objection is correct and unanswerable from the evidence FAM3 collected, so this lane collects the
evidence that answers it, BEFORE it generates an endpoint.

The decisive test is HG-3, and it is available only because decoding is greedy. Under greedy
decoding the model's own free-running caption IS its own continuation from its own prefix at any
cut (Appendix A.1 of the paper leans on exactly this). So: cut the model's OWN caption at the
census cut, splice the first half back in as an assistant-turn prefill, and the model must
re-emit the second half. A harness that lands the prefill outside the assistant turn, closes the
turn, corrupts M-RoPE or attends a pad CANNOT pass this. A model that is merely terse on an
out-of-distribution stimulus CAN.

GATES (thresholds fixed in SPRINT/QWEN3_PREREG.md section 3 before this file was ever run):

  HG-1 TEMPLATE        prompt is byte-identical to the frozen processor-rendered template
  HG-2 PREFILL_LANDS   on realised spliced token ids: the prefill sits immediately after the
                       generation-prompt suffix, introduces no special/control token, and does
                       not change the turn-marker counts of the template
  HG-3 SELF_CONT       self-prefill re-emits the model's own caption remainder
                       (a) median continuation length      >= 0.50 x median remainder length
                       (b) median first-16-token agreement >= 0.75
                       (c) empty-continuation rate         <= 0.10
  HG-4 FREE_LEN        median free-running caption length  >= 32 tokens
  HG-5 ASSEMBLED       on the real cap1/cap5 stimulus at the census cut, in BOTH arms
                       (a) median continuation length      >= 12 tokens
                       (b) <=3-token rate                  <= 0.30
  HG-6 PREFIX_BINDS    the assembled prefill changes the continuation vs free running

CALIBRATION. HG-3's thresholds are run unchanged on LLaVA-1.5-7B, the model that carries the
paper's lead result, in this same job. A gate the paper's own headline model fails is a broken
gate, not a finding (feedback_calibrate_gates_against_own_corpus). HG-5's thresholds are
calibrated against the published per-arm degeneracy screens of families 1 and 4: the shortest
median in any arm the paper interprets is 16 (LLaVA-1.5-7B cap1g) and the highest <=3-token rate
is 0.209 (Kosmos-2 cap5g). FAM3's Qwen2.5-VL arms (median 4-5) fail both.

DECISION (registered):
  HG-1 or HG-2 fail  -> the harness is wrong. STOP. No endpoint.
  HG-3 fails         -> the model cannot continue a pre-filled answer under a harness that is
                        otherwise proven correct. STOP, report with this evidence. No endpoint.
  HG-3 passes, HG-5 fails -> the harness is proven correct on a self-prefill, so short
                        continuations on the assembled stimulus are the model's response to the
                        STIMULUS, not a template bug. The endpoint IS run and the row is reported
                        as degenerate-and-attributable, with this evidence.
  all pass           -> run the endpoint.

Writes out/q3_gate.json. Sentinel printed by Python; the exit code is NEVER the success signal.
"""
import os, sys, json, time
sys.path.insert(0, "/data/alexmueller/sprint/qwen3/code")
import numpy as np
from q3_common import (M5, M5_SHA, OUT, WHICH, QUESTION, load_m5, q3_cont, q3_free,
                       q3_generate_spliced, load_coco_sample, ChairScorer, SYN, MIN_LEN,
                       cut_of, BS_FREE, BS_CONT, MAX_NEW_FREE, MAX_NEW_384, jload, sentinel)
from gen_common import prompt_for
from j5_common import build_prefix_ids
from wia_redun_common import (sources, permuted, assemble_nodouble, asm_units,
                              S_CAPPICK, S_JOIN_CAP)

N_GATE = 24                    # images for HG-3 / HG-5 / HG-6
ASM_SEED_BASE = 1950
AGREE_K = 16                   # HG-3(b) window
CAL_CUT_M1 = 0.5               # LLaVA-1.5-7B's own registered primary cut

HG3_LEN_FRAC = 0.50
HG3_AGREE = 0.75
HG3_EMPTY = 0.10
HG4_MEDIAN_FREE = 32
HG5_MEDIAN = 12
HG5_LE3 = 0.30


def _agree(a, b, k=AGREE_K):
    """Fraction of the first k tokens of the expected remainder that the continuation matches."""
    m = min(k, len(b))
    if m == 0:
        return None
    return sum(1 for i in range(m) if i < len(a) and a[i] == b[i]) / m


def self_prefill_block(tag, free_rows, cut_f, gen_cont_fn, by_id):
    """HG-3 for one model. `gen_cont_fn(items) -> rows` must generate at the endpoint batch size."""
    items, exp = [], {}
    for r in free_rows:
        L = len(r["gen_ids"])
        c = cut_of(L, cut_f)
        if c < 1 or c >= L:
            continue
        items.append({"image_id": r["image_id"], "file": by_id[r["image_id"]]["file"],
                      "arm": "self", "prefix_ids": list(r["gen_ids"][:c]), "cut": c, "L": L})
        exp[r["image_id"]] = list(r["gen_ids"][c:])
    rows = gen_cont_fn(items)
    lens, agr, rem, exact = [], [], [], 0
    per = []
    for x in rows:
        e = exp[x["image_id"]]
        a = _agree(x["cont_ids"], e)
        lens.append(x["cont_len"]); rem.append(len(e))
        if a is not None:
            agr.append(a)
        exact += (list(x["cont_ids"]) == e)
        per.append({"image_id": x["image_id"], "L": x["L"], "cut": x["cut"],
                    "remainder": len(e), "cont_len": x["cont_len"], "agree16": a})
    med_len, med_rem = float(np.median(lens)), float(np.median(rem))
    med_agr = float(np.median(agr)) if agr else 0.0
    empty = float(np.mean([l == 0 for l in lens]))
    a_ok = med_len >= HG3_LEN_FRAC * med_rem
    b_ok = med_agr >= HG3_AGREE
    c_ok = empty <= HG3_EMPTY
    out = {"tag": tag, "cut_f": cut_f, "n": len(rows),
           "median_cont_len": med_len, "mean_cont_len": float(np.mean(lens)),
           "median_remainder_len": med_rem, "mean_remainder_len": float(np.mean(rem)),
           "median_agree16": med_agr, "mean_agree16": float(np.mean(agr)) if agr else None,
           "exact_full_match": exact, "empty_rate": empty,
           "a_len_ok": bool(a_ok), "b_agree_ok": bool(b_ok), "c_empty_ok": bool(c_ok),
           "pass": bool(a_ok and b_ok and c_ok),
           "thresholds": {"len_frac": HG3_LEN_FRAC, "agree": HG3_AGREE, "empty": HG3_EMPTY},
           "per_row": per}
    print(f"[HG-3:{tag}] median cont {med_len} vs 0.5*remainder {0.5*med_rem:.1f} | "
          f"median agree16 {med_agr:.3f} | empty {empty:.3f} | exact {exact}/{len(rows)} | "
          f"{'PASS' if out['pass'] else 'FAIL'}", flush=True)
    for x in rows[:2]:
        print(f"[HG-3:{tag}] PFX {x['prefix_text'][-120:]!r}", flush=True)
        print(f"[HG-3:{tag}] CNT {x['cont_text'][:200]!r}", flush=True)
    return out


def main():
    t0 = time.time()
    assert os.uname().nodename == "vgi1", f"FATAL_WRONG_NODE {os.uname().nodename}"
    R = {"sentinel": True, "model": M5, "revision": M5_SHA, "n_gate_images": N_GATE,
         "bs_free": BS_FREE, "bs_cont": BS_CONT, "max_new_cont": MAX_NEW_384,
         "thresholds": {"HG3": {"len_frac": HG3_LEN_FRAC, "agree": HG3_AGREE,
                                "empty": HG3_EMPTY},
                        "HG4_median_free": HG4_MEDIAN_FREE,
                        "HG5": {"median": HG5_MEDIAN, "le3": HG5_LE3}}}

    chair = ChairScorer(SYN)
    data = load_coco_sample(chair, n=500)
    by_id = {d["image_id"]: d for d in data}
    sids = sorted(by_id)
    pos = {i: j for j, i in enumerate(sids)}
    cells = json.load(open(f"{OUT}/q3_cells.json"))
    assert cells["sentinel"], "FATAL_NO_CENSUS_SENTINEL"
    CUT = cells["cut"]
    R["cut_f"] = CUT
    keep = sorted({c["image_id"] for c in cells["cells"]})[:N_GATE]
    print(f"[gate] CUT={CUT} gate images {len(keep)}", flush=True)

    # ---------------------------------------------------------------- CALIBRATION on LLaVA-1.5
    # Run HG-3 unchanged on the model that carries the paper's lead result. Loaded and freed
    # FIRST so that only one 7-8B checkpoint is resident at a time on the 24 GB card.
    import torch
    from gen_common import load_any, generate_spliced, gen_free
    m1, p1, t1_, meta1 = load_any("m1")
    print("[gate] calib model", json.dumps(meta1, default=str), flush=True)
    recs_g = [by_id[i] for i in keep]
    f1_free = gen_free("m1", m1, p1, t1_, recs_g)

    def cont_m1(items):
        pr = prompt_for("m1", p1, QUESTION)
        items = sorted(items, key=lambda it: len(it["prefix_ids"]))
        out = []
        from PIL import Image
        for s in range(0, len(items), BS_CONT):
            b = items[s:s + BS_CONT]
            ims = [Image.open(it["file"]).convert("RGB") for it in b]
            ids_l, _ = generate_spliced(m1, p1, t1_, ims, [pr] * len(b),
                                        prefix_ids=[it["prefix_ids"] for it in b],
                                        max_new=MAX_NEW_384, sample=False, seed=None)
            for it, g in zip(b, ids_l):
                r = {k: v for k, v in it.items() if k not in ("prefix_ids", "file")}
                r["cont_ids"] = g
                r["cont_text"] = t1_.decode(g, skip_special_tokens=True)
                r["cont_len"] = len(g)
                r["prefix_text"] = t1_.decode(it["prefix_ids"], skip_special_tokens=True)
                out.append(r)
        return out

    R["HG3_calibration_llava15"] = self_prefill_block("llava-1.5-7b", f1_free, CAL_CUT_M1,
                                                      cont_m1, by_id)
    R["HG3_calibration_llava15"]["free_median_len"] = float(
        np.median([r["gen_len"] for r in f1_free]))
    import gc, gen_common as _gc
    _gc._CACHE.pop("m1", None)
    del m1, p1, t1_
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[gate] cuda reserved after free: "
          f"{torch.cuda.memory_reserved()/2**30:.2f} GiB", flush=True)
    print(f"[gate] calibration done, m1 freed  t={time.time()-t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- the lane's own model
    model, proc, tok, meta = load_m5()
    R["meta"] = meta
    print("[gate] meta", json.dumps(meta, default=str), flush=True)
    assert meta["cls"] == "Qwen3VLForConditionalGeneration", f"FATAL_CLS {meta['cls']}"

    # ---- HG-1 TEMPLATE ---------------------------------------------------------------------
    frozen = json.load(open(f"{OUT}/q3_template.json"))
    p = prompt_for(WHICH, proc, QUESTION)
    p_nogen = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": QUESTION}]}],
        add_generation_prompt=False)
    hg1 = {"pass": bool(p == frozen["template"]), "template": p,
           "template_no_gen_prompt": p_nogen,
           "frozen_matches": bool(p == frozen["template"]),
           "used_apply_chat_template": True}
    assert isinstance(p_nogen, str), "FATAL_TEMPLATE_NOT_A_STRING"
    gen_suffix = p[len(p_nogen):] if p.startswith(p_nogen) else None
    hg1["generation_prompt_suffix"] = gen_suffix
    hg1["pass"] = bool(hg1["pass"] and gen_suffix is not None and len(gen_suffix) > 0)
    R["HG1_template"] = hg1
    print(f"[HG-1] {'PASS' if hg1['pass'] else 'FAIL'} suffix={gen_suffix!r}", flush=True)

    # ---- HG-5 stimulus + HG-2 / HG-6 all come from ONE assembled-prefill generation ---------
    items = []
    for iid in keep:
        rc = next(c for c in cells["cells"] if c["image_id"] == iid and c["k"] == 1)
        b = rc["budget"]
        srcs = sources(sids, pos, iid, 1)
        assert srcs == rc["srcs"], "FATAL_CELL_DRIFT"
        a_ids, a_txt, a_cyc = build_prefix_ids(tok, by_id[srcs[0]]["human_caps"], b,
                                               seed=iid * 7 + ASM_SEED_BASE + 1)
        per_src = [permuted(asm_units(by_id[s]["human_caps"]),
                            iid * 7 + S_CAPPICK + 1 + 10 * j) for j, s in enumerate(srcs)]
        mc = assemble_nodouble(tok, per_src, b, iid * 7 + S_JOIN_CAP + 1)
        assert len(a_ids) == len(mc["ids"]) == b, "FATAL_BUDGET_MISMATCH"
        for arm, pids in (("cap1", a_ids), ("cap5", mc["ids"])):
            items.append({"image_id": iid, "file": by_id[iid]["file"], "arm": arm,
                          "budget": b, "prefix_ids": list(pids)})
    rows = q3_cont(model, proc, tok, items, gray=False, max_new=MAX_NEW_384, bs=BS_CONT,
                   keep_diag=True)

    # ---- HG-2 PREFILL LANDS IN THE ASSISTANT TURN (on realised spliced ids) ------------------
    # Clause (ii) of the registered HG-2 -- "the prefill sits immediately after the
    # generation-prompt suffix" -- is tested in TOKEN-ID space, which is what the clause
    # actually means and what the model actually sees. Testing it on the decoded string instead
    # would make it hostage to how the tokenizer re-inserts whitespace around control tokens, a
    # decode artefact that has nothing to do with where the prefill landed. The string form is
    # kept and reported as a diagnostic, never as the pass condition. Amendment made before the
    # gate job started and recorded in QWEN3_PREREG.md section 8.
    spec = set(tok.all_special_ids)
    tmpl_ids = tok(p, add_special_tokens=False)["input_ids"]
    tmpl_specials = [s for s in sorted(spec) if s in set(tmpl_ids)]
    tmpl_counts = {str(s): tmpl_ids.count(s) for s in tmpl_specials}
    gs_ids = tok(gen_suffix, add_special_tokens=False)["input_ids"] if gen_suffix else None
    bad_special = bad_adj = bad_counts = bad_tail_str = 0
    ex = None
    for x in rows:
        ids = x.pop("_spliced_ids")
        pref_len = x["budget"]
        pre, tail = ids[:-pref_len], ids[-pref_len:]
        # (a) the prefill introduces no special / control token
        bad_special += bool(set(tail) & spec)
        # (b) AUTHORITATIVE: the generation-prompt ids sit immediately before the splice point
        if gs_ids is None or pre[-len(gs_ids):] != list(gs_ids):
            bad_adj += 1
        # (c) the prefill opens and closes no turn: non-image special counts conserved
        c = {str(s): pre.count(s) + tail.count(s) for s in tmpl_specials}
        nonimg = {s for s in tmpl_specials
                  if c[str(s)] != tmpl_counts[str(s)]
                  and "image" not in (tok.convert_ids_to_tokens(int(s)) or "").lower()
                  and "vision" not in (tok.convert_ids_to_tokens(int(s)) or "").lower()}
        bad_counts += bool(nonimg)
        # diagnostic only
        txt = tok.decode(ids, skip_special_tokens=False)
        tail_txt = tok.decode(tail, skip_special_tokens=False)
        bad_tail_str += bool(gen_suffix is None or not txt.endswith(gen_suffix + tail_txt))
        if ex is None:
            ex = {"decoded_tail": txt[-400:], "special_counts": c,
                  "template_counts": tmpl_counts, "nonconserved": sorted(nonimg),
                  "gen_suffix_ids": list(gs_ids) if gs_ids else None,
                  "pre_tail_ids": pre[-8:], "prefill_first_ids": tail[:8], "budget": pref_len}
    hg2 = {"pass": bool(bad_special == 0 and bad_adj == 0 and bad_counts == 0),
           "n_rows": len(rows), "rows_with_special_in_prefill": bad_special,
           "rows_prefill_not_after_generation_prompt": bad_adj,
           "rows_turn_markers_changed": bad_counts,
           "diagnostic_rows_failing_string_form": bad_tail_str, "example": ex}
    R["HG2_prefill_lands"] = hg2
    print(f"[HG-2] {'PASS' if hg2['pass'] else 'FAIL'} special={bad_special} "
          f"adjacency={bad_adj} counts={bad_counts} (string-form diag {bad_tail_str})",
          flush=True)
    print(f"[HG-2] example spliced tail {ex['decoded_tail']!r}", flush=True)

    # ---- HG-3 SELF-PREFILL, this model -------------------------------------------------------
    af = [r for r in jload(f"{OUT}/q3_af.jsonl") if r["image_id"] in set(keep)]
    assert len(af) == len(keep), f"FATAL_AF_SUBSET {len(af)} != {len(keep)}"
    R["HG3_self_continuation"] = self_prefill_block(
        "qwen3-vl-8b", af, CUT,
        lambda its: q3_cont(model, proc, tok, its, gray=False, max_new=MAX_NEW_384, bs=BS_CONT),
        by_id)

    # ---- HG-4 FREE-RUNNING LENGTH ------------------------------------------------------------
    allaf = jload(f"{OUT}/q3_af.jsonl")
    Lall = np.array([r["gen_len"] for r in allaf])
    hg4 = {"pass": bool(float(np.median(Lall)) >= HG4_MEDIAN_FREE),
           "n": len(Lall), "median": float(np.median(Lall)), "mean": float(Lall.mean()),
           "sd": float(Lall.std()), "min": int(Lall.min()), "max": int(Lall.max()),
           "frac_le3": float((Lall <= 3).mean()),
           "at_max_256": float((Lall >= MAX_NEW_FREE).mean()),
           "threshold_median": HG4_MEDIAN_FREE}
    R["HG4_free_length"] = hg4
    print(f"[HG-4] {'PASS' if hg4['pass'] else 'FAIL'} median {hg4['median']} "
          f"mean {hg4['mean']:.2f}", flush=True)

    # ---- HG-5 ASSEMBLED-PREFILL SANITY -------------------------------------------------------
    hg5 = {"thresholds": {"median": HG5_MEDIAN, "le3": HG5_LE3}, "arms": {}}
    ok = True
    for a in ("cap1", "cap5"):
        L = [x["cont_len"] for x in rows if x["arm"] == a]
        d = {"n": len(L), "median_cont_len": float(np.median(L)),
             "mean_cont_len": float(np.mean(L)), "frac_le3": float(np.mean([v <= 3 for v in L])),
             "empty_rate": float(np.mean([v == 0 for v in L])),
             "at_max_384": float(np.mean([v >= MAX_NEW_384 for v in L]))}
        d["pass"] = bool(d["median_cont_len"] >= HG5_MEDIAN and d["frac_le3"] <= HG5_LE3)
        ok = ok and d["pass"]
        hg5["arms"][a] = d
        print(f"[HG-5:{a}] median {d['median_cont_len']} le3 {d['frac_le3']:.3f} "
              f"empty {d['empty_rate']:.3f} {'PASS' if d['pass'] else 'FAIL'}", flush=True)
    hg5["pass"] = bool(ok)
    R["HG5_assembled"] = hg5
    for x in rows[:2]:
        print(f"[HG-5] ARM {x['arm']} PFX {x['prefix_text'][-140:]!r}", flush=True)
        print(f"[HG-5] ARM {x['arm']} CNT {x['cont_text'][:240]!r}", flush=True)

    # ---- HG-6 PREFIX BINDS -------------------------------------------------------------------
    freeby = {r["image_id"]: r["gen_ids"] for r in af}
    diff = 0
    for x in rows:
        f = freeby[x["image_id"]]
        diff += (list(x["cont_ids"]) != list(f[:len(x["cont_ids"])]))
    hg6 = {"pass": bool(diff >= len(rows) - 1), "n_differ": diff, "n": len(rows)}
    R["HG6_prefix_binds"] = hg6
    print(f"[HG-6] {'PASS' if hg6['pass'] else 'FAIL'} {diff}/{len(rows)}", flush=True)

    # ---- VERDICT -----------------------------------------------------------------------------
    hard = R["HG1_template"]["pass"] and R["HG2_prefill_lands"]["pass"]
    selfc = R["HG3_self_continuation"]["pass"]
    calib = R["HG3_calibration_llava15"]["pass"]
    if not calib:
        verdict = "GATE-INVALID"
        why = ("HG-3 thresholds reject LLaVA-1.5-7B, the model carrying the paper's lead "
               "result. The gate, not the model, is wrong; thresholds must be re-derived "
               "before this lane reports anything.")
    elif not hard:
        verdict = "HARNESS-BROKEN"
        why = "HG-1/HG-2 failed: the prompt or the prefill placement is wrong for this model."
    elif not selfc:
        verdict = "HARNESS-OR-MODEL-CANNOT-CONTINUE"
        why = ("HG-3 failed under a template and prefill placement that HG-1/HG-2 verify and "
               "that LLaVA-1.5-7B passes: the model does not re-emit its own caption remainder "
               "from its own prefix. No endpoint is run.")
    elif not (R["HG4_free_length"]["pass"] and R["HG5_assembled"]["pass"]
              and R["HG6_prefix_binds"]["pass"]):
        verdict = "RUN-ENDPOINT-DEGENERACY-ATTRIBUTED"
        why = ("HG-1/HG-2/HG-3 pass, so the harness continues a pre-filled answer correctly on "
               "this model. A HG-4/HG-5/HG-6 failure is therefore a property of the stimulus or "
               "the model, not of our prompt. The endpoint runs and the row is reported with "
               "this evidence.")
    else:
        verdict = "RUN-ENDPOINT"
        why = "All harness gates pass."
    import torch as _t
    R["gpu_peak_gib"] = round(_t.cuda.max_memory_allocated() / 2**30, 2)
    R["gpu_total_gib"] = round(_t.cuda.get_device_properties(0).total_memory / 2**30, 2)
    print(f"[gate] GPU peak allocated {R['gpu_peak_gib']} GiB of "
          f"{R['gpu_total_gib']} GiB", flush=True)
    R["VERDICT"] = verdict
    R["why"] = why
    R["secs"] = round(time.time() - t0, 1)
    json.dump(R, open(f"{OUT}/q3_gate.json", "w"), indent=1)
    print(f"[gate] VERDICT = {verdict}\n[gate] {why}", flush=True)
    sentinel("GATE")


if __name__ == "__main__":
    main()
