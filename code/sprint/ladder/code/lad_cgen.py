"""L2 -- the ablation ladder on standard CHAIR (LADDER_PREREG.md section 4).

Appendix G's setup, unchanged: LLaVA-1.5-7B, 500 COCO val2017 images, PAI's own prompt,
greedy, 512-token budget, alpha in {0.0, 0.5}, gamma in {1.0, 1.1}, layers [2, 32).

The decode loop is `md_gen.md_two_stream`, IMPORTED, not re-typed -- it carries pa_gen's gated
attention kernel and its mandatory causality repair. The ONLY change here is the pixels:
`encode()` applies a ladder rung in the encoder's canonical geometry instead of flat grey.
Nothing under mitdecomp/, paialpha/ or j5_gates/ is written.

ENV
  LAD_CARM     vanilla | pai05            (default vanilla)
  LAD_CRUNGS   comma-separated rung names (default: every rung)
  LAD_CLIMIT   optional int, first N images -- gates and PC4 only, tagged so it can never be
               mistaken for a registered arm
  LAD_CBS      batch size, default 4 (md_sighted.sbatch's own MD_BS)

Writes out/lad_c_<arm>_<rung>[_limN].jsonl + _meta.json. Sentinel printed by Python.
"""
import os, sys, json, time, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/alexmueller/mitdecomp/code")
sys.path.insert(0, "/data/alexmueller/paialpha/code")
sys.path.insert(0, "/data/alexmueller/sc1_interv/code")
sys.path.insert(0, "/data/alexmueller/j5_gates/code")
os.environ.setdefault("PA_ARM", "vanilla")

import numpy as np
import torch

from lad_common import (RungImages, ALL_RUNGS, RUNGS, UNIFORM_RUNGS, OUT, sentinel,
                        jdump, canon, derangement, MISMATCH_SEED)
from md_common import (MODEL, SYN, N_IMAGES, SAMPLE_SEED, MAX_NEW, LAYER_LO, LAYER_HI,
                       ARMS, prompt_text)
from md_gen import md_two_stream
from pa_gen import ST, load_model_pa, build_uncond
from j5_common import load_coco_sample, ChairScorer, JsonlWriter, image_token_id

CBS = int(os.environ.get("LAD_CBS", "4"))


def encode(proc, chunk, imgs):
    ims = [imgs.get(d["file"], d["image_id"]) for d in chunk]
    enc = proc(images=ims, text=[prompt_text()] * len(chunk), return_tensors="pt",
               padding=True)
    return {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda"))
            for k, v in enc.items()}


def clean_pix(proc, chunk):
    from PIL import Image
    return proc.image_processor(
        images=[canon(proc, Image.open(d["file"]).convert("RGB")) for d in chunk],
        return_tensors="pt")["pixel_values"]


def run_arm(arm, rung, model, proc, tok, data, limit, chair, sub=None):
    spec = ARMS[arm]
    alpha, gamma = spec["alpha"], spec["gamma"]
    assert 0.0 <= alpha < 1.0, f"FATAL_ALPHA_OUT_OF_BOUND {alpha}"
    tag = f"lad_c_{arm}_{rung}" + (f"_lim{limit}" if limit else "")
    path, mpath = f"{OUT}/{tag}.jsonl", f"{OUT}/{tag}_meta.json"
    want = len(data)
    have = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    mn = -1
    if os.path.exists(mpath):
        try:
            mn = json.load(open(mpath)).get("n", -1)
        except Exception:
            mn = -1
    if have == want and mn == want:
        print(f"[lad_cgen] {tag} already complete ({have} rows), skipping", flush=True)
        return None
    print(f"[lad_cgen] === {tag} alpha={alpha} gamma={gamma} (have {have}, meta {mn}) ===",
          flush=True)

    imgs = RungImages(proc, rung, sub=sub)
    submeta = None
    if rung == "mismatch":
        assert sub, "FATAL_MISMATCH_WITHOUT_SUBSTITUTION_MAP"
        assert all(int(v[0]) != int(k) for k, v in sub.items()), "FATAL_MISMATCH_SELF_PAIRED"
        assert len({v[0] for v in sub.values()}) == len(sub), "FATAL_MISMATCH_NOT_BIJECTIVE"
        submeta = {"n": len(sub), "seed": MISMATCH_SEED,
                   "map_sample": {str(k): int(v[0]) for k, v in list(sub.items())[:10]}}
        print(f"[lad_cgen] GATE_MISMATCH_DERANGED ok, n={len(sub)}", flush=True)
    pad, eos = tok.pad_token_id, tok.eos_token_id
    w = JsonlWriter(path)
    t0 = time.time()
    n_trunc = 0
    gate = {"rows": 0, "rows_changed": 0, "pairs": 0, "cross_image_distinct": 0,
            "grey_rows_equal": 0}
    grey_ref = None
    for b0 in range(0, len(data), CBS):
        chunk = data[b0:b0 + CBS]
        enc = encode(proc, chunk, imgs)
        pix = enc["pixel_values"]

        if b0 == 0:
            cp = clean_pix(proc, chunk).to(pix.device, pix.dtype)
            for i in range(len(chunk)):
                gate["rows"] += 1
                gate["rows_changed"] += int(not bool(torch.equal(pix[i], cp[i])))
            for i in range(len(chunk) - 1):
                gate["pairs"] += 1
                gate["cross_image_distinct"] += int(not bool(torch.equal(pix[i], pix[i + 1])))
            assert gate["rows_changed"] == gate["rows"], f"FATAL_RUNG_NOT_APPLIED {rung} {gate}"
            if rung in UNIFORM_RUNGS:
                assert gate["cross_image_distinct"] == 0, \
                    f"FATAL_UNIFORM_RUNG_NOT_CONSTANT {rung} {gate}"
            else:
                assert gate["cross_image_distinct"] == gate["pairs"], \
                    f"FATAL_RUNG_CONSTANT_ACROSS_IMAGES {rung} {gate}"
            print(f"[lad_cgen] GATE_RUNG_APPLIED {rung} {json.dumps(gate)}", flush=True)
        if rung in UNIFORM_RUNGS:
            # md_gen's own G6: a flat grey of any input size resizes to the SAME tensor.
            if grey_ref is None:
                grey_ref = pix[0].clone()
            assert bool((pix == grey_ref[None]).all()), "FATAL_UNIFORM_PIX_ROWS_DIFFER"
            gate["grey_rows_equal"] += len(chunk)

        ids_u, am_u = build_uncond(enc["input_ids"], enc["attention_mask"], ST["img_tok"], pad)
        gen, mass = md_two_stream(model, tok, enc, pix, ids_u, am_u, alpha, gamma,
                                  MAX_NEW, eos, pad, pix_u=None)
        for i, d in enumerate(chunk):
            ids = [int(x) for x in gen[i].tolist()]
            hit_eos = eos in ids
            ids = ids[:ids.index(eos)] if hit_eos else ids
            ids = [x for x in ids if x != pad]
            text = tok.decode(ids, skip_special_tokens=True)
            trunc = (not hit_eos) and len(ids) >= MAX_NEW
            n_trunc += int(trunc)
            w.write({"image_id": d["image_id"], "arm": arm, "rung": rung, "alpha": alpha,
                     "gamma": gamma, "second": "noimg", "layers": [LAYER_LO, LAYER_HI],
                     "text": text, "gen_ids": ids, "n_tokens": len(ids),
                     "hit_eos": bool(hit_eos), "truncated": bool(trunc), "gt": d["gt"],
                     "sub_image_id": (None if sub is None
                                      else int(sub[int(d["image_id"])][0])),
                     "img_mass": [float(x) for x in mass[i]]})
        if (b0 // CBS) % 25 == 0 or b0 + len(chunk) >= len(data):
            print(f"[lad_cgen] {tag} {min(b0+CBS,len(data))}/{len(data)} "
                  f"{time.time()-t0:.0f}s trunc={n_trunc}", flush=True)
    w.close()

    rows = [json.loads(l) for l in open(path) if l.strip()]
    assert len(rows) == len(data), f"FATAL_ROWS {len(rows)} != {len(data)}"
    ment = [chair.mentions(r["text"]) for r in rows]
    occ = int(sum(len(m) for m in ment))
    ncap = int(sum(1 for m in ment if m))
    hal = int(sum(1 for m, r in zip(ment, rows) for (nd, _c) in m if nd not in set(r["gt"])))
    meta = {"sentinel": True, "arm": arm, "rung": rung, "alpha": alpha, "gamma": gamma,
            "layers": [LAYER_LO, LAYER_HI], "model": MODEL, "n": len(rows),
            "max_new": MAX_NEW, "sample_seed": SAMPLE_SEED, "n_truncated": n_trunc,
            "prompt": prompt_text(), "bs": CBS, "limit": limit,
            "gate_rung_applied": gate, "rung_cache": imgs.stats(), "mismatch": submeta,
            "attn_calls": ST["calls"], "pai_applied_steps": ST["applied"],
            "n_edited_first": ST["n_edited"], "mask_present": ST["mask_present"],
            "causal_repairs": ST["causal_repairs"], "nomask_decode": ST["nomask_decode"],
            "mask_leak": ST["mask_leak"],
            "mention_occurrences": occ, "captions_with_mention": ncap,
            "hallucinated_occurrences": hal,
            "chair_i_pooled": (None if occ == 0 else 100.0 * hal / occ),
            "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])),
            "median_tokens": float(np.median([r["n_tokens"] for r in rows])),
            "n_distinct_texts": len({r["text"] for r in rows}),
            "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "prereg": "SPRINT/LADDER_PREREG.md @ f186bcf", "secs": time.time() - t0}
    assert ST["mask_leak"] == 0, f"FATAL_MASK_LEAK {ST['mask_leak']}"
    assert ST["mask_present"] + ST["causal_repairs"] > 0, "FATAL_NO_PREFILL_SEEN"
    if alpha > 0:
        assert ST["applied"] > 0, "FATAL_PAI_NEVER_APPLIED"
    jdump(mpath, meta)
    print(f"[lad_cgen] {tag} occ={occ} caps_with_mention={ncap}/{len(rows)} "
          f"chair_i={meta['chair_i_pooled']} median_tok={meta['median_tokens']} "
          f"distinct={meta['n_distinct_texts']} secs={meta['secs']:.0f}", flush=True)
    return meta


def main():
    assert socket.gethostname().startswith("vgi1"), f"FATAL_WRONG_NODE {socket.gethostname()}"
    arm = os.environ.get("LAD_CARM", "vanilla")
    assert arm in ("vanilla", "pai05"), f"FATAL_UNKNOWN_ARM {arm}"
    rungs = [r.strip() for r in
             os.environ.get("LAD_CRUNGS", ",".join(RUNGS)).split(",") if r.strip()]
    for r in rungs:
        assert r in ALL_RUNGS, f"FATAL_UNKNOWN_RUNG {r}"
    assert "identity" not in rungs, "FATAL_IDENTITY_IS_THE_SIGHTED_ARM"
    limit = int(os.environ.get("LAD_CLIMIT", "0"))

    print(f"[lad_cgen] host={socket.gethostname()} arm={arm} rungs={rungs} limit={limit} "
          f"bs={CBS} max_new={MAX_NEW}", flush=True)
    chair = ChairScorer(SYN)
    data = load_coco_sample(chair, n=N_IMAGES, seed=SAMPLE_SEED)
    data.sort(key=lambda d: d["image_id"])
    if limit:
        data = data[:limit]
    assert data, "FATAL_NO_ITEMS"

    model, proc, tok = load_model_pa()
    ST["img_tok"] = image_token_id(model)
    sub = None
    if "mismatch" in rungs:
        by_id = {int(d["image_id"]): d for d in data}
        dg = derangement(sorted(by_id))
        sub = {int(k): (int(v), by_id[int(v)]["file"]) for k, v in dg.items()}
    out = []
    for r in rungs:
        m = run_arm(arm, r, model, proc, tok, data, limit, chair,
                    sub=(sub if r == "mismatch" else None))
        out.append({"rung": r, "skipped": m is None,
                    "occ": None if m is None else m["mention_occurrences"],
                    "caps": None if m is None else m["captions_with_mention"]})
        print(f"[lad_cgen] progress {json.dumps(out)}", flush=True)
    sentinel("GEN_C_LIM" if limit else "GEN_C")


if __name__ == "__main__":
    main()
