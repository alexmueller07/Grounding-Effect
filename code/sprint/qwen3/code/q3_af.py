"""Lane `qwen3` stage `af`: the model's own free-running captions, real and gray.

Port of f3_af.py (WIA-FAM3 section 8.1 step 2 / 3.2), model swapped.

  real -> out/q3_af.jsonl       gives L_i, the per-cell budget base of the census
  gray -> out/q3_af_gray.jsonl  the blind arm of GATE_ATTENDS

GATE_ATTENDS: rec = |objset(caption) & G(i)| / |G(i)|, defined for EVERY image because G(i) is
non-empty by the eligibility rule. Registered statistic Delta_rec = mean rec(real) -
mean rec(gray), paired image bootstrap, B = 4000. PASSES iff lower bound > 0 AND the
identical-caption rate is <= 0.50.

No prefix is constructed here and no `gap` is computed.
"""
import os, sys, json, time
sys.path.insert(0, "/data/alexmueller/sprint/qwen3/code")
import numpy as np
from q3_common import (M5, M5_SHA, OUT, WHICH, QUESTION, load_m5, q3_free, load_coco_sample,
                       ChairScorer, SYN, MIN_LEN, BS_FREE, MAX_NEW_FREE, jdump, jload, sentinel)
from gen_common import prompt_for

NBOOT = 4000
SEED = 20260919
SMOKE = os.environ.get("Q3_SMOKE", "0") == "1"
NS = 8


def main():
    t0 = time.time()
    assert os.uname().nodename == "vgi1", f"FATAL_WRONG_NODE {os.uname().nodename}"
    chair = ChairScorer(SYN)
    recs = load_coco_sample(chair, n=500)
    if SMOKE:
        recs = recs[:NS]
    objset = lambda t: set(n for (n, _c) in chair.mentions(t))
    gt = {r["image_id"]: set(r["gt"]) for r in recs}

    model, proc, tok, meta = load_m5()
    print("[af] meta", json.dumps(meta, default=str), flush=True)
    assert meta["model"] == M5 and meta["revision"] == M5_SHA, "FATAL_WRONG_MODEL"
    assert meta["cls"] == "Qwen3VLForConditionalGeneration", f"FATAL_CLS {meta['cls']}"
    assert 7.0e9 < meta["n_params"] < 9.5e9, f"FATAL_PARAM_COUNT {meta['n_params']}"

    frozen = json.load(open(f"{OUT}/q3_template.json"))
    assert frozen["sentinel"], "FATAL_NO_TEMPLATE_SENTINEL"
    p = prompt_for(WHICH, proc, QUESTION)
    assert p == frozen["template"], f"FATAL_TEMPLATE_DRIFT {p!r}"
    print("[af] PROMPT", repr(p), flush=True)

    real = q3_free(model, proc, tok, recs, gray=False, bs=BS_FREE, max_new=MAX_NEW_FREE)
    print(f"[af] real done {time.time()-t0:.0f}s", flush=True)
    gray = q3_free(model, proc, tok, recs, gray=True, bs=BS_FREE, max_new=MAX_NEW_FREE)
    print(f"[af] gray done {time.time()-t0:.0f}s", flush=True)
    sfx = "_smoke" if SMOKE else ""
    jdump(f"{OUT}/q3_af{sfx}.jsonl", real)
    jdump(f"{OUT}/q3_af_gray{sfx}.jsonl", gray)

    L = np.array([r["gen_len"] for r in real])
    Lg = np.array([r["gen_len"] for r in gray])
    short = [r["image_id"] for r in real if r["gen_len"] < MIN_LEN]
    ident = [a["image_id"] for a, b in zip(real, gray) if a["gen_ids"] == b["gen_ids"]]

    rr, rg, pr, pg = [], [], [], []
    for a, b in zip(real, gray):
        oa, ob = objset(a["text"]), objset(b["text"])
        g = gt[a["image_id"]]
        assert g, f"FATAL_EMPTY_GT {a['image_id']}"
        rr.append(len(oa & g) / len(g)); rg.append(len(ob & g) / len(g))
        if oa and ob:
            pr.append(len(oa & g) / len(oa)); pg.append(len(ob & g) / len(ob))
    rr, rg = np.array(rr), np.array(rg)
    pr, pg = np.array(pr), np.array(pg)
    d = float(rr.mean() - rg.mean())
    rng = np.random.default_rng(SEED)
    n = len(rr)
    vals = []
    for b in range(NBOOT):
        sel = rng.integers(0, n, n)
        assert len(sel) == n, "FATAL_BOOTSTRAP_LENGTH"
        if b < 20 and n >= 20:
            assert len(set(sel.tolist())) < n, "FATAL_BOOTSTRAP_NO_DUPLICATES"
        vals.append(rr[sel].mean() - rg[sel].mean())
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    ident_rate = len(ident) / len(real)
    gate = bool(lo > 0 and ident_rate <= 0.50)

    out = {"sentinel": True, "model": M5, "revision": M5_SHA, "n_images": len(real),
           "smoke": SMOKE, "prompt": p, "bs_free": BS_FREE, "max_new_free": MAX_NEW_FREE,
           "free_real": {"mean_len": float(L.mean()), "median_len": float(np.median(L)),
                         "sd": float(L.std()), "min": int(L.min()), "max": int(L.max()),
                         "at_max_256": float((L >= MAX_NEW_FREE).mean()),
                         "frac_le3": float((L <= 3).mean()),
                         "n_below_MIN_LEN": len(short), "below_MIN_LEN_ids": short[:50]},
           "free_gray": {"mean_len": float(Lg.mean()), "median_len": float(np.median(Lg)),
                         "sd": float(Lg.std()), "min": int(Lg.min()), "max": int(Lg.max()),
                         "at_max_256": float((Lg >= MAX_NEW_FREE).mean()),
                         "frac_le3": float((Lg <= 3).mean())},
           "GATE_ATTENDS": {"pass": gate, "statistic": "delta_recall", "delta_recall": d,
                            "ci": [lo, hi], "mean_recall_real": float(rr.mean()),
                            "mean_recall_gray": float(rg.mean()), "n_images_scored": n,
                            "descriptive_precision": {
                                "mean_prec_real": float(pr.mean()) if len(pr) else None,
                                "mean_prec_gray": float(pg.mean()) if len(pg) else None,
                                "n_pairs_both_define_precision": int(len(pr))},
                            "identical_caption_rate": ident_rate,
                            "identical_ids": ident[:50], "nboot": NBOOT, "seed": SEED,
                            "rule": "PASS iff ci_lo > 0 and identical_caption_rate <= 0.50"},
           "mean_objects_per_caption": {
               "real": float(np.mean([len(objset(r["text"])) for r in real])),
               "gray": float(np.mean([len(objset(r["text"])) for r in gray]))},
           "secs": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/q3_attends{sfx}.json", "w"), indent=1)
    print("[af] GATE_ATTENDS", json.dumps(out["GATE_ATTENDS"]), flush=True)
    print("[af] free_real", json.dumps(out["free_real"]), flush=True)
    print("[af] free_gray", json.dumps(out["free_gray"]), flush=True)
    for i in range(min(2, len(real))):
        print(f"[af] REAL[{i}] {real[i]['text'][:300]!r}", flush=True)
        print(f"[af] GRAY[{i}] {gray[i]['text'][:300]!r}", flush=True)
    sentinel("AF")


if __name__ == "__main__":
    main()
