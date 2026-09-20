"""Load-path equivalence check.

q3_common.load_m5 now streams the checkpoint straight to the GPU instead of materialising it
on the CPU first, so the lane's jobs reserve ~12 GB of host RAM on a node four other lanes are
sharing rather than ~32 GB. The argument that this is placement-only (the checkpoint is natively
bfloat16, so `dtype` casts nothing on either path) is an argument, not evidence.

This asserts the evidence: regenerate the 8 free-running captions of the plumbing smoke, which
was produced by the OLD `.to("cuda")` path at the same batch size, and require the greedy token
ids to be BYTE-IDENTICAL row for row. Not "similar": identical. If they are not, the lane
reverts to the old path and pays the host RAM.
"""
import os, sys, json, time
sys.path.insert(0, "/data/alexmueller/sprint/qwen3/code")
from q3_common import (M5, OUT, load_m5, q3_free, load_coco_sample, ChairScorer, SYN,
                       BS_FREE, MAX_NEW_FREE, jload, sentinel)

N = 8


def main():
    t0 = time.time()
    assert os.uname().nodename == "vgi1", f"FATAL_WRONG_NODE {os.uname().nodename}"
    ref = jload(f"{OUT}/q3_af_smoke.jsonl")
    assert len(ref) == N, f"FATAL_REF_ROWS {len(ref)}"
    chair = ChairScorer(SYN)
    recs = load_coco_sample(chair, n=500)[:N]
    model, proc, tok, meta = load_m5()
    print("[loadeq] meta", json.dumps(meta, default=str), flush=True)
    assert meta["load_path"] == "device_map_cuda0", f"FATAL_LOAD_PATH {meta['load_path']}"
    new = q3_free(model, proc, tok, recs, gray=False, bs=BS_FREE, max_new=MAX_NEW_FREE)
    refby = {r["image_id"]: r["gen_ids"] for r in ref}
    ident = sum(1 for r in new if list(r["gen_ids"]) == list(refby[r["image_id"]]))
    lens = [r["gen_len"] for r in new]
    import torch
    out = {"sentinel": True, "n": len(new), "identical": ident,
           "pass": bool(ident == len(new)), "lens_new": lens,
           "lens_ref": [r["gen_len"] for r in ref], "meta": meta,
           "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
           "secs": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/q3_loadeq.json", "w"), indent=1)
    print(f"[loadeq] byte-identical {ident}/{len(new)}  GPU peak {out['gpu_peak_gib']} GiB "
          f"of {out['gpu_total_gib']}  {'PASS' if out['pass'] else 'FAIL'}", flush=True)
    assert out["pass"], f"FATAL_LOAD_PATH_CHANGES_OUTPUT {ident}/{len(new)}"
    sentinel("LOADEQ")


if __name__ == "__main__":
    main()
