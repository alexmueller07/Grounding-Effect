"""Lane `qwen3` stage 0: fetch the checkpoint and FREEZE the chat template.

Downloads Qwen/Qwen3-VL-8B-Instruct at the pinned revision into HF_HOME, reports `df -h /`
before and after (the volume is shared and near full), and writes out/q3_template.json:
the prompt string produced by the PROCESSOR'S OWN `apply_chat_template(..., add_generation_
prompt=True)`. Every later stage byte-compares its prompt against that frozen string, so the
template can never drift between the gate and the endpoint, and it is never hand-rolled.

No model weights are moved to a GPU here, nothing is generated, no endpoint is touched.
"""
import os, sys, json, time, shutil, subprocess
sys.path.insert(0, "/data/alexmueller/sprint/qwen3/code")
from q3_common import M5, M5_SHA, OUT, WHICH, QUESTION, sentinel

MIN_FREE_GB_AFTER = 25.0     # refuse to leave the shared volume below this


def dfroot():
    t = shutil.disk_usage("/")
    return {"total_gb": round(t.total / 2**30, 2), "used_gb": round(t.used / 2**30, 2),
            "free_gb": round(t.free / 2**30, 2),
            "pct_used": round(100.0 * t.used / t.total, 2)}


def main():
    t0 = time.time()
    before = dfroot()
    print("[fetch] df / BEFORE", json.dumps(before), flush=True)
    print(subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout, flush=True)

    from huggingface_hub import snapshot_download, HfApi
    info = HfApi().model_info(M5, revision=M5_SHA, files_metadata=True)
    size = sum(f.size or 0 for f in info.siblings)
    print(f"[fetch] repo {M5}@{M5_SHA[:12]} {len(info.siblings)} files "
          f"{size/2**30:.3f} GiB", flush=True)
    assert info.sha == M5_SHA, f"FATAL_REVISION_DRIFT {info.sha}"
    need = size / 2**30
    assert before["free_gb"] - need > MIN_FREE_GB_AFTER, (
        f"FATAL_DISK_WOULD_DROP_BELOW_FLOOR free={before['free_gb']} need={need:.2f} "
        f"floor={MIN_FREE_GB_AFTER}")

    path = snapshot_download(M5, revision=M5_SHA)
    print(f"[fetch] snapshot at {path}  t={time.time()-t0:.0f}s", flush=True)

    from transformers import AutoProcessor, AutoConfig
    cfg = AutoConfig.from_pretrained(M5, revision=M5_SHA)
    proc = AutoProcessor.from_pretrained(M5, revision=M5_SHA)
    tok = proc.tokenizer
    from gen_common import prompt_for
    p = prompt_for(WHICH, proc, QUESTION)
    assert isinstance(p, str), f"FATAL_TEMPLATE_NOT_A_STRING {type(p)}"
    print("[fetch] TEMPLATE", repr(p), flush=True)

    # Structural template assertions -- fixed in the registration, not chosen after seeing it.
    assert p.rstrip().endswith("assistant") or p.endswith("assistant\n") or \
        "assistant" in p.split("<|im_start|>")[-1], "FATAL_TEMPLATE_NO_TRAILING_ASSISTANT_TURN"
    assert QUESTION in p, "FATAL_TEMPLATE_NO_QUESTION"
    ws = tok.convert_ids_to_tokens(tok(" cat", add_special_tokens=False)["input_ids"])[0]
    print(f"[fetch] word-start marker for ' cat' -> {ws!r}", flush=True)

    out = {"sentinel": True, "model": M5, "revision": M5_SHA, "template": p,
           "question": QUESTION, "which": WHICH,
           "tok_class": type(tok).__name__, "proc_class": type(proc).__name__,
           "vocab": len(tok), "pad_id": tok.pad_token_id, "eos_id": tok.eos_token_id,
           "word_start_token_for_space_cat": ws,
           "cfg_class": type(cfg).__name__,
           "text_config": {k: v for k, v in
                           (cfg.text_config.to_dict().items() if hasattr(cfg, "text_config")
                            else [])
                           if k in ("num_hidden_layers", "hidden_size", "num_attention_heads",
                                    "num_key_value_heads", "rope_scaling", "vocab_size")},
           "repo_bytes": size, "repo_gib": round(size / 2**30, 3),
           "df_before": before, "snapshot_path": path}
    out["df_after"] = dfroot()
    print("[fetch] df / AFTER", json.dumps(out["df_after"]), flush=True)
    print(subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout, flush=True)
    assert out["df_after"]["free_gb"] > MIN_FREE_GB_AFTER, "FATAL_DISK_BELOW_FLOOR_AFTER"
    out["secs"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/q3_template.json", "w"), indent=1)
    sentinel("FETCH")


if __name__ == "__main__":
    main()
