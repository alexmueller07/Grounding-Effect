"""POPE lane -- item list and the batch plan.

THE BATCH PLAN IS PART OF THE EXPERIMENT.  bf16 attention is batch-shape sensitive, and this
project has already had a replay criterion broken by a different batch shape.  Items are
therefore grouped by EXACT prompt token length, so every batch is rectangular with ZERO
padding, and the plan depends only on the text -- never on the pixels, the method or the
arm.  Every arm therefore runs byte-identical token tensors in byte-identical batches.
"""
import hashlib
import json
import os

from pope_common import DATA, IMGDIR, POPE_SHA256, SPLITS, prompt_text


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_items(splits=SPLITS, limit_images=0):
    """The official POPE rows, checksum-gated, in a fixed order."""
    items = []
    for si, s in enumerate(splits):
        p = f"{DATA}/coco_pope_{s}.json"
        got = _sha256(p)
        assert got == POPE_SHA256[s], f"FATAL_POPE_FILE_CHANGED {s} {got}"
        rows = [json.loads(l) for l in open(p)]
        assert len(rows) == 3000, f"FATAL_POPE_N {s} {len(rows)}"
        for r in rows:
            assert r["label"] in ("yes", "no"), f"FATAL_LABEL {r}"
            items.append({
                "split": s, "split_idx": si, "question_id": int(r["question_id"]),
                "image": r["image"], "image_id": int(r["image"].split("_")[-1].split(".")[0]),
                "text": r["text"], "label": r["label"],
            })
    if limit_images:
        keep = sorted({it["image_id"] for it in items})[:limit_images]
        keep = set(keep)
        items = [it for it in items if it["image_id"] in keep]
    for it in items:
        f = os.path.join(IMGDIR, it["image"])
        assert os.path.exists(f), f"FATAL_MISSING_IMAGE {f}"
        it["file"] = f
    return items


def batch_plan(items, tok, bs):
    """Group by exact prompt token length -> rectangular, zero-padding batches.
    Deterministic and text-only, so identical in every arm."""
    lens = {}
    for i, it in enumerate(items):
        n = len(tok(prompt_text(it["text"]))["input_ids"])
        lens.setdefault(n, []).append(i)
    plan = []
    for n in sorted(lens):
        idx = sorted(lens[n], key=lambda i: (items[i]["split_idx"], items[i]["question_id"]))
        for b0 in range(0, len(idx), bs):
            plan.append(idx[b0:b0 + bs])
    total = sum(len(b) for b in plan)
    assert total == len(items), f"FATAL_PLAN_SIZE {total} != {len(items)}"
    h = hashlib.sha256(json.dumps(plan).encode()).hexdigest()
    return plan, h, {str(k): len(v) for k, v in sorted(lens.items())}
