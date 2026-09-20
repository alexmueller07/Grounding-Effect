"""Dry-run pope_score.py end-to-end on SYNTHETIC cells. Nothing here touches out/."""
import hashlib, json, os, shutil, sys
sys.path.insert(0, "/data/alexmueller/sprint/pope/code")
import numpy as np
from pope_items import load_items
import pope_score as S

TMP = "/data/alexmueller/sprint/pope/_scoretest_out"
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)
S.OUT = TMP

items = load_items()
rng = np.random.default_rng(7)
# p(answer yes) per arm; pai arms get a small gain, mismatch sits near the answer prior
P = {("vanilla", "sighted"): (0.88, 0.12), ("vanilla", "grey"): (0.63, 0.60),
     ("vanilla", "mismatch"): (0.66, 0.58),
     ("pai_full", "sighted"): (0.89, 0.09), ("pai_full", "grey"): (0.64, 0.59),
     ("pai_full", "mismatch"): (0.67, 0.56),
     ("pai_attn", "sighted"): (0.87, 0.10), ("pai_attn", "grey"): (0.62, 0.61),
     ("pai_attn", "mismatch"): (0.65, 0.58),
     # deliberately DEGENERATE: constant-yes, to prove the screen and the trap-guard fire
     ("pai_both", "sighted"): (0.90, 0.08), ("pai_both", "grey"): (1.00, 1.00),
     ("pai_both", "mismatch"): (0.68, 0.55)}
for (meth, arm), (ph, pf) in P.items():
    rows = []
    for it in items:
        p = ph if it["label"] == "yes" else pf
        yes = rng.random() < p
        ans = "Yes" if yes else "No"
        rows.append({"split": it["split"], "question_id": it["question_id"],
                     "image": it["image"], "image_id": it["image_id"],
                     "label": it["label"], "question": it["text"],
                     "method": meth, "pixarm": arm, "answer": ans,
                     "gen_ids": [1, 2], "n_tokens": 2, "hit_eos": True, "truncated": False,
                     "img_mass": 0.2,
                     "pid_sha": hashlib.sha256(
                         json.dumps([it["split"], it["question_id"]]).encode()
                     ).hexdigest()[:16]})
    rows.sort(key=lambda r: (r["split"], r["question_id"]))
    t = f"pope_{meth}_{arm}"
    with open(f"{TMP}/{t}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    json.dump({"tag": t, "method": meth, "pixarm": arm, "n_rows": len(rows),
               "trunc_rate": 0.0, "empty_rate": 0.0, "mean_tokens": 2.0,
               "median_tokens": 2.0, "n_distinct_answers": 2, "mean_img_mass": 0.2,
               "pai_applied_steps": 0 if meth == "vanilla" else 500,
               "pai_applied_prefill": 0,
               "answer_dist": Counter(r["answer"] for r in rows).most_common(20),
               "answer_dist_by_split": {}},
              open(f"{TMP}/{t}_meta.json", "w"))
print("synthetic cells written:", len(P))
S.main()
res = json.load(open(f"{TMP}/pope_results.json"))
print("\n--- identity ---", res["identity_check"])
print("--- PRIMARY ---", json.dumps(res["verdict"]["PRIMARY"], indent=1))
print("--- grey verdict ---", res["verdict"]["PRIMARY_grey"]["outcome"])
print("--- mismatch verdict ---", res["verdict"]["COPRIMARY_mismatch"]["outcome"])
print("--- pai_both grey (should be BLIND-ARM-DEGENERATE) ---",
      res["verdict"]["S_pai_both|grey|pooled"]["outcome"],
      res["screens"]["pai_both|grey"]["flags"])
print("--- S5 grey ---", res["verdict"]["S5_blindability_grey"]["outcome"])
shutil.rmtree(TMP, ignore_errors=True)
print("\nSCORETEST_OK (temp dir removed)")
