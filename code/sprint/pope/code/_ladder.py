import glob, json, os, sys
sys.path.insert(0, "/data/alexmueller/sprint/pope/code")
import numpy as np
import pope_score as S
OUT = "/data/alexmueller/sprint/pope/out"
ORDER = ["sighted", "noise25", "blur4", "noise50", "lowres", "blur16", "pshuffle",
         "noise100", "mismatch", "grey"]
print("%-10s %-9s %7s %7s %7s %7s %8s %6s %7s %6s"
      % ("arm", "method", "acc", "yes", "H", "F", "J", "d'", "medTok", "nAns"))
for meth in ("vanilla", "pai_full", "pai_attn"):
    for arm in ORDER:
        f = f"{OUT}/pope_{meth}_{arm}.jsonl"
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f)]
        pred = np.array([S.parse_lenient(r["answer"]) == "yes" for r in rows])
        lab = np.array([r["label"] == "yes" for r in rows])
        C = np.array([int((pred & lab).sum()), int((pred & ~lab).sum()),
                      int((~pred & ~lab).sum()), int((~pred & lab).sum())])
        m = {k: float(v) for k, v in S.metrics(C).items()}
        meta = json.load(open(f"{OUT}/pope_{meth}_{arm}_meta.json"))
        yr = m["yes_ratio"]
        flag = "  <-- DEGENERATE-CONSTANT" if (yr >= 0.95 or yr <= 0.05) else ""
        print("%-10s %-9s %7.2f %7.4f %7.4f %7.4f %+8.4f %+6.3f %7.0f %6d%s"
              % (arm, meth, m["accuracy"], yr, m["H"], m["F"], m["J"], m["dprime"],
                 meta["median_tokens"], meta["n_distinct_answers"], flag))
    print()
