"""Registered gate G_PC only: the positive control on vanilla/sighted, plus the degeneracy
screens for whatever cells exist. No contrast, no verdict, no Delta of any kind."""
import glob, json, os, sys
sys.path.insert(0, "/data/alexmueller/sprint/pope/code")
import numpy as np
from pope_common import (OUT, SPLITS, PC_ACC_CENTER, PC_ACC_HALF, PC_F1_CENTER, PC_F1_HALF,
                         DEGEN_YES_HI, DEGEN_YES_LO, DEGEN_UNPARSE)
import pope_score as S

for f in sorted(glob.glob(f"{OUT}/pope_*_meta.json")):
    m = json.load(open(f))
    print("%-28s n=%d trunc=%.4f meanTok=%.2f medTok=%.0f distinct=%d imgmass=%.4f "
          "pai=%d prefill=%d" % (m["tag"], m["n_rows"], m["trunc_rate"], m["mean_tokens"],
                                 m["median_tokens"], m["n_distinct_answers"],
                                 m["mean_img_mass"], m["pai_applied_steps"],
                                 m["pai_applied_prefill"]))
    print("    top answers:", m["answer_dist"][:5])

p = f"{OUT}/pope_vanilla_sighted.jsonl"
if not os.path.exists(p):
    print("\nvanilla_sighted not finished yet -- positive control not computable")
    sys.exit(0)
rows = [json.loads(l) for l in open(p)]
pred = np.array([S.parse_lenient(r["answer"]) == "yes" for r in rows])
lab = np.array([r["label"] == "yes" for r in rows])
spl = np.array([r["split"] for r in rows])
strict = [S.parse_strict(r["answer"]) for r in rows]
print("\n=== G_PC: positive control, vanilla / sighted ===")
print("unparseable (strict):", round(float(np.mean([x is None for x in strict])), 5))
accs = {}
for s in list(SPLITS) + ["pooled"]:
    m_ = np.ones(len(rows), bool) if s == "pooled" else (spl == s)
    C = np.array([int((pred[m_] & lab[m_]).sum()), int((pred[m_] & ~lab[m_]).sum()),
                  int((~pred[m_] & ~lab[m_]).sum()), int((~pred[m_] & lab[m_]).sum())])
    mm = {k: float(v) for k, v in S.metrics(C).items()}
    idc = S.identity_check(C, mm)
    accs[s] = mm
    print("  %-12s acc=%6.2f prec=%6.2f rec=%6.2f f1=%6.2f yes=%.3f | H=%.4f F=%.4f "
          "J=%+.4f d'=%+.3f c=%+.3f | identity %.2e"
          % (s, mm["accuracy"], mm["precision"], mm["recall"], mm["f1"], mm["yes_ratio"],
             mm["H"], mm["F"], mm["J"], mm["dprime"], mm["c"],
             idc.get("worst_abs_deviation", float("nan"))))
a, f1 = accs["pooled"]["accuracy"], accs["pooled"]["f1"]
pc_a = abs(a - PC_ACC_CENTER) <= PC_ACC_HALF and abs(f1 - PC_F1_CENTER) <= PC_F1_HALF
pc_a2 = accs["random"]["accuracy"] >= accs["popular"]["accuracy"] >= accs["adversarial"]["accuracy"]
print("\nPC-A  target PAI T2 single-turn greedy vanilla: acc 84.76 [%.2f,%.2f], F1 85.51 [%.2f,%.2f]"
      % (PC_ACC_CENTER - PC_ACC_HALF, PC_ACC_CENTER + PC_ACC_HALF,
         PC_F1_CENTER - PC_F1_HALF, PC_F1_CENTER + PC_F1_HALF))
print("PC-A  measured acc %.2f  F1 %.2f  ->  %s" % (a, f1, "PASS" if pc_a else "FAIL"))
print("PC-A2 random>=popular>=adversarial -> %s" % ("PASS" if pc_a2 else "FAIL"))
