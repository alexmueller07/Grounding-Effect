import sys
sys.path.insert(0, "/data/alexmueller/sprint/pope/code")
import numpy as np
import pope_score as S

worst = 0.0
rng = np.random.default_rng(0)
cases = []
for m in (1500, 4500):
    for _ in range(300):
        tp = int(rng.integers(0, m + 1)); fp = int(rng.integers(0, m + 1))
        cases.append((tp, fp, m - fp, m - tp))
# the degenerate corners the screen must survive
cases += [(1500, 1500, 0, 0), (0, 0, 1500, 1500), (1500, 0, 1500, 0), (0, 1500, 0, 1500)]
for C in cases:
    C = np.array(C, dtype=np.int64)
    lv = {k: float(v) for k, v in S.metrics(C).items()}
    r = S.identity_check(C, lv)
    if r.get("balanced"):
        worst = max(worst, r["worst_abs_deviation"])
print("cases:", len(cases), "worst abs identity deviation:", worst)
# unbalanced must be skipped, not asserted
print("unbalanced skipped:", S.identity_check(np.array([10, 5, 20, 3]), 
      {k: float(v) for k, v in S.metrics(np.array([10, 5, 20, 3])).items()}))
# the always-No responder Lan et al. report
C = np.array([0, 0, 1500, 1500])
lv = {k: float(v) for k, v in S.metrics(C).items()}
print("always-No: acc=%.2f prec=%s rec=%.2f yes_ratio=%.2f H=%.3f F=%.3f J=%.3f"
      % (lv["accuracy"], lv["precision"], lv["recall"], lv["yes_ratio"],
         lv["H"], lv["F"], lv["J"]))
C = np.array([1500, 1500, 0, 0])
lv = {k: float(v) for k, v in S.metrics(C).items()}
print("always-Yes: acc=%.2f prec=%.2f rec=%.2f yes_ratio=%.2f H=%.3f F=%.3f J=%.3f"
      % (lv["accuracy"], lv["precision"], lv["recall"], lv["yes_ratio"],
         lv["H"], lv["F"], lv["J"]))
