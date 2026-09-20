import json
r = json.load(open("/data/alexmueller/sprint/pope/out/pope_results.json"))
print("identity:", r["identity_check"], "| cells:", len(r["cells"]))
for m in ("pai_full", "pai_attn", "pai_both"):
    for ab in ("mismatch", "grey"):
        k = "%s|%s|pooled" % (m, ab)
        if k not in r["contrasts"]:
            continue
        c = r["contrasts"][k]
        v = r["verdict"].get("S_%s|%s|pooled" % (m, ab), {})
        print("\n=== %s / %s / pooled ===  VERDICT: %s  %s"
              % (m, ab, v.get("outcome"), v.get("note", "")))
        for q in ("J", "accuracy", "dprime", "c", "H", "F"):
            d = c[q]
            print("   %-8s dS=%+0.5f [%+0.5f,%+0.5f]  dA=%+0.5f [%+0.5f,%+0.5f]  dImg=%+0.5f [%+0.5f,%+0.5f]"
                  % (q, d["delta_sighted"]["point"], d["delta_sighted"]["ci95"][0],
                     d["delta_sighted"]["ci95"][1], d["delta_ablated"]["point"],
                     d["delta_ablated"]["ci95"][0], d["delta_ablated"]["ci95"][1],
                     d["delta_image"]["point"], d["delta_image"]["ci95"][0],
                     d["delta_image"]["ci95"][1]))
        rho = c["J"]["rho_surviving"]
        print("   rho=%+0.3f [%+0.3f,%+0.3f]  half_gain_line=%s"
              % (rho["point"], rho["ci95"][0], rho["ci95"][1], v.get("half_gain_line")))
print("\n=== per-split, pai_attn / mismatch ===")
for s in ("random", "popular", "adversarial"):
    k = "pai_attn|mismatch|%s" % s
    if k in r["contrasts"]:
        d = r["contrasts"][k]["J"]
        v = r["verdict"].get("S_pai_attn|mismatch|%s" % s, {})
        print("  %-12s dS=%+0.4f [%+0.4f,%+0.4f]  dA=%+0.4f [%+0.4f,%+0.4f]  dImg=%+0.4f [%+0.4f,%+0.4f] -> %s"
              % (s, d["delta_sighted"]["point"], d["delta_sighted"]["ci95"][0],
                 d["delta_sighted"]["ci95"][1], d["delta_ablated"]["point"],
                 d["delta_ablated"]["ci95"][0], d["delta_ablated"]["ci95"][1],
                 d["delta_image"]["point"], d["delta_image"]["ci95"][0],
                 d["delta_image"]["ci95"][1], v.get("outcome")))
print("\n=== alt seed (primary, both ablations) ===")
a = r.get("alt_seed_primary", {})
for ab in ("grey", "mismatch"):
    if ab in a:
        d = a[ab]["delta_image"]
        print("  %-9s dImg=%+0.5f [%+0.5f,%+0.5f]" % (ab, d["point"], d["ci95"][0], d["ci95"][1]))
print("\n=== all levels, pooled ===")
for k in sorted(r["levels"]):
    if not k.endswith("|pooled"):
        continue
    v = r["levels"][k]
    f1 = ("%.2f" % v["f1"]) if v["f1"] == v["f1"] else "nan"
    print("  %-28s acc=%6.2f f1=%6s yes=%.4f H=%.4f F=%.4f J=%+0.4f d=%+0.3f c=%+0.3f"
          % (k, v["accuracy"], f1, v["yes_ratio"], v["H"], v["F"], v["J"], v["dprime"], v["c"]))
print("\n=== screens ===")
for k, v in sorted(r["screens"].items()):
    print("  %-24s yes=%.4f unparse=%.4f trunc=%.4f medTok=%3.0f distinct=%5d flags=%s"
          % (k, v["yes_ratio"], v["unparseable_rate"], v["trunc_rate"], v["median_tokens"],
             v["n_distinct_answers"], v["flags"]))
print("\n=== PRIMARY ===")
print(json.dumps(r["verdict"]["PRIMARY"], indent=1))
print("\n=== S5 ===")
for ab in ("grey", "mismatch"):
    b = r["verdict"].get("S5_blindability_%s" % ab, {})
    if b:
        print(" ", ab, b.get("outcome"),
              {k: round(x["point"], 2) for k, x in b.get("vanilla_accuracy", {}).items()},
              "yes_ratio=", round(b.get("yes_ratio", float("nan")), 4))
