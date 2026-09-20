"""Render LADDER.md tables straight from lad_l1.json / lad_l2.json / lad_dose.json.

Numbers in the report are never retyped by hand: every cell here is read from the JSON the
scorers wrote, so a transcription error cannot enter the write-up.
Usage: python lad_report.py <dir-with-the-json-files>
"""
import json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else "."


def ld(n):
    p = os.path.join(D, n)
    return json.load(open(p)) if os.path.exists(p) else None


def ci(d, k=4):
    if d is None or d.get("point") is None:
        return "n/a"
    lo, hi = d["ci"]
    if lo is None:
        return f"{d['point']:+.{k}f} [no CI]"
    return f"{d['point']:+.{k}f} [{lo:+.{k}f}, {hi:+.{k}f}]"


l1, l2, dose = ld("lad_l1.json"), ld("lad_l2.json"), ld("lad_dose.json")
out = []

if dose:
    out.append("### Realised severity of each rung (measured, n=%d images)\n" % dose["n"])
    out.append("| rung | emb_cos (pooled) | emb_cos (per-patch) | pixel RMSE |")
    out.append("|---|---|---|---|")
    order = sorted(dose["emb_cos"], key=lambda r: -dose["emb_cos"][r])
    for r in order:
        out.append(f"| `{r}` | {dose['emb_cos'][r]:.4f} | {dose['emb_cos_tok'][r]:.4f} | "
                   f"{dose['pix_rmse'][r]:.4f} |")
    out.append("")

if l1:
    out.append("### L1 positive controls\n")
    p1 = l1["PC1"]
    out.append("| control | measured | reference | abs dev | tol | pass |")
    out.append("|---|---|---|---|---|---|")
    for k in p1["reference"]:
        out.append(f"| PC1 {k} | {p1['measured'][k]:.8f} | {p1['reference'][k]} | "
                   f"{p1['abs_dev'][k]:.2e} | {p1['tol']} | {'PASS' if p1['PASS'] else 'FAIL'} |")
    p2 = l1.get("PC2") or {}
    if p2.get("n_compared"):
        out.append(f"| PC2a token identity | {p2['n_identical']}/{p2['n_compared']} "
                   f"({p2['frac_identical']:.4f}) | >= {p2['min_frac']} | - | - | "
                   f"{'PASS' if p2['PC2a_PASS'] else 'FAIL'} |")
        out.append(f"| PC2b Delta_image(grey re-gen) | {p2['Delta_image_regen_grey']:+.6f} | "
                   f"-0.0160 | {p2['abs_dev_vs_ref']:.2e} | {p2['tol']} | "
                   f"{'PASS' if p2['PC2b_PASS'] else 'FAIL'} |")
    out.append("")

    out.append("### L1 degeneracy screens (every arm; Amendment 3 rule)\n")
    out.append("| arm | n | median len | mean len | cap-hit | empty | <=3 tok | "
               "distinct-8gram | distinct-tokid | distinct texts | top text share | "
               "true/false units | screen |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a, s in l1["screens"].items():
        ts = s.get("top_text_share")
        out.append(f"| `{a}` | {s['n']} | {s['median_cont_len']:.0f} | "
                   f"{s['mean_cont_len']:.1f} | {s.get('cap_hit_rate', float('nan')):.3f} | "
                   f"{s['empty_rate']:.3f} | {s.get('le3_token_rate', float('nan')):.3f} | "
                   f"{s['mean_ngram8_ratio']:.3f} | "
                   f"{s['mean_tokid_ratio']:.3f} | {s['n_distinct_texts']} | "
                   f"{('%.3f' % ts) if ts is not None else '-'} | "
                   f"{s['n_true']}/{s['n_false']} | "
                   f"{'DEGENERATE ' + '; '.join(s['degenerate_why']) if s['DEGENERATE'] else 'ok'} |")
    out.append("")
    if l1.get("AMENDMENT3_verdicts_changed") is not None:
        out.append("### Which verdicts the Amendment-3 fix changed\n")
        ch = l1["AMENDMENT3_verdicts_changed"]
        if not ch:
            out.append("No rung's verdict differs between the original and corrected "
                       "collapse rule.")
        else:
            out.append("| rung | corrected rule | ORIGINAL rule | flags that would have fired |")
            out.append("|---|---|---|---|")
            for r, c in ch.items():
                out.append(f"| `{r}` | **{c['corrected']}** | {c['original_rule']} | "
                           f"{'; '.join(c['comparative_flags_that_would_have_fired'])} |")
        out.append("")

    out.append("### L1 primary: Delta_image per rung\n")
    iv = l1["interval"]
    out.append(f"Delta_primary (frozen sighted) = {ci(iv['D[sighted]'])}\n")
    out.append("| rung | Delta_rung | Delta_image (95%) | width | Bonferroni 99.29% | verdict |")
    out.append("|---|---|---|---|---|---|")
    v = l1["verdict_per_rung"]
    pg = iv.get("Dimg[pubgrey]")
    out.append(f"| `grey` (published, frozen) | {ci(iv['D[pubgrey]'])} | {ci(pg)} | "
               f"{pg['ci_width']:.4f} | - | positive control |")
    if "D[grey]" in iv:
        out.append(f"| `grey` (this lane, re-generated) | {ci(iv['D[grey]'])} | "
                   f"{ci(iv['Dimg[grey]'])} | {iv['Dimg[grey]']['ci_width']:.4f} | - | "
                   f"positive control |")
    for rg, d in v.items():
        if rg in ("mismatch",):
            continue
        out.append(f"| `{rg}` | {ci(d['Delta_rung'])} | {ci(d['Delta_image'])} | "
                   f"{d['Delta_image']['ci_width']:.4f} | "
                   f"{ci(d['Delta_image_bonferroni'])} | **{d['verdict']}** |")
    if "mismatch" in v:
        out.append(f"| `mismatch` (real but wrong image) | "
                   f"{ci(v['mismatch']['Delta_rung'])} | "
                   f"{ci(v['mismatch']['Delta_image'])} | "
                   f"{v['mismatch']['Delta_image']['ci_width']:.4f} | "
                   f"{ci(v['mismatch']['Delta_image_bonferroni'])} | "
                   f"**{v['mismatch']['verdict']}** |")
    lm = l1.get("Delta_image_ladder_mean")
    if lm:
        out.append(f"\nPooled ladder mean Delta_image (seven degradation rungs only) = {ci(lm)}")
    if l1.get("POSTHOC_vs_grey"):
        out.append("\n**Post hoc** — each ablation's Delta_image minus grey's, paired on the "
                   "same bootstrap draw:\n")
        out.append("| rung | Delta_image(rung) - Delta_image(grey) |")
        out.append("|---|---|")
        for k, d in l1["POSTHOC_vs_grey"].items():
            out.append(f"| `{k[len('Dimg_minus_greyDimg['):-1]}` | {ci(d)} |")
    out.append(f"\n**LADDER_VERDICT (seven degradation rungs) = {l1['LADDER_VERDICT']}**")
    out.append(f"\n**MISMATCH_VERDICT = {l1.get('MISMATCH_VERDICT')}**")
    out.append(f"\n**OVERALL_VERDICT = {l1.get('OVERALL_VERDICT')}**")
    if "spearman_severity_vs_Dimage" in l1:
        out.append(f"\nSpearman(severity, Delta_image) = "
                   f"{l1['spearman_severity_vs_Dimage']:+.3f} (descriptive, not a gate)")
    out.append("")

    out.append("### L1 rate split: what the image does to H and F\n")
    out.append("| rung | image-attributable dH | image-attributable dF |")
    out.append("|---|---|---|")
    for rg, d in v.items():
        out.append(f"| `{rg}` | {ci(d.get('dH_image'))} | {ci(d.get('dF_image'))} |")
    out.append("")

    if l1.get("lenmatch_cross"):
        out.append("### L1 secondaries: length matching\n")
        out.append("| rung | within-condition | cross-condition |")
        out.append("|---|---|---|")
        lw = l1.get("lenmatch_within", {})
        for rg in list(v) + (["grey"] if "grey" in l1.get("lenmatch_cross", {}) else []):
            a = lw.get(f"Dimg[{rg}]")
            b = (l1["lenmatch_cross"].get(rg) or {}).get("Delta_image")
            out.append(f"| `{rg}` | {ci(a)} | {ci(b)} |")
        out.append("")

    out.append("### L1 collapse screen (Amendment 1)\n")
    out.append("| rung | more degenerate than grey | more degenerate than weaker reference |")
    out.append("|---|---|---|")
    for rg, d in v.items():
        out.append(f"| `{rg}` | {'; '.join(d['more_degenerate_than_grey']) or 'no'} | "
                   f"{'; '.join(d['more_degenerate_than_weaker_reference']) or 'no'} |")
    out.append("")

if l2:
    out.append("### L2 positive control (PC3)\n")
    p3 = l2["PC3"]
    out.append("| quantity | measured | reference | abs dev |")
    out.append("|---|---|---|---|")
    for k in p3["reference"]:
        out.append(f"| {k} | {p3['measured'][k]:.6f} | {p3['reference'][k]} | "
                   f"{p3['abs_dev'][k]:.2e} |")
    out.append(f"\nPC3 PASS = {p3['PASS']}. Sighted PAI gain on CHAIR_i = "
               f"{ci(l2['sighted_contrast']['gain_i'], 3)}, "
               f"dJ = {ci(l2['sighted_contrast']['dJ'])}\n")

    out.append("### L2 denominator screen (vanilla arm, 500 captions)\n")
    out.append("| rung | mention occurrences | captions with >=1 mention | CHAIR_i | "
               "CHAIR_s | median tokens | distinct texts | verdict |")
    out.append("|---|---|---|---|---|---|---|---|")
    for rg, s in l2["denominator_screen"].items():
        cii = "n/a (0/0)" if s.get("CHAIR_i") is None else f"{s['CHAIR_i']:.2f}"
        css = "-" if s.get("CHAIR_s") is None else f"{s['CHAIR_s']:.2f}"
        ec = s.get("emb_cos"); mt = s.get("MATERIAL")
        out.append(f"| `{rg}` | {('%.4f' % ec) if ec is not None else '-'} | "
                   f"{'yes' if mt else ('no' if mt is False else '-')} | "
                   f"{s['mention_occurrences']} | "
                   f"{s['captions_with_mention']}/{s['n']} | {cii} | {css} | "
                   f"{s['median_tokens']:.0f} | {s['n_distinct_texts']} | "
                   f"**{s['verdict']}** |")
    out.append("")

    if l2.get("contrast_at_usable_rungs"):
        out.append("### L2 PAI contrast at usable rungs\n")
        out.append("| rung | CHAIR_i gain at rung | image-attributable part of the gain | "
                   "dJ at rung | image-attributable dJ | verdict |")
        out.append("|---|---|---|---|---|---|")
        for rg, c in l2["contrast_at_usable_rungs"].items():
            if "gain_i_rung" not in c:
                out.append(f"| `{rg}` | - | - | - | - | {c['verdict']} |")
                continue
            out.append(f"| `{rg}` | {ci(c['gain_i_rung'], 3)} | {ci(c['gain_i_image'], 3)} | "
                       f"{ci(c['dJ_rung'])} | {ci(c['dJ_image'])} | **{c['verdict']}** |")
        out.append("")
        out.append("| rung | arm | CHAIR_i | CHAIR_s | H | F | J | d' | c |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for rg, c in l2["contrast_at_usable_rungs"].items():
            lv = c.get("levels")
            if not lv:
                continue
            for arm in ("vanilla", "pai05"):
                z = lv[arm]
                out.append(f"| `{rg}` | {arm} | {lv['CHAIR_i_' + arm]:.2f} | "
                           f"{lv['CHAIR_s_' + arm]:.2f} | {z['H']:.4f} | {z['F']:.4f} | "
                           f"{z['J']:.4f} | {z['dprime']:.3f} | {z['c']:.3f} |")
        out.append("")
    out.append(f"**L2_VERDICT = {l2['L2_VERDICT']}**\n")

print("\n".join(out))
