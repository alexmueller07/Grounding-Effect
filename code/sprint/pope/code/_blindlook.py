import json, collections, sys
sys.path.insert(0, "/data/alexmueller/sprint/pope/code")
import pope_score as S
for arm in ("vanilla_grey", "vanilla_sighted", "pai_full_grey"):
    rows = [json.loads(l) for l in open(f"/data/alexmueller/sprint/pope/out/pope_{arm}.jsonl")]
    keys = set()
    for r in rows[:5]:
        keys |= set(r)
    yes = sum(1 for r in rows if S.parse_lenient(r["answer"]) == "yes")
    empty = sum(1 for r in rows if not str(r["answer"]).strip())
    print("%-18s n=%d yes_rate=%.4f empty=%d has_text_key=%s"
          % (arm, len(rows), yes / len(rows), empty, "text" in keys))
print()
rows = [json.loads(l) for l in open("/data/alexmueller/sprint/pope/out/pope_vanilla_grey.jsonl")]
print("most common grey answers (vanilla):")
for a, c in collections.Counter(r["answer"] for r in rows).most_common(8):
    print("  %5d  %s" % (c, a[:140]))
print()
print("does the blind model describe the grey field itself?")
kw = ("gray", "grey", "blank", "plain", "solid", "empty", "no image", "background")
hit = sum(1 for r in rows if any(k in r["answer"].lower() for k in kw))
print("  answers mentioning a grey/blank/plain field: %d / %d (%.1f%%)"
      % (hit, len(rows), 100 * hit / len(rows)))
lab_yes = [r for r in rows if r["label"] == "yes"]
print("  of the 4500 label=yes items (object IS in the original photo), yes-answers: %d"
      % sum(1 for r in lab_yes if S.parse_lenient(r["answer"]) == "yes"))
