"""WIA-GARBLE shared construction (WIA_GARBLE_PREREG.md section 3): token-level sentence split,
seeded partial derangement, level assembly, and the per-cell checks. Imported by
wia_garblecensus.py, wia_garblegen.py, wia_garblescore.py, wia_garblenll.py, wia_garble_e1.py.

Nothing here touches the model. Pure functions of (tokenizer, prefix_ids).
"""
import math, random

WS = "▁"                       # SentencePiece word-start marker
SEED_BASE = 2000                    # Random(i*7 + 2000 + k + 100*ell)
LN_LEVELS = (("L25", 0.25, 1), ("L50", 0.50, 2), ("L100", 1.00, 3))   # (arm, g, ell)
OWN_LEVEL = ("L100own", 1.00, 4)
OUT_G = "/data/alexmueller/sc1_interv/out"
LN_IN = f"{OUT_G}/at_ln.jsonl"                 # job 18979 rows: wrongln / wrongasm / selfwrong_ln
CELLS_PATH = f"{OUT_G}/garble_cells.json"      # written by the census, re-derived by the generator
MIN_IMAGES_GATE = 350


def level_seed(iid, k, ell):
    return iid * 7 + SEED_BASE + k + 100 * ell


def split_sentences(tok, ids):
    """Cut `ids` after every boundary period. Boundary = the bare '.' token that is last or is
    followed by a word-start piece. Returns (segments, fragment); every segment ends in a
    boundary period; the fragment (possibly empty) is whatever follows the last boundary."""
    ids = [int(i) for i in ids]
    toks = tok.convert_ids_to_tokens(ids)
    segs, cur, n = [], [], len(ids)
    for p, (i, t) in enumerate(zip(ids, toks)):
        cur.append(i)
        if t == "." and (p == n - 1 or toks[p + 1].startswith(WS)):
            segs.append(cur)
            cur = []
    return segs, cur


def text_sentence_count(text):
    """The design's text-level rule: '. ' occurrences plus a terminal '.'."""
    return text.count(". ") + (1 if text.endswith(".") else 0)


def pieces_text(tok, segs, frag):
    parts = [tok.decode(s) for s in segs]
    if frag:
        parts.append(tok.decode(frag))
    return " ".join(parts)


def check_split(tok, ids, prefix_text):
    """Per-cell checks of section 3.3. Returns dict with n_sent, frag_len, ok flags, reasons."""
    segs, frag = split_sentences(tok, ids)
    n_text = text_sentence_count(prefix_text)
    dec_full = tok.decode(list(ids))
    dec_join = pieces_text(tok, segs, frag)
    reasons = []
    if len(segs) != n_text:
        reasons.append(f"count token={len(segs)} text={n_text}")
    if dec_full != prefix_text:
        reasons.append("decode(ids) != prefix_text")
    if dec_join != dec_full:
        reasons.append("join(decode(segments)) != decode(ids)")
    if any(len(s) < 2 for s in segs):
        reasons.append("segment shorter than 2 tokens")
    return {"n_sent": len(segs), "frag_len": len(frag), "ok": not reasons, "reasons": reasons,
            "segs": segs, "frag": frag}


def m_of(g, n):
    """Displaced-sentence count: max(2, round_half_up(g*n)) for g < 1; n for g = 1."""
    if g >= 1.0:
        return n
    return max(2, int(math.floor(g * n + 0.5)))


def derange_order(n, m, rng):
    """new_order[q] = index of the original sentence placed at position q. Exactly the m selected
    positions receive a sentence other than their own (a derangement of the selected set)."""
    assert 2 <= m <= n, (m, n)
    sel = sorted(rng.sample(range(n), m))
    while True:
        perm = list(sel)
        rng.shuffle(perm)
        if all(a != b for a, b in zip(perm, sel)):
            break
    order = list(range(n))
    for j, p in enumerate(sel):
        order[p] = perm[j]
    assert sorted(order) == list(range(n))
    return order, sel


def build_level(tok, ids, segs, frag, g, seed):
    """Assemble one garbled level. Asserts multiset identity, length, and decode equality."""
    ids = [int(i) for i in ids]
    n = len(segs)
    m = m_of(g, n)
    rng = random.Random(seed)
    order, sel = derange_order(n, m, rng)
    new_segs = [segs[o] for o in order]
    out = [i for s in new_segs for i in s] + list(frag)
    assert len(out) == len(ids), f"FATAL_BUDGET_MISMATCH level len {len(out)} != {len(ids)}"
    assert sorted(out) == sorted(ids), "FATAL_MULTISET_DRIFT"
    text = tok.decode(out)
    assert text == pieces_text(tok, new_segs, frag), "FATAL_DECODE_MISMATCH"
    displaced = sum(1 for q, o in enumerate(order) if q != o)
    assert displaced == m, (displaced, m)
    pairs = n * (n - 1) / 2
    kendall = sum(1 for a in range(n) for b in range(a + 1, n) if order[a] > order[b]) / pairs
    invisible = sum(1 for q, o in enumerate(order) if q != o and segs[q] == segs[o])
    return out, text, {"n_sent": n, "m": m, "g": g, "order": order, "sel": sel,
                       "displaced_frac": displaced / n, "kendall": kendall,
                       "invisible_displacements": invisible, "seed": seed,
                       "frag_len": len(frag)}


def build_cell_levels(tok, iid, k, ln_ids, ln_text, own_ids, own_text):
    """Everything the census records and the generator re-derives for one (image, k) cell."""
    ln = check_split(tok, ln_ids, ln_text)
    own = check_split(tok, own_ids, own_text)
    rec = {"image_id": iid, "k": k, "budget": len(ln_ids),
           "ln": {"n_sent": ln["n_sent"], "frag_len": ln["frag_len"], "ok": ln["ok"],
                  "reasons": ln["reasons"], "levels": {}},
           "own": {"n_sent": own["n_sent"], "frag_len": own["frag_len"], "ok": own["ok"],
                   "reasons": own["reasons"], "levels": {}}}
    if ln["ok"]:
        for arm, g, ell in LN_LEVELS:
            need = 3 if g < 1.0 else 2            # S3 for L25/L50, S2 for L100
            if ln["n_sent"] >= need:
                ids, text, meta = build_level(tok, ln_ids, ln["segs"], ln["frag"], g,
                                              level_seed(iid, k, ell))
                rec["ln"]["levels"][arm] = {"ids": ids, "text": text, **meta}
    if own["ok"] and own["n_sent"] >= 2:          # O2
        arm, g, ell = OWN_LEVEL
        ids, text, meta = build_level(tok, own_ids, own["segs"], own["frag"], g,
                                      level_seed(iid, k, ell))
        rec["own"]["levels"][arm] = {"ids": ids, "text": text, **meta}
    return rec


def load_tok():
    """Tokenizer exactly as the generator sees it (AutoProcessor path first, as wia_lngen.py)."""
    from transformers import AutoProcessor
    try:
        tok = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf").tokenizer
    except Exception as e:
        print(f"[garble] AutoProcessor failed ({e!r}); using AutoTokenizer", flush=True)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
    print(f"[garble] tokenizer class = {tok.__class__.__name__}  vocab={tok.vocab_size}", flush=True)
    return tok
