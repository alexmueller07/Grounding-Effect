"""J5 shared harness: model, COCO val2017, CHAIR-port scorer, prefix builder, D_t scorer.

Prereg: J5_PREREG.md (committed before any GPU computation). Sentinels are printed by
Python; exit codes are never the success signal.
"""
import os, re, json, math, random, sys, time

os.environ.setdefault("HF_HOME", "/data/alexmueller/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

MODEL = "llava-hf/llava-1.5-7b-hf"
ROOT = "/data/alexmueller/j5_gates"
OUT = os.environ.get("J5_OUT", f"{ROOT}/out")   # PV 2026-09-08: env override, default identical
COCO = "/data/datasets/coco"
ANN_INST = f"{COCO}/annotations/instances_val2017.json"
ANN_CAPS = f"{COCO}/annotations/captions_val2017.json"
IMG_DIRS = [f"{COCO}/val2014", f"{COCO}/images_val"]  # both hold 2017-named jpgs on this server
SEED = 20260817
N_IMAGES = int(os.environ.get("J5_N", "500"))
MAX_NEW = 256
MAX_NEW_CONT = 192
CUTS = tuple(float(x) for x in os.environ.get("J5_CUTS", "0.25,0.5,0.75").split(","))  # PV: default identical
PRIMARY_CUT = 0.5
TSTAR = 48
REFW = 16
BS = int(os.environ.get("J5_BS", "16"))

SYS_STR = os.environ.get("J5_SYS_STR",   # PV 2026-09-08: env override, default byte-identical
           "A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions. ")
QUESTION = os.environ.get("J5_QUESTION", "Please describe this image in detail.")  # PV: default identical


def prompt_text():
    return f"{SYS_STR}USER: <image>\n{QUESTION} ASSISTANT:"


def prompt_text_noimg():
    return f"{SYS_STR}USER: \n{QUESTION} ASSISTANT:"


def sentinel(name):
    print(f"J5_SENTINEL_{name}_OK", flush=True)


class JsonlWriter:
    def __init__(self, path, mode="w"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, mode, buffering=1)

    def write(self, obj):
        self.f.write(json.dumps(obj) + "\n")

    def flush(self):
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.flush()
        self.f.close()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- CHAIR port
_SING_KEEP = ("ss", "us", "is")


def _singularize(w):
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if w.endswith("ses") and not w.endswith("sses"):
        return w[:-2]
    if w.endswith("s") and not w.endswith(_SING_KEEP):
        return w[:-1]
    return w


class ChairScorer:
    """Faithful port of Rohrbach et al. chair.py (synonyms.txt verbatim; regex tokenizer and
    rule-based singularizer instead of nltk/pattern -- identical across all arms)."""

    def __init__(self, syn_path):
        syns = []
        for line in open(syn_path):
            line = line.strip()
            if not line:
                continue
            syns.append([w.strip() for w in line.split(",") if w.strip()])
        self.objects = set()
        self.inverse = {}
        for group in syns:
            for w in group:
                self.objects.add(w)
                self.inverse.setdefault(w, group[0])

        coco_double_words = [
            'motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light',
            'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter',
            'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket',
            'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier',
            'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'home plate',
            'train track']
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                        'zebra', 'giraffe', 'animal', 'cub']
        vehicle_words = ['jet', 'train']
        self.double = {}
        for dw in coco_double_words:
            self.double[dw] = dw
        for a in animal_words:
            self.double['baby %s' % a] = a
            self.double['adult %s' % a] = a
        for v in vehicle_words:
            self.double['passenger %s' % v] = v
        self.double['bow tie'] = 'tie'
        self.double['toilet seat'] = 'toilet'
        self.double['wine glas'] = 'wine glass'

    def node_for(self, w):
        """Return canonical node for a (possibly plural) word, or None."""
        if w in self.inverse:
            return self.inverse[w]
        s = _singularize(w)
        if s in self.inverse:
            return self.inverse[s]
        return None

    def mentions(self, text):
        """Return [(node_word, char_start), ...] for every MSCOCO object mention."""
        raw = [(m.group(0), m.start()) for m in re.finditer(r"[a-z]+", text.lower())]
        words = [( _singularize(w), c, w) for (w, c) in raw]
        # double-word merge on singularized stream (canonical order), keep char of first word
        merged = []
        i = 0
        while i < len(words):
            if i + 1 < len(words):
                dw = f"{words[i][0]} {words[i+1][0]}"
                dwr = f"{words[i][2]} {words[i+1][2]}"  # raw pair too (e.g. 'sports ball')
                hit = self.double.get(dw) or self.double.get(dwr)
                if hit:
                    merged.append((hit, words[i][1], hit))
                    i += 2
                    continue
            merged.append(words[i])
            i += 1
        toks = [m[0] for m in merged]
        if 'toilet' in toks and 'seat' in toks:
            merged = [m for m in merged if m[0] != 'seat']
        out = []
        for (w, c, rawlike) in merged:
            node = self.node_for(w) or (self.node_for(rawlike) if rawlike != w else None)
            if node is not None:
                out.append((node, c))
        return out


# ---------------------------------------------------------------- COCO data
def load_coco_sample(chair, n=N_IMAGES, seed=SEED):
    """Return list of dicts: image_id, file (abs path), gt (set of node words),
    human_caps (list of str)."""
    inst = json.load(open(ANN_INST))
    caps = json.load(open(ANN_CAPS))
    id2file = {im["id"]: im["file_name"] for im in inst["images"]}
    cat2name = {c["id"]: c["name"] for c in inst["categories"]}
    inst_by_img = {}
    for a in inst["annotations"]:
        inst_by_img.setdefault(a["image_id"], set()).add(a["category_id"])
    caps_by_img = {}
    for a in caps["annotations"]:
        caps_by_img.setdefault(a["image_id"], []).append(a["caption"])

    img_dir = None
    probe = next(iter(id2file.values()))
    for d in IMG_DIRS:
        if os.path.isfile(os.path.join(d, probe)):
            img_dir = d
            break
    assert img_dir, f"no image dir contains {probe}"

    eligible = sorted(i for i in id2file
                      if i in inst_by_img and i in caps_by_img
                      and os.path.isfile(os.path.join(img_dir, id2file[i])))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(np.array(eligible), size=n, replace=False).tolist())

    out = []
    for i in chosen:
        gt = set()
        for cid in inst_by_img[i]:
            name = cat2name[cid]
            node = chair.node_for(name) or chair.node_for(name.replace(" ", ""))
            if node is None:
                # multiword category names ('dining table', ...) go through mentions()
                ms = chair.mentions(name)
                node = ms[0][0] if ms else None
            if node:
                gt.add(node)
        for c in caps_by_img[i]:
            for (node, _c) in chair.mentions(c):
                gt.add(node)
        out.append({"image_id": int(i), "file": os.path.join(img_dir, id2file[i]),
                    "gt": sorted(gt), "human_caps": caps_by_img[i][:8]})
    return out


# ---------------------------------------------------------------- model
def load_model(attn_override=None):
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"
    if proc.tokenizer.pad_token is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    model = LlavaForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    if attn_override:
        lm = model.model.language_model
        lm.config._attn_implementation = attn_override
        try:
            model.config.text_config._attn_implementation = attn_override
        except Exception:
            pass
    return model, proc, proc.tokenizer


def encode_batch(proc, images, prompts):
    import torch
    enc = proc(images=images, text=prompts, return_tensors="pt", padding=True)
    return {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda"))
            for k, v in enc.items()}


def image_token_id(model):
    cfg = model.config
    return getattr(cfg, "image_token_index", None) or getattr(cfg, "image_token_id", None)


# ---------------------------------------------------------------- prefixes
def build_prefix_ids(tok, human_caps, target_len, seed):
    r = random.Random(seed)
    caps = [c.strip().rstrip(".").strip() for c in human_caps if c.strip()]
    r.shuffle(caps)
    text = ". ".join(caps) + "."
    ids = tok(text, add_special_tokens=False)["input_ids"]
    cycles = 0
    while len(ids) < target_len:
        cycles += 1
        text = text + " " + text
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if cycles > 6:
            break
    ids = ids[:target_len]
    return ids, tok.decode(ids), cycles


def glue(prefix_text, cont_text):
    if cont_text and not cont_text[0] in " .,;:!?":
        return prefix_text + " " + cont_text, len(prefix_text) + 1
    return prefix_text + cont_text, len(prefix_text)


def mention_positions(tok, text, mentions):
    """Map mention char offsets to token indices in the retokenized text."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    starts = [o[0] for o in offs]
    out = []
    for (node, c) in mentions:
        idx = 0
        for j, (a, b) in enumerate(offs):
            if a <= c < max(b, a + 1):
                idx = j
                break
            if a > c:
                idx = max(0, j - 1)
                break
        else:
            idx = len(offs) - 1
        out.append((node, c, idx))
    return out, len(offs)


# ---------------------------------------------------------------- repetition guard
def degenerate_repetition(text, n=4, k=3):
    ws = text.lower().split()
    if len(ws) < n * k:
        return False
    grams = [" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    run = 1
    for i in range(n, len(grams), n):
        if grams[i] == grams[i - n]:
            run += 1
            if run >= k:
                return True
        else:
            run = 1
    return False
