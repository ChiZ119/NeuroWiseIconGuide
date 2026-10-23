"""
pictogramIndex.py
=================

Library-agnostic picture matching for NeuroWise IconGuide.

Given an object from a step ("toothbrush") and the step sentence it came from
("Wet your toothbrush under the tap."), find the best picture in ANY library
built with buildPictogramIndex.py (ARASAAC, the old size-512 library, a folder of
your own pictures, or a merge of these).

Four matching modes, so they can be compared in the paper:

    "text"          entity text vs. picture keywords only
                    (≈ the thesis method: exact / whole-word / MiniLM similarity)
    "text+context"  + how well the whole step sentence fits the picture's
                    keywords, definition and categories (fixes "tap", "bowl", ...)
    "image"         CLIP: entity text vs. the picture pixels only
    "hybrid"        weighted mix of all three (default)

It also finds an ACTION pictogram for a step ("brush teeth", "put on socks"),
which the old object-only library could not do.

Typical use (the server does this for you when PICTO_LIBRARY is set):

    from pictogramIndex import get_active_index
    index = get_active_index()
    index.search("toothbrush", context="Wet your toothbrush under the tap.", k=5)
    index.best_action("Brush your teeth for two minutes.", nlp)
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict
from urllib.parse import quote

import numpy as np

from pictogramLibraries import item_image_path, read_library

INDEX_FILE = "index.npz"
BUILD_INFO_FILE = "build_info.json"

MODES = ("text", "text+context", "image", "hybrid")

# Weights for the fused score. Starting values only: tune them on your labelled
# set with `python evalRetrieval.py score ... --grid` and paste the result here
# (or set PICTO_WEIGHTS="0.6,0.2,0.2").
DEFAULT_WEIGHTS = {"text": 0.65, "context": 0.15, "image": 0.20}

# Minimum fused score to show a picture (below it, the step stays text-only,
# exactly like the thesis version). Also tunable via evalRetrieval.py.
DEFAULT_THRESHOLDS = {"text": 0.50, "text+context": 0.50, "image": 0.50, "hybrid": 0.45}

# CLIP similarities are small and compressed (~0.15-0.35), so they are turned into
# a 0..1 confidence by how far the picture stands out from the whole library
# (z-score): z <= Z_LO -> 0, z >= Z_HI -> 1.
Z_LO, Z_HI = 2.0, 6.0

# Action pictograms: a verb+object phrase must reach this text similarity;
# a bare verb is only accepted on an exact keyword hit.
ACTION_PHRASE_MIN = 0.75

_SEMANTIC_CAP = 0.9  # same ceiling as the thesis code: only string matches reach >0.9

# Bump when the matching logic changes: cached pictures in knowledgeBase.json are
# keyed on library build + this version, so they get recomputed automatically.
MATCHER_VERSION = "1"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", _norm(s))


def _encode_text(model, texts: list[str], batch_size: int = 128) -> np.ndarray:
    v = model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=len(texts) > 2000)
    return np.asarray(v, dtype=np.float32).reshape(len(texts), -1)


def _to_rgb_on_white(im):
    """Transparent PNGs turn black in CLIP unless composited on white first."""
    from PIL import Image
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        return bg.convert("RGB")
    return im.convert("RGB")


def _encode_images(model, paths: list[str], batch_size: int = 32,
                   log=print) -> tuple[np.ndarray | None, np.ndarray]:
    from PIL import Image
    vecs, ok = [None] * len(paths), np.zeros(len(paths), dtype=bool)
    t0 = time.time()
    for start in range(0, len(paths), batch_size):
        batch_idx, imgs = [], []
        for i in range(start, min(start + batch_size, len(paths))):
            try:
                with Image.open(paths[i]) as im:
                    imgs.append(_to_rgb_on_white(im))
                batch_idx.append(i)
            except Exception:
                pass  # missing / broken file -> no image vector
        if imgs:
            emb = model.encode(imgs, batch_size=batch_size, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
            for j, i in enumerate(batch_idx):
                vecs[i] = np.asarray(emb[j], dtype=np.float32)
                ok[i] = True
        done = min(start + batch_size, len(paths))
        if done % (batch_size * 20) < batch_size or done == len(paths):
            log(f"  ... CLIP-encoded {done}/{len(paths)} images ({time.time() - t0:.0f}s)")
    if not ok.any():
        return None, ok
    dim = next(v for v in vecs if v is not None).shape[0]
    out = np.zeros((len(paths), dim), dtype=np.float32)
    for i, v in enumerate(vecs):
        if v is not None:
            out[i] = v
    return out, ok


def _item_text(it: dict) -> str:
    """The text that represents a picture as a whole (used for context matching)."""
    parts = [it.get("title", "")]
    kws = [k for k in it.get("keywords", []) if k.lower() != it.get("title", "").lower()]
    if kws:
        parts.append(", ".join(kws[:6]))
    if it.get("definition"):
        parts.append(it["definition"])
    if it.get("categories"):
        parts.append(", ".join(it["categories"][:4]))
    return ". ".join(p for p in parts if p)


class _LRU(OrderedDict):
    def __init__(self, maxsize=4096):
        super().__init__()
        self.maxsize = maxsize

    def get_or(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return self[key]
        val = fn()
        self[key] = val
        if len(self) > self.maxsize:
            self.popitem(last=False)
        return val


class PictogramIndex:
    # ── construction ────────────────────────────────────────────────────────

    def __init__(self, library_dir: str, info: dict, items: list[dict], build_info: dict,
                 kw_texts: list[str], kw_item: np.ndarray, kw_emb: np.ndarray,
                 desc_emb: np.ndarray, img_emb: np.ndarray | None, has_img: np.ndarray | None,
                 text_model, clip_text_model=None):
        self.library_dir = os.path.abspath(library_dir)
        self.info = info
        self.items = items
        self.build_info = build_info
        self.kw_texts = kw_texts
        self.kw_item = kw_item
        self.kw_emb = kw_emb
        self.desc_emb = desc_emb
        self.img_emb = img_emb
        self.has_img = has_img if has_img is not None else np.zeros(len(items), dtype=bool)
        self.text_model = text_model
        self.clip_text_model = clip_text_model

        self.name = info.get("name", os.path.basename(self.library_dir))
        self.library_id = f"{self.name}@{build_info.get('built_on', '?')}/m{MATCHER_VERSION}"
        self.clip_prompt = info.get("clip_prompt", "a pictogram of {}")
        self.default_mode = "hybrid"
        self.weights = dict(DEFAULT_WEIGHTS)
        self.thresholds = dict(DEFAULT_THRESHOLDS)

        # exact keyword lookup + token postings for whole-word containment
        self._exact: dict[str, list[int]] = {}
        self._kw_tokens: list[list[str]] = []
        self._postings: dict[str, set[int]] = {}
        for r, (t, i) in enumerate(zip(kw_texts, kw_item)):
            self._exact.setdefault(t, []).append(int(i))
            toks = _tokens(t)
            self._kw_tokens.append(toks)
            for tok in set(toks):
                self._postings.setdefault(tok, set()).add(r)

        self._item_rows: list[list[int]] = [[] for _ in items]
        for r, i in enumerate(kw_item):
            self._item_rows[int(i)].append(r)

        kinds = [set(it.get("kinds") or ["noun"]) for it in items]
        self._has_verb = np.array(["verb" in k for k in kinds], dtype=bool)
        self._verb_only = np.array([k == {"verb"} for k in kinds], dtype=bool)
        self._id_to_idx = {it["id"]: i for i, it in enumerate(items)}

        self._text_cache = _LRU()
        self._qvec_cache = _LRU()
        self._ctx_cache = _LRU(1024)
        self._img_cache = _LRU()
        self._warned_no_clip = False

    @classmethod
    def build(cls, library_dir: str, text_model, text_model_name: str,
              clip_image_model=None, clip_model_name: str | None = None,
              clip_text_model_name: str | None = None, batch_size: int = 32,
              log=print) -> "PictogramIndex":
        info, items = read_library(library_dir)
        log(f"[index] building '{info.get('name')}' with {len(items)} pictures")

        kw_texts, kw_item = [], []
        for i, it in enumerate(items):
            for kw in it.get("keywords") or [it.get("title", "")]:
                t = _norm(kw)
                if t:
                    kw_texts.append(t)
                    kw_item.append(i)
        t0 = time.time()
        kw_emb = _encode_text(text_model, kw_texts)
        desc_emb = _encode_text(text_model, [_item_text(it) for it in items])
        log(f"[index] text embeddings: {len(kw_texts)} keywords + {len(items)} "
            f"descriptions ({time.time() - t0:.0f}s)")

        img_emb, has_img = None, np.zeros(len(items), dtype=bool)
        if clip_image_model is not None:
            paths = [item_image_path(library_dir, info, it) for it in items]
            img_emb, has_img = _encode_images(clip_image_model, paths, batch_size, log)
            log(f"[index] image embeddings for {int(has_img.sum())}/{len(items)} pictures")

        arrays = {
            "kw_texts": np.array(kw_texts),
            "kw_item": np.array(kw_item, dtype=np.int32),
            "kw_emb": kw_emb.astype(np.float16),
            "desc_emb": desc_emb.astype(np.float16),
            "has_img": has_img,
        }
        if img_emb is not None:
            arrays["img_emb"] = img_emb.astype(np.float16)
        np.savez_compressed(os.path.join(library_dir, INDEX_FILE), **arrays)

        build_info = {
            "library": info.get("name"),
            "n_items": len(items),
            "n_keywords": len(kw_texts),
            "text_model": text_model_name,
            "text_dim": int(kw_emb.shape[1]),
            "clip_model": clip_model_name if img_emb is not None else None,
            "clip_text_model": (clip_text_model_name or clip_model_name) if img_emb is not None else None,
            "n_images_embedded": int(has_img.sum()),
            "built_on": time.strftime("%Y-%m-%dT%H:%M"),
        }
        with open(os.path.join(library_dir, BUILD_INFO_FILE), "w", encoding="utf-8") as f:
            json.dump(build_info, f, indent=2)
        log(f"[index] saved -> {os.path.join(library_dir, INDEX_FILE)}")
        return cls(library_dir, info, items, build_info, kw_texts, np.array(kw_item, dtype=np.int32),
                   kw_emb, desc_emb, img_emb, has_img, text_model, None)

    @classmethod
    def load(cls, library_dir: str, text_model, clip_text_model=None) -> "PictogramIndex":
        info, items = read_library(library_dir)
        with open(os.path.join(library_dir, BUILD_INFO_FILE), encoding="utf-8") as f:
            build_info = json.load(f)
        z = np.load(os.path.join(library_dir, INDEX_FILE), allow_pickle=False)
        kw_texts = [str(t) for t in z["kw_texts"]]
        kw_emb = z["kw_emb"].astype(np.float32)
        probe = _encode_text(text_model, ["probe"])
        if probe.shape[1] != kw_emb.shape[1]:
            raise ValueError(
                f"Index was built with '{build_info.get('text_model')}' "
                f"({kw_emb.shape[1]}-d) but the running text model is {probe.shape[1]}-d. "
                f"Rebuild the index or load the same model.")
        img_emb = z["img_emb"].astype(np.float32) if "img_emb" in z.files else None
        idx = cls(library_dir, info, items, build_info, kw_texts, z["kw_item"], kw_emb,
                  z["desc_emb"].astype(np.float32), img_emb, z["has_img"], text_model,
                  clip_text_model)
        print(f"[pictogramIndex] loaded '{idx.name}': {len(items)} pictures, "
              f"{len(kw_texts)} keywords, image vectors: "
              f"{int(idx.has_img.sum()) if img_emb is not None else 0}")
        return idx

    # ── helpers used by the server ─────────────────────────────────────────

    def image_path(self, item: dict) -> str:
        return item_image_path(self.library_dir, self.info, item)

    def image_path_for_id(self, item_id: str) -> str | None:
        i = self._id_to_idx.get(item_id)
        return None if i is None else self.image_path(self.items[i])

    @staticmethod
    def image_url(item: dict) -> str:
        return "/pictos/" + quote(item["id"], safe="")

    @property
    def has_image_channel(self) -> bool:
        return self.img_emb is not None and self.clip_text_model is not None

    # ── the three score channels ───────────────────────────────────────────

    def _text_scores(self, entity: str) -> np.ndarray:
        e = _norm(entity)

        def compute():
            n = len(self.items)
            q = self._query_vec(e)
            sims = self.kw_emb @ q
            sem = np.full(n, 0.0, dtype=np.float32)
            np.maximum.at(sem, self.kw_item, sims)
            scores = np.clip(sem, 0.0, _SEMANTIC_CAP)
            for row, val in self._containment(e):
                i = int(self.kw_item[row])
                scores[i] = max(scores[i], val)
            for i in self._exact.get(e, []):
                scores[i] = 1.0
            return scores

        return self._text_cache.get_or(e, compute)

    def _query_vec(self, e: str) -> np.ndarray:
        return self._qvec_cache.get_or(e, lambda: _encode_text(self.text_model, [e])[0])

    def matched_keyword(self, entity: str, item_index: int) -> str:
        """The keyword of a picture that best explains the match (for hover/debug)."""
        e = _norm(entity)
        rows = self._item_rows[item_index]
        if not rows:
            return self.items[item_index].get("title", "")
        for r in rows:
            if self.kw_texts[r] == e:
                return self.kw_texts[r]
        sims = self.kw_emb[rows] @ self._query_vec(e)
        return self.kw_texts[rows[int(np.argmax(sims))]]

    def _containment(self, e: str) -> list[tuple[int, float]]:
        """Whole-word string matches, like the thesis code:
        keyword contains the entity ("brush" in "tooth brush")      -> 0.95
        entity contains a keyword that includes its head noun
            ("laundry bag" / "bag" in "mesh laundry bag")           -> 0.80..0.95
            (the more of the entity the keyword covers, the higher)
        entity contains only a modifier ("hand" in "hand wraps")    -> 0.60"""
        toks = _tokens(e)
        if not toks:
            return []
        posting_sets = [self._postings.get(t, set()) for t in toks]
        out = []
        if all(posting_sets):
            pat = re.compile(rf"\b{re.escape(e)}\b")
            for r in set.intersection(*posting_sets):
                if self.kw_texts[r] != e and pat.search(self.kw_texts[r]):
                    out.append((r, 0.95))
        if len(toks) > 1:
            tokset, head = set(toks), toks[-1]
            for r in set().union(*posting_sets):
                kt = self._kw_tokens[r]
                if kt and len(kt) < len(toks) and set(kt) <= tokset and \
                        re.search(rf"\b{re.escape(self.kw_texts[r])}\b", e):
                    covers_head = head in kt or head.rstrip("s") in kt
                    out.append((r, 0.80 + 0.15 * len(kt) / len(toks) if covers_head else 0.60))
        return out

    def _context_scores(self, context: str) -> np.ndarray | None:
        c = _norm(context)
        if not c:
            return None
        return self._ctx_cache.get_or(
            c, lambda: np.clip(self.desc_emb @ _encode_text(self.text_model, [c])[0], 0.0, 1.0))

    def _image_scores(self, entity: str) -> np.ndarray | None:
        if not self.has_image_channel:
            if not self._warned_no_clip and self.img_emb is not None:
                print("[pictogramIndex] no CLIP text model loaded -> image channel off")
                self._warned_no_clip = True
            return None
        e = _norm(entity)

        def compute():
            q = _encode_text(self.clip_text_model, [self.clip_prompt.format(e)])[0]
            sims = self.img_emb @ q
            valid = self.has_img
            mu, sd = float(sims[valid].mean()), float(sims[valid].std()) + 1e-6
            conf = np.clip(((sims - mu) / sd - Z_LO) / (Z_HI - Z_LO), 0.0, 1.0)
            conf[~valid] = 0.0
            return conf.astype(np.float32)

        return self._img_cache.get_or(e, compute)

    # ── search ─────────────────────────────────────────────────────────────

    def _kind_mask(self, kinds: str) -> np.ndarray:
        if kinds == "action":
            return self._has_verb
        if kinds == "object":
            return ~self._verb_only
        return np.ones(len(self.items), dtype=bool)

    def search(self, entity: str, context: str = "", k: int = 5, mode: str | None = None,
               kinds: str = "object", weights: dict | None = None) -> list[dict]:
        """Top-k pictures for `entity`, best first.

        Each hit: {"item", "index", "keyword", "score", "text", "context", "image"}.
        kinds: "object" (skip verb-only pictograms), "action", or "any".
        """
        mode = mode or self.default_mode
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        w = {**self.weights, **(weights or {})}

        text = self._text_scores(entity)
        ctx = self._context_scores(context) if mode in ("text+context", "hybrid") else None
        img = self._image_scores(entity) if mode in ("image", "hybrid") else None

        if mode == "text":
            final = text
        elif mode == "image":
            final = img if img is not None else text
        else:
            parts = [(w["text"], text)]
            if ctx is not None:
                parts.append((w["context"], ctx))
            if img is not None:
                parts.append((w["image"], img))
            total = sum(p[0] for p in parts) or 1.0
            final = sum(p[0] * p[1] for p in parts) / total

        final = np.where(self._kind_mask(kinds), final, -1.0)
        k = max(1, min(k, len(final)))
        top = np.argpartition(-final, k - 1)[:k]
        top = top[np.argsort(-final[top], kind="stable")]
        return [{
            "item": self.items[i],
            "index": int(i),
            "keyword": self.matched_keyword(entity, int(i)),
            "score": round(float(final[i]), 4),
            "text": round(float(text[i]), 4),
            "context": None if ctx is None else round(float(ctx[i]), 4),
            "image": None if img is None else round(float(img[i]), 4),
        } for i in top if final[i] > -1.0]

    def best_match(self, entity: str, context: str = "", mode: str | None = None,
                   threshold: float | None = None) -> dict | None:
        mode = mode or self.default_mode
        thr = self.thresholds[mode] if threshold is None else threshold
        hits = self.search(entity, context=context, k=1, mode=mode)
        return hits[0] if hits and hits[0]["score"] >= thr else None

    # ── actions ────────────────────────────────────────────────────────────

    @staticmethod
    def action_phrases(doc, instruction_verbs: set[str] | None = None) -> list[str]:
        """'Put on your socks.' -> ['put on socks', 'put socks', ..., 'put on', 'put']

        Steps are imperatives, which small spaCy models often mis-tag
        ('Spit' as a proper noun), so the first word counts as the verb if it is
        tagged VERB, is a known instruction verb, or governs an object/particle.
        Objects: the direct object, then nouns inside it ('surfaces of your
        teeth' -> 'surfaces', 'teeth'), else the first nouns of the sentence."""
        if len(doc) == 0:
            return []

        def looks_like_verb(t):
            return (t.pos_ == "VERB"
                    or (instruction_verbs and t.lemma_.lower() in instruction_verbs)
                    or (t.dep_ == "ROOT" and any(c.dep_ in ("dobj", "obj", "prt")
                                                 for c in t.children)))

        verb = doc[0] if looks_like_verb(doc[0]) else \
            next((t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)
        if verb is None:
            return []
        v = verb.lemma_.lower()
        if v in {"be", "have", "do"}:
            return []
        prt = " ".join(c.text.lower() for c in verb.children if c.dep_ == "prt")

        objs = []
        dobj = next((c for c in verb.children if c.dep_ in ("dobj", "obj")), None)
        if dobj is not None:
            objs.append(dobj)
            objs += [t for t in dobj.subtree if t.pos_ == "NOUN" and t.i != dobj.i]
        if not objs:
            objs = [t for t in doc[verb.i + 1:] if t.pos_ == "NOUN"][:2]

        phrases = []
        for n, obj in enumerate(objs[:3]):
            variants = [obj.text.lower(), obj.lemma_.lower()]
            if n == 0:
                comp = " ".join(c.text.lower() for c in obj.children if c.dep_ == "compound")
                if comp:
                    variants.insert(0, f"{comp} {obj.text.lower()}")
            for o in dict.fromkeys(variants):
                if prt:
                    phrases.append(f"{v} {prt} {o}")
                phrases.append(f"{v} {o}")
        if prt:
            phrases.append(f"{v} {prt}")
        phrases.append(v)
        return [_norm(p) for p in dict.fromkeys(phrases)]

    def best_action(self, sentence: str, nlp, mode: str | None = None,
                    instruction_verbs: set[str] | None = None) -> dict | None:
        """Action pictogram for a step, or None. Returns the hit plus 'phrase'."""
        if not self._has_verb.any():
            return None  # library has no action pictures (e.g. the legacy one)
        phrases = self.action_phrases(nlp(sentence), instruction_verbs)
        if not phrases:
            return None
        mode = mode or self.default_mode
        thr = self.thresholds[mode]
        verb_only = phrases[-1]
        for ph in phrases:
            hits = self.search(ph, context=sentence, k=1, mode=mode, kinds="action")
            if not hits:
                continue
            h = hits[0]
            need_text = 1.0 if ph == verb_only else ACTION_PHRASE_MIN
            if h["text"] >= need_text and h["score"] >= thr:
                return {**h, "phrase": ph}
        return None


# ── one shared index for the whole app ───────────────────────────────────────

_ACTIVE: PictogramIndex | None = None
_ACTIVE_LOADED = False


def load_clip_model(name: str):
    from sentence_transformers import SentenceTransformer
    print(f"[pictogramIndex] loading CLIP model {name} ...")
    return SentenceTransformer(name)


def _parse_weights(s: str) -> dict:
    vals = [float(x) for x in s.split(",")]
    return dict(zip(("text", "context", "image"), vals))


def get_active_index() -> PictogramIndex | None:
    """The library chosen with the PICTO_LIBRARY environment variable, loaded once.

    PICTO_LIBRARY   folder built with buildPictogramIndex.py, e.g. pictograms/arasaac
                    (unset -> None, and the app uses the old size-512 code path)
    PICTO_MODE      text | text+context | image | hybrid      (default: hybrid)
    PICTO_WEIGHTS   "text,context,image" weights, e.g. "0.6,0.2,0.2"
    PICTO_THRESHOLD minimum score to show a picture for the chosen mode
    """
    global _ACTIVE, _ACTIVE_LOADED
    if _ACTIVE_LOADED:
        return _ACTIVE
    _ACTIVE_LOADED = True
    path = os.environ.get("PICTO_LIBRARY", "").strip()
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    from modelLoader import EMBED_MODEL

    mode = os.environ.get("PICTO_MODE", "hybrid").strip() or "hybrid"
    with open(os.path.join(path, BUILD_INFO_FILE), encoding="utf-8") as f:
        clip_name = json.load(f).get("clip_text_model")
    clip = load_clip_model(clip_name) if (clip_name and mode in ("image", "hybrid")) else None

    _ACTIVE = PictogramIndex.load(path, EMBED_MODEL, clip)
    _ACTIVE.default_mode = mode
    if os.environ.get("PICTO_WEIGHTS"):
        _ACTIVE.weights.update(_parse_weights(os.environ["PICTO_WEIGHTS"]))
    if os.environ.get("PICTO_THRESHOLD"):
        _ACTIVE.thresholds[mode] = float(os.environ["PICTO_THRESHOLD"])
    print(f"[pictogramIndex] active library '{_ACTIVE.name}', mode={mode}, "
          f"weights={_ACTIVE.weights}, threshold={_ACTIVE.thresholds[mode]}")
    return _ACTIVE
