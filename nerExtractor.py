import re
from functools import lru_cache
from sentence_transformers import util
from modelLoader import NLP, EMBED_MODEL

_ABSTRACT_NOUNS = {
    # time
    "minute", "minutes", "second", "seconds", "hour", "hours",
    "time", "times", "moment", "moments", "day", "days", "week", "weeks",
    "while", "duration",
    # quantity / measurement
    "amount", "number", "quantity", "level", "size", "way", "ways",
    "step", "steps", "part", "parts", "piece", "pieces",
    "bit", "bits", "lot", "lots",
    # sensation / state
    "heat", "cold", "warmth", "temperature",
    "pressure", "force", "motion", "direction", "side", "sides",
    "end", "ends", "top", "bottom", "middle", "center", "centre",
    "front", "back", "edge", "edges", "corner", "corners",
    "surface", "surfaces", "area", "areas",
    # discourse
    "example", "examples", "tip", "tips", "note", "notes",
    "option", "options", "choice", "choices", "result", "results",
    "instruction", "instructions", "method", "methods",
    # generic placeholders
    "thing", "things", "item", "items", "object", "objects",
    "stuff", "kind", "type", "types", "sort", "sorts",
}

_INSTRUCTION_VERBS = {
    "add", "apply", "boil", "break", "bring", "brush", "build", "buy",
    "carry", "check", "choose", "chop", "clean", "close", "collect",
    "combine", "connect", "cook", "cool", "cover", "cross", "cut",
    "dip", "do", "drag", "draw", "drop", "dry", "empty",
    "fasten", "feed", "fill", "find", "finish", "fold", "follow", "fry",
    "gather", "get", "give", "glue", "grab", "grasp", "grip", "hand",
    "hang", "heat", "help", "hold", "iron", "join", "knot", "lay",
    "leave", "lift", "line", "load", "lock", "look", "make", "mark",
    "match", "measure", "melt", "mix", "move", "open", "pack", "paint",
    "pat", "peel", "pick", "place", "plug", "pour", "press", "pull",
    "push", "put", "raise", "reach", "read", "remove", "repeat", "rinse",
    "roll", "rub", "saw", "scrub", "seal", "select", "separate", "set",
    "shake", "shape", "share", "shop", "show", "shut", "sit", "slice",
    "slide", "snap", "soak", "spread", "squeeze", "stack", "stand",
    "start", "steam", "stick", "stir", "stop", "store", "stretch",
    "swap", "sweep", "tape", "tap", "test", "throw", "tie", "tilt",
    "toss", "touch", "transfer", "try", "tuck", "turn", "twist", "use",
    "wait", "walk", "warm", "wash", "watch", "water", "wave", "wear",
    "weigh", "wet", "wipe", "work", "wrap", "write", "zip",
}

def _normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _detect_imperative(doc) -> bool:
    if len(doc) == 0:
        return False
    first = doc[0]
    if first.lemma_.lower() in _INSTRUCTION_VERBS:
        return True
    if first.pos_ == "VERB":
        return True
    return False


def extract_entities(sentence: str) -> list[dict]:
    doc = NLP(sentence)
    is_imperative = _detect_imperative(doc)

    results: list[dict] = []
    seen_surfaces: set[str] = set()
    consumed_token_ids: set[int] = set()

    # 1. Real named entities first.
    _SKIP_NER_LABELS = {"DATE", "TIME", "PERCENT", "MONEY",
                        "QUANTITY", "ORDINAL", "CARDINAL"}
    for ent in doc.ents:
        if ent.label_ in _SKIP_NER_LABELS:
            continue
        surface = ent.text.strip()
        norm = _normalize_phrase(surface)
        if not norm or len(norm) <= 2:
            continue
        if norm in seen_surfaces:
            continue
        if all(tok.lemma_.lower() in _ABSTRACT_NOUNS for tok in ent):
            continue
        if is_imperative and ent.start == 0 and len(ent) == 1:
            if ent[0].lemma_.lower() in _INSTRUCTION_VERBS:
                continue
        seen_surfaces.add(norm)
        for tok in ent:
            consumed_token_ids.add(tok.i)
        results.append({
            "surface": surface,
            "lemma":   norm,
            "start":   ent.start_char,
            "end":     ent.end_char,
        })

    # 2. Noun chunks.
    for chunk in doc.noun_chunks:
        if any(t.i in consumed_token_ids for t in chunk):
            continue

        root = chunk.root
        if root.pos_ == "PRON":
            continue
        if root.pos_ not in ("NOUN", "PROPN"):
            continue
        if root.lemma_.lower() in _ABSTRACT_NOUNS:
            continue

        if is_imperative and chunk.start == 0:
            if root.lemma_.lower() in _INSTRUCTION_VERBS:
                continue

        if len(chunk) == 1 and root.lemma_.lower() in _INSTRUCTION_VERBS:
            has_object = any(t.dep_ in ("dobj", "obj") for t in root.children)
            if has_object or root.dep_ == "ROOT":
                continue

        kept_tokens = [t for t in chunk
                       if t.pos_ not in ("DET", "PRON")
                       and not t.is_punct]
        if not kept_tokens:
            continue

        if len(kept_tokens) > 3:
            kept_tokens = kept_tokens[-3:]

        surface = " ".join(t.text for t in kept_tokens)
        norm = _normalize_phrase(surface)
        if not norm or len(norm) <= 2:
            continue
        if norm in seen_surfaces:
            continue
        if norm in _ABSTRACT_NOUNS:
            continue

        start_char = kept_tokens[0].idx
        end_char = kept_tokens[-1].idx + len(kept_tokens[-1].text)

        seen_surfaces.add(norm)
        results.append({
            "surface": surface,
            "lemma":   norm,
            "start":   start_char,
            "end":     end_char,
        })

    results.sort(key=lambda r: r["start"])
    return results


@lru_cache(maxsize=8192)
def _embed(text: str):
    return EMBED_MODEL.encode(text, convert_to_tensor=True)


def score_entity_against_title(entity: str, title: str) -> float:
    # Kept for /api/test_match and ad-hoc use. NOT used in the hot path anymore.
    e = _normalize_phrase(entity)
    t = _normalize_phrase(title)
    if not e or not t:
        return 0.0
    if e == t:
        return 1.0
    if re.search(rf"\b{re.escape(e)}\b", t) or re.search(rf"\b{re.escape(t)}\b", e):
        return 0.95
    sim = float(util.cos_sim(_embed(e), _embed(t))[0][0])
    sim = max(0.0, min(0.9, sim))
    return round(sim, 3)

_TITLE_CACHE = {
    "id": None, "len": None,
    "raw": None, "norm": None, "files": None, "emb": None,
}


def _ensure_title_matrix(titles_with_files) -> None:
    # Recompute only if the titles list actually changed (it's a stable singleton
    # in the server, so this runs exactly once).
    if (_TITLE_CACHE["id"] == id(titles_with_files)
            and _TITLE_CACHE["len"] == len(titles_with_files)
            and _TITLE_CACHE["emb"] is not None):
        return
    raw   = [t for t, _f in titles_with_files]
    files = [f for _t, f in titles_with_files]
    norm  = [_normalize_phrase(t) for t in raw]
    emb   = EMBED_MODEL.encode(norm, convert_to_tensor=True)  # ONE batched call
    _TITLE_CACHE.update(id=id(titles_with_files), len=len(titles_with_files),
                        raw=raw, norm=norm, files=files, emb=emb)
    print(f"[nerExtractor] Title matrix embedded once: {len(raw)} titles")


def best_image_for_entity(entity, titles_with_files, threshold=0.5):
    if not titles_with_files:
        return None
    _ensure_title_matrix(titles_with_files)

    e = _normalize_phrase(entity)
    if not e:
        return None

    raw   = _TITLE_CACHE["raw"]
    norm  = _TITLE_CACHE["norm"]
    files = _TITLE_CACHE["files"]

    # 1. Cheap string pass: exact (1.0) and word-boundary substring (0.95).
    #    Pure string ops over the list — no embeddings, very fast.
    best_str_idx, best_str_score = -1, 0.0
    for i, t in enumerate(norm):
        if e == t:
            return {"title": raw[i], "file_name": files[i], "score": 1.0}
        if re.search(rf"\b{re.escape(e)}\b", t) or re.search(rf"\b{re.escape(t)}\b", e):
            if 0.95 > best_str_score:
                best_str_idx, best_str_score = i, 0.95

    # 2. ONE batched semantic pass over the whole library (clamped to 0.9).
    sims = util.cos_sim(_embed(e), _TITLE_CACHE["emb"])[0]
    sem_idx = int(sims.argmax())
    sem_score = max(0.0, min(0.9, float(sims[sem_idx])))

    # 3. Take the stronger of substring vs semantic (substring 0.95 can beat
    #    the semantic cap of 0.9, preserving the old priority).
    if best_str_score >= sem_score:
        best_idx, best_score = best_str_idx, best_str_score
    else:
        best_idx, best_score = sem_idx, sem_score

    if best_idx < 0 or best_score < threshold:
        return None
    return {"title": raw[best_idx], "file_name": files[best_idx],
            "score": round(best_score, 3)}


def extract_and_match(sentence, titles_with_files, threshold=0.5):
    """
    Returns dicts with the fields the frontend needs, INCLUDING char offsets
    (`start`, `end`) so the frontend can highlight the entity in the step text.
    """
    out = []
    for ent in extract_entities(sentence):
        m = best_image_for_entity(ent["lemma"], titles_with_files, threshold)
        base = {
            "chunk": ent["surface"],
            "start": ent["start"],
            "end":   ent["end"],
        }
        if m is None:
            base.update({
                "image": None,
                "file_name": None,
                "image_url": None,
                "score": 0.0,
                "matched": False,
            })
        else:
            base.update({
                "image":     m["title"].title(),
                "file_name": m["file_name"],
                "image_url": f"/images/{m['file_name']}",
                "score":     m["score"],
                "matched":   True,
            })
        out.append(base)
    return out

# ── New library-agnostic path (used when PICTO_LIBRARY is set) ────────────────
# Same output shape as extract_and_match(), so the frontend needs no changes,
# plus `pictogram_id` and the per-channel scores for debugging / evaluation.

def extract_and_match_index(sentence, index, mode=None, threshold=None):
    out = []
    for ent in extract_entities(sentence):
        hit = index.best_match(ent["lemma"], context=sentence, mode=mode,
                               threshold=threshold)
        base = {
            "chunk": ent["surface"],
            "start": ent["start"],
            "end":   ent["end"],
        }
        if hit is None:
            base.update({
                "image": None,
                "file_name": None,
                "image_url": None,
                "score": 0.0,
                "matched": False,
            })
        else:
            item = hit["item"]
            base.update({
                "image":        hit["keyword"].title(),
                "file_name":    item["file_name"],
                "image_url":    index.image_url(item),
                "pictogram_id": item["id"],
                "score":        round(min(hit["score"], 1.0), 3),
                "channels":     {"text": hit["text"], "context": hit["context"],
                                 "image": hit["image"]},
                "matched":      True,
            })
        out.append(base)
    return out


def match_action_index(sentence, index, mode=None):
    """Action pictogram for the whole step ("brush teeth"), or None."""
    hit = index.best_action(sentence, NLP, mode=mode,
                            instruction_verbs=_INSTRUCTION_VERBS)
    if hit is None:
        return None
    item = hit["item"]
    return {
        "phrase":       hit["phrase"],
        "image":        hit["keyword"].title(),
        "file_name":    item["file_name"],
        "image_url":    index.image_url(item),
        "pictogram_id": item["id"],
        "score":        round(min(hit["score"], 1.0), 3),
    }
