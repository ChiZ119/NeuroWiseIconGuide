import os
import json
import asyncio
import threading
import time
from collections import defaultdict

from flask import Flask, jsonify, send_from_directory, request
import modelLoader  # noqa: F401
from modelLoader import EMBED_MODEL
from sentence_transformers import util

from debugPaChong import run_retrieval_system
from nerExtractor import extract_and_match

from openai import OpenAI

_CHAT_HITS = defaultdict(list)     # ip (recent timestamps)
_PER_IP_PER_HOUR = 15              # one visitor: 15 chat messages/hour
_GLOBAL_DAILY_CAP = 1000           # whole site: 1000 chat calls/day
_global_day = {"date": None, "count": 0}

def _chat_allowed(ip: str) -> bool:
    now = time.time()
    today = time.strftime("%Y-%m-%d")
    if _global_day["date"] != today:
        _global_day["date"], _global_day["count"] = today, 0
    if _global_day["count"] >= _GLOBAL_DAILY_CAP:
        return False
    hits = _CHAT_HITS[ip]
    hits[:] = [t for t in hits if t > now - 3600]   # keep only the last hour
    if len(hits) >= _PER_IP_PER_HOUR:
        return False
    hits.append(now)
    _global_day["count"] += 1
    return True

_AI_CLIENT = OpenAI(
    api_key=os.environ.get("ChiBscProject", ""),
    base_url="https://api.deepseek.com",   # OpenAI-compatible endpoint
)
AI_MODEL = "deepseek-v4-flash"

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_LIBRARY_DIR = os.path.join(BASE_DIR, "size-512")
IMAGE_FOLDER      = os.path.join(IMAGE_LIBRARY_DIR, "images")

_live_search_lock = threading.Lock()
SCORE_THRESHOLD = 0.5

FUZZY_STRONG_THRESHOLD = 0.7   # accept cached knowledgeBase entry directly
FUZZY_WEAK_THRESHOLD   = 0.55  # only use as fallback if WikiHow fails

# How many chips to show on the homepage.
POPULAR_LIMIT = 10


def load_image_index():
    meta_path = os.path.join(IMAGE_LIBRARY_DIR, "meta.json")
    if not os.path.exists(meta_path):
        print(f"meta.json not found at {meta_path}")
        return {}, []
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    index, all_titles = {}, []
    for item in meta.get("items", []):
        title = item.get("title", "").strip()
        slug  = item.get("slug",  "").strip().lower()
        fname = item.get("file_name", "")
        if not title or not fname:
            continue
        tl = title.lower()
        all_titles.append((tl, fname))
        index[tl] = fname
        if slug and slug not in index:
            index[slug] = fname
    print(f"📷  Image index: {len(index)} keys, {len(all_titles)} images")
    return index, all_titles


IMAGE_INDEX, ALL_TITLES = load_image_index()


# ── KB enrichment ───────────────────────────────────────────────────────────

def find_all_entities_for_step(step_text: str) -> list[dict]:
    return extract_and_match(step_text, ALL_TITLES, threshold=SCORE_THRESHOLD)


def enrich_steps(entry: dict, query: str = "") -> tuple[dict, bool]:
    if not entry.get("query") and query:
        entry["query"] = query.title()

    was_enriched = False
    for step in entry.get("steps", []):
        existing = step.get("entities")
        is_fresh = (
            isinstance(existing, list)
            and all(isinstance(e, dict) and "image_url" in e and "start" in e
                    for e in existing)
        )
        if not is_fresh:
            step["entities"] = find_all_entities_for_step(step.get("step", ""))
            was_enriched = True

        matched = [e for e in step["entities"] if e.get("matched")]
        step["image_url"] = matched[0]["image_url"] if matched else None
    return entry, was_enriched


# ── Knowledge base I/O ──────────────────────────────────────────────────────

KB_PATH = os.path.join(BASE_DIR, "knowledgeBase.json")


def load_kb() -> dict:
    if not os.path.exists(KB_PATH):
        return {}
    with open(KB_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_kb(kb: dict) -> None:
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)


# ── Search popularity counter ─────────────────────────────────────────────────
# Every successful search bumps a counter. /api/popular returns the top N so the
# homepage only shows the most-used chips; everything else stays in the KB and is
# still reachable by typing in the search box.

COUNTS_PATH = os.path.join(BASE_DIR, "searchCounts.json")
_counts_lock = threading.Lock()


def _load_counts() -> dict:
    if not os.path.exists(COUNTS_PATH):
        return {}
    try:
        with open(COUNTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _bump_count(query_key: str) -> None:
    if not query_key:
        return
    with _counts_lock:
        counts = _load_counts()
        counts[query_key] = counts.get(query_key, 0) + 1
        with open(COUNTS_PATH, "w", encoding="utf-8") as f:
            json.dump(counts, f, ensure_ascii=False, indent=2)


def prewarm_kb():
    kb = load_kb()
    if not kb:
        print("[prewarm] no KB to prewarm")
        return
    t0 = time.time()
    changed = False
    for k, v in kb.items():
        _, was_enriched = enrich_steps(v, k)
        if was_enriched:
            changed = True
    if changed:
        save_kb(kb)
    print(f"[prewarm] KB entity extraction done in {time.time() - t0:.2f}s "
          f"({'updated' if changed else 'all fresh, no save'})")


# ── Semantic fuzzy match ────────────────────────────────────────────────────

def semantic_kb_match(query: str, kb: dict) -> tuple[str | None, float]:
    keys = list(kb.keys())
    if not keys:
        return None, 0.0
    q_vec = EMBED_MODEL.encode(query, convert_to_tensor=True)
    k_vecs = EMBED_MODEL.encode(keys, convert_to_tensor=True)
    sims = util.cos_sim(q_vec, k_vecs)[0]
    best_idx = int(sims.argmax())
    return keys[best_idx], float(sims[best_idx])


# ── Live WikiHow ────────────────────────────────────────────────────────────

def run_live_search(query: str) -> dict | None:
    with _live_search_lock:
        try:
            result = asyncio.run(run_retrieval_system(live_query=query))
        except Exception as e:
            print(f"[live search] crashed: {e}")
            return None

    if result.get("perfect_matches"):
        first = next(iter(result["perfect_matches"].values()))
        return {
            "query":  query.title(),
            "source": first["url"],
            "origin": "crawler_live",
            "url":    first["url"],
            "steps":  first["steps"],
        }
    if result.get("manual_review_list"):
        best = result["manual_review_list"][0]
        return {
            "query":  query.title(),
            "source": best["url"],
            "origin": "crawler_live_partial",
            "url":    best["url"],
            "steps":  best.get("steps", []),
        }
    return None


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    html_path = os.path.join(BASE_DIR, "stepCardDemo.html")
    if not os.path.exists(html_path):
        return f"<h2>stepCardDemo.html not found in {BASE_DIR}</h2>", 404
    return send_from_directory(BASE_DIR, "stepCardDemo.html")


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


@app.route("/api/knowledgeBase")
def knowledgeBase():
    kb = load_kb()
    if not kb:
        return jsonify({"error": "knowledgeBase.json not found or empty"}), 404
    result, any_changed = {}, False
    for k, v in kb.items():
        enriched, was_enriched = enrich_steps(v, k)
        result[k] = enriched
        if was_enriched:
            kb[k] = enriched
            any_changed = True
    if any_changed:
        save_kb(kb)
    return jsonify(result)


@app.route("/api/popular")
def popular():
    """Top N query labels by search count, padded with KB entries if needed."""
    kb = load_kb()
    counts = _load_counts()

    ranked = sorted(
        ((k, c) for k, c in counts.items() if k in kb),
        key=lambda kv: kv[1], reverse=True,
    )
    top_labels = [kb[k].get("query", k.title()) for k, _ in ranked[:POPULAR_LIMIT]]

    # If fewer than POPULAR_LIMIT have ever been searched, pad with other KB
    # entries so the homepage never looks empty.
    if len(top_labels) < POPULAR_LIMIT:
        for k, v in kb.items():
            label = v.get("query", k.title())
            if label not in top_labels:
                top_labels.append(label)
            if len(top_labels) >= POPULAR_LIMIT:
                break

    return jsonify(top_labels)


@app.route("/api/search")
def search():
    q = request.args.get("q", "").lower().strip()
    if not q:
        return jsonify({"error": "empty query"}), 400

    kb = load_kb()

    # Stage 1: exact KB hit
    if q in kb:
        print(f"[search] '{q}' — exact KB hit")
        _bump_count(q)
        enriched, was_enriched = enrich_steps(kb[q], q)
        if was_enriched:
            kb[q] = enriched
            save_kb(kb)
        return jsonify(enriched)

    # Stage 2: STRONG semantic match against cached KB keys.
    best_key, best_sim = semantic_kb_match(q, kb)
    if best_key and best_sim >= FUZZY_STRONG_THRESHOLD:
        print(f"[search] '{q}' — strong fuzzy hit on '{best_key}' (sim={best_sim:.2f})")
        _bump_count(best_key)   # credit the entry the user actually got
        enriched, was_enriched = enrich_steps(kb[best_key], q)
        if was_enriched:
            kb[best_key] = enriched
            save_kb(kb)
        return jsonify(enriched)

    # Stage 3: live WikiHow.
    print(f"[search] '{q}' — falling back to live WikiHow "
          f"(best cached sim was {best_sim:.2f} on '{best_key}')")
    live_entry = run_live_search(q)
    if live_entry:
        enriched, _ = enrich_steps(live_entry, q)
        kb[q] = enriched
        save_kb(kb)
        _bump_count(q)
        print(f"[search] '{q}' — live search succeeded, cached")
        return jsonify(enriched)

    # Stage 4: WEAK fuzzy fallback.
    if best_key and best_sim >= FUZZY_WEAK_THRESHOLD:
        print(f"[search] '{q}' — weak fuzzy fallback on '{best_key}' "
              f"(sim={best_sim:.2f}); flagging as approximate")
        enriched, _ = enrich_steps(kb[best_key], q)
        enriched = dict(enriched)
        enriched["approximate_match"] = True
        enriched["approximate_for"]   = q
        return jsonify(enriched)

    # Stage 5: real failure.
    print(f"[search] '{q}' — no results from any stage")
    return jsonify({
        "error":  "not_found",
        "origin": "no_results",
        "query":  q,
        "message": (
            f"We couldn't find anything for \"{q}\" on WikiHow yet. "
            f"This topic will be added in a future update."
        ),
    }), 404


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Tiny AI helper. The frontend sends the child's message plus the task and the
    on-screen step texts, so the model can answer questions about the current
    card ("what does step 3 mean?") instead of guessing.
    """
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()
    if not _chat_allowed(ip):
        return jsonify({"reply": "You've reached the question limit for now. "
                                 "Please take a short break and try again later, thanks."}), 200
    if not _AI_CLIENT.api_key:
        return jsonify({"reply": "The assistant isn't set up yet "
                                 "(missing DEEPSEEK_API_KEY)."}), 200

    data = request.get_json(force=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"reply": "Ask me anything about this task!"}), 200

    task  = data.get("task", "")
    steps = data.get("steps", [])
    if isinstance(steps, list):
        steps_text = " | ".join(str(s) for s in steps)
    else:
        steps_text = str(steps)

    system = (
        "You are a gentle, friendly helper for children learning everyday tasks, "
        "including children with autism. Use very short sentences and simple words. "
        "Always reply in the same language the child writes in. "
        "Be calm, kind, and encouraging. Never give unsafe advice. "
        "If the question is not about the task, gently steer back. "
        f"The child is learning: '{task}'. The steps shown are: {steps_text}"
    )

    try:
        resp = _AI_CLIENT.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=300,
            temperature=0.5,
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        print(f"[chat] AI call failed: {e}")
        reply = "Sorry, I can't answer right now. Please try again in a moment."

    return jsonify({"reply": reply})


@app.route("/api/test_match")
def test_match():
    q = request.args.get("q", "brush teeth").lower().strip()
    kb = load_kb()
    entry = kb.get(q)
    if not entry:
        best_key, _ = semantic_kb_match(q, kb)
        entry = kb.get(best_key) if best_key else None
    if not entry:
        return jsonify({"error": f"'{q}' not found in knowledge base"})
    results = []
    for step in entry.get("steps", []):
        entities = find_all_entities_for_step(step.get("step", ""))
        matched   = [e for e in entities if e["matched"]]
        unmatched = [e for e in entities if not e["matched"]]
        results.append({
            "step":      step.get("step"),
            "entities":  entities,
            "matched":   len(matched),
            "missing":   len(unmatched),
        })
    return jsonify({"query": q, "steps": results})


@app.route("/api/debug")
def debug():
    kb = load_kb()
    return jsonify({
        "base_dir":               BASE_DIR,
        "image_library_dir":      IMAGE_LIBRARY_DIR,
        "image_folder":           IMAGE_FOLDER,
        "knowledge_base_found":   os.path.exists(KB_PATH),
        "knowledge_base_entries": list(kb.keys()),
        "html_found":             os.path.exists(os.path.join(BASE_DIR, "stepCardDemo.html")),
        "meta_json_found":        os.path.exists(os.path.join(IMAGE_LIBRARY_DIR, "meta.json")),
        "image_index_size":       len(IMAGE_INDEX),
        "image_library_size":     len(ALL_TITLES),
        "score_threshold":        SCORE_THRESHOLD,
        "fuzzy_strong":           FUZZY_STRONG_THRESHOLD,
        "fuzzy_weak":             FUZZY_WEAK_THRESHOLD,
        "search_counts":          _load_counts(),
        "ai_model":               AI_MODEL,
        "ai_key_present":         bool(_AI_CLIENT.api_key),
    })


# ── Boot ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" Starting WikiHow+AI server")
    print("=" * 60)
    print(f"Project folder  : {BASE_DIR}")
    print(f"Image library   : {IMAGE_LIBRARY_DIR}")
    kb = load_kb()
    if kb:
        print(f"Knowledge base  : {list(kb.keys())}")
    else:
        print("knowledgeBase.json not found — run buildKnowledgeBase.py first!")

    prewarm_kb()

    print("=" * 60)
    print(" Server ready at http://127.0.0.1:8080")
    print("=" * 60)
    app.run(debug=False, port=8080, use_reloader=False)