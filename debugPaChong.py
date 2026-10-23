import json
import re
import asyncio
from urllib.parse import urlparse
import os

import aiohttp
from bs4 import BeautifulSoup
from sentence_transformers import util
import pandas as pd
import torch

from modelLoader import NLP as _nlp, EMBED_MODEL as _model

_BASE = os.path.dirname(os.path.abspath(__file__))
IMAGE_LIBRARY_XLSX = os.path.join(_BASE, "xlsxFiles", "Processed_Library_info.xlsx")

USE_BROWSER_FALLBACK = False

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def load_image_meta(file_path: str) -> list[dict]:
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"读取 meta 数据失败: {e}")
        return []


# ── Load the image library + its embeddings ONCE, at import time. ─────────────
# This is the single biggest speed change. Everything downstream reuses these.
# When PICTO_LIBRARY is set, coverage is scored against that library instead
# (the same index the server uses), and the old spreadsheet is not loaded.
from pictogramIndex import get_active_index
_PICTO_INDEX = get_active_index()

if _PICTO_INDEX is None:
    _IMAGE_META: list[dict] = load_image_meta(IMAGE_LIBRARY_XLSX)
else:
    _IMAGE_META = []
_IMAGE_TEXTS: list[str] = [str(img.get("entities", "")) for img in _IMAGE_META]
if _IMAGE_TEXTS:
    _IMAGE_EMB = _model.encode(_IMAGE_TEXTS, convert_to_tensor=True)
    print(f"[debugPaChong] Image library embedded once: {len(_IMAGE_TEXTS)} items")
else:
    _IMAGE_EMB = None
    if _PICTO_INDEX is None:
        print("[debugPaChong] WARNING: image library empty — no image matching available")


# Semantic categories that should NEVER yield a picture even if they parse as
# nouns.  Keep this list short; rely on the parser first.
_ABSTRACT_NOUNS = {
    # time & quantity
    "minute", "minutes", "second", "seconds", "hour", "hours",
    "time", "times", "moment", "day", "days", "week", "weeks",
    "amount", "number", "quantity", "level", "size", "way", "step",
    # sensation / state (no physical referent)
    "wet", "dry", "clean", "dirty", "smooth", "rough",
    "heat", "cold", "warmth", "temperature",
    "pressure", "force", "motion", "direction", "side",
    # discourse
    "example", "tip", "note", "option", "choice", "result",
}


def extract_noun_chunks(sentence: str) -> list[str]:
    doc = _nlp(sentence)
    chunks = []
    seen = set()

    for chunk in doc.noun_chunks:
        root = chunk.root.lemma_.lower()

        # Drop pronouns, abstract nouns, very short tokens
        if chunk.root.pos_ == "PRON":
            continue
        if root in _ABSTRACT_NOUNS:
            continue
        if len(root) <= 2:
            continue

        # Prefer the full noun phrase (e.g. "washing machine") but cap at 3 words
        # to avoid overly specific phrases that won't match the image library.
        words = [t.text.lower() for t in chunk
                 if not t.is_stop and t.pos_ not in ("DET", "PRON")]
        phrase = " ".join(words[:3]).strip()
        if not phrase:
            phrase = root

        if phrase not in seen:
            seen.add(phrase)
            chunks.append(phrase)

        # Also add the root word alone so we get both
        # "washing machine" AND "machine" as candidates
        if root != phrase and root not in seen:
            seen.add(root)
            chunks.append(root)

    return chunks


def find_all_entities_for_step(
    step_text: str,
    threshold: float = 0.55,
) -> list[dict]:
    """
    Extract noun chunks from a step and match each against the (precomputed)
    image library in a single batched encode + one cos_sim matrix multiply.
    """
    if _PICTO_INDEX is not None:
        from nerExtractor import extract_and_match_index
        return [{
            "chunk": e["chunk"],
            "entity": e["image"] if e["matched"] else None,
            "score": e["score"],
            "matched": e["matched"],
        } for e in extract_and_match_index(step_text, _PICTO_INDEX)]

    chunks = extract_noun_chunks(step_text)
    if not chunks or _IMAGE_EMB is None:
        return [{"chunk": c, "entity": None, "score": 0.0, "matched": False}
                for c in chunks]

    # ── Batched encode: one model call for ALL chunks in this step. ──
    chunk_embs = _model.encode(chunks, convert_to_tensor=True)
    sim = util.cos_sim(chunk_embs, _IMAGE_EMB)  # shape: [len(chunks), len(library)]

    raw_results = []
    for i, chunk in enumerate(chunks):
        scores = sim[i]
        best_idx = int(torch.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= threshold:
            img = _IMAGE_META[best_idx]
            raw_results.append({
                "chunk": chunk,
                "entity": img.get("entities"),
                "score": round(best_score, 3),
                "matched": True,
            })
            print(f"    '{chunk}' → {img.get('entities')} (score={best_score:.3f})")
        else:
            raw_results.append({
                "chunk": chunk, "entity": None, "score": 0.0, "matched": False,
            })
            print(f"    '{chunk}' → no match")

    # Deduplicate matched entities by score (keep best); preserve order via index.
    best_per_entity: dict[str, dict] = {}
    for r in raw_results:
        if not r["matched"]:
            continue
        entity = r["entity"]
        if entity not in best_per_entity or r["score"] > best_per_entity[entity]["score"]:
            best_per_entity[entity] = r

    results = []
    seen_entities: set[str] = set()
    for r in raw_results:
        if r["matched"]:
            entity = r["entity"]
            if entity in seen_entities:
                continue
            seen_entities.add(entity)
            results.append(best_per_entity[entity])
        else:
            results.append(r)

    return results


async def extract_wikihow_steps(url: str, session: aiohttp.ClientSession, query_emb=None,) -> list[str]:
    try:
        async with session.get(url, headers=HEADERS) as response:
            html = await response.text()

        soup = BeautifulSoup(html, "html.parser")
        method_sections = soup.select("div.section.steps")

        if len(method_sections) > 1 and query_emb is not None:
            headings = []
            for s in method_sections:
                h = s.select_one(".mw-headline, .altblock")
                headings.append(h.get_text(" ", strip=True) if h else "")
            sims = util.cos_sim(
                query_emb, _model.encode(headings, convert_to_tensor=True)
            )[0]
            step_container = method_sections[int(torch.argmax(sims))]
        elif method_sections:
            step_container = method_sections[0]    # first or the only method
        else:
            step_container = soup                  # no method wrapper found

        steps: list[str] = []
        for el in step_container.select(".step"):
            headline = _extract_headline(el)
            if headline and len(headline) > 5:
                steps.append(headline)

        print(f"  Steps found: {len(steps)} (from {len(method_sections) or 1} method section(s))")
        if len(steps) < 2:
            return []
        return steps[:12]
    except Exception as e:
        print(f"  extract_wikihow_steps EXCEPTION: {e}")
        return []


def _extract_headline(step_el) -> str:
    for junk in step_el.find_all(["ul", "ol", "sup", "script", "style"]):
        junk.extract()

    text = step_el.get_text(separator=" ", strip=True)
    text = re.sub(r"\[\d+\]", "", text) 
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)
    headline = sentences[0] if sentences else text
    if len(headline) < 25 and len(sentences) > 1:
        headline = (headline + " " + sentences[1]).strip()
    return headline


# Search: aiohttp first (fast), Playwright only as a lazy fallback
async def _search_wikihow_aiohttp(query: str, session: aiohttp.ClientSession) -> list[str]:
    try:
        url = f"https://en.wikihow.com/wikiHowTo?search={query.replace(' ', '+')}"
        print(f"  → (aiohttp) {url}")
        async with session.get(url, headers=HEADERS) as r:
            html = await r.text()
        soup = BeautifulSoup(html, "html.parser")
        box = soup.find("div", id="searchresults_list")
        if not box:
            return []
        links = [a.get("href", "") for a in box.find_all("a", class_="result_link")]
        print(f"  → {len(links)} links found (aiohttp)")
        return list(dict.fromkeys(links))[:3]
    except Exception as e:
        print(f"  → aiohttp search EXCEPTION: {type(e).__name__}: {e}")
        return []


async def _search_wikihow_browser(query: str, get_page) -> list[str]:
    """Fallback only. `get_page` lazily launches one shared browser/page."""
    try:
        page = await get_page()
        search_url = f"https://en.wikihow.com/wikiHowTo?search={query.replace(' ', '+')}"
        print(f"  → (browser fallback) {search_url}")
        await page.goto(search_url)
        await page.wait_for_selector("#searchresults_list", timeout=8000)
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        box = soup.find("div", id="searchresults_list")
        if not box:
            return []
        links = [a.get("href", "") for a in box.find_all("a", class_="result_link")]
        print(f"  → {len(links)} links found (browser)")
        return list(dict.fromkeys(links))[:3]
    except Exception as e:
        print(f"  → browser search EXCEPTION: {type(e).__name__}: {e}")
        return []


def _url_relevance_score(url: str, query_emb) -> float:
    """Numeric relevance score. query_emb is precomputed ONCE per query."""
    slug = urlparse(url).path.strip("/").replace("-", " ")
    slug_emb = _model.encode(slug, convert_to_tensor=True)
    return float(util.cos_sim(query_emb, slug_emb)[0][0])


async def process_single_query(
    query: str,
    session: aiohttp.ClientSession,
    get_page,
) -> dict:
    print(f"Query: {query}")

    # aiohttp search first; only fall back to a browser if it found nothing.
    article_urls = await _search_wikihow_aiohttp(query, session)
    if not article_urls and USE_BROWSER_FALLBACK and get_page is not None:
        article_urls = await _search_wikihow_browser(query, get_page)

    # Encode the query ONCE; reuse for every URL relevance score.
    query_emb = _model.encode(query, convert_to_tensor=True)

    scored_urls = [(url, _url_relevance_score(url, query_emb)) for url in article_urls]
    scored_urls.sort(key=lambda x: x[1], reverse=True)

    print("\n  Ranked candidates:")
    for url, score in scored_urls:
        slug = urlparse(url).path.strip("/").replace("-", " ")
        print(f"    {score:.3f}  {slug}")

    URL_REL_THRESHOLD = 0.55

    for url, url_score in scored_urls:
        if url_score < URL_REL_THRESHOLD:
            print(f"\n  Stopping: remaining URLs below relevance threshold ({URL_REL_THRESHOLD})")
            break

        print(f"\n  Processing top-ranked URL ({url_score:.3f}): {url}")
        steps = await extract_wikihow_steps(url, session, query_emb)
        if not steps:
            print("  No steps extracted, trying next URL")
            continue

        all_steps: list[dict] = []
        matched_count = 0

        for step in steps:
            print(f"\n  Step: {step}")
            entities = find_all_entities_for_step(step)
            matched_entities = [e for e in entities if e["matched"]]
            unmatched_entities = [e for e in entities if not e["matched"]]

            fully_matched = bool(matched_entities) and not unmatched_entities
            if fully_matched:
                matched_count += 1

            all_steps.append({
                "step": step,
                "entities": entities,
                "image": matched_entities[0]["entity"] if matched_entities else None,
                "fully_matched": fully_matched,
                "missing_chunks": [e["chunk"] for e in unmatched_entities],
            })

        total = len(steps)
        rate = matched_count / total if total else 0.0
        status = "perfect" if rate == 1.0 else "partial"

        print(f"\n  Coverage: {rate:.0%} ({matched_count}/{total}) — returning this article ({status})")

        return {
            "query": query,
            "status": status,
            "url": url,
            "rate": rate,
            "url_relevance": round(url_score, 3),
            "steps": all_steps,
        }

    return {
        "query": query,
        "status": "failed",
        "url": None,
        "rate": 0.0,
        "url_relevance": 0.0,
        "matched_steps": [],
        "unmatched_steps": [],
    }


async def run_retrieval_system(live_query: str | None = None) -> dict:
    # NOTE: image_meta + embeddings are already loaded at module import — we do
    # NOT reload the Excel file here anymore.

    if live_query:
        queries = [live_query.strip()]
        print(f"[Live mode] Running single query: '{live_query}'")
    else:
        queries = [
            "Pack a backpack", "Wash dishes", "Do laundry", "Ride a bike",
            "Set the table", "Put on headphones", "Use a glue stick",
            "Use scissors to cut", "Wear mask", "Put on clothes", "Comb hair",
        ]
        print(f"[Batch mode] Running {len(queries)} queries")

    perfect_matches: dict = {}
    manual_review_list: list[dict] = []
    failed_queries: list[str] = []

    # Lazy browser: only launched if aiohttp search comes back empty.
    holder = {"pw": None, "browser": None, "page": None}

    async def get_page():
        if holder["page"] is None:
            from playwright.async_api import async_playwright
            holder["pw"] = await async_playwright().start()
            holder["browser"] = await holder["pw"].chromium.launch(headless=True)
            holder["page"] = await holder["browser"].new_page()
        return holder["page"]

    async with aiohttp.ClientSession() as session:
        for query in queries:
            result = await process_single_query(query, session, get_page)

            if result["status"] == "perfect":
                perfect_matches[query] = {
                    "url": result["url"],
                    "steps": result["steps"],
                }
            elif result["status"] == "partial":
                manual_review_list.append({
                    "query": query,
                    "url": result["url"],
                    "rate": result["rate"],
                    "url_relevance": result.get("url_relevance", 0.0),
                    "steps": result["steps"],
                })
            else:
                failed_queries.append(query)

    # Tidy up the browser if the fallback ever launched it.
    if holder["browser"] is not None:
        await holder["browser"].close()
    if holder["pw"] is not None:
        await holder["pw"].stop()

    print("\n\nCLASS 1: Perfect match (all steps, all entities covered)")
    for q, data in perfect_matches.items():
        print(f"\n  {q}  ({data['url']})")
        for i, s in enumerate(data["steps"], 1):
            print(f"    Step {i}: {s['step']}")
            for e in s["entities"]:
                tag = f"{e['entity']} ({e['score']:.2f})" if e["matched"] else f"'{e['chunk']}' — no image"
                print(f"      {tag}")

    print("\n\nCLASS 2: Needs manual review (partial match)")
    for item in manual_review_list:
        print(f"\n  Query: {item['query']}  ({item['url']})")
        print(f"  Match rate: {item['rate']:.0%}")

    print("\n\nCLASS 3: Failed queries")
    print(failed_queries)

    output = {
        "perfect_matches": perfect_matches,
        "manual_review_list": manual_review_list,
        "failed_queries": failed_queries,
    }
    with open("crawlerOutput.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nSaved to crawlerOutput.json")

    return output


if __name__ == "__main__":
    import sys
    live = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_retrieval_system(live_query=live))