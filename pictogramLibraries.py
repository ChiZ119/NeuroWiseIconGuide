"""
pictogramLibraries.py
=====================

Turns any picture library into ONE common format, so the rest of the app never
needs to know where the pictures came from.

Every library ends up as a folder like this:

    pictograms/<name>/
        items.json        <- library info + one record per picture (written here)
        images/           <- the picture files (or "image_dir" points elsewhere)
        index.npz         <- embeddings (written by pictogramIndex.py)

One record ("item") looks like:

    {
      "id":         "arasaac:2694",
      "file_name":  "2694.png",            # relative to the library's image dir
      "title":      "toothbrush",          # shown in the UI
      "keywords":   ["toothbrush", "toothbrushes", "tooth brush", ...],
      "definition": "...",                 # optional, helps with polysemy
      "categories": ["hygiene product", "routine"],
      "tags":       [...],
      "kinds":      ["noun"],              # noun / verb / adj / other
      "source":     "arasaac",
      "source_url": "https://arasaac.org/pictograms/en/2694"
    }

Supported sources
-----------------
* ARASAAC   -> fetch_arasaac()       (downloads metadata + images from the API)
* legacy    -> load_legacy_meta()    (the old size-512/meta.json format)
* any folder of images -> load_image_folder()  (optional captions.json)

Run it through buildPictogramIndex.py rather than calling these directly.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Iterable

ARASAAC_API = "https://api.arasaac.org/v1"
ARASAAC_STATIC = "https://static.arasaac.org/pictograms"

# Text required by the ARASAAC licence (CC BY-NC-SA). Shown in the app footer
# and should also appear in the paper (e.g. in the acknowledgements / figure captions).
ARASAAC_ATTRIBUTION = (
    "Pictograms: Sergio Palao. Origin: ARASAAC (https://arasaac.org). "
    "Licence: CC BY-NC-SA. Owner: Government of Aragón (Spain)."
)

# ARASAAC keyword "type" codes. 2 (common noun) and 3 (verb) are the ones that
# matter here; the rest are kept as "other".
_ARASAAC_TYPE_TO_KIND = {1: "noun", 2: "noun", 3: "verb", 4: "adj", 5: "other", 6: "other"}

# ARASAAC "meaning" strings start with dictionary abbreviations such as "tr." —
# strip them so the definition embeds cleanly.
_DICT_ABBREV = re.compile(r"^(?:(?:tr|intr|prnl|adj|adv|m|f|n|U|tc)\.\s*)+", re.IGNORECASE)


def _dedup(seq: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for s in seq:
        s = (s or "").strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)
    return out


def write_library(library_dir: str, info: dict, items: list[dict]) -> str:
    """Write items.json for a library folder. Returns the path."""
    os.makedirs(library_dir, exist_ok=True)
    path = os.path.join(library_dir, "items.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"library": info, "items": items}, f, ensure_ascii=False, indent=1)
    print(f"[library] wrote {len(items)} items -> {path}")
    return path


def read_library(library_dir: str) -> tuple[dict, list[dict]]:
    with open(os.path.join(library_dir, "items.json"), encoding="utf-8") as f:
        data = json.load(f)
    return data.get("library", {}), data.get("items", [])


# ── ARASAAC ──────────────────────────────────────────────────────────────────

def arasaac_to_items(raw: list[dict], lang: str = "en",
                     drop_sensitive: bool = True) -> list[dict]:
    """Convert the ARASAAC API JSON (list of pictograms) into common items.

    drop_sensitive: skip pictograms ARASAAC flags as sex- or violence-related.
    The app is child-facing, so these are excluded by default.
    """
    items: list[dict] = []
    for p in raw:
        if drop_sensitive and (p.get("sex") or p.get("violence")):
            continue
        kws = [k for k in p.get("keywords", []) if (k.get("keyword") or "").strip()]
        if not kws:
            continue
        pid = str(p.get("_id"))

        keywords, kinds, meanings = [], [], []
        for k in kws:
            keywords.append(k["keyword"])
            if k.get("plural"):
                keywords.append(k["plural"])
            kinds.append(_ARASAAC_TYPE_TO_KIND.get(k.get("type"), "other"))
            if k.get("meaning"):
                meanings.append(_DICT_ABBREV.sub("", k["meaning"]).strip())

        categories = p.get("categories", []) or []
        if "verb" in categories and "verb" not in kinds:
            kinds.append("verb")

        definition = (p.get("desc") or "").strip() or (meanings[0] if meanings else "")
        items.append({
            "id": f"arasaac:{pid}",
            "file_name": f"{pid}.png",
            "title": kws[0]["keyword"].strip(),
            "keywords": _dedup(keywords),
            "definition": definition,
            "categories": categories,
            "tags": p.get("tags", []) or [],
            "kinds": _dedup(kinds) or ["other"],
            "schematic": bool(p.get("schematic")),
            "source": "arasaac",
            "source_url": f"https://arasaac.org/pictograms/{lang}/{pid}",
        })
    return items


async def _download_many(urls_and_paths: list[tuple[str, str]], concurrency: int = 8,
                         retries: int = 3) -> tuple[int, list[str]]:
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    done, failed = 0, []
    t0 = time.time()

    async def one(session, url, path):
        nonlocal done
        async with sem:
            for attempt in range(retries):
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
                        if r.status == 200:
                            data = await r.read()
                            tmp = path + ".part"
                            with open(tmp, "wb") as f:
                                f.write(data)
                            os.replace(tmp, path)
                            done += 1
                            if done % 500 == 0:
                                print(f"  ... {done}/{len(urls_and_paths)} images "
                                      f"({time.time() - t0:.0f}s)")
                            return
                        if r.status == 404:
                            break
                except Exception:
                    pass
                await asyncio.sleep(1.5 * (attempt + 1))
            failed.append(url)

    headers = {"User-Agent": "NeuroWise-IconGuide research prototype (VU Amsterdam)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        await asyncio.gather(*(one(session, u, p) for u, p in urls_and_paths))
    return done, failed


def fetch_arasaac(library_dir: str, lang: str = "en", resolution: int = 500,
                  download_images: bool = True, concurrency: int = 8,
                  raw_json_path: str | None = None) -> tuple[dict, list[dict]]:
    """Download ARASAAC metadata (+ images) and write items.json.

    Re-running is safe: images already on disk are skipped.
    raw_json_path: use a previously downloaded API dump instead of the network.
    """
    import urllib.request

    os.makedirs(library_dir, exist_ok=True)
    raw_path = raw_json_path or os.path.join(library_dir, f"arasaac_all_{lang}.json")
    if not os.path.exists(raw_path):
        url = f"{ARASAAC_API}/pictograms/all/{lang}"
        print(f"[arasaac] downloading metadata: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "NeuroWise-IconGuide"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        with open(raw_path, "wb") as f:
            f.write(data)
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"[arasaac] {len(raw)} pictograms in metadata")

    items = arasaac_to_items(raw, lang=lang)
    print(f"[arasaac] {len(items)} usable pictograms after filtering")

    image_dir = os.path.join(library_dir, "images")
    if download_images:
        os.makedirs(image_dir, exist_ok=True)
        todo = []
        for it in items:
            pid = it["id"].split(":", 1)[1]
            path = os.path.join(image_dir, it["file_name"])
            if not os.path.exists(path):
                todo.append((f"{ARASAAC_STATIC}/{pid}/{pid}_{resolution}.png", path))
        print(f"[arasaac] {len(items) - len(todo)} images already present, "
              f"{len(todo)} to download")
        if todo:
            ok, failed = asyncio.run(_download_many(todo, concurrency=concurrency))
            print(f"[arasaac] downloaded {ok}, failed {len(failed)}")
            if failed:
                with open(os.path.join(library_dir, "failed_downloads.txt"), "w") as f:
                    f.write("\n".join(failed))
        # keep only items whose image actually exists
        items = [it for it in items if os.path.exists(os.path.join(image_dir, it["file_name"]))]

    info = {
        "name": f"arasaac-{lang}",
        "source": "arasaac",
        "language": lang,
        "image_dir": "images",
        "license": "CC BY-NC-SA",
        "attribution": ARASAAC_ATTRIBUTION,
        "clip_prompt": "a pictogram of {}",
        "fetched_on": time.strftime("%Y-%m-%d"),
    }
    write_library(library_dir, info, items)
    return info, items


def download_missing_images(library_dir: str, resolution: int = 500,
                            concurrency: int = 8) -> int:
    """Fetch any missing ARASAAC pictures for an EXISTING items.json, without
    touching items.json or the index. Use this at deploy time (e.g. in the
    Dockerfile), so the ~13k PNGs don't have to go through Git LFS."""
    info, items = read_library(library_dir)
    image_dir = os.path.join(library_dir, info.get("image_dir", "images"))
    os.makedirs(image_dir, exist_ok=True)
    todo = []
    for it in items:
        if not it["id"].startswith("arasaac:"):
            continue
        path = os.path.join(image_dir, it["file_name"])
        if not os.path.exists(path):
            pid = it["id"].split(":", 1)[1]
            todo.append((f"{ARASAAC_STATIC}/{pid}/{pid}_{resolution}.png", path))
    print(f"[arasaac] {len(todo)} missing images to download")
    if todo:
        ok, failed = asyncio.run(_download_many(todo, concurrency=concurrency))
        print(f"[arasaac] downloaded {ok}, failed {len(failed)}")
    return len(todo)


# ── Legacy library (size-512/meta.json) ──────────────────────────────────────

def load_legacy_meta(meta_path: str, library_dir: str, image_dir: str) -> tuple[dict, list[dict]]:
    """The old 8,000-picture library. Pictures stay where they are; items.json
    just points at them, so nothing is copied."""
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    items = []
    for it in meta.get("items", []):
        title = (it.get("title") or "").strip()
        fname = it.get("file_name") or ""
        if not title or not fname:
            continue
        items.append({
            "id": f"legacy:{it.get('slug') or fname}",
            "file_name": fname,
            "title": title,
            "keywords": _dedup([title]),
            "definition": "",
            "categories": [it["category"]] if it.get("category") else [],
            "tags": it.get("tags", []) or [],
            "kinds": ["noun"],
            "source": "legacy",
            "source_url": "",
        })
    info = {
        "name": "legacy-size-512",
        "source": "legacy",
        "language": "en",
        "image_dir": os.path.relpath(os.path.abspath(image_dir), os.path.abspath(library_dir)),
        "license": "unknown",
        "attribution": "",
        "clip_prompt": "an icon of {}",
    }
    write_library(library_dir, info, items)
    return info, items


# ── Any folder of images (e.g. AI-generated gap fillers) ─────────────────────

_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def load_image_folder(image_dir: str, library_dir: str, name: str,
                      captions_path: str | None = None, license: str = "",
                      attribution: str = "") -> tuple[dict, list[dict]]:
    """Index a plain folder of pictures.

    Titles come from the file names ("hand-soap.png" -> "hand soap").
    captions.json (optional) improves matching a lot:
        { "hand-soap.png": {"title": "hand soap",
                            "keywords": ["soap", "liquid soap"],
                            "definition": "a pump bottle of soap for washing hands",
                            "kinds": ["noun"]} , ... }
    """
    captions = {}
    if captions_path and os.path.exists(captions_path):
        with open(captions_path, encoding="utf-8") as f:
            captions = json.load(f)
    items = []
    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() not in _IMG_EXT:
            continue
        stem = os.path.splitext(fname)[0]
        cap = captions.get(fname, {})
        title = cap.get("title") or re.sub(r"[-_]+", " ", stem).strip()
        items.append({
            "id": f"{name}:{stem}",
            "file_name": fname,
            "title": title,
            "keywords": _dedup([title] + list(cap.get("keywords", []))),
            "definition": cap.get("definition", ""),
            "categories": cap.get("categories", []),
            "tags": cap.get("tags", []),
            "kinds": cap.get("kinds", ["noun"]),
            "source": name,
            "source_url": "",
            "generated": bool(cap.get("generated", False)),
        })
    info = {
        "name": name,
        "source": "folder",
        "language": "en",
        "image_dir": os.path.relpath(os.path.abspath(image_dir), os.path.abspath(library_dir)),
        "license": license,
        "attribution": attribution,
        "clip_prompt": "a pictogram of {}",
    }
    write_library(library_dir, info, items)
    return info, items


# ── Combine libraries (e.g. ARASAAC + your own gap-filler pictures) ──────────

def item_image_path(library_dir: str, info: dict, item: dict) -> str:
    """Absolute path of an item's picture file."""
    if item.get("image_path"):
        return os.path.normpath(os.path.join(library_dir, item["image_path"]))
    return os.path.normpath(os.path.join(library_dir, info.get("image_dir", "images"),
                                         item["file_name"]))


def merge_libraries(out_dir: str, source_dirs: list[str], name: str) -> tuple[dict, list[dict]]:
    """Create one library that points at the pictures of several others.

    Each item gets an "image_path" relative to out_dir, so no files are copied.
    Attributions of all sources are joined for the app footer.
    """
    os.makedirs(out_dir, exist_ok=True)
    items, attributions, licenses, langs = [], [], [], []
    for src in source_dirs:
        info, src_items = read_library(src)
        for it in src_items:
            it = dict(it)
            abs_path = item_image_path(src, info, it)
            it["image_path"] = os.path.relpath(abs_path, os.path.abspath(out_dir))
            items.append(it)
        if info.get("attribution"):
            attributions.append(info["attribution"])
        licenses.append(f"{info.get('name', src)}: {info.get('license') or 'unknown'}")
        langs.append(info.get("language", "en"))
    info = {
        "name": name,
        "source": "merged",
        "sources": [os.path.basename(os.path.normpath(s)) for s in source_dirs],
        "language": langs[0] if langs else "en",
        "image_dir": ".",
        "license": "; ".join(licenses),
        "attribution": " ".join(_dedup(attributions)),
        "clip_prompt": "a pictogram of {}",
    }
    write_library(out_dir, info, items)
    return info, items
