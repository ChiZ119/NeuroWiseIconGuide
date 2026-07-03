---
title: NeuroWise IconGuide
emoji: 🧩
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: generating visual how-to instructions for autism support
---
# NeuroWise IconGuide

**Generating visual how-to instructions for autism support.**

NeuroWise IconGuide is a web application built for a BSc AI thesis at Vrije Universiteit Amsterdam. A user types an everyday task, such as `brush teeth` or `make the bed`, and the system returns an illustrated, step-by-step card sequence. Every step is parsed for the concrete objects it mentions, and each object is matched to an icon from a private icon library (includes 8,000 images), so each instruction is paired with a clear, consistent picture.

The motivation is that many autistic children learn daily-living tasks best through visual supports, yet most instructions are text-only, or story cards are made by hand, so it's time consuming. NeuroWise IconGuide automatically converts arbitrary task queries into a consistent, low-anxiety, picture-supported format.

> **Current Status:** the system is deployed on Hugging Face Spaces and is currently in the user-study phase. The README documents the code as it stands in this repository.

---

## What it does

1. Accepts a typed task query through a friendly, child-facing single-page UI.
2. Resolves the query to a set of instructional steps using a staged retrieval system including semantic match and live WikiHow crawl.
3. Extracts the real-world objects from each step with spaCy NER, POS tagging plus noun-chunk analysis.
4. Matches each object to the closest icon in the image library using sentence-embedding similarity.
5. Renders each step as a card with the matched entity highlighted in the sentence and its icon shown beside it, plus a progress bar and an optional gentle AI helper.

---

## Repository layout

| File | Role |
|------|------|
| `modelLoader.py` | Loads the spaCy model and the Sentence-Transformer model **once** as module-level singletons, with a warm-up pass so the first real request is fast. |
| `nerExtractor.py` | Extracts entities from a sentence (NER + noun chunks, with abstract/time words filtered out) and matches each entity to a library icon using a pre-computed title embedding matrix. Returns character offsets so the frontend can highlight matched entities. |
| `debugPaChong.py` | Asynchronous WikiHow retrieval. Searches WikiHow, ranks candidate articles by URL relevance, extracts and cleans step headlines, scores per-step icon coverage, and classifies each article as **perfect / partial / failed**. Writes `crawlerOutput.json`. |
| `buildKnowledgeBase.py` | Assembles `knowledgeBase.json` from crawler perfect matches, approved manual-review items, and a hand-written fallback dictionary for tasks the crawler can't cover. |
| `serverOfNER.py` | The Flask application: serves the UI and images, exposes the search/knowledge-base/chat API, runs the staged search, applies rate limiting, and maintains a search-popularity counter. |
| `stepCardDemo.html` | The single-page frontend: informed-consent gate, search box and popular-task chips, illustrated step cards with entity highlighting, progress bar, AI assistant chat panel, and post-task survey button. |
| `knowledgeBase.json` | from the cached task to steps store consumed by the server (38 entries in this snapshot). |
| `crawlerOutput.json` | Raw output of a crawler run, used as input to `buildKnowledgeBase.py`. |

---

## How it works

### Retrieval pipeline (`/api/search`)

The server tries the cheapest source first and stops at the first hit:

1. **Exact KB hit** : the normalized query is a key in `knowledgeBase.json`; served immediately.
2. **Strong semantic match** : MiniLM cosine similarity between the query and cached task keys; if the best score is >= `0.70` (`FUZZY_STRONG_THRESHOLD`), the cached entry is served.
3. **Live WikiHow crawl** : `debugPaChong.run_retrieval_system(live_query=...)` searches WikiHow, extracts steps, matches icons, and the result is cached back into the KB for next time.
4. **Weak semantic fallback** : if the live crawl fails but a cached key scores >= `0.55` (`FUZZY_WEAK_THRESHOLD`), that entry is returned and flagged as `approximate_match`.
5. **No result** : a friendly `not_found` message is returned.

### Step extraction (crawler)

WikiHow articles often contain several "methods". The crawler scopes extraction to `div.section.steps`, and when an article has multiple method sections it picks the one whose heading is most similar to the query (again via embeddings). Each step's headline is cleaned (citation markers stripped, whitespace normalized) and capped at 12 steps per article (at least 2 steps tho).

### Icon matching

Each step sentence is run through spaCy to pull out concrete objects while dropping verbs, pronouns, and abstract words (time, amount, side, etc). Every surviving phrase is matched to the library:

- **Exact / or it's a substring** title match to instant, score `1.0` / `0.95`.
- Otherwise **one batched cosine pass** over the pre-embedded library titles (`all-MiniLM-L6-v2`, 384-dimensional), clamped to a `0.9` ceiling.
- The stronger candidate wins; anything below the score threshold (`0.5`) leaves the step text-only rather than showing a wrong picture.

Library titles are embedded once at load time, so matching a whole step is a single matrix multiply. It's fast enough to enrich live crawl results on the fly.

---

## API endpoints

| Method & path | Purpose |
|---------------|---------|
| `GET /` | Serves the `stepCardDemo.html` frontend. |
| `GET /images/<filename>` | Serves a Icon PNG from the image library. |
| `GET /api/search?q=<task>` | The main endpoint: runs the staged retrieval pipeline and returns enriched steps. |
| `GET /api/knowledgeBase` | Returns the whole knowledge base (entities enriched and cached on first call). |
| `GET /api/popular` | Top-N task labels by search count, padded with other KB entries. |
| `POST /api/chat` | Rate-limited DeepSeek-backed helper that answers questions about the current task/steps. |
| `GET /api/test_match?q=<task>` | Diagnostic: shows entity matching per step for a task. |
| `GET /api/debug` | Diagnostic: paths, KB keys, library size, thresholds, and config flags. |

The chat endpoint is rate-limited to **15 messages per IP per hour** and **1,000 calls per day** site-wide.

---

## Data assets

The Python files in this repository are the application logic. To run end-to-end, the following data assets are also needed at runtime (they are large and may live outside the repo / in Git LFS):

- **`size-512/images/`** — the Icon PNGs served at `/images/<filename>`.
- **`size-512/meta.json`** — image index consumed by the server, with items shaped `{ "title", "slug", "file_name" }`.
- **`xlsxFiles/Processed_Library_info.xlsx`** — used by the crawler; the `entities` column supplies the text that the library embeddings are built from.
- **`searchCounts.json`** — created automatically to track search popularity.

`knowledgeBase.json` and `crawlerOutput.json` are included.

---

## Setup and running locally

### 1. Install dependencies

```bash
pip install flask spacy sentence-transformers torch \
            beautifulsoup4 aiohttp pandas openpyxl openai
python -m spacy download en_core_web_sm
# (optional, better entities) python -m spacy download en_core_web_md
```

`sentence-transformers` downloads `all-MiniLM-L6-v2` automatically on first use. The Playwright browser fallback in the crawler is disabled by default (`USE_BROWSER_FALLBACK = False`); enable and `pip install playwright` only if you need it.

### 2. Provide the image library

Place the Icons under `size-512/images/` and the index at `size-512/meta.json`, and the crawler spreadsheet at `xlsxFiles/Processed_Library_info.xlsx` (see **Data assets** above).

### 3. Set the AI assistant key (optional)

The chat helper talks to a DeepSeek endpoint via the OpenAI-compatible client. The key is read from the environment variable **`ChiBscProject`**:

```bash
export ChiBscProject="" # From the researcher: I stored the deepseek api key in HF space (in the Secrets) actually, not inside the code. But in case anyone wants to clone my code and insert their api plz feel free to do it yourself
```

Without it, search and cards work normally; only the chat helper is disabled.

### 4. (Re)build the knowledge base — optional

```bash
python buildKnowledgeBase.py        # writes knowledgeBase.json
```

### 5. Run a crawl — optional

```bash
python debugPaChong.py "wash dishes"   # single live query
python debugPaChong.py                 # batch over the built-in query list
```

### 6. Start the server

```bash
python serverOfNER.py
# → http://127.0.0.1:8080
```

On startup the server loads the models, pre-warms the KB (extracting and caching entities for every cached task), and reports library and KB sizes.

---

## Deployment (Hugging Face Spaces)

The repository is configured as a **Docker** Space (see the YAML frontmatter at the top of this file) and exposes port **7860**. The image library is large, so it is handled with Git LFS; zipping the library before upload avoids LFS rate limits, and the container unpacks it on build. Common deployment gotchas already addressed in the code include avoiding hard-coded local paths and ensuring `os` is imported where filesystem paths are built.

---

## Frontend and ethics

The UI is intentionally calm and rounded with soft pastel palette, friendly display type, generous spacing, to keep cognitive and sensory load low. Before the app is usable, an **informed-consent gate** explains the study, what data is collected, and that participation is voluntary. After results appear, a **post-task survey** invitation is revealed. Use of the third-party DeepSeek service for the AI helper is disclosed to participants.

The evaluation is two single-group usability studies and one study that compares NeuroWise IconGuide and original WikiHow articles. The system's mean SUS score is compared against the established benchmark of **68** using a one-sample t-test, following a pilot with neurotypical adult proxy participants.

---

## Limitations and future work

- Icon matching is similarity-based, so abstract or rare objects can map to an imperfect picture.
- The current evaluation uses adult proxy participants; testing with the target user group (autistic children) is planned separately and carries its own ethical requirements.
- Live crawling depends on WikiHow's page structure, so parser selectors may need maintenance over time.

---

## Acknowledgements

Bachelor's thesis project, Vrije Universiteit Amsterdam (2026). Author: Chi Zhang. Supervisor: Jiahuan Pei and Yifan Mo. Step content is retrieved from WikiHow; please respect the terms of the underlying content and Icon sources when reusing this project.