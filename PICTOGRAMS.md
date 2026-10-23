# Picture libraries and matching (post-thesis version)

The app no longer depends on one particular picture library. Any library is turned
into a common format once, and the server loads whichever one you point it at.
**If you set nothing, the app runs exactly like the thesis version** (size-512,
text matching), which was checked to give identical results on all 40 tasks.

## What's new

| File | What it does |
|---|---|
| `pictogramLibraries.py` | Loaders: ARASAAC (API download), the old `size-512/meta.json`, any folder of pictures (+ optional `captions.json`), and merging several libraries. |
| `pictogramIndex.py` | The matcher. Text + context + CLIP image scores, four modes, and **action pictograms** ("brush teeth", "put on socks"). |
| `buildPictogramIndex.py` | Command line tool to download / index a library. |
| `evalRetrieval.py` | Labelling page + metrics (Hit@1, Hit@5, MRR) to compare the modes for the paper. |
| `serverOfNER.py`, `debugPaChong.py`, `nerExtractor.py`, `stepCardDemo.html` | Small additions, so the new path is used when `PICTO_LIBRARY` is set. The UI shows the action picture first in each step (purple border) and the library's licence attribution at the bottom. |

## Quick start (on your Mac, in the project folder)

```bash
# 1. ARASAAC in English: ~13k pictograms, metadata + images + embeddings.
#    First run downloads the CLIP model (~600 MB); encoding takes a while on CPU.
python buildPictogramIndex.py arasaac --lang en

# 2. Run the app with it
PICTO_LIBRARY=pictograms/arasaac-en python serverOfNER.py
```

Then open http://127.0.0.1:8080 and try a task. `/api/debug` shows which library,
mode, weights and threshold are active. `/api/test_match?q=brush teeth` shows the
matching per step.

No new packages are needed: CLIP runs through `sentence-transformers`, which
the project already uses.

## Settings (environment variables)

| Variable | Meaning | Default |
|---|---|---|
| `PICTO_LIBRARY` | Folder built by `buildPictogramIndex.py` | unset = thesis version |
| `PICTO_MODE` | `text`, `text+context`, `image` or `hybrid` | `hybrid` |
| `PICTO_WEIGHTS` | `"text,context,image"`, e.g. `"0.6,0.2,0.2"` | `0.65,0.15,0.20` |
| `PICTO_THRESHOLD` | Minimum score to show a picture (else the step stays text-only) | 0.50 (`hybrid`: 0.45) |

What the modes mean:

- **text**: the entity against picture keywords (exact, whole word, or MiniLM similarity). This is the thesis method.
- **text+context**: also scores how well the *whole step sentence* fits the picture's keywords, definition and categories. This is what separates *tap* (faucet) from *tap* (dance) and *bowl* from *bowling*.
- **image**: CLIP compares the entity text with the picture pixels. It needs no metadata, so it works on any folder of pictures.
- **hybrid**: a weighted mix of all three.

The weights and threshold are starting values. Tune them with the evaluation below.

Cached pictures in `knowledgeBase.json` are tagged with the library build and the
matcher version. Switching libraries or updating the matcher recomputes them
automatically.

## Evaluation for the paper

```bash
# 1. Export ~250 random objects/actions with the top-5 candidates of every mode
python evalRetrieval.py export --library pictograms/arasaac-en --sample 250

# 2. Open eval/label.html in a browser and click every correct picture
#    ("none of these fits" if nothing fits). Progress is kept in the browser.
#    Then "Download labels.csv" and move it into eval/.
#    Have a second person (ideally a teacher/therapist) label too, for agreement.

# 3. Metrics for every mode, plus hybrid weights tuned on half of the tasks
#    and reported on the other half
python evalRetrieval.py score --library pictograms/arasaac-en --grid
```

Tips:
- Remove tasks you won't study from the KB first, or pass `--tasks tasks.txt`.
- `--kind object` or `--kind action` scores the two separately.
- `judged@1` tells you how many top-1 results were actually labelled. If it drops after tuning, export and label again.
- To compare with the thesis setup, run the same steps on `pictograms/legacy-size-512` (built with `python buildPictogramIndex.py legacy`). Keep that one local only, since its licence is unknown.

## Filling gaps with your own / AI-generated pictures

1. Put the pictures in a folder, e.g. `gapfill/images/hand-soap.png`.
2. Optional but recommended: `gapfill/captions.json`
   ```json
   {"hand-soap.png": {"title": "hand soap", "keywords": ["soap", "liquid soap"],
                      "definition": "pump bottle of soap for washing hands",
                      "kinds": ["noun"], "generated": true}}
   ```
   Use `"kinds": ["verb"]` for action pictures.
3. Index it and merge it with ARASAAC:
   ```bash
   python buildPictogramIndex.py folder --images gapfill/images --name gapfill \
          --captions gapfill/captions.json --license "CC BY-NC-SA" \
          --attribution "Supplementary pictures generated for NeuroWise IconGuide."
   python buildPictogramIndex.py merge --sources pictograms/arasaac-en pictograms/gapfill \
          --name arasaac-plus-gapfill
   PICTO_LIBRARY=pictograms/arasaac-plus-gapfill python serverOfNER.py
   ```

Keep `"generated": true` on every AI-made picture, so the paper can report how many
there are and that each was checked by a person.

## Licence and attribution (ARASAAC)

ARASAAC pictograms are **CC BY-NC-SA**: non-commercial use, with attribution, and
derivatives under the same licence. The app footer shows the attribution
automatically. Use it in the paper too (acknowledgements and figure captions):

> Pictograms: Sergio Palao. Origin: ARASAAC (https://arasaac.org). Licence: CC BY-NC-SA. Owner: Government of Aragón (Spain).

Pictograms that ARASAAC flags as sexual or violent are excluded by default, because
the app is child-facing.

## Deploying to Hugging Face Spaces

Commit only `items.json`, `index.npz` and `build_info.json` (the images and the raw
API dump are git-ignored). Let the Docker build download the pictures and the CLIP
model. When you are ready to switch, the relevant Dockerfile lines become:

```dockerfile
COPY --chown=user . .
# was: RUN unzip -q size-512-images.zip && rm size-512-images.zip
RUN python buildPictogramIndex.py images --library pictograms/arasaac-en
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('clip-ViT-B-32')"
ENV PICTO_LIBRARY=pictograms/arasaac-en
```

Also stop shipping `size-512-images.zip` and `size-512/` (remove them from the
repo, or add them to `.dockerignore`). The image ZIP stays in the repo's git
history, so before making a Space public again, start a fresh Space (or squash
the history) rather than just deleting the file.

## Known limits

- CLIP was trained mostly on photos, not flat pictograms, and it is weaker on
  actions. That is why `hybrid` mixes it with text rather than relying on it.
  Whether it actually helps on ARASAAC is an empirical question for `evalRetrieval.py`.
- Entity and action extraction use English spaCy. For Dutch, build ARASAAC with
  `--lang nl` and `--clip-text-model sentence-transformers/clip-ViT-B-32-multilingual-v1`,
  and use a Dutch spaCy model and text model (not wired up yet).
- Action detection assumes imperative steps ("Brush your teeth…"). That's another
  reason the AI step-rewriting idea (one action per step, plain wording) pairs well with this.
