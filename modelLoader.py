import time
import spacy
from sentence_transformers import SentenceTransformer


def _load_spacy():
    print("[modelLoader] Loading spaCy en_core_web_md ...")
    t0 = time.time()
    try:
        #nlp = spacy.load("en_core_web_md")
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("[modelLoader] WARNING: using en_core_web_sm. "
                  "Run `python -m spacy download en_core_web_md` for better results.")
        except OSError:
            raise SystemExit(
                "[modelLoader] No spaCy model found.\n"
                "Run: python -m spacy download en_core_web_md"
            )
    # warm it up so the first real sentence is fast
    _ = nlp("Brush your teeth with toothpaste.")
    print(f"[modelLoader] spaCy ready in {time.time() - t0:.2f}s")
    return nlp


def _load_st():
    print("[modelLoader] Loading SentenceTransformer all-MiniLM-L6-v2 ...")
    t0 = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # warm Torch graph so the first encode() is fast
    _ = model.encode("warmup", convert_to_tensor=True)
    print(f"[modelLoader] SentenceTransformer ready in {time.time() - t0:.2f}s")
    return model


# Module-level singletons. Importing this module triggers the load exactly once.
NLP = _load_spacy()
EMBED_MODEL = _load_st()