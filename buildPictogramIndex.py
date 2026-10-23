"""
buildPictogramIndex.py — prepare a picture library for NeuroWise IconGuide.

Run this once per library on your own computer (it needs internet for ARASAAC
and for downloading the models the first time). It writes everything into
pictograms/<name>/, which the server then loads when PICTO_LIBRARY points to it.

Examples
--------
# 1) ARASAAC in English: metadata + around 13k pictures + text and CLIP embeddings
python buildPictogramIndex.py arasaac --lang en

# 2) the old 8,000-picture library (for comparison runs only)
python buildPictogramIndex.py legacy --meta size-512/meta.json --images size-512/images

# 3) your own folder, e.g. AI-generated gap fillers, with optional captions.json
python buildPictogramIndex.py folder --images gapfill/images --name gapfill \
       --captions gapfill/captions.json --license "CC BY-NC-SA" \
       --attribution "Supplementary pictures generated for NeuroWise IconGuide."

# 4) combine libraries into one (pictures are not copied)
python buildPictogramIndex.py merge --sources pictograms/arasaac-en pictograms/gapfill \
       --name arasaac-plus-gapfill

# 5) deploy time: only download missing ARASAAC pictures for an index you
#    already built (no models needed, items.json/index.npz are not touched)
python buildPictogramIndex.py images --library pictograms/arasaac-en

Useful flags: --no-clip (text embeddings only, much faster),
              --clip-text-model sentence-transformers/clip-ViT-B-32-multilingual-v1
              (lets Dutch queries hit the same CLIP image vectors).
Re-running `arasaac` skips pictures that are already downloaded.
"""

import argparse
import os
import sys

from pictogramLibraries import (download_missing_images, fetch_arasaac, load_image_folder,
                                load_legacy_meta, merge_libraries)

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEXT_MODEL = "all-MiniLM-L6-v2"  # must match modelLoader.py
DEFAULT_CLIP_MODEL = "clip-ViT-B-32"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="source", required=True)

    a = sub.add_parser("arasaac", help="download ARASAAC and index it")
    a.add_argument("--lang", default="en")
    a.add_argument("--resolution", type=int, default=500, choices=[300, 500, 2500])
    a.add_argument("--concurrency", type=int, default=8,
                   help="parallel downloads; please keep this polite")
    a.add_argument("--metadata-only", action="store_true",
                   help="skip image download (and therefore CLIP)")
    a.add_argument("--raw-json", help="use an already downloaded /pictograms/all/<lang> dump")

    l = sub.add_parser("legacy", help="index the old size-512 library")
    l.add_argument("--meta", default=os.path.join(BASE, "size-512", "meta.json"))
    l.add_argument("--images", default=os.path.join(BASE, "size-512", "images"))

    f = sub.add_parser("folder", help="index any folder of pictures")
    f.add_argument("--images", required=True)
    f.add_argument("--name", required=True)
    f.add_argument("--captions")
    f.add_argument("--license", default="")
    f.add_argument("--attribution", default="")

    m = sub.add_parser("merge", help="combine already-built libraries")
    m.add_argument("--sources", nargs="+", required=True)
    m.add_argument("--name", required=True)

    im = sub.add_parser("images", help="download missing ARASAAC pictures only")
    im.add_argument("--library", required=True)
    im.add_argument("--resolution", type=int, default=500, choices=[300, 500, 2500])
    im.add_argument("--concurrency", type=int, default=8)

    for sp in (a, l, f, m):
        sp.add_argument("--out", help="output folder (default: pictograms/<name>)")
        sp.add_argument("--no-clip", action="store_true", help="skip image embeddings")
        sp.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
        sp.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
        sp.add_argument("--clip-text-model", default=None,
                        help="CLIP text encoder for queries (default: same as --clip-model)")
        sp.add_argument("--batch-size", type=int, default=32)

    args = p.parse_args(argv)

    if args.source == "images":
        download_missing_images(args.library, args.resolution, args.concurrency)
        return 0

    if args.source == "arasaac":
        out = args.out or os.path.join(BASE, "pictograms", f"arasaac-{args.lang}")
        fetch_arasaac(out, lang=args.lang, resolution=args.resolution,
                      download_images=not args.metadata_only,
                      concurrency=args.concurrency, raw_json_path=args.raw_json)
        if args.metadata_only:
            args.no_clip = True
    elif args.source == "legacy":
        out = args.out or os.path.join(BASE, "pictograms", "legacy-size-512")
        load_legacy_meta(args.meta, out, args.images)
    elif args.source == "folder":
        out = args.out or os.path.join(BASE, "pictograms", args.name)
        load_image_folder(args.images, out, args.name, args.captions,
                          license=args.license, attribution=args.attribution)
    else:
        out = args.out or os.path.join(BASE, "pictograms", args.name)
        merge_libraries(out, args.sources, args.name)

    from sentence_transformers import SentenceTransformer
    from pictogramIndex import PictogramIndex

    print(f"[build] text model: {args.text_model}")
    text_model = SentenceTransformer(args.text_model)
    clip = None
    if not args.no_clip:
        print(f"[build] CLIP model: {args.clip_model} (first run downloads ~600 MB)")
        clip = SentenceTransformer(args.clip_model)

    PictogramIndex.build(out, text_model, args.text_model,
                         clip_image_model=clip,
                         clip_model_name=args.clip_model if clip else None,
                         clip_text_model_name=args.clip_text_model or args.clip_model,
                         batch_size=args.batch_size)
    rel = os.path.relpath(out, BASE)
    print("\nDone. Start the app with this library:")
    print(f"  PICTO_LIBRARY={rel} python serverOfNER.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
