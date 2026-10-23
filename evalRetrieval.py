"""
evalRetrieval.py — measure how well each matching mode picks the right picture.

This gives the paper a small, clean technical evaluation:
"text-only (thesis method) vs. text+context vs. CLIP image-only vs. hybrid".

Workflow
--------
1) Export a pool of candidate pictures for every object (and action) in your
   tasks, using ALL modes, and open the labelling page:

       python evalRetrieval.py export --library pictograms/arasaac-en --sample 250
       open eval/label.html              # click every correct picture

   Tip: first remove tasks you won't study from knowledgeBase.json, or pass
   --tasks tasks.txt (one task per line) to export only those.

   The page saves progress in the browser; when done, click
   "Download labels.csv" and put the file in eval/.
   Ideally a second person (e.g. a special-education teacher) labels too,
   so you can report agreement.

2) Score every mode, and tune the hybrid weights on half of the tasks while
   reporting on the other half (so the numbers are not overfitted):

       python evalRetrieval.py score --library pictograms/arasaac-en --grid

Metrics: Hit@1, Hit@5, MRR@10, and precision/coverage of the shown picture at
the app's threshold. Pictures that nobody labelled count as wrong, so the
script also reports how many top-1 results were labelled ("judged@1"); if that
drops, export again and label the new candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))


def _qid(task: str, step_idx: int, kind: str, entity: str) -> str:
    return hashlib.sha1(f"{task}|{step_idx}|{kind}|{entity}".encode()).hexdigest()[:12]


def _load_index(library: str):
    """Load the index with the app's own text model (+ CLIP if the index has images)."""
    from pictogramIndex import BUILD_INFO_FILE, PictogramIndex, load_clip_model
    from modelLoader import EMBED_MODEL
    lib = library if os.path.isabs(library) else os.path.join(BASE, library)
    with open(os.path.join(lib, BUILD_INFO_FILE), encoding="utf-8") as f:
        clip_name = json.load(f).get("clip_text_model")
    clip = load_clip_model(clip_name) if clip_name else None
    return PictogramIndex.load(lib, EMBED_MODEL, clip)


def _available_modes(index) -> list[str]:
    modes = ["text", "text+context"]
    if index.has_image_channel:
        modes += ["image", "hybrid"]
    else:
        print("[eval] index has no image vectors / CLIP model -> skipping image & hybrid")
    return modes


# ── export ───────────────────────────────────────────────────────────────────

def build_queries(kb: dict, index, nlp, extract_entities, instruction_verbs) -> list[dict]:
    queries = []
    for task_key, entry in kb.items():
        task = entry.get("query", task_key)
        for si, step in enumerate(entry.get("steps", [])):
            text = step.get("step", "")
            if not text:
                continue
            for ent in extract_entities(text):
                queries.append({"qid": _qid(task, si, "object", ent["lemma"]), "task": task,
                                "step_idx": si, "step": text, "entity": ent["lemma"],
                                "kind": "object"})
            phrases = index.action_phrases(nlp(text), instruction_verbs)
            if phrases:
                ph = phrases[0]
                queries.append({"qid": _qid(task, si, "action", ph), "task": task,
                                "step_idx": si, "step": text, "entity": ph, "kind": "action"})
    seen, uniq = set(), []
    for q in queries:
        if q["qid"] not in seen:
            seen.add(q["qid"])
            uniq.append(q)
    return uniq


def export(args):
    from nerExtractor import NLP, _INSTRUCTION_VERBS, extract_entities
    index = _load_index(args.library)
    out = args.out
    os.makedirs(out, exist_ok=True)
    with open(args.kb, encoding="utf-8") as f:
        kb = json.load(f)
    if args.tasks:
        wanted = {t.strip().lower() for t in open(args.tasks, encoding="utf-8") if t.strip()}
        kb = {k: v for k, v in kb.items() if k.lower() in wanted}

    queries = build_queries(kb, index, NLP, extract_entities, _INSTRUCTION_VERBS)
    if args.sample and args.sample < len(queries):
        import random
        rng = random.Random(args.seed)
        queries = sorted(rng.sample(queries, args.sample),
                         key=lambda q: (q["task"], q["step_idx"], q["kind"]))
        print(f"[eval] random sample of {args.sample} queries (seed {args.seed})")
    modes = _available_modes(index)
    print(f"[eval] {len(queries)} queries from {len(kb)} tasks; pooling top-{args.k} "
          f"of modes {modes}")

    rows, page = [], []
    for q in queries:
        kinds = "action" if q["kind"] == "action" else "object"
        pool: dict[str, dict] = {}
        for mode in modes:
            for rank, h in enumerate(index.search(q["entity"], context=q["step"], k=args.k,
                                                  mode=mode, kinds=kinds), 1):
                it = h["item"]
                c = pool.setdefault(it["id"], {"item_id": it["id"], "title": it["title"],
                                               "img": os.path.relpath(index.image_path(it), out),
                                               "found_by": []})
                c["found_by"].append(f"{mode}#{rank}")
        cands = list(pool.values())
        page.append({**q, "cands": cands})
        for c in cands:
            rows.append({**{k: q[k] for k in ("qid", "task", "step_idx", "step", "entity", "kind")},
                         "item_id": c["item_id"], "title": c["title"],
                         "found_by": " ".join(c["found_by"])})

    with open(os.path.join(out, "queries.json"), "w", encoding="utf-8") as f:
        json.dump({"library": index.library_id, "queries": queries}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "pool.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["qid"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "label.html"), "w", encoding="utf-8") as f:
        f.write(_label_page(page, index.library_id))
    print(f"[eval] wrote {out}/queries.json, pool.csv ({len(rows)} candidates) and label.html")
    print(f"[eval] open {os.path.join(out, 'label.html')} in a browser to label")


def _label_page(page: list[dict], library_id: str) -> str:
    data = json.dumps(page, ensure_ascii=False).replace("</", "<\\/")
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pictogram Labelling</title>
<style>
:root{--bg:#f7f9fc;--card:#fff;--ink:#1f2d3d;--muted:#6b7c8f;--ok:#00a389;--no:#e35d5b;--line:#dfe6ee}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#12161c;--card:#1b222b;--ink:#e6edf3;--muted:#93a1b0;--line:#2c3743}}
:root[data-theme="dark"]{--bg:#12161c;--card:#1b222b;--ink:#e6edf3;--muted:#93a1b0;--line:#2c3743}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
header b{font-size:16px}button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:6px 12px;cursor:pointer}
button.primary{background:var(--ok);color:#fff;border-color:var(--ok)}main{max-width:1100px;margin:0 auto;padding:16px}
.q{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:12px}
.q.done{border-left:5px solid var(--ok)}.meta{color:var(--muted);font-size:13px}.step mark{background:#ffe08a;color:#1f2d3d;border-radius:3px;padding:0 2px}
.kind{display:inline-block;font-size:12px;border-radius:6px;padding:0 6px;margin-left:6px;background:#ede7f6;color:#5e35b1}
.cands{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}.c{width:118px;border:3px solid var(--line);border-radius:10px;padding:6px;text-align:center;cursor:pointer;background:#fff;color:#1f2d3d}
.c img{width:100%;height:96px;object-fit:contain}.c .t{font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.c.yes{border-color:var(--ok);box-shadow:0 0 0 2px var(--ok)}.none{margin-top:8px;font-size:13px}.none.on{color:var(--no);font-weight:700}
label.filter{font-size:13px;color:var(--muted)}
</style></head><body>
<header><b>Which picture fits the highlighted word?</b>
<span id="prog" class="meta"></span>
<label class="filter"><input type="checkbox" id="todo"> show unlabelled only</label>
<button class="primary" id="dl">Download labels.csv</button>
<span class="meta">Library: """ + html.escape(library_id) + """</span></header>
<main id="list"></main>
<script>
const DATA = """ + data + """;
const KEY = 'picto-labels-' + """ + json.dumps(library_id) + """;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} render(); }
function esc(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function hl(step, ent){
  const i = step.toLowerCase().indexOf(ent.toLowerCase());
  if (i < 0) return esc(step);
  return esc(step.slice(0,i)) + '<mark>' + esc(step.slice(i,i+ent.length)) + '</mark>' + esc(step.slice(i+ent.length));
}
function render(){
  const onlyTodo = document.getElementById('todo').checked;
  const done = DATA.filter(q => state[q.qid]).length;
  document.getElementById('prog').textContent = done + ' / ' + DATA.length + ' labelled';
  document.getElementById('list').innerHTML = DATA.filter(q => !onlyTodo || !state[q.qid]).map(q => {
    const s = state[q.qid] || null;
    const cards = q.cands.map(c => `<div class="c ${s && s.yes.includes(c.item_id) ? 'yes' : ''}" data-q="${q.qid}" data-i="${esc(c.item_id)}">
        <img src="${esc(c.img)}" alt="${esc(c.title)}" loading="lazy"><div class="t" title="${esc(c.title)}">${esc(c.title)}</div></div>`).join('');
    return `<div class="q ${s ? 'done' : ''}">
      <div class="meta">${esc(q.task)} · step ${q.step_idx + 1}${q.kind === 'action' ? '<span class="kind">action</span>' : ''}</div>
      <div class="step">${q.kind === 'action' ? esc(q.step) + ' → <mark>' + esc(q.entity) + '</mark>' : hl(q.step, q.entity)}</div>
      <div class="cands">${cards}</div>
      <div class="none ${s && s.none ? 'on' : ''}"><label><input type="checkbox" data-none="${q.qid}" ${s && s.none ? 'checked' : ''}> none of these fits</label></div>
    </div>`;
  }).join('');
}
document.addEventListener('click', e => {
  const c = e.target.closest('.c'); if (!c) return;
  const q = c.dataset.q, i = c.dataset.i;
  const s = state[q] || {yes: [], none: false};
  s.none = false;
  s.yes = s.yes.includes(i) ? s.yes.filter(x => x !== i) : s.yes.concat([i]);
  if (s.yes.length) state[q] = s; else delete state[q];
  save();
});
document.addEventListener('change', e => {
  if (e.target.id === 'todo') return render();
  const q = e.target.dataset.none; if (!q) return;
  if (e.target.checked) state[q] = {yes: [], none: true}; else delete state[q];
  save();
});
document.getElementById('dl').onclick = () => {
  const lines = ['qid,item_id,label'];
  for (const q of DATA) {
    const s = state[q.qid]; if (!s) continue;
    for (const c of q.cands) lines.push(`${q.qid},"${c.item_id.replace(/"/g,'""')}",${s.yes.includes(c.item_id) ? 1 : 0}`);
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\\n')], {type: 'text/csv'}));
  a.download = 'labels.csv'; a.click();
};
render();
</script></body></html>
"""


# ── score ────────────────────────────────────────────────────────────────────

def _read_labels(path: str) -> dict[str, dict[str, int]]:
    labels: dict[str, dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            labels.setdefault(r["qid"], {})[r["item_id"]] = int(r["label"])
    return labels


def _split(task: str) -> str:
    return "dev" if int(hashlib.md5(task.encode()).hexdigest(), 16) % 2 == 0 else "test"


def _precompute(index, queries):
    pre = []
    for q in queries:
        kinds = "action" if q["kind"] == "action" else "object"
        text = index._text_scores(q["entity"])
        ctx = index._context_scores(q["step"])
        img = index._image_scores(q["entity"])
        pre.append((text, ctx, img, index._kind_mask(kinds)))
    return pre


def _evaluate(pre, queries, labels, w, threshold, idx_of, items):
    wt, wc, wi = w
    hit1 = hit5 = rr = judged1 = accepted = acc_ok = 0
    for (text, ctx, img, mask), q in zip(pre, queries):
        parts = [(wt, text)]
        if wc and ctx is not None:
            parts.append((wc, ctx))
        if wi and img is not None:
            parts.append((wi, img))
        tot = sum(p[0] for p in parts) or 1.0
        final = sum(p[0] * p[1] for p in parts) / tot
        final = np.where(mask, final, -1.0)
        lab = labels[q["qid"]]
        rel = [idx_of[i] for i, v in lab.items() if v == 1 and i in idx_of]
        top = int(np.argmax(final))
        top_id = items[top]["id"]
        judged1 += top_id in lab
        is_ok = top_id in lab and lab[top_id] == 1
        if final[top] >= threshold:
            accepted += 1
            acc_ok += is_ok
        if rel:
            best_rel = max(final[r] for r in rel)
            rank = 1 + int((final > best_rel).sum())
            hit1 += rank == 1
            hit5 += rank <= 5
            rr += 1.0 / rank if rank <= 10 else 0.0
    n = len(queries) or 1
    return {"n": len(queries), "hit@1": hit1 / n, "hit@5": hit5 / n, "mrr@10": rr / n,
            "judged@1": judged1 / n, "shown": accepted / n,
            "precision_shown": (acc_ok / accepted) if accepted else 0.0}


def score(args):
    index = _load_index(args.library)
    items = index.items
    idx_of = {it["id"]: i for i, it in enumerate(index.items)}
    with open(os.path.join(args.eval_dir, "queries.json"), encoding="utf-8") as f:
        qdata = json.load(f)
    labels = _read_labels(os.path.join(args.eval_dir, "labels.csv"))
    queries = [q for q in qdata["queries"] if q["qid"] in labels]
    if args.kind != "all":
        queries = [q for q in queries if q["kind"] == args.kind]
    print(f"[eval] {len(queries)} labelled queries "
          f"({sum(q['kind'] == 'object' for q in queries)} objects, "
          f"{sum(q['kind'] == 'action' for q in queries)} actions)")
    if not queries:
        return
    pre = _precompute(index, queries)
    w = index.weights
    weights = {
        "text": (1.0, 0.0, 0.0),
        "text+context": (w["text"], w["context"], 0.0),
        "image": (0.0, 0.0, 1.0),
        "hybrid": (w["text"], w["context"], w["image"]),
    }
    modes = _available_modes(index)
    results = {}
    print(f"\n{'mode':<14}{'Hit@1':>8}{'Hit@5':>8}{'MRR':>8}{'judged@1':>10}"
          f"{'shown':>8}{'prec.shown':>12}")
    for m in modes:
        r = _evaluate(pre, queries, labels, weights[m], index.thresholds[m], idx_of, items)
        results[m] = r
        print(f"{m:<14}{r['hit@1']:>8.3f}{r['hit@5']:>8.3f}{r['mrr@10']:>8.3f}"
              f"{r['judged@1']:>10.2f}{r['shown']:>8.2f}{r['precision_shown']:>12.3f}")

    if args.grid and "hybrid" in modes:
        dev = [i for i, q in enumerate(queries) if _split(q["task"]) == "dev"]
        test = [i for i, q in enumerate(queries) if _split(q["task"]) == "test"]
        sub = lambda ids: ([pre[i] for i in ids], [queries[i] for i in ids])
        best = None
        for wt in np.arange(0.3, 1.0, 0.1):
            for wc in np.arange(0.0, 0.45, 0.05):
                for wi in np.arange(0.0, 0.55, 0.05):
                    if wt + wc + wi == 0:
                        continue
                    r = _evaluate(*sub(dev), labels, (wt, wc, wi), 0.0, idx_of, items)
                    key = (r["mrr@10"], r["hit@1"])
                    if best is None or key > best[0]:
                        best = (key, (round(float(wt), 2), round(float(wc), 2), round(float(wi), 2)))
        bw = best[1]
        s = sum(bw)
        bw_n = tuple(round(float(x) / s, 3) for x in bw)
        print(f"\n[grid] best hybrid weights on dev tasks ({len(dev)} queries): "
              f"text,context,image = {bw_n}")
        for name, wts in (("hybrid (default)", weights["hybrid"]), ("hybrid (tuned)", bw)):
            r = _evaluate(*sub(test), labels, wts, index.thresholds["hybrid"], idx_of, items)
            print(f"[grid] TEST tasks ({len(test)} queries) {name:<17} Hit@1 {r['hit@1']:.3f}  "
                  f"Hit@5 {r['hit@5']:.3f}  MRR {r['mrr@10']:.3f}")
            results[f"test::{name}"] = r
        results["tuned_weights"] = bw_n
        print(f"[grid] to use them: PICTO_WEIGHTS=\"{bw_n[0]},{bw_n[1]},{bw_n[2]}\"")

    # threshold sweep for the default mode: how often a picture is shown vs. how often it's right
    m = index.default_mode if index.default_mode in modes else "text"
    print(f"\n[threshold sweep, mode={m}]  thr  shown  precision")
    sweep = []
    for thr in np.arange(0.30, 0.95, 0.05):
        r = _evaluate(pre, queries, labels, weights[m], thr, idx_of, items)
        sweep.append({"threshold": round(float(thr), 2), "shown": r["shown"],
                      "precision": r["precision_shown"]})
        print(f"                              {thr:.2f}  {r['shown']:.2f}   {r['precision_shown']:.3f}")
    results["threshold_sweep"] = {"mode": m, "rows": sweep}

    out = os.path.join(args.eval_dir, "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] saved {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="pool candidates and write the labelling page")
    e.add_argument("--library", required=True)
    e.add_argument("--kb", default=os.path.join(BASE, "knowledgeBase.json"))
    e.add_argument("--tasks", help="optional text file: one task name per line (subset of the KB)")
    e.add_argument("--out", default=os.path.join(BASE, "eval"))
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--sample", type=int, default=0,
                   help="label only a random sample of N queries (e.g. 250); 0 = all")
    e.add_argument("--seed", type=int, default=13)
    s = sub.add_parser("score", help="compute metrics from eval/labels.csv")
    s.add_argument("--library", required=True)
    s.add_argument("--eval-dir", default=os.path.join(BASE, "eval"))
    s.add_argument("--kind", choices=["all", "object", "action"], default="all")
    s.add_argument("--grid", action="store_true", help="tune hybrid weights (dev/test split by task)")
    args = p.parse_args(argv)
    (export if args.cmd == "export" else score)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
