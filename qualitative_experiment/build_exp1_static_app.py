#!/usr/bin/env python3
"""Build the Experiment-1 close-reading app: a self-contained static HTML page.

For each of the 5 papers in selection.json (exp1_close_reading) the page shows
three introductions one after another — the authors' original, the LLM "basic
prompt" version, and the LLM "CARS-guided prompt" version (claude-opus-4-8;
the model is not named in the app) — every one rendered with the SAME colored
CARS highlights used in the discrimination study (annotations.chatgpt,
collapse convention, single source of truth: the discrimination renderer).
Below the three versions sits one big free-comment box per paper.

No installation for the reader: they open index.html in any browser
(double-click), comments autosave to localStorage, and a button downloads all
comments as one JSON file to send back. NOT blinded — the three versions are
labeled; if the reader is also the discrimination-study annotator, they must
finish that study first (the app reveals the two prompting modes).

Usage:
    python qualitative_experiment/build_exp1_static_app.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "discrimination_experiment"))
sys.path.insert(0, str(QUAL_DIR))

from build_argilla_datasets import CARS_LABELS, collapse_segments, render_intro_html  # noqa: E402
from render_verify import verify_roundtrip  # noqa: E402

APP_VERSION = "exp1-v1"
GENERATOR = "claude-opus-4-8"

ROLES = [
    ("original", None, "Original — written by the article's authors", "#2e7d32"),
    ("basic", "basic", "AI-generated — basic prompt (no writing instructions)", "#b26a00"),
    ("cars_guided", "cars_guided",
     "AI-generated — CARS-guided prompt (given Swales' move scheme to plan with)", "#6a4fa3"),
]

GUIDANCE_HTML = """
<b>Things to compare across the three versions</b> (free form — note whatever you find,
these are only pointers):
<ul style="margin:8px 0 0 0;padding-left:22px;">
 <li><b>Rhetorical structure</b> — which moves/steps occur and in what order (the colored
     annotation helps); does an AI version &ldquo;tidy up&rdquo; or normalise the original's
     structure?</li>
 <li><b>The niche</b> — how is the gap / motivation built? Earned from the literature or
     simply asserted?</li>
 <li><b>Use of the literature</b> — specific engagement vs. name-dropping; citations that
     look misplaced or invented.</li>
 <li><b>Content</b> — does each AI version promise the same study the original announces?</li>
 <li><b>Style</b> — register, hedging, formulaic phrasing, anything that gives the writer
     away.</li>
</ul>
Please name the versions explicitly in your comment (&ldquo;the original&rdquo;,
&ldquo;the basic version&rdquo;, &ldquo;the CARS-guided version&rdquo;)."""

APP_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Close reading — human vs AI introductions</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #e9edf2; color: #1c1c1c;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  a { color: #35507a; }
  #topbar { position: sticky; top: 0; z-index: 10; background: #223148; color: #fff;
            padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  #topbar h1 { font-size: 1.02em; margin: 0; font-weight: 600; flex: 1 1 260px; }
  #annotator { border: none; border-radius: 5px; padding: 6px 9px; font-size: 0.9em; width: 170px; }
  .wrap { max-width: 900px; margin: 0 auto; padding: 18px 14px 60px; }
  .card { background: #ffffff; border-radius: 10px; padding: 16px 20px; margin: 14px 0;
          box-shadow: 0 1px 3px rgba(20,30,50,0.08); }
  .paperhead h2 { margin: 0 0 4px; font-size: 1.12em; }
  .meta { color: #55606e; font-size: 0.88em; }
  .chip { display: inline-block; background: #e3e9f2; color: #34435c; border-radius: 4px;
          padding: 1px 8px; font-size: 0.82em; margin-left: 6px; }
  .navrow { display: flex; align-items: center; gap: 8px; margin: 16px 0 4px; flex-wrap: wrap; }
  .navrow button { background: #35507a; color: #fff; border: none; border-radius: 6px;
                   padding: 8px 16px; font-size: 0.95em; cursor: pointer; }
  .navrow button:disabled { background: #a9b4c4; cursor: default; }
  .pill { border: 1px solid #8fa0b8; background: #f4f7fb; color: #34435c; border-radius: 15px;
          padding: 4px 12px; font-size: 0.88em; cursor: pointer; }
  .pill.active { background: #35507a; color: #fff; border-color: #35507a; }
  .pill.done::after { content: " \\2713"; color: #2e7d32; }
  .pill.active.done::after { color: #b5e0b8; }
  .rolebanner { border-left: 6px solid; border-radius: 6px; background: #f4f6f9;
                padding: 8px 12px; margin: 22px 0 8px; font-weight: 600; font-size: 0.98em; }
  .rolemeta { font-weight: 400; color: #55606e; font-size: 0.85em; }
  .intro-wrap { font-family: Georgia, "Times New Roman", serif; font-size: 15px;
                border: 1px solid #d4dbe4; border-radius: 10px; }
  .guidance { background: #f2f6ee; border: 1px solid #cfdcc2; border-radius: 8px;
              padding: 12px 16px; font-size: 0.92em; margin-top: 26px; }
  textarea { width: 100%; min-height: 240px; border: 1px solid #b8c2cf; border-radius: 8px;
             padding: 12px; font-size: 0.98em; font-family: inherit; line-height: 1.5;
             resize: vertical; }
  textarea:focus { outline: 2px solid #35507a; }
  .savednote { color: #55606e; font-size: 0.85em; margin-top: 6px; min-height: 1.2em; }
  .dl { background: #2e7d32; color: #fff; border: none; border-radius: 6px;
        padding: 9px 18px; font-size: 0.95em; cursor: pointer; }
  .footer { text-align: center; margin-top: 26px; color: #55606e; font-size: 0.9em; }
</style>
</head>
<body>
<div id="topbar">
  <h1>Close reading — human vs AI introductions</h1>
  <input id="annotator" placeholder="Your name" title="Your name (stored in the comments file)">
  <button class="dl" onclick="downloadComments()">Download comments</button>
</div>
<div class="wrap">
  <div class="card" id="introcard">
    <b>How this works.</b> There are <span id="npapers"></span> papers. For each paper you
    will read three versions of its introduction, shown one after another: the
    <b>original</b> written by the article's authors, and two versions generated by a large
    language model that was given the article's content and cited works but not its
    introduction — one from a <b>basic prompt</b> (just &ldquo;write the introduction&rdquo;)
    and one from a <b>CARS-guided prompt</b> (the model was additionally given Swales'
    CARS move scheme and asked to plan its rhetorical moves first). All three carry
    colored move annotations produced by the same automatic annotator. Read the three
    versions, then write your observations in the comment box at the bottom — there are
    no fixed questions; comparisons, oddities, and single striking details are all
    welcome. Comments save automatically in this browser; when you are done (or as a
    backup), press <b>Download comments</b> and send the file to the research team.
    Please use the same browser on the same computer throughout, and don't use
    private/incognito mode.
  </div>
  <div class="navrow" id="nav-top"></div>
  <div id="main"></div>
  <div class="navrow" id="nav-bottom"></div>
  <div class="footer">
    <button class="dl" onclick="downloadComments()">Download comments</button>
    <div style="margin-top:8px;">Send the downloaded file to the research team.
    App __APP_VERSION__.</div>
  </div>
</div>
<script>
const DATA = __DATA_JSON__;
const KEY = "qual_triplets___APP_VERSION__";
let cur = 0;

function defaultState() { return { annotator: "", comments: {} }; }
function loadState() {
  try { return JSON.parse(localStorage.getItem(KEY)) || defaultState(); }
  catch (e) { return defaultState(); }
}
let state = loadState();
function saveState() { localStorage.setItem(KEY, JSON.stringify(state)); }

function escapeHtml(t) {
  return String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function hasComment(pk) {
  const c = state.comments[pk];
  return c && c.comment && c.comment.trim().length > 0;
}

function navHtml() {
  let pills = DATA.papers.map((p, i) =>
    '<span class="pill' + (i === cur ? " active" : "") +
    (hasComment(p.paper_key) ? " done" : "") + '" onclick="go(' + i + ')">' +
    "Paper " + (i + 1) + "</span>").join("");
  return '<button onclick="go(cur-1)"' + (cur === 0 ? " disabled" : "") + ">&larr; Previous</button>" +
         pills +
         '<button onclick="go(cur+1)"' + (cur === DATA.papers.length - 1 ? " disabled" : "") +
         ">Next &rarr;</button>";
}

function render() {
  const p = DATA.papers[cur];
  document.getElementById("nav-top").innerHTML = navHtml();
  document.getElementById("nav-bottom").innerHTML = navHtml();
  let h = '<div class="card paperhead"><h2>Paper ' + (cur + 1) + " of " + DATA.papers.length +
          " &mdash; " + escapeHtml(p.title) + "</h2>" +
          '<div class="meta">id ' + escapeHtml(p.paper_key) +
          '<span class="chip">journal quartile ' + escapeHtml(p.quartile) + "</span></div></div>";
  for (const intro of p.intros) {
    h += '<div class="rolebanner" style="border-color:' + intro.color + '">' +
         escapeHtml(intro.role_label) +
         ' <span class="rolemeta">&mdash; ' + intro.word_count + " words, " +
         intro.n_units + ' move units</span></div>' +
         '<div class="intro-wrap">' + intro.html + "</div>";
  }
  h += '<div class="guidance">' + DATA.guidance_html + "</div>";
  const saved = state.comments[p.paper_key];
  h += '<div class="card"><b>Your comments on this paper</b> (autosaves as you type)<br><br>' +
       '<textarea id="comment" placeholder="Write your observations about the three versions here…"></textarea>' +
       '<div class="savednote" id="savednote"></div></div>';
  document.getElementById("main").innerHTML = h;
  const box = document.getElementById("comment");
  box.value = saved ? saved.comment : "";
  box.addEventListener("input", onEdit);
  updateSavedNote();
  window.scrollTo(0, 0);
}

function onEdit() {
  const p = DATA.papers[cur];
  state.comments[p.paper_key] = {
    comment: document.getElementById("comment").value,
    updated_at: new Date().toISOString()
  };
  saveState();
  updateSavedNote();
  document.getElementById("nav-top").innerHTML = navHtml();
  document.getElementById("nav-bottom").innerHTML = navHtml();
}

function updateSavedNote() {
  const c = state.comments[DATA.papers[cur].paper_key];
  document.getElementById("savednote").textContent =
    c && c.comment.trim() ? "Saved in this browser at " + new Date(c.updated_at).toLocaleString() : "";
}

function go(i) {
  if (i < 0 || i >= DATA.papers.length) return;
  cur = i;
  render();
}

function downloadComments() {
  state.annotator = document.getElementById("annotator").value.trim();
  saveState();
  const missing = DATA.papers.filter(p => !hasComment(p.paper_key)).length;
  if (missing > 0 &&
      !confirm(missing + " of " + DATA.papers.length +
               " papers have no comment yet. Download anyway (as a backup)?")) return;
  const payload = {
    study: "qualitative_triplets",
    app_version: "__APP_VERSION__",
    annotator: state.annotator,
    downloaded_at: new Date().toISOString(),
    comments: state.comments
  };
  const blob = new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "qualitative_triplet_comments.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

document.getElementById("npapers").textContent = DATA.papers.length;
document.getElementById("annotator").value = state.annotator || "";
document.getElementById("annotator").addEventListener("input", function () {
  state.annotator = this.value; saveState();
});
render();
</script>
</body>
</html>
"""

README = """# Experiment 1 — close reading of human vs AI introductions (reader guide)

You received a folder with one file that matters: `index.html`.

1. Double-click `index.html` — it opens in your web browser. Nothing to install,
   no internet needed.
2. Work paper by paper (5 papers). For each paper, read the three versions of its
   introduction (original, AI basic, AI CARS-guided — all labeled) and write your
   observations in the big comment box at the bottom. Everything autosaves in the
   browser; you can close the page and continue later.
3. Please always use the SAME browser on the SAME computer, and not a
   private/incognito window (those forget the autosaved comments).
4. When you are done — and once in between as a backup — press **Download
   comments** and send the downloaded `qualitative_triplet_comments.json` to the
   research team.

Note: if you are also taking part in the "who wrote this introduction?" judgment
study, please FINISH that study before opening this one — this app reveals
information that would bias those judgments.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", default=str(QUAL_DIR / "selection.json"))
    parser.add_argument("--dataset", default=str(REPO / "data" / "master_dataset.json"))
    parser.add_argument("--out-dir", default=str(QUAL_DIR / "exp1_close_reading"))
    args = parser.parse_args()

    selection = json.loads(Path(args.selection).read_text())
    master = json.loads(Path(args.dataset).read_text())
    intro_index = {i["intro_id"]: i for p in master["papers"] for i in p["intros"]}

    papers = []
    for row in selection["exp1_close_reading"]:
        pk = row["paper_key"]
        intros = []
        for role_id, mode, role_label, color in ROLES:
            intro_id = f"{pk}_original" if mode is None else f"{pk}_{GENERATOR}_{mode}"
            intro = intro_index[intro_id]
            segments = intro["annotations"]["chatgpt"]["segments"]
            unknown = {s["label"] for s in segments} - set(CARS_LABELS)
            if unknown:
                raise SystemExit(f"{intro_id}: unknown CARS labels {unknown}")
            rendered = render_intro_html(intro["text"], segments)
            verify_roundtrip(rendered, intro["text"], intro_id)
            intros.append(
                {
                    "role": role_id,
                    "role_label": role_label,
                    "color": color,
                    "word_count": intro["word_count"],
                    "n_units": len(collapse_segments(segments, intro["text"])),
                    "html": rendered,
                }
            )
        papers.append(
            {
                "paper_key": pk,
                "title": row["title"],
                "quartile": row["quartile"],
                "intros": intros,
            }
        )

    data = {
        "built": date.today().isoformat(),
        "guidance_html": GUIDANCE_HTML,
        "papers": papers,
    }
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = APP_TEMPLATE.replace("__DATA_JSON__", data_json).replace("__APP_VERSION__", APP_VERSION)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page)
    (out_dir / "README.md").write_text(README)
    size_mb = len(page.encode()) / 1e6
    print(
        f"wrote {out_dir}/index.html ({size_mb:.2f} MB): "
        f"{len(papers)} papers x 3 intros, all roundtrip-verified"
    )


if __name__ == "__main__":
    main()
