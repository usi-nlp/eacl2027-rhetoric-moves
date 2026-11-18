#!/usr/bin/env python3
"""Build the Experiment-2 close-reading app: a self-contained static HTML page.

Shows the four papers in `selection.json`'s exp2_swales_aware list (one per
quartile, hand-picked by the researcher after auditing every
master_dataset.json swales_topic flag for false positives — see EXP2_NOTES in
select_papers.py). Each is an original (human-written) introduction whose
paper genuinely engages Swales' CARS model or rhetorical move analysis
somewhere in the article; the app tells the reader exactly where that
engagement appears (inside this introduction, or elsewhere in the paper) so
they can judge accordingly. No control/distractor papers.

Same look and mechanics as the exp-1 app: colored CARS highlights
(annotations.chatgpt), one big per-paper comment box, localStorage autosave,
"Download comments" -> one JSON file.

Usage:
    python qualitative_experiment/build_exp2_static_app.py
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

APP_VERSION = "exp2-v2"

GUIDANCE_HTML = """
<b>What to look at</b> (free form — these are only pointers):
<ul style="margin:8px 0 0 0;padding-left:22px;">
 <li><b>Move structure</b> — canonical or idiosyncratic? A deliberate,
     signposted niche? Self-conscious use of the funnel?</li>
 <li><b>Metadiscourse</b> — does the author audibly steer the reader through
     the introduction?</li>
 <li><b>Use of the literature</b> and how the gap is built.</li>
 <li>Anything where you <b>disagree with the colored annotation</b> — say so.</li>
 <li>Where relevant, note whether the paper's Swales/CARS engagement (see the
     note above each text) seems to leave a trace in this introduction or
     not.</li>
</ul>"""

APP_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Does knowing the CARS model change how authors write?</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #e9edf2; color: #1c1c1c;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  #topbar { position: sticky; top: 0; z-index: 10; background: #3a2b55; color: #fff;
            padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  #topbar h1 { font-size: 1.02em; margin: 0; font-weight: 600; flex: 1 1 260px; }
  #annotator { border: none; border-radius: 5px; padding: 6px 9px; font-size: 0.9em; width: 170px; }
  .wrap { max-width: 900px; margin: 0 auto; padding: 18px 14px 60px; }
  .card { background: #ffffff; border-radius: 10px; padding: 16px 20px; margin: 14px 0;
          box-shadow: 0 1px 3px rgba(20,30,50,0.08); }
  .paperhead h2 { margin: 0 0 4px; font-size: 1.12em; }
  .meta { color: #55606e; font-size: 0.88em; }
  .chip { display: inline-block; background: #eee3f7; color: #46345c; border-radius: 4px;
          padding: 1px 8px; font-size: 0.82em; margin-left: 6px; }
  .navrow { display: flex; align-items: center; gap: 8px; margin: 16px 0 4px; flex-wrap: wrap; }
  .navrow button { background: #574080; color: #fff; border: none; border-radius: 6px;
                   padding: 8px 16px; font-size: 0.95em; cursor: pointer; }
  .navrow button:disabled { background: #a9b4c4; cursor: default; }
  .pill { border: 1px solid #9a8cb8; background: #f6f4fb; color: #46345c; border-radius: 15px;
          padding: 4px 12px; font-size: 0.88em; cursor: pointer; }
  .pill.active { background: #574080; color: #fff; border-color: #574080; }
  .pill.done::after { content: " \\2713"; color: #2e7d32; }
  .pill.active.done::after { color: #b5e0b8; }
  .note { border-left: 6px solid #6a4fa3; border-radius: 6px; background: #f4f1fa;
          padding: 10px 14px; margin: 16px 0; font-size: 0.92em; }
  .intro-wrap { font-family: Georgia, "Times New Roman", serif; font-size: 15px;
                border: 1px solid #d4dbe4; border-radius: 10px; margin-top: 14px; }
  .guidance { background: #f3eff9; border: 1px solid #d5c8e8; border-radius: 8px;
              padding: 12px 16px; font-size: 0.92em; margin-top: 26px; }
  textarea { width: 100%; min-height: 220px; border: 1px solid #b8c2cf; border-radius: 8px;
             padding: 12px; font-size: 0.98em; font-family: inherit; line-height: 1.5;
             resize: vertical; }
  textarea:focus { outline: 2px solid #574080; }
  .savednote { color: #55606e; font-size: 0.85em; margin-top: 6px; min-height: 1.2em; }
  .dl { background: #2e7d32; color: #fff; border: none; border-radius: 6px;
        padding: 9px 18px; font-size: 0.95em; cursor: pointer; }
  .footer { text-align: center; margin-top: 26px; color: #55606e; font-size: 0.9em; }
</style>
</head>
<body>
<div id="topbar">
  <h1>Does knowing the CARS model change how authors write?</h1>
  <input id="annotator" placeholder="Your name" title="Your name (stored in the comments file)">
  <button class="dl" onclick="downloadComments()">Download comments</button>
</div>
<div class="wrap">
  <div class="card">
    <b>How this works.</b> You will read <b>4 introductions from published research
    articles</b> — all written by the articles' human authors, none AI-generated, one per
    journal quartile (Q1&hellip;Q4). Each comes from an article that itself engages with
    <b>Swales' CARS model or rhetorical move analysis</b> somewhere — a note above each text
    says exactly where that engagement appears (sometimes inside the introduction shown
    here, sometimes elsewhere in the paper). The question we want your view on: <b>does
    familiarity with the CARS framework show in how the authors structure their own
    introduction?</b> The colored highlights are the rhetorical moves as identified by an
    automatic annotator, the same system for every text — treat them as an imperfect aid.
    Write a comment for each paper. Comments save automatically in this browser; when you
    are done (or as a backup), press <b>Download comments</b> and send the file to the
    research team. Please use the same browser on the same computer throughout, not in
    private/incognito mode. All texts are shown as a single continuous paragraph on
    purpose; minor formatting glitches come from PDF text extraction and are not
    meaningful.
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
const KEY = "qual_swales___APP_VERSION__";
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
    escapeHtml(p.quartile) + "</span>").join("");
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
          '<span class="chip">journal quartile ' + escapeHtml(p.quartile) + "</span>" +
          '<span class="chip">' + p.word_count + " words, " + p.n_units + " move units</span></div>" +
          '<div class="note"><b>Where the Swales/CARS engagement appears:</b> ' +
          escapeHtml(p.researcher_note) + "</div></div>" +
          '<div class="intro-wrap">' + p.html + "</div>";
  h += '<div class="guidance">' + DATA.guidance_html + "</div>";
  const saved = state.comments[p.paper_key];
  h += '<div class="card"><b>Your comments on this paper</b> (autosaves as you type)<br><br>' +
       '<textarea id="comment" placeholder="Write your observations about this introduction here…"></textarea>' +
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
  const comments = {};
  for (const p of DATA.papers) {
    const c = state.comments[p.paper_key];
    if (!c || !c.comment.trim()) continue;
    comments[p.paper_key] = { quartile: p.quartile, comment: c.comment, updated_at: c.updated_at };
  }
  const payload = {
    study: "exp2_swales_aware",
    app_version: "__APP_VERSION__",
    annotator: state.annotator,
    downloaded_at: new Date().toISOString(),
    comments: comments
  };
  const blob = new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "swales_awareness_comments.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

document.getElementById("annotator").value = state.annotator || "";
document.getElementById("annotator").addEventListener("input", function () {
  state.annotator = this.value; saveState();
});
render();
</script>
</body>
</html>
"""

README = """# Experiment 2 — Swales-aware introductions (reader guide)

You received a folder with one file that matters: `index.html`.

1. Double-click `index.html` — it opens in your web browser. Nothing to install,
   no internet needed.
2. Work through the 4 papers (one per journal quartile). For each, read the
   introduction and write your observations in the comment box below it —
   the note above the text tells you where in the paper its Swales/CARS
   engagement actually appears (sometimes it's in this introduction,
   sometimes elsewhere in the article). Everything autosaves in the browser;
   you can close the page and continue later.
3. Please always use the SAME browser on the SAME computer, and not a
   private/incognito window (those forget the autosaved comments).
4. When you are done — and once in between as a backup — press **Download
   comments** and send the downloaded `swales_awareness_comments.json` to the
   research team.

Note: if you are also taking part in the "who wrote this introduction?" judgment
study, please FINISH that study before opening this one.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", default=str(QUAL_DIR / "selection.json"))
    parser.add_argument("--dataset", default=str(REPO / "data" / "master_dataset.json"))
    parser.add_argument("--out-dir", default=str(QUAL_DIR / "exp2_swales_aware"))
    args = parser.parse_args()

    selection = json.loads(Path(args.selection).read_text())
    master = json.loads(Path(args.dataset).read_text())
    intro_index = {i["intro_id"]: i for p in master["papers"] for i in p["intros"]}

    papers = []
    for row in selection["exp2_swales_aware"]:
        pk = row["paper_key"]
        intro = intro_index[f"{pk}_original"]
        segments = intro["annotations"]["chatgpt"]["segments"]
        unknown = {s["label"] for s in segments} - set(CARS_LABELS)
        if unknown:
            raise SystemExit(f"{pk}: unknown CARS labels {unknown}")
        rendered = render_intro_html(intro["text"], segments)
        verify_roundtrip(rendered, intro["text"], f"{pk}_original")
        papers.append(
            {
                "paper_key": pk,
                "title": row["title"],
                "quartile": row["quartile"],
                "researcher_note": row["researcher_note"],
                "word_count": intro["word_count"],
                "n_units": len(collapse_segments(segments, intro["text"])),
                "html": rendered,
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
        f"{len(papers)} papers, all roundtrip-verified"
    )


if __name__ == "__main__":
    main()
