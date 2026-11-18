#!/usr/bin/env python3
"""
Interactive citation review app.

Select text in the introduction → the app suggests the best-matching reference
→ click Assign to save it to the JSON file.

Run from the repo root:
  python data_processing/citation_review_app.py
Then open http://localhost:5000 in your browser.
"""

import json
import re
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
REFS_DIR   = _REPO_ROOT / "data/raw_data/references_json_with_citations"
EXCEL_PATH = _REPO_ROOT / "data/raw_data/80-papers_corpus_nativeness_quartiles.xlsx"
PORT = 5000

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_excel_cache: dict[int, dict] | None = None
_paper_index: dict[int, Path] | None = None   # paper_id → json path


def _load_excel() -> dict[int, dict]:
    try:
        import openpyxl
    except ImportError:
        print("pip install openpyxl"); sys.exit(1)

    wb = openpyxl.load_workbook(EXCEL_PATH)
    ws = wb.active
    headers = [c.value for c in ws[1]]
    pid_col   = headers.index("paper_id")
    intro_col = headers.index("introduction")
    title_col = headers.index("title")
    nat_col   = headers.index("nativeness")
    q_col     = headers.index("quartile")

    result: dict[int, dict] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        pid = row[pid_col]
        if pid is None:
            continue
        result[int(pid)] = {
            "intro":      str(row[intro_col] or "").strip(),
            "title":      str(row[title_col] or "").strip(),
            "nativeness": str(row[nat_col]   or "").strip(),
            "quartile":   str(row[q_col]     or "").strip(),
        }
    return result


def _build_paper_index() -> dict[int, Path]:
    idx: dict[int, Path] = {}
    for jf in sorted(REFS_DIR.glob("*.json")):
        if jf.name.startswith("_"):
            continue
        try:
            with open(jf) as fh:
                d = json.load(fh)
            pid = d.get("paper_id")
            if pid is not None:
                idx[int(pid)] = jf
        except Exception:
            pass
    return idx


def excel() -> dict[int, dict]:
    global _excel_cache
    if _excel_cache is None:
        _excel_cache = _load_excel()
    return _excel_cache


def paper_index() -> dict[int, Path]:
    global _paper_index
    if _paper_index is None:
        _paper_index = _build_paper_index()
    return _paper_index


def load_paper_json(paper_id: int) -> dict | None:
    idx = paper_index()
    jf = idx.get(paper_id)
    if jf is None:
        return None
    with open(jf) as fh:
        return json.load(fh)


def save_paper_json(paper_id: int, data: dict) -> None:
    jf = paper_index()[paper_id]
    with open(jf, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Reference matching helpers
# ---------------------------------------------------------------------------

def _extract_year(text: str) -> str | None:
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None


def _extract_surnames(text: str) -> list[str]:
    """Pull capitalised words that look like surnames from a citation string."""
    return re.findall(r"\b([A-ZÁÉÍÓÚÀÂÄÔÜÇÑŞĞ][a-záéíóúàâäôüçñşğ\-]{2,})\b", text)


def score_match(selected: str, ref: dict) -> float:
    """Score how well `selected` (highlighted text) matches `ref`."""
    score = 0.0
    sel_year = _extract_year(selected)
    sel_surnames = {s.lower() for s in _extract_surnames(selected)}

    # Year match
    ref_year = str(ref.get("year", "") or "")
    ref_str  = ref.get("reference") or ""
    ref_year2_m = re.search(r"\b(19|20\d{2})\b", ref_str)
    ref_years = {ref_year[:4]} if len(ref_year) >= 4 else set()
    if ref_year2_m:
        ref_years.add(ref_year2_m.group(0))
    if sel_year and sel_year in ref_years:
        score += 3.0

    # Surname match from authors list
    ref_authors = ref.get("authors") or []
    ref_surnames: set[str] = set()
    for name in ref_authors:
        for part in name.split():
            part = part.strip(".,")
            if len(part) > 2 and part[0].isupper():
                ref_surnames.add(part.lower())
    # Also from raw reference string (first 120 chars before year)
    before_year = ref_str[:120]
    for s in _extract_surnames(before_year):
        ref_surnames.add(s.lower())

    overlap = sel_surnames & ref_surnames
    score += len(overlap) * 2.0

    # Partial match (one surname appears as substring of another — handles diacritics)
    for ss in sel_surnames:
        for rs in ref_surnames:
            if ss in rs or rs in ss:
                score += 0.5

    return score


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

def extract_context(intro: str, citation_text: str) -> str:
    """Return the sentence(s) around `citation_text` in `intro`."""
    pos = intro.find(citation_text)
    if pos == -1:
        return ""
    start = max(0, pos - 300)
    end = min(len(intro), pos + len(citation_text) + 300)
    chunk = intro[start:end]
    local = pos - start

    # find sentence start
    before = chunk[:local]
    s = max(before.rfind("\n"), before.rfind(". "), before.rfind("! "), before.rfind("? "))
    sent_start = s + 1 if s >= 0 else 0

    # find sentence end
    after = chunk[local + len(citation_text):]
    candidates = [after.find(x) for x in (".\n", ". ", "! ", "? ", "\n\n") if after.find(x) >= 0]
    sent_end = (local + len(citation_text) + min(candidates) + 1) if candidates else len(chunk)

    return chunk[sent_start:sent_end].strip()


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/papers")
def api_papers():
    ex = excel()
    idx = paper_index()
    papers = []
    for pid in sorted(idx):
        data = load_paper_json(pid)
        if data is None:
            continue
        refs = data.get("references", [])
        cited = sum(1 for r in refs if r.get("cited_in_introduction"))
        meta = ex.get(pid, {})
        papers.append({
            "paper_id":   pid,
            "title":      meta.get("title") or data.get("paper_title", ""),
            "nativeness": meta.get("nativeness", ""),
            "quartile":   meta.get("quartile", ""),
            "total_refs": len(refs),
            "cited_refs": cited,
        })
    return jsonify(papers)


@app.route("/api/paper/<int:pid>")
def api_paper(pid: int):
    data = load_paper_json(pid)
    if data is None:
        return jsonify({"error": "not found"}), 404
    ex = excel()
    meta = ex.get(pid, {})
    return jsonify({
        "paper_id":   pid,
        "title":      meta.get("title") or data.get("paper_title", ""),
        "nativeness": meta.get("nativeness", ""),
        "quartile":   meta.get("quartile", ""),
        "intro":      meta.get("intro", ""),
        "references": data.get("references", []),
    })


@app.route("/api/paper/<int:pid>/citation", methods=["POST"])
def api_add_citation(pid: int):
    """
    Body: { ref_index: int, citation_text: str, context: str }
    Adds the citation to the reference and sets cited_in_introduction = true.
    """
    body = request.get_json(force=True)
    ref_index   = int(body["ref_index"])
    cit_text    = str(body["citation_text"]).strip()
    context_str = str(body.get("context", "")).strip()

    data = load_paper_json(pid)
    if data is None:
        return jsonify({"error": "paper not found"}), 404

    # Find the reference by its index field
    ref = next((r for r in data.get("references", []) if r.get("index") == ref_index), None)
    if ref is None:
        return jsonify({"error": f"ref index {ref_index} not found"}), 404

    # Build context if not supplied
    if not context_str:
        ex = excel()
        intro = ex.get(pid, {}).get("intro", "")
        context_str = extract_context(intro, cit_text)

    # Check for duplicate
    existing = ref.get("introduction_citation_contexts") or []
    for ctx in existing:
        if ctx.get("citation_text") == cit_text:
            return jsonify({"status": "already_exists", "ref_index": ref_index})

    existing.append({"citation_text": cit_text, "context": context_str})
    ref["introduction_citation_contexts"] = existing
    ref["cited_in_introduction"] = True

    save_paper_json(pid, data)
    return jsonify({"status": "ok", "ref_index": ref_index, "citation_text": cit_text})


@app.route("/api/paper/<int:pid>/citation", methods=["DELETE"])
def api_remove_citation(pid: int):
    """
    Body: { ref_index: int, citation_text: str }
    Removes a specific citation context from a reference.
    """
    body = request.get_json(force=True)
    ref_index = int(body["ref_index"])
    cit_text  = str(body["citation_text"]).strip()

    data = load_paper_json(pid)
    if data is None:
        return jsonify({"error": "paper not found"}), 404

    ref = next((r for r in data.get("references", []) if r.get("index") == ref_index), None)
    if ref is None:
        return jsonify({"error": f"ref index {ref_index} not found"}), 404

    existing = ref.get("introduction_citation_contexts") or []
    ref["introduction_citation_contexts"] = [c for c in existing if c.get("citation_text") != cit_text]
    ref["cited_in_introduction"] = bool(ref["introduction_citation_contexts"])

    save_paper_json(pid, data)
    return jsonify({"status": "ok"})


@app.route("/api/paper/<int:pid>/suggest")
def api_suggest(pid: int):
    """
    ?text=... — return top-5 reference suggestions for the selected text.
    """
    selected = request.args.get("text", "")
    data = load_paper_json(pid)
    if data is None:
        return jsonify([])

    scored = []
    for ref in data.get("references", []):
        s = score_match(selected, ref)
        if s > 0:
            scored.append((s, ref.get("index"), ref.get("reference") or ""))

    scored.sort(key=lambda x: -x[0])
    return jsonify([
        {"score": s, "index": idx, "reference": raw}
        for s, idx, raw in scored[:5]
    ])


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Citation Review</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       height: 100vh; display: flex; flex-direction: column; background: #f0f2f5; }

/* ── Header ── */
#header {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 18px; background: #1a3a5c; color: #fff;
  flex-shrink: 0;
}
#header h1 { font-size: 1em; font-weight: 600; color: #adc8e8; flex-shrink: 0; }
#paper-select { flex: 1; max-width: 480px; padding: 5px 8px; border-radius: 4px;
                border: none; font-size: 0.88em; }
#header button { padding: 5px 12px; border: 1px solid #4a7aa8; background: #1e4a72;
                  color: #fff; border-radius: 4px; cursor: pointer; font-size: 0.85em; }
#header button:hover { background: #2563a0; }
#progress { font-size: 0.8em; color: #adc8e8; flex-shrink: 0; }
#save-status { font-size: 0.8em; color: #7fc; flex-shrink: 0; min-width: 80px; text-align: right; }

/* ── Main split ── */
#main { display: flex; flex: 1; overflow: hidden; gap: 0; }

/* ── Left: intro ── */
#left {
  flex: 0 0 58%; display: flex; flex-direction: column;
  border-right: 1px solid #ccc; background: #fff; overflow: hidden;
}
#left-header {
  padding: 10px 16px; background: #f7f8fa; border-bottom: 1px solid #e0e0e0;
  font-size: 0.82em; color: #555; flex-shrink: 0;
}
#left-header strong { color: #222; }
#intro-container { flex: 1; overflow-y: auto; padding: 18px 20px; }
#intro-text {
  font-size: 0.88em; line-height: 1.85; color: #222; white-space: pre-wrap;
  word-break: break-word; user-select: text;
}
mark.cited {
  background: #c8f7c5; border-radius: 3px; padding: 1px 1px;
  cursor: pointer; position: relative;
}
mark.cited:hover::after {
  content: attr(data-tip);
  position: absolute; bottom: 100%; left: 0;
  background: #333; color: #fff; font-size: 0.75em;
  padding: 3px 7px; border-radius: 3px; white-space: nowrap;
  z-index: 100; pointer-events: none;
}
mark.pending { background: #fff3cd; border-radius: 3px; padding: 1px 1px; }

/* ── Right: references ── */
#right {
  flex: 1; display: flex; flex-direction: column;
  background: #fff; overflow: hidden;
}
#right-header {
  padding: 10px 14px; background: #f7f8fa; border-bottom: 1px solid #e0e0e0;
  flex-shrink: 0;
}
#right-header h2 { font-size: 0.92em; color: #333; margin-bottom: 6px; }
#ref-search {
  width: 100%; padding: 5px 8px; border: 1px solid #ccc; border-radius: 4px;
  font-size: 0.82em;
}
#ref-list { flex: 1; overflow-y: auto; }
.ref-item {
  padding: 9px 14px; border-bottom: 1px solid #f0f0f0; cursor: pointer;
  transition: background 0.1s;
}
.ref-item:hover { background: #f0f7ff; }
.ref-item.cited-item { background: #f0fff4; }
.ref-item.cited-item:hover { background: #e0f7ea; }
.ref-item.suggested { background: #fffbea; }
.ref-item.suggested:hover { background: #fff3c0; }
.ref-item .ref-top { display: flex; align-items: flex-start; gap: 8px; }
.ref-item .ref-num { font-size: 0.75em; color: #999; flex-shrink: 0; padding-top: 2px; }
.ref-item .ref-raw { font-size: 0.78em; color: #333; line-height: 1.5; }
.ref-item .ref-raw em { font-style: normal; color: #1a5c2a; font-weight: 600; }
.badge { display: inline-block; font-size: 0.7em; border-radius: 3px;
         padding: 1px 6px; margin-right: 4px; flex-shrink: 0; }
.b-cited { background: #27ae60; color: #fff; }
.b-none  { background: #ddd; color: #666; }
.b-match { background: #f39c12; color: #fff; }
.cit-ctx { font-size: 0.73em; color: #666; margin-top: 4px; border-left: 3px solid #27ae60;
            padding-left: 6px; }
.assign-btn {
  display: inline-block; margin-top: 5px; padding: 3px 10px;
  background: #2563a0; color: #fff; border: none; border-radius: 3px;
  cursor: pointer; font-size: 0.76em;
}
.assign-btn:hover { background: #1a4a80; }
.remove-btn {
  display: inline-block; padding: 2px 7px; background: none;
  color: #c0392b; border: 1px solid #e0a0a0; border-radius: 3px;
  cursor: pointer; font-size: 0.7em; margin-left: 6px;
}
.remove-btn:hover { background: #fdecea; }

/* ── Selection panel ── */
#sel-panel {
  display: none; flex-shrink: 0;
  background: #1e3a5c; color: #fff;
  padding: 10px 16px; border-top: 2px solid #4a7aa8;
  font-size: 0.85em;
}
#sel-panel.active { display: block; }
#sel-text { font-style: italic; color: #adc8e8; margin-bottom: 6px; font-size: 0.88em; }
#sel-hint { color: #cde; font-size: 0.82em; }
#sel-panel button.cancel-btn {
  float: right; margin-top: -4px;
  padding: 3px 10px; background: none; border: 1px solid #5588aa;
  color: #adc8e8; border-radius: 3px; cursor: pointer; font-size: 0.8em;
}

/* ── No intro ── */
.no-intro { color: #999; font-style: italic; font-size: 0.88em; padding: 20px 0; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }
</style>
</head>
<body>

<div id="header">
  <h1>Citation Review</h1>
  <select id="paper-select" onchange="loadPaper(+this.value)"></select>
  <button onclick="prevPaper()">← Prev</button>
  <button onclick="nextPaper()">Next →</button>
  <span id="progress"></span>
  <span id="save-status"></span>
</div>

<div id="main">
  <div id="left">
    <div id="left-header">
      <strong id="paper-title">Loading…</strong><br>
      <span id="paper-meta"></span>
    </div>
    <div id="intro-container">
      <div id="intro-text"></div>
    </div>
    <div id="sel-panel">
      <button class="cancel-btn" onclick="clearSelection()">✕ Cancel</button>
      <div id="sel-text"></div>
      <div id="sel-hint">← Click a reference on the right to assign this citation</div>
    </div>
  </div>

  <div id="right">
    <div id="right-header">
      <h2 id="ref-count">References</h2>
      <input id="ref-search" type="text" placeholder="Search references…" oninput="filterRefs(this.value)">
    </div>
    <div id="ref-list"></div>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────

let papers = [];         // [{paper_id, title, cited_refs, total_refs}, ...]
let currentPid = null;
let currentData = null;  // full paper object from API
let pendingSelection = null;  // {text, start, end} — highlighted text

// ── Init ───────────────────────────────────────────────────────────────────

async function init() {
  const res = await fetch("/api/papers");
  papers = await res.json();

  const sel = document.getElementById("paper-select");
  papers.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p.paper_id;
    opt.textContent = `${String(p.paper_id).padStart(3,'0')}. ${p.title.slice(0,60)} (${p.cited_refs}/${p.total_refs})`;
    sel.appendChild(opt);
  });

  if (papers.length) loadPaper(papers[0].paper_id);
}

// ── Paper loading ──────────────────────────────────────────────────────────

async function loadPaper(pid) {
  clearSelection();
  currentPid = pid;
  document.getElementById("paper-select").value = pid;
  updateProgress();

  const res = await fetch(`/api/paper/${pid}`);
  currentData = await res.json();

  document.getElementById("paper-title").textContent = currentData.title;
  document.getElementById("paper-meta").textContent =
    `Nativeness: ${currentData.nativeness}  |  Quartile: ${currentData.quartile}`;

  renderIntro();
  renderRefs();
}

function updateProgress() {
  const idx = papers.findIndex(p => p.paper_id === currentPid);
  document.getElementById("progress").textContent = `${idx + 1} / ${papers.length}`;
}

function prevPaper() {
  const idx = papers.findIndex(p => p.paper_id === currentPid);
  if (idx > 0) loadPaper(papers[idx - 1].paper_id);
}
function nextPaper() {
  const idx = papers.findIndex(p => p.paper_id === currentPid);
  if (idx < papers.length - 1) loadPaper(papers[idx + 1].paper_id);
}

// ── Intro rendering ────────────────────────────────────────────────────────

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderIntro() {
  const intro = currentData.intro || "";
  const refs  = currentData.references || [];

  if (!intro) {
    document.getElementById("intro-text").innerHTML = '<p class="no-intro">No introduction text available.</p>';
    return;
  }

  // Collect all matched citation spans: {start, end, refIndex, citText}
  const spans = [];
  refs.forEach(ref => {
    (ref.introduction_citation_contexts || []).forEach(ctx => {
      const cit = ctx.citation_text || "";
      if (!cit) return;
      let pos = 0;
      while (true) {
        const idx = intro.indexOf(cit, pos);
        if (idx === -1) break;
        spans.push({ start: idx, end: idx + cit.length, refIndex: ref.index,
                     citText: cit, refRaw: (ref.reference || "").slice(0, 80) });
        pos = idx + 1;
      }
    });
  });

  // Sort and de-overlap
  spans.sort((a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start));
  const merged = [];
  let lastEnd = -1;
  spans.forEach(s => {
    if (s.start >= lastEnd) { merged.push(s); lastEnd = s.end; }
  });

  // Build HTML
  let html = "";
  let pos = 0;
  merged.forEach(s => {
    if (s.start > pos) html += escHtml(intro.slice(pos, s.start));
    const tip = `Ref #${s.refIndex}: ${s.refRaw}…`;
    html += `<mark class="cited" data-ref="${s.refIndex}" data-cit="${escHtml(s.citText)}" data-tip="${escHtml(tip)}" onclick="scrollToRef(${s.refIndex})">${escHtml(s.citText)}</mark>`;
    pos = s.end;
  });
  if (pos < intro.length) html += escHtml(intro.slice(pos));

  document.getElementById("intro-text").innerHTML = html.replace(/\n/g, "<br>\n");
}

// ── Reference list rendering ───────────────────────────────────────────────

function renderRefs(filter = "", suggestedIndices = []) {
  const refs = (currentData && currentData.references) || [];
  const cited = refs.filter(r => r.cited_in_introduction).length;
  document.getElementById("ref-count").textContent =
    `References — ${cited} cited / ${refs.length} total`;

  const fl = filter.toLowerCase();
  const list = document.getElementById("ref-list");
  list.innerHTML = "";

  refs.forEach(ref => {
    const raw = ref.reference || "";
    if (fl && !raw.toLowerCase().includes(fl)) return;

    const isCited = ref.cited_in_introduction;
    const isSugg  = suggestedIndices.includes(ref.index);
    const item    = document.createElement("div");
    item.className = "ref-item" + (isCited ? " cited-item" : "") + (isSugg ? " suggested" : "");
    item.id = `ref-${ref.index}`;

    let badge = isCited
      ? `<span class="badge b-cited">✓ cited</span>`
      : `<span class="badge b-none">not cited</span>`;
    if (isSugg && pendingSelection)
      badge += `<span class="badge b-match">best match</span>`;

    // Existing citation contexts
    let ctxHtml = "";
    (ref.introduction_citation_contexts || []).forEach(ctx => {
      const cit = escHtml(ctx.citation_text || "");
      ctxHtml += `
        <div class="cit-ctx">
          <em>${cit}</em>
          <button class="remove-btn" onclick="removeCitation(event,${ref.index},'${escHtml(ctx.citation_text || '')}')">✕ remove</button>
        </div>`;
    });

    // Assign button (shown when there's a pending selection)
    const assignBtn = pendingSelection
      ? `<button class="assign-btn" onclick="assignCitation(event, ${ref.index})">Assign →</button>`
      : "";

    item.innerHTML = `
      <div class="ref-top">
        <span class="ref-num">#${ref.index}</span>
        <span class="ref-raw">${badge} ${escHtml(raw)}</span>
      </div>
      ${ctxHtml}
      ${assignBtn}`;

    item.addEventListener("click", e => {
      if (e.target.classList.contains("assign-btn") ||
          e.target.classList.contains("remove-btn")) return;
      if (pendingSelection) assignCitation(e, ref.index);
    });
    list.appendChild(item);
  });
}

function filterRefs(q) {
  renderRefs(q, []);
}

function scrollToRef(refIndex) {
  const el = document.getElementById(`ref-${refIndex}`);
  if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ── Text selection → citation panel ───────────────────────────────────────

document.addEventListener("mouseup", async (e) => {
  // Only care about selections inside the intro text
  const introEl = document.getElementById("intro-text");
  if (!introEl.contains(e.target)) return;

  const sel = window.getSelection();
  const text = sel ? sel.toString().trim() : "";
  if (!text || text.length < 3) return;

  pendingSelection = { text };

  document.getElementById("sel-text").textContent = `Selected: "${text}"`;
  document.getElementById("sel-panel").classList.add("active");

  // Highlight the selection as pending
  try {
    const range = sel.getRangeAt(0);
    const mark = document.createElement("mark");
    mark.className = "pending";
    mark.id = "pending-mark";
    range.surroundContents(mark);
    sel.removeAllRanges();
  } catch(ex) { /* selection spans multiple elements — ignore visual highlight */ }

  // Fetch suggestions
  const res = await fetch(`/api/paper/${currentPid}/suggest?text=${encodeURIComponent(text)}`);
  const suggestions = await res.json();
  const suggIndices = suggestions.map(s => s.index);

  renderRefs("", suggIndices);

  // Scroll first suggestion into view
  if (suggIndices.length) {
    setTimeout(() => scrollToRef(suggIndices[0]), 100);
  }
});

function clearSelection() {
  pendingSelection = null;
  document.getElementById("sel-panel").classList.remove("active");
  // Remove pending mark
  const mark = document.getElementById("pending-mark");
  if (mark) {
    const parent = mark.parentNode;
    while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
    parent.removeChild(mark);
  }
  renderRefs();
}

// ── Save / Remove ──────────────────────────────────────────────────────────

async function assignCitation(e, refIndex) {
  e.stopPropagation();
  if (!pendingSelection) return;

  const citText = pendingSelection.text;
  showStatus("Saving…");

  const res = await fetch(`/api/paper/${currentPid}/citation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ref_index: refIndex, citation_text: citText, context: "" }),
  });
  const data = await res.json();

  if (data.status === "ok" || data.status === "already_exists") {
    showStatus("✓ Saved");
    // Update local state
    const ref = currentData.references.find(r => r.index === refIndex);
    if (ref && data.status === "ok") {
      if (!ref.introduction_citation_contexts) ref.introduction_citation_contexts = [];
      ref.introduction_citation_contexts.push({ citation_text: citText, context: "" });
      ref.cited_in_introduction = true;
    }
    clearSelection();
    renderIntro();
    renderRefs();
    // Update paper list stats
    const pm = papers.find(p => p.paper_id === currentPid);
    if (pm && data.status === "ok") pm.cited_refs++;
    updateSelectOption();
  } else {
    showStatus("✗ Error");
  }
}

async function removeCitation(e, refIndex, citText) {
  e.stopPropagation();
  if (!confirm(`Remove citation "${citText}" from reference #${refIndex}?`)) return;

  showStatus("Removing…");
  const res = await fetch(`/api/paper/${currentPid}/citation`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ref_index: refIndex, citation_text: citText }),
  });
  const data = await res.json();
  if (data.status === "ok") {
    showStatus("✓ Removed");
    const ref = currentData.references.find(r => r.index === refIndex);
    if (ref) {
      ref.introduction_citation_contexts = (ref.introduction_citation_contexts || [])
        .filter(c => c.citation_text !== citText);
      ref.cited_in_introduction = ref.introduction_citation_contexts.length > 0;
    }
    renderIntro();
    renderRefs();
    const pm = papers.find(p => p.paper_id === currentPid);
    if (pm && !currentData.references.find(r => r.index === refIndex)?.cited_in_introduction)
      pm.cited_refs = Math.max(0, pm.cited_refs - 1);
    updateSelectOption();
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────

function showStatus(msg) {
  const el = document.getElementById("save-status");
  el.textContent = msg;
  setTimeout(() => { el.textContent = ""; }, 2500);
}

function updateSelectOption() {
  const sel = document.getElementById("paper-select");
  const pm = papers.find(p => p.paper_id === currentPid);
  if (!pm) return;
  const opt = sel.querySelector(`option[value="${currentPid}"]`);
  if (opt) opt.textContent =
    `${String(pm.paper_id).padStart(3,'0')}. ${pm.title.slice(0,60)} (${pm.cited_refs}/${pm.total_refs})`;
}

// ── Keyboard shortcuts ─────────────────────────────────────────────────────

document.addEventListener("keydown", e => {
  if (e.key === "Escape") clearSelection();
  if (e.key === "ArrowLeft"  && e.altKey) prevPaper();
  if (e.key === "ArrowRight" && e.altKey) nextPaper();
});

// ── Start ──────────────────────────────────────────────────────────────────

init();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--refs-dir", default=str(REFS_DIR))
    parser.add_argument("--excel", default=str(EXCEL_PATH))
    args = parser.parse_args()

    REFS_DIR = Path(args.refs_dir)
    EXCEL_PATH = Path(args.excel)

    print(f"Starting Citation Review App on http://localhost:{args.port}")
    print(f"  refs-dir : {REFS_DIR}")
    print(f"  excel    : {EXCEL_PATH}")
    print(f"  Alt+← / Alt+→  to navigate papers")
    print(f"  Esc to cancel a selection")
    app.run(port=args.port, debug=False)
