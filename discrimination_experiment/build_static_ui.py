#!/usr/bin/env python3
"""Build the FREE static annotation app for the discrimination experiment.

Fallback to the Argilla route (HF now paywalls Docker/Gradio Spaces behind
PRO; static Spaces stay free): generates a single self-contained index.html
— the same rendered intros, questions and guidelines as the other builders —
and (with --push) uploads it to a PRIVATE static HF Space.

App behaviour:
- one list of 120 items, order strictly enforced (next item unlocks only after
  the current one is submitted; submitted answers are view-only); the annotator
  works at their own pace and can stop/continue anytime.
- Q1 provenance (5-point), Q2 basis (multi-select, >=1), optional comment.
- Every answer autosaves to localStorage; the annotator downloads a responses
  JSON file every ~30 items (and at the end) and sends it back.

Blinding: the page embeds ONLY item_code / position / label_condition + the
rendered HTML. intro_type / mode / intro_id never leave allocation.json.

Usage:
    python discrimination_experiment/build_static_ui.py                # build only
    python discrimination_experiment/build_static_ui.py --push        # + upload to HF
"""

import argparse
import html as html_lib
import json
import re
from datetime import datetime
from pathlib import Path

from build_argilla_datasets import (  # same renderer + content, single source of truth
    build_guidelines,
    build_record_payloads,
    load_intro_index,
    load_items,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = Path(__file__).resolve().parent

SPACE_ID = "your-hf-username/cars-intro-study"  # override with --space-id
APP_VERSION = "2.0"


# Inline styles (not a <style> block) so the markup survives any sanitizer, and
# an explicit light background + dark text so it reads in dark-themed viewers too.
GSTYLE = {
    "wrap": (
        "background:#ffffff;color:#1c1c1c;font-family:system-ui,-apple-system,sans-serif;"
        "line-height:1.65;padding:26px 30px;border-radius:10px;max-width:840px;"
    ),
    "h2": "margin:0 0 14px;font-size:1.45em;color:#1c1c1c;",
    "h3": (
        "margin:36px 0 14px;font-size:1.15em;color:#3d4a7a;"
        "border-bottom:2px solid #dfe3f0;padding-bottom:6px;"
    ),
    "p": "margin:0 0 14px;color:#1c1c1c;",
    "list": "margin:0 0 18px;padding-left:24px;color:#1c1c1c;",
    "li": "margin:0 0 12px;",
    "sublist": "margin:8px 0 2px;padding-left:20px;",
    "subli": "margin:0 0 7px;",
    "callout": (
        "background:#eef1fa;color:#1c1c1c;border-left:4px solid #4e5d94;"
        "padding:14px 18px;border-radius:0 8px 8px 0;margin:10px 0 16px;"
    ),
}


def guidelines_html(medium: str = "static", wrap: bool = True) -> str:
    """Convert the (fully self-authored, simple) guidelines markdown to styled,
    theme-proof HTML: spaced sections, nested lists, callout for the summary."""
    md = build_guidelines(medium)
    out = []
    for block in re.split(r"\n\n+", md):
        block = block.strip("\n")
        if not block.strip():
            continue
        if block.startswith("## "):
            out.append(f'<h3 style="{GSTYLE["h3"]}">{html_lib.escape(block[3:].strip())}</h3>')
        elif block.startswith("# "):
            out.append(f'<h2 style="{GSTYLE["h2"]}">{html_lib.escape(block[2:].strip())}</h2>')
        elif re.match(r"^(\d+\.|-)\s", block):
            out.append(_list_html(block.split("\n")))
        elif block.startswith("In short"):
            out.append(f'<div style="{GSTYLE["callout"]}"><b>In short:</b>{_inline_md(block[len("In short:"):])}</div>')
        else:
            out.append(f'<p style="{GSTYLE["p"]}">{_inline_md(" ".join(ln.strip() for ln in block.split(chr(10))))}</p>')
    body = "\n".join(out)
    return f'<div style="{GSTYLE["wrap"]}">{body}</div>' if wrap else body


def _list_html(lines: list[str]) -> str:
    """Top-level numbered/bulleted list with indented '- ' lines as nested items."""
    tag = "ol" if lines[0][0].isdigit() else "ul"
    items: list[list] = []  # [text, [subitems]]
    for ln in lines:
        if re.match(r"^\s+-\s", ln):
            items[-1][1].append(re.sub(r"^\s+-\s+", "", ln))
        else:
            items.append([re.sub(r"^(\d+\.|-)\s+", "", ln), []])
    parts = []
    for text, subs in items:
        li = _inline_md(text)
        if subs:
            sub_lis = "".join(f'<li style="{GSTYLE["subli"]}">{_inline_md(s)}</li>' for s in subs)
            li += f'<ul style="{GSTYLE["sublist"]}">{sub_lis}</ul>'
        parts.append(f'<li style="{GSTYLE["li"]}">{li}</li>')
    return f'<{tag} style="{GSTYLE["list"]}">{"".join(parts)}</{tag}>'


def _inline_md(text: str) -> str:
    text = html_lib.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


APP_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Introduction Study</title>
<style>
  :root { --accent:#4e5d94; --bg:#f7f7f5; --card:#ffffff; --ink:#1c1c1c; --muted:#666; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:Georgia,'Times New Roman',serif; line-height:1.55; }
  header { position:sticky; top:0; z-index:5; background:var(--accent); color:#fff;
           padding:10px 18px; display:flex; align-items:center; gap:14px;
           font-family:system-ui,sans-serif; }
  header .title { font-weight:600; }
  header .spacer { flex:1; }
  header button { background:#fff2; color:#fff; border:1px solid #fff6; border-radius:6px;
                  padding:5px 12px; cursor:pointer; font-size:0.85rem; }
  header button:hover { background:#fff3; }
  main { max-width:860px; margin:0 auto; padding:24px 18px 80px; }
  .card { background:var(--card); border:1px solid #ddd; border-radius:10px;
          padding:20px 24px; margin-bottom:18px; }
  h2,h3 { font-family:system-ui,sans-serif; }
  .btn { background:var(--accent); color:#fff; border:none; border-radius:6px;
         padding:8px 16px; cursor:pointer; font-size:0.95rem;
         font-family:system-ui,sans-serif; }
  .btn:disabled { background:#aaa; cursor:default; }
  .btn.secondary { background:#e8e8e4; color:var(--ink); }
  .progressbar { height:6px; background:#e3e3df; border-radius:3px; margin:10px 0 18px; }
  .progressbar > div { height:100%; background:var(--accent); border-radius:3px;
                       transition:width .3s; }
  .itemmeta { font-family:system-ui,sans-serif; color:var(--muted); font-size:0.9rem;
              margin-bottom:10px; display:flex; gap:16px; }
  .intro-box { font-size:1.02rem; }
  fieldset { border:1px solid #ddd; border-radius:8px; margin:16px 0; padding:12px 16px;
             font-family:system-ui,sans-serif; }
  legend { font-weight:600; padding:0 6px; }
  label.opt { display:block; padding:6px 4px; cursor:pointer; border-radius:6px; }
  label.opt:hover { background:#f2f2ee; }
  textarea { width:100%; min-height:70px; font-family:inherit; font-size:0.95rem;
             border:1px solid #ccc; border-radius:6px; padding:8px; }
  .locked { opacity:0.75; pointer-events:none; }
  .navrow { display:flex; gap:12px; align-items:center; margin-top:14px;
            font-family:system-ui,sans-serif; }
  .warn { color:#b00020; font-family:system-ui,sans-serif; font-size:0.9rem; }
  .done-note { background:#eef4ee; border:1px solid #cfe0cf; border-radius:8px;
               padding:12px 16px; font-family:system-ui,sans-serif; }
  .guidelines h2 { margin-top:0; }
  .pill { display:inline-block; font-size:0.75rem; padding:1px 8px; border-radius:10px;
          background:#e8e8e4; font-family:system-ui,sans-serif; }
  input.namebox { font-size:1rem; padding:6px 10px; border:1px solid #ccc;
                  border-radius:6px; font-family:system-ui,sans-serif; }
</style>
</head>
<body>
<header>
  <span class="title">Introduction Study</span>
  <span id="hdr-context"></span>
  <span class="spacer"></span>
  <button onclick="showHome()">Overview</button>
  <button onclick="showGuidelines()">Guidelines</button>
  <button onclick="exportAll()">Save progress file</button>
</header>
<main id="main"></main>
<script>
"use strict";
const DATA = __DATA_JSON__;
const ITEMS = DATA.items;
const APP_VERSION = "__APP_VERSION__";
const STORE_KEY = "cars_intro_study_v1";

function loadState() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {annotator:"", responses:{}}; }
  catch (e) { return {annotator:"", responses:{}}; }
}
function saveState() { localStorage.setItem(STORE_KEY, JSON.stringify(state)); }
let state = loadState();

const $main = document.getElementById("main");
const $ctx = document.getElementById("hdr-context");

function answered(code) { return state.responses[code] && state.responses[code].provenance; }
function nAnswered() { return ITEMS.filter(it => answered(it.item_code)).length; }
function firstOpenIndex() {
  for (let i = 0; i < ITEMS.length; i++) if (!answered(ITEMS[i].item_code)) return i;
  return ITEMS.length;
}

/* ------------------------------ home ------------------------------ */
function showHome() {
  $ctx.textContent = "";
  const done = nAnswered(), total = ITEMS.length;
  const finished = done === total;
  $main.innerHTML = `
    <div class="card">
      <h2>Welcome</h2>
      <p style="font-family:system-ui,sans-serif">You will read introductions to research
      articles and judge, for each one, whether it was written by the article's human
      authors or generated by a large language model (LLM). Please read the
      <a href="#" onclick="showGuidelines();return false;">guidelines</a> before starting.
      Work at your own pace — stop and continue whenever you like, your progress is
      saved in this browser.</p>
      <p style="font-family:system-ui,sans-serif">Your name:
        <input class="namebox" id="annotator" value="${escapeAttr(state.annotator)}"
               placeholder="your name" onchange="state.annotator=this.value;saveState();">
      </p>
      <div class="progressbar"><div style="width:${Math.round(100*done/total)}%"></div></div>
      <p style="font-family:system-ui,sans-serif">${done} / ${total} answered${finished ? " — all done, thank you! Please download the final file below and send it back." : ""}</p>
      <p>
        ${finished
          ? `<button class="btn" onclick="exportAll()">Download final responses file</button>`
          : `<button class="btn" onclick="start()">${done ? "Continue" : "Start"} annotating</button>
             <button class="btn secondary" onclick="exportAll()">Download progress file</button>`}
      </p>
      <p style="font-family:system-ui,sans-serif;color:var(--muted);font-size:0.9rem">
      Answers save automatically in this browser. Please click <b>Save progress file</b>
      every 30 items or so (the app will also remind you) and send the downloaded file to
      the research team — it is your backup. Please use the same browser and device
      throughout.</p>
    </div>`;
}

/* --------------------------- guidelines --------------------------- */
function showGuidelines() {
  $ctx.textContent = "Guidelines";
  $main.innerHTML = `<div class="card guidelines">__GUIDELINES_HTML__
    <p><button class="btn" onclick="showHome()">Back to overview</button></p></div>`;
  window.scrollTo(0, 0);
}

/* ------------------------------ items ------------------------------ */
let currentIndex = null;

function start() { currentIndex = firstOpenIndex(); renderItem(); }

function renderItem() {
  if (currentIndex >= ITEMS.length) { allDone(); return; }
  const it = ITEMS[currentIndex];
  const resp = state.responses[it.item_code] || {};
  const locked = !!resp.provenance;
  if (!locked && !resp.opened_at) {
    state.responses[it.item_code] = Object.assign({}, resp, {
      position: it.position, label_condition: it.label_condition,
      opened_at: new Date().toISOString()
    });
    saveState();
  }
  $ctx.textContent = `Item ${currentIndex + 1} of ${ITEMS.length}`;
  const pct = Math.round(100 * nAnswered() / ITEMS.length);
  $main.innerHTML = `
    <div class="progressbar"><div style="width:${pct}%"></div></div>
    <div class="card">
      <div class="itemmeta">
        <span>Item ${it.item_code.slice(-3)}</span>
        <span class="pill">${it.label_condition === "labeled" ? "with move annotations" : "plain text"}</span>
      </div>
      <div class="intro-box">${it.html}</div>
    </div>
    <div class="card ${locked ? "locked" : ""}" id="qcard">
      <fieldset>
        <legend>Q1 — Who wrote this introduction?</legend>
        ${["1|Sure it's human","2|Most likely human","3|Can't decide","4|Most likely AI","5|Sure it's AI"]
          .map(o => { const [v, t] = o.split("|");
            return `<label class="opt"><input type="radio" name="prov" value="${v}"
              ${String(resp.provenance) === v ? "checked" : ""}> ${v} — ${t}</label>`; }).join("")}
      </fieldset>
      <fieldset>
        <legend>Q2 — What did you base your decision on? (tick all that apply)</legend>
        ${[["argumentation_structure","Argumentation / Swales CARS structure"],
           ["writing_style","Writing style"],["guess","Just a guess"]]
          .map(([v, t]) => `<label class="opt"><input type="checkbox" name="basis" value="${v}"
            ${(resp.basis || []).includes(v) ? "checked" : ""}> ${t}</label>`).join("")}
      </fieldset>
      <fieldset>
        <legend>Optional comment</legend>
        <textarea id="comment" placeholder="anything that struck you…">${escapeHtml(resp.comment || "")}</textarea>
      </fieldset>
    </div>
    <div class="navrow">
      ${currentIndex > 0 ? `<button class="btn secondary" onclick="nav(-1)">← Previous (view only)</button>` : ""}
      ${locked
        ? `<button class="btn" onclick="nav(1)">Next →</button>
           <span style="font-family:system-ui,sans-serif;color:var(--muted)">answer submitted</span>`
        : `<button class="btn" onclick="submitItem()">Submit answer</button>`}
      <span class="warn" id="warn"></span>
    </div>`;
  window.scrollTo(0, 0);
}

function nav(d) {
  const target = currentIndex + d;
  if (target < 0) return;
  if (target > firstOpenIndex()) return; // never skip ahead
  currentIndex = target;
  renderItem();
}

function submitItem() {
  const it = ITEMS[currentIndex];
  const prov = document.querySelector("input[name=prov]:checked");
  const basis = [...document.querySelectorAll("input[name=basis]:checked")].map(b => b.value);
  const warn = document.getElementById("warn");
  if (!prov) { warn.textContent = "Please answer Q1."; return; }
  if (basis.length === 0) { warn.textContent = "Please tick at least one option in Q2."; return; }
  const comment = document.getElementById("comment").value.trim() || null;
  state.responses[it.item_code] = Object.assign({}, state.responses[it.item_code], {
    position: it.position, label_condition: it.label_condition,
    provenance: parseInt(prov.value, 10), basis, comment,
    submitted_at: new Date().toISOString()
  });
  saveState();
  const done = nAnswered();
  currentIndex += 1;
  if (done > 0 && done % 30 === 0 && done < ITEMS.length) {
    backupReminder(done);
  } else {
    renderItem();
  }
}

function backupReminder(done) {
  $ctx.textContent = `${done} of ${ITEMS.length} done`;
  $main.innerHTML = `<div class="card">
      <div class="done-note"><b>${done} items done — nice progress!</b><br>
      Please download a progress file now and send it to the research team (it is your
      backup), then continue whenever you like.</div>
      <p><button class="btn" onclick="exportAll()">Download progress file</button>
         <button class="btn secondary" onclick="renderItem()">Continue →</button></p>
    </div>`;
  window.scrollTo(0, 0);
}

function allDone() {
  $ctx.textContent = "All items complete";
  $main.innerHTML = `<div class="card">
      <div class="done-note"><b>All ${ITEMS.length} items complete — thank you!</b><br>
      Please download the final responses file and send it to the research team.</div>
      <p><button class="btn" onclick="exportAll()">Download final responses file</button>
         <button class="btn secondary" onclick="showHome()">Back to overview</button></p>
    </div>`;
}

/* ------------------------------ export ----------------------------- */
function exportAll() {
  const responses = {};
  for (const it of ITEMS) if (state.responses[it.item_code]) responses[it.item_code] = state.responses[it.item_code];
  const n = Object.values(responses).filter(r => r.provenance).length;
  const payload = {meta: {exported: new Date().toISOString(), annotator: state.annotator || null,
                          app_version: APP_VERSION, n_answered: n}, responses};
  const blob = new Blob([JSON.stringify(payload, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `responses_${n}of${ITEMS.length}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ------------------------------ utils ------------------------------ */
function escapeHtml(t) { return t.replace(/&/g,"&amp;").replace(/</g,"&lt;"); }
function escapeAttr(t) { return (t || "").replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/</g,"&lt;"); }

showHome();
</script>
</body>
</html>
"""


def build_page() -> str:
    items = load_items(EXP_DIR / "allocation.json")
    intro_index = load_intro_index(REPO_ROOT / "data" / "master_dataset.json")
    records = build_record_payloads(items, intro_index)

    items_data = [
        {
            "item_code": rec["item_code"],
            "position": rec["metadata"]["position"],
            "label_condition": rec["metadata"]["label_condition"],
            "html": rec["html"],
        }
        for rec in records
    ]
    data = {"items": items_data, "built": datetime.now().isoformat(timespec="seconds")}

    # blinding audit on the embedded payload
    forbidden = {"intro_type", "mode", "quartile", "nativeness", "intro_id", "paper_key", "block", "session"}
    for rec in items_data:
        leaked = set(rec) & forbidden
        if leaked:
            raise SystemExit(f"blinding violation in {rec['item_code']}: {leaked}")

    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (
        APP_TEMPLATE.replace("__DATA_JSON__", data_json)
        .replace("__GUIDELINES_HTML__", guidelines_html(wrap=False))
        .replace("__APP_VERSION__", APP_VERSION)
    )


def push_space(index_path: Path, space_id: str):
    import os

    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv(REPO_ROOT / ".env")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="static", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(index_path),
        path_in_repo="index.html",
        repo_id=space_id,
        repo_type="space",
        commit_message=f"annotation app v{APP_VERSION}",
    )
    print(f"pushed -> https://huggingface.co/spaces/{space_id} (private)")
    print("grant the annotator access: Space settings -> Collaborators -> add their HF username")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(EXP_DIR / "static_ui" / "index.html"))
    parser.add_argument("--push", action="store_true", help="upload to the private static HF Space")
    parser.add_argument("--space-id", default=SPACE_ID)
    args = parser.parse_args()

    page = build_page()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"built {out} ({len(page)/1e6:.2f} MB)")
    if args.push:
        push_space(out, args.space_id)


if __name__ == "__main__":
    main()
