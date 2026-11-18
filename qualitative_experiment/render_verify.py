"""Roundtrip check for rendered intro HTML (shared by both experiment builders).

Confirms that the text shown to the reader is exactly the source intro text:
strips the legend box and the move chips (renderer furniture), removes all
tags, unescapes, and compares whitespace-collapsed strings. Newlines were
flattened to spaces at render time, so whitespace collapsing is the right
equivalence.
"""

from __future__ import annotations

import html as html_lib
import re
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    """Collect text, skipping the legend <div> and the bold chip <span>s."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        style = dict(attrs).get("style", "")
        skip_root = ("border:1px solid #999" in style) or ("font-weight:bold" in style)
        if self._skip_depth or skip_root:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)

    def handle_entityref(self, name):
        if not self._skip_depth:
            self.parts.append(html_lib.unescape(f"&{name};"))

    def handle_charref(self, name):
        if not self._skip_depth:
            self.parts.append(html_lib.unescape(f"&#{name};"))


def stripped_text(rendered_html: str) -> str:
    parser = _TextExtractor()
    parser.feed(rendered_html)
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def verify_roundtrip(rendered_html: str, source_text: str, label: str) -> None:
    got = stripped_text(rendered_html)
    want = re.sub(r"\s+", " ", source_text).strip()
    if got != want:
        for i, (a, b) in enumerate(zip(got, want)):
            if a != b:
                context = f"...{got[max(0, i - 40) : i + 40]!r} vs {want[max(0, i - 40) : i + 40]!r}"
                break
        else:
            context = f"length {len(got)} vs {len(want)}"
        raise SystemExit(f"roundtrip FAIL for {label}: {context}")
