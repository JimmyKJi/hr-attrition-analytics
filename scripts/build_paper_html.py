"""Render paper/writeup.md to a single self-contained paper/writeup.html.

No pandoc / LaTeX required. The output is one portable file with the figures
embedded as base64 and the maths rendered client-side by MathJax (CDN), so a
non-technical reviewer can open it in any browser and "Print → Save as PDF".

Why hand-roll instead of `markdown` alone: python-markdown mangles LaTeX inside
`$...$` (underscores become <em>, backslashes get eaten). So code spans/blocks
and maths are *extracted* to placeholders before conversion and restored after —
code as escaped <pre>/<code>, maths as raw `$...$` for MathJax to typeset.
"""
from __future__ import annotations

import base64
import html as _html
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "paper" / "writeup.md"
OUT = ROOT / "paper" / "writeup.html"

CSS = """
:root { --ink:#1a1a1a; --muted:#666; --rule:#e2e2e2; --link:#1f5fae; }
* { box-sizing: border-box; }
body { color: var(--ink); background:#fff;
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  max-width: 50rem; margin: 2.5rem auto; padding: 0 1.25rem; }
h1 { font-size: 1.85rem; line-height: 1.2; margin: 0 0 .6rem; }
h2 { font-size: 1.35rem; margin: 2.2rem 0 .6rem; padding-bottom: .25rem;
  border-bottom: 1px solid var(--rule); }
h3 { font-size: 1.08rem; margin: 1.6rem 0 .4rem; }
p, li { margin: .55rem 0; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
img { display: block; max-width: 100%; height: auto; margin: 1.1rem auto;
  border: 1px solid var(--rule); border-radius: 4px; }
blockquote { margin: 1rem 0; padding: .4rem 1rem; color: var(--muted);
  border-left: 3px solid var(--rule); background: #fafafa; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .88em; background: #f4f4f4; padding: .1em .35em; border-radius: 3px; }
pre { background: #f7f7f7; border: 1px solid var(--rule); border-radius: 6px;
  padding: .85rem 1rem; overflow-x: auto; line-height: 1.45; }
pre code { background: none; padding: 0; font-size: .82rem; }
table { border-collapse: collapse; width: 100%; margin: 1.1rem 0; font-size: .93rem; }
th, td { border: 1px solid var(--rule); padding: .45rem .6rem; text-align: left;
  vertical-align: top; }
th { background: #f4f6f8; }
hr { border: none; border-top: 1px solid var(--rule); margin: 2rem 0; }
.synthetic-note { font-size: .9rem; }
@media print { body { margin: 0; max-width: none; } a { color: var(--ink); }
  h2 { page-break-after: avoid; } img, pre, table { page-break-inside: avoid; } }
"""

MATHJAX = """
<script>
window.MathJax = {
  tex: { inlineMath: [['$','$']], displayMath: [['$$','$$']] },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }
};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
{mathjax}
</head>
<body>
{body}
<hr>
<p class="synthetic-note"><em>Rendered from <code>paper/writeup.md</code> by
<code>scripts/build_paper_html.py</code> — figures embedded, maths via MathJax.
Synthetic data throughout: every quantity is illustrative of method, not an
estimate of a real workforce.</em></p>
</body>
</html>
"""


def _protect(text: str) -> tuple[str, dict[str, str]]:
    """Stash fenced code, inline code, then display/inline maths behind tokens."""
    store: dict[str, str] = {}

    def stash(kind: str, s: str) -> str:
        key = f"ZZ{kind}{len(store)}ZZ"
        store[key] = s
        return key

    text = re.sub(r"```.*?```",
                  lambda m: "\n\n" + stash("CB", m.group(0)) + "\n\n",
                  text, flags=re.S)
    text = re.sub(r"`[^`]*`", lambda m: stash("CI", m.group(0)), text)
    text = re.sub(r"\$\$.*?\$\$", lambda m: stash("MD", m.group(0)), text, flags=re.S)
    text = re.sub(r"\$[^$\n]+?\$", lambda m: stash("MI", m.group(0)), text)
    return text, store


def _restore(html_str: str, store: dict[str, str]) -> str:
    for key, raw in store.items():
        if key.startswith("ZZCB"):
            inner = re.sub(r"^```[^\n]*\n?", "", raw)
            inner = re.sub(r"\n?```$", "", inner)
            repl = f"<pre><code>{_html.escape(inner)}</code></pre>"
            html_str = html_str.replace(f"<p>{key}</p>", repl)
        elif key.startswith("ZZCI"):
            repl = f"<code>{_html.escape(raw.strip('`'))}</code>"
        else:  # maths — reinsert raw for MathJax
            repl = raw
        html_str = html_str.replace(key, repl)
    return html_str


def _embed_images(html_str: str) -> str:
    def repl(m: re.Match) -> str:
        name = Path(m.group(1)).name
        p = ROOT / "figures" / name
        if not p.exists():
            return m.group(0)
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'src="data:image/png;base64,{b64}"'
    return re.sub(r'src="([^"]+)"', repl, html_str)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    m = re.search(r"^#\s+(.+)$", text, flags=re.M)
    title = m.group(1).strip() if m else "Writeup"

    protected, store = _protect(text)
    body = markdown.markdown(protected, extensions=["tables", "sane_lists"])
    body = _restore(body, store)
    body = _embed_images(body)

    OUT.write_text(PAGE.format(title=_html.escape(title), css=CSS,
                               mathjax=MATHJAX, body=body), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    n_img = body.count("data:image/png;base64,")
    print(f"wrote {OUT}  ({size_kb:.0f} KB, {n_img} figures embedded)")
    print("Open in a browser; File → Print → Save as PDF for a shareable PDF.")


if __name__ == "__main__":
    main()
