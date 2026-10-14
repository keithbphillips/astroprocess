"""Write <output_root>/index.html: every processed picture with its key numbers, for quick review."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Astro Gallery</title>
<style>
:root {{ --bg:#0d0f14; --card:#161a22; --fg:#e6e8ee; --muted:#9aa3b2; --accent:#7fb2ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,sans-serif; padding:16px; }}
h1 {{ font-size:20px; margin:4px 0 16px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:16px; }}
.card {{ background:var(--card); border-radius:10px; overflow:hidden; }}
.card img {{ width:100%; height:300px; object-fit:cover; display:block; background:#000; }}
.meta {{ padding:10px 12px 12px; }}
.meta h2 {{ font-size:16px; margin:0 0 4px; }}
.meta p {{ margin:2px 0; color:var(--muted); font-size:13px; }}
.meta a {{ color:var(--accent); text-decoration:none; margin-right:10px; font-size:13px; }}
</style></head><body>
<h1>Astro gallery <span style="color:var(--muted);font-weight:400">({n} images)</span></h1>
<div class="grid">{cards}</div></body></html>
"""

CARD = """<div class="card"><a href="{full}"><img loading="lazy" src="{web}" alt="{title}"></a>
<div class="meta"><h2>{title}</h2><p>{line1}</p><p>{line2}</p><p>{links}</p></div></div>"""


def build(root: Path) -> Path:
    cards = []
    for report in sorted(root.glob("*/*_report.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        name = report.name.removesuffix("_report.json")
        folder = report.parent
        try:
            meta = json.loads(report.read_text())
        except json.JSONDecodeError:
            continue
        web = folder / f"{name}_web.jpg"
        if not web.exists():
            continue
        rel = lambda p: os.path.relpath(p, root)  # noqa: E731
        st = meta.get("stack", {})
        line1 = " · ".join(x for x in [
            meta.get("camera") or meta.get("instrument", ""), meta.get("filter", ""), meta.get("type", ""),
        ] if x)
        used = st.get("frames_stacked")
        line2 = (f"{used}/{st.get('lights')} frames · {st.get('integration_used_min', '?')} min · "
                 f"colour: {meta.get('color', '?')}") if used else f"colour: {meta.get('color', '?')}"
        links = [f'<a href="{html.escape(rel(folder / (name + ".jpg")))}">full JPEG</a>']
        for suffix, label in (("_hoo_web.jpg", "HOO"), ("_vs_seestar.jpg", "vs Seestar"), (".tif", "TIFF")):
            p = folder / f"{name}{suffix}"
            if p.exists():
                links.append(f'<a href="{html.escape(rel(p))}">{label}</a>')
        cards.append(CARD.format(full=html.escape(rel(folder / (name + ".jpg"))), web=html.escape(rel(web)),
                                 title=html.escape(meta.get("target", name)), line1=html.escape(line1),
                                 line2=html.escape(line2), links=" ".join(links)))
    out = root / "index.html"
    out.write_text(PAGE.format(n=len(cards), cards="\n".join(cards)))
    return out
