"""Analysis report: every analysis the user has looked at, as one self-contained HTML file (no scripts, no internet;
charts are inline SVG). Open it in a browser and print to PDF to share."""
import html
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

ORDER = ["데이터", "데이터 점검", "상관분석", "이변량 회귀", "시차 상관", "소프트센서"]
INK, MUTED, BLUE, ORANGE = "#1f1f1f", "#8a8780", "#2a6fdb", "#e0782f"


def md(text):
    """The small markdown subset the summaries use (**bold**, `code`) → safe HTML."""
    t = html.escape(str(text))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def lines_html(lines):
    out, items = [], []
    for ln in lines:
        for part in str(ln).split("\n"):
            if part.startswith("- "):
                items.append(f"<li>{md(part[2:])}</li>")
                continue
            if items:
                out.append("<ul>" + "".join(items) + "</ul>")
                items = []
            if part.strip():
                out.append(f"<p>{md(part)}</p>")
    if items:
        out.append("<ul>" + "".join(items) + "</ul>")
    return "".join(out)


def _scale(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo or 1) * (b - a)


def svg_lines(df, colors=(BLUE, ORANGE), w=720, h=220, points=600):
    """Time-series lines (NaN breaks the line; sparse series drawn as dots), y range and first/last time labelled."""
    d = df.iloc[:: max(1, len(df) // points)] if len(df) > points else df
    vals = d.to_numpy(float)
    if not np.isfinite(vals).any():
        return ""
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    L, R, T, B = 56, w - 10, 10, h - 26
    x = np.linspace(L, R, len(d))
    parts = []
    for (name, s), color in zip(d.items(), colors):
        y = _scale(s.to_numpy(float), lo, hi, B, T)
        ok = np.isfinite(y)
        if ok.mean() < 0.5:  # lab results: dots
            parts += [f'<circle cx="{a:.1f}" cy="{b:.1f}" r="2.5" fill="{color}"/>' for a, b in zip(x[ok], y[ok])]
            continue
        path, pen = [], "M"
        for a, b, good in zip(x, y, ok):
            if good:
                path.append(f"{pen}{a:.1f},{b:.1f}")
                pen = "L"
            else:
                pen = "M"
        parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    legend = " ".join(f'<tspan fill="{c}">■</tspan> {html.escape(str(n))}' for n, c in zip(d.columns, colors))
    t0, t1 = d.index[0], d.index[-1]
    fmt = (lambda t: f"{t:%Y-%m-%d %H:%M}") if isinstance(t0, pd.Timestamp) else str
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" font-size="11" font-family="sans-serif">'
            f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{MUTED}"/>'
            f'<text x="{L - 4}" y="{T + 8}" text-anchor="end" fill="{INK}">{hi:.4g}</text>'
            f'<text x="{L - 4}" y="{B}" text-anchor="end" fill="{INK}">{lo:.4g}</text>'
            f'<text x="{L}" y="{h - 8}" fill="{MUTED}">{fmt(t0)}</text>'
            f'<text x="{R}" y="{h - 8}" text-anchor="end" fill="{MUTED}">{fmt(t1)}</text>'
            f'<text x="{(L + R) / 2}" y="{h - 8}" text-anchor="middle" fill="{INK}">{legend}</text>'
            + "".join(parts) + "</svg>")


def svg_scatter(x, y, fit=None, xlabel="", ylabel="", w=480, h=320, points=800):
    """Scatter of y against x with an optional fitted curve (DataFrame with columns x, 적합)."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if d.empty:
        return ""
    d = d.iloc[:: max(1, len(d) // points)]
    xs = [d["x"]] + ([fit["x"]] if fit is not None else [])
    ys = [d["y"]] + ([fit["적합"]] if fit is not None else [])
    xlo, xhi = min(s.min() for s in xs), max(s.max() for s in xs)
    ylo, yhi = min(s.min() for s in ys), max(s.max() for s in ys)
    L, R, T, B = 56, w - 10, 10, h - 36
    sx = lambda v: _scale(v, xlo, xhi, L, R)  # noqa: E731
    sy = lambda v: _scale(v, ylo, yhi, B, T)  # noqa: E731
    dots = "".join(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="2.2" fill="{BLUE}" fill-opacity="0.45"/>'
                   for a, b in zip(d["x"], d["y"]))
    line = ""
    if fit is not None:
        line = '<path d="' + " ".join(f"{'M' if i == 0 else 'L'}{sx(a):.1f},{sy(b):.1f}" for i, (a, b) in
                                      enumerate(zip(fit["x"], fit["적합"]))) + f'" fill="none" stroke="{INK}" stroke-width="2"/>'
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" role="img" font-size="11" font-family="sans-serif">'
            f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{MUTED}"/><line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="{MUTED}"/>'
            f'<text x="{L}" y="{B + 14}" fill="{INK}">{xlo:.4g}</text><text x="{R}" y="{B + 14}" text-anchor="end" fill="{INK}">{xhi:.4g}</text>'
            f'<text x="{(L + R) / 2}" y="{h - 6}" text-anchor="middle" fill="{INK}">{html.escape(xlabel)}</text>'
            f'<text x="{L - 4}" y="{T + 8}" text-anchor="end" fill="{INK}">{yhi:.4g}</text>'
            f'<text x="{L - 4}" y="{B}" text-anchor="end" fill="{INK}">{ylo:.4g}</text>'
            f'<text x="12" y="{(T + B) / 2}" transform="rotate(-90 12 {(T + B) / 2})" text-anchor="middle" fill="{INK}">'
            f'{html.escape(ylabel)}</text>' + dots + line + "</svg>")


def build(sections, title="공정 데이터 분석 보고서"):
    """sections: {name: {"lines": [...], "table": DataFrame | None, "svg": str, "code": str}} → HTML text."""
    body = []
    for name in [n for n in ORDER if n in sections] + [n for n in sections if n not in ORDER]:
        s = sections[name]
        body.append(f"<section><h2>{html.escape(name)}</h2>{lines_html(s.get('lines', []))}")
        if s.get("svg"):
            body.append(f'<figure>{s["svg"]}</figure>')
        if s.get("code"):
            body.append(f"<pre>{html.escape(s['code'])}</pre>")
        if s.get("table") is not None and len(s["table"]):
            t = s["table"]
            body.append(t.to_html(border=0, float_format=lambda v: f"{v:.4g}", na_rep="-", escape=True,
                                  index=not isinstance(t.index, pd.RangeIndex)))
        body.append("</section>")
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{font-family:"Malgun Gothic","Apple SD Gothic Neo",sans-serif;color:{INK};background:#fff;max-width:860px;margin:24px auto;padding:0 16px;line-height:1.55}}
h1{{font-size:22px;margin-bottom:2px}} h2{{font-size:17px;border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:28px}}
table{{border-collapse:collapse;font-size:12px;margin:8px 0}} th,td{{border-bottom:1px solid #e3e3e3;padding:3px 8px;text-align:left}}
pre{{background:#f5f5f3;padding:8px;font-size:12px;white-space:pre-wrap}} code{{background:#f0f0ee;padding:0 3px}}
.meta{{color:{MUTED};font-size:12px}} figure{{margin:8px 0}} section{{break-inside:avoid-page}}
</style></head><body><h1>{html.escape(title)}</h1>
<p class="meta">작성: {datetime.now():%Y-%m-%d %H:%M} · 이 파일은 인터넷 없이 열립니다. 브라우저의 인쇄 → PDF로 저장해 공유할 수 있습니다.
상관·회귀 결과는 '함께 움직인다'는 뜻이며 원인을 증명하지 않습니다.</p>
{''.join(body)}</body></html>"""


def put(name, lines=(), table=None, svg="", code=""):
    """Record the latest result of one analysis for the report (called where the result is drawn)."""
    st.session_state.setdefault("_report", {})[name] = {"lines": list(lines), "table": table, "svg": svg, "code": code}


def sections():
    return st.session_state.get("_report", {})


if __name__ == "__main__":
    idx = pd.date_range("2026-01-01", periods=50, freq="h")
    s = pd.DataFrame({"실측": np.where(np.arange(50) % 5 == 0, np.arange(50.0), np.nan), "예측": np.arange(50.0)}, index=idx)
    svg = svg_lines(s)
    assert svg.count("<circle") == 10 and "<path" in svg and "2026-01-01 00:00" in svg
    sc = svg_scatter(s["예측"], s["예측"] * 2, pd.DataFrame({"x": [0, 49], "적합": [0, 98]}), "x<1>", "y")
    assert sc.count("<circle") == 50 and "x&lt;1&gt;" in sc
    doc = build({"소프트센서": {"lines": ["**요약** <script>", "- 항목 `a`"], "svg": svg, "code": "y = 2x",
                               "table": pd.DataFrame({"R²": [0.98765]})},
                 "데이터": {"lines": ["파일: t.csv"]}})
    assert doc.index("<h2>데이터") < doc.index("<h2>소프트센서") and "<b>요약</b> &lt;script&gt;" in doc
    assert "<li>항목 <code>a</code></li>" in doc and "0.9877" in doc and "<script" not in doc and "<th>0</th>" not in doc
    print("ok")
