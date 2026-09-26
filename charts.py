import altair as alt
import pandas as pd
import streamlit as st

MUTED, TIME_FMT = "#898781", "%Y-%m-%d %H:%M"
class _PerViewer:
    """Module globals are shared by every session (and thread); each viewer's palette lives in their session state."""

    def __init__(self, key):
        self.key = key

    def __getitem__(self, i):
        return st.session_state["_theme"][self.key][i]

    def __iter__(self):
        return iter(st.session_state["_theme"][self.key])

    def __len__(self):
        return len(st.session_state["_theme"][self.key])


SERIES, DIVERGING, THEME = _PerViewer("series"), _PerViewer("diverging"), _PerViewer("theme")


def apply_theme():
    """Validated reference palette (dataviz skill): the slot order is what keeps adjacent series CVD-safe."""
    dark = st.context.theme.type == "dark"
    series = (["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"] if dark else
              ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"])
    st.session_state["_theme"] = {
        "series": series, "diverging": [series[0], "#383835" if dark else "#f0efec", series[7]],
        "theme": {"ink": "#ffffff" if dark else "#0b0b0b", "surface": "#0e1117" if dark else "#ffffff"}}


def trend_chart(df, colors, extra=(), height=300, log=False):
    """Lines with a hover crosshair. colors: {series: hex} in fixed slot order. extra: more Altair layers."""
    long = df.rename_axis("_t").reset_index().melt("_t", var_name="_s", value_name="_v")
    hover = alt.selection_point(fields=["_t"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(long).encode(x=alt.X("_t:T", title="시간"))
    single = len(colors) == 1  # one series: the axis title names it, no legend box
    lines = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("_v:Q", title=next(iter(colors)) if single else None,
                scale=alt.Scale(type="log") if log else alt.Scale(zero=False)),
        color=alt.Color("_s:N", title=None, scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                        legend=None if single else alt.Legend(orient="top")),
    ).add_params(alt.selection_interval(bind="scales", encodings=["x"]))
    points = lines.mark_point(size=64, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("_t:T", title="시간", format=TIME_FMT), alt.Tooltip("_s:N", title="태그"),
                 alt.Tooltip("_v:Q", title="값", format=".5g")],
    ).add_params(hover)
    rule = base.mark_rule(color=MUTED).transform_filter(hover)
    extra = [extra] if isinstance(extra, alt.TopLevelMixin) else list(extra)
    st.altair_chart(alt.layer(*extra, lines, points, rule).properties(height=height), width="stretch")


def hline(y, label, above=False):
    """Dashed reference/limit line with a direct label (dashed reads as a threshold, never as grid).
    The label sits under the line by default: an upper limit is often at the top edge, where a label above is clipped."""
    d = pd.DataFrame({"y": [y], "label": [label]})
    y = alt.Y("y:Q", title=None)
    rule = alt.Chart(d).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y=y)
    text = alt.Chart(d).mark_text(align="left", dx=4, dy=-6 if above else 12, color=THEME["ink"], fontSize=11).encode(y=y, text="label:N", x=alt.value(0))
    return rule + text


def shade(spans, label):
    """Recessive background bands for time spans (DataFrame with 시작/끝)."""
    d = spans.rename(columns={"시작": "s", "끝": "e"})[["s", "e"]].assign(label=label)
    return alt.Chart(d).mark_rect(color=MUTED, opacity=0.15).encode(
        x=alt.X("s:T", title=None), x2="e:T", tooltip=[alt.Tooltip("label:N", title="구간"), alt.Tooltip("s:T", title="시작", format=TIME_FMT),
                                    alt.Tooltip("e:T", title="끝", format=TIME_FMT)])


def bar_chart(s, title, height=None, horizontal=True, fmt=".4g"):
    """One series → one color (slot 1). Rounded data-ends, value in tooltip; table view lives next to it."""
    d = s.rename_axis("_k").reset_index(name="_v")
    enc = dict(tooltip=[alt.Tooltip("_k:N", title=""), alt.Tooltip("_v:Q", title=title, format=fmt)])
    if horizontal:
        chart = alt.Chart(d).mark_bar(color=SERIES[0], cornerRadiusEnd=4, height={"band": 0.7}).encode(
            y=alt.Y("_k:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)), x=alt.X("_v:Q", title=title), **enc)
        height = height or max(160, 40 * len(d))
    else:
        chart = alt.Chart(d).mark_bar(color=SERIES[0], cornerRadiusEnd=4, width={"band": 0.7}).encode(
            x=alt.X("_k:N", sort=None, title=None), y=alt.Y("_v:Q", title=title), **enc)
        height = height or 260
    st.altair_chart(chart.properties(height=height), width="stretch")


def corr_heatmap(corr, order, label_min=0.7):
    """Diverging −1…+1 heatmap. Cell numbers only while cells are big enough to read (≤ 20 variables); tooltips always."""
    n = len(order)
    cells = corr.loc[order, order].rename_axis("a").reset_index().melt("a", var_name="b", value_name="r")
    axis = alt.Axis(labelLimit=180)
    heat = alt.Chart(cells).encode(x=alt.X("b:N", sort=order, title=None, axis=axis), y=alt.Y("a:N", sort=order, title=None, axis=axis))
    layers = [heat.mark_rect(stroke=THEME["surface"], strokeWidth=1 if n > 20 else 2).encode(
        color=alt.Color("r:Q", title="r", scale=alt.Scale(domain=[-1, 0, 1], range=list(DIVERGING), interpolate="lab")),
        tooltip=[alt.Tooltip("a:N", title="변수 1"), alt.Tooltip("b:N", title="변수 2"), alt.Tooltip("r:Q", title="상관계수", format=".3f")])]
    if n <= 20:  # label only strong pairs; the full matrix is in the table view
        layers.append(heat.mark_text(color=THEME["ink"], fontSize=11 if n <= 12 else 9).encode(
            text=alt.Text("r:Q", format=".2f")).transform_filter(f"abs(datum.r) >= {label_min} && datum.a != datum.b"))
    st.altair_chart(alt.layer(*layers).properties(height=max(320, min(1400, 26 * n))), width="stretch")
