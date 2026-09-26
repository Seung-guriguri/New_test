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
    st.altair_chart(alt.layer(*extra, lines, points, rule).properties(height=height), use_container_width=True)


def hline(y, label):
    """Dashed reference/limit line with a direct label (dashed reads as a threshold, never as grid)."""
    d = pd.DataFrame({"y": [y], "label": [label]})
    y = alt.Y("y:Q", title=None)
    rule = alt.Chart(d).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y=y)
    text = alt.Chart(d).mark_text(align="left", dx=4, dy=-6, color=THEME["ink"], fontSize=11).encode(y=y, text="label:N", x=alt.value(0))
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
            y=alt.Y("_k:N", sort=None, title=None), x=alt.X("_v:Q", title=title), **enc)
        height = height or max(160, 40 * len(d))
    else:
        chart = alt.Chart(d).mark_bar(color=SERIES[0], cornerRadiusEnd=4, width={"band": 0.7}).encode(
            x=alt.X("_k:N", sort=None, title=None), y=alt.Y("_v:Q", title=title), **enc)
        height = height or 260
    st.altair_chart(chart.properties(height=height), use_container_width=True)
