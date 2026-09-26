import io
import json
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from analysis import apply_model, export_model, fit_soft_sensor, lag_scan, load, parse_model, to_timeseries

st.set_page_config(page_title="증류탑 공정데이터 분석", layout="wide")

# Validated reference palette (dataviz skill): the slot order is what keeps adjacent series CVD-safe.
DARK = st.context.theme.type == "dark"
SERIES = (["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"] if DARK else
          ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"])
DIVERGING = [SERIES[0], "#383835" if DARK else "#f0efec", SERIES[7]]
MUTED, INK, SURFACE = "#898781", "#ffffff" if DARK else "#0b0b0b", "#0e1117" if DARK else "#ffffff"
TIME_FMT = "%Y-%m-%d %H:%M"


# In-memory only; ttl/max_entries bound how long uploaded plant data lingers in the server process.
@st.cache_data(show_spinner=False, ttl="1h", max_entries=3)
def read(data: bytes, name: str):
    return load(io.BytesIO(data), name)


@st.cache_data(show_spinner="시간 컬럼 해석 중…", ttl="1h", max_entries=3)
def parse(data: bytes, name: str, time_col: str):
    return to_timeseries(read(data, name), time_col)


def equation(target, intercept, coef, lag):
    terms = "\n".join(f"    {c:+.6g} × {n}" for n, c in coef.items())
    st.code(f"{target} = {intercept:.6g}\n{terms}" + (f"\n\n※ 입력은 모두 {lag} 샘플 이전 값" if lag else ""), language=None)


def trend_chart(df, colors, extra=None, height=300):
    """Lines with a hover crosshair. colors: {series: hex} in fixed slot order."""
    long = df.rename_axis("_t").reset_index().melt("_t", var_name="_s", value_name="_v")
    hover = alt.selection_point(fields=["_t"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(long).encode(x=alt.X("_t:T", title="시간"))
    single = len(colors) == 1  # one series: the axis title names it, no legend box
    lines = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("_v:Q", title=next(iter(colors)) if single else None, scale=alt.Scale(zero=False)),
        color=alt.Color("_s:N", title=None, scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                        legend=None if single else alt.Legend(orient="top")),
    ).add_params(alt.selection_interval(bind="scales", encodings=["x"]))
    points = lines.mark_point(size=64, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("_t:T", title="시간", format=TIME_FMT), alt.Tooltip("_s:N", title="태그"),
                 alt.Tooltip("_v:Q", title="값", format=".5g")],
    ).add_params(hover)
    rule = base.mark_rule(color=MUTED).transform_filter(hover)
    chart = alt.layer(lines, points, rule, *([extra] if extra is not None else [])).properties(height=height)
    st.altair_chart(chart, use_container_width=True)


st.title("증류탑 공정데이터 분석")
st.caption("v1.0")

file = st.sidebar.file_uploader("CSV / Excel 업로드", type=["csv", "xlsx"])
if not file:
    st.info("왼쪽에서 파일을 업로드하세요. 형식: 첫 행 헤더, 시간 컬럼 1개 + 태그별 컬럼.")
    st.stop()

data = file.getvalue()
try:
    raw = read(data, file.name)
except Exception as e:  # malformed upload: show why instead of a traceback
    st.error(f"파일을 읽을 수 없습니다: {e}")
    st.stop()
time_col = st.sidebar.selectbox("시간 컬럼", raw.columns)
df = parse(data, file.name, time_col)
if df.empty:
    st.error(f"'{time_col}' 컬럼을 시간으로 해석할 수 없거나 숫자 태그가 없습니다.")
    st.stop()

rule = st.sidebar.selectbox("리샘플링(평균)", ["원본", "1min", "10min", "1h", "1D"])
if rule != "원본":
    df = df.resample(rule).mean()

lo, hi = df.index.min().to_pydatetime(), df.index.max().to_pydatetime()
if lo < hi:
    start, end = st.sidebar.slider("기간", lo, hi, (lo, hi), timedelta(minutes=1), "YYYY-MM-DD HH:mm")
    df = df.loc[start:end]
st.sidebar.caption(f"{len(df):,}행 · 태그 {df.shape[1]}개")

trend, soft, apply = st.tabs(["트렌드 · 통계", "소프트센서", "모델 적용"])

with trend:
    tags = st.multiselect("태그 (최대 8개)", df.columns, default=list(df.columns[:4]), max_selections=8)
    if tags:
        # A tag keeps its color while selected, so adding/removing others never repaints it.
        prev = st.session_state.get("slots", {})
        slots = {t: prev[t] for t in tags if t in prev}
        free = [i for i in range(8) if i not in slots.values()]
        slots |= {t: free.pop(0) for t in tags if t not in slots}
        st.session_state["slots"] = slots

        view = df[tags]
        mode = st.radio("보기", ["겹쳐 보기", "태그별 분리", "정규화(z-score)"], horizontal=True,
                        help="단위가 다른 태그는 '태그별 분리'(각자 세로축) 또는 '정규화'로 비교하세요.")
        if mode == "태그별 분리":
            for t in tags:
                trend_chart(view[[t]], {t: SERIES[slots[t]]}, height=160)
        else:
            trend_chart((view - view.mean()) / view.std() if mode.startswith("정규화") else view,
                        {t: SERIES[slots[t]] for t in tags})
        with st.expander("표로 보기"):
            st.dataframe(view)

        c1, c2 = st.columns(2)
        c1.subheader("기초통계")
        c1.dataframe(view.describe().T.round(3))
        c2.subheader("상관계수")
        corr = view.corr()
        cells = corr.rename_axis("a").reset_index().melt("a", var_name="b", value_name="r")
        heat = alt.Chart(cells).encode(x=alt.X("b:N", sort=tags, title=None), y=alt.Y("a:N", sort=tags, title=None))
        c2.altair_chart(alt.layer(
            heat.mark_rect(stroke=SURFACE, strokeWidth=2).encode(
                color=alt.Color("r:Q", title="r", scale=alt.Scale(domain=[-1, 0, 1], range=DIVERGING, interpolate="lab")),
                tooltip=[alt.Tooltip("a:N", title="태그 1"), alt.Tooltip("b:N", title="태그 2"),
                         alt.Tooltip("r:Q", title="상관계수", format=".2f")]),
            # Label only strong pairs; the full matrix is in the table view.
            heat.mark_text(color=INK).encode(text=alt.Text("r:Q", format=".2f")).transform_filter(
                "abs(datum.r) >= 0.7 && datum.a != datum.b"),
        ).properties(height=320), use_container_width=True)
        with c2.expander("상관계수 표"):
            st.dataframe(corr.round(2))

with soft:
    target = st.selectbox("예측 대상 (품질 변수: 순도, 조성 등)", df.columns, index=df.shape[1] - 1)
    inputs = st.multiselect("입력 변수 (온도, 압력, 유량 등)", [c for c in df.columns if c != target])
    c1, c2 = st.columns(2)
    method = c1.radio("모델", ["OLS (선형회귀)", "PLS (부분최소제곱)"], horizontal=True,
                      help="입력 변수끼리 상관이 강하면(예: 인접 단 온도) PLS가 계수가 안정적입니다.")
    k = c2.number_input("PLS 성분 수", 1, len(inputs), min(2, len(inputs))) if method.startswith("PLS") and inputs else None
    c1, c2 = st.columns(2)
    lag = c1.number_input("입력 지연 (샘플 수) — 분석계 지연·dead time 보정", 0, 10_000, key="lag")
    frac = c2.slider("학습 비율 (앞쪽 시간 구간으로 학습, 뒤쪽으로 검증)", 0.5, 0.9, 0.7)

    if inputs:
        with st.expander("최적 지연 자동 탐색"):
            max_lag = st.number_input("탐색할 최대 지연 (샘플 수)", 1, 2000, 60)
            key = (target, tuple(inputs), frac, k, len(df), max_lag)

            def run_scan():
                scan = lag_scan(df, target, inputs, max_lag, frac, k)
                st.session_state["scan"] = (key, scan)
                if not scan.empty:
                    st.session_state["lag"] = int(scan.idxmax())

            st.button("탐색하고 최적 지연 적용", on_click=run_scan)
            if st.session_state.get("scan", (None,))[0] == key:
                scan = st.session_state["scan"][1]
                if scan.empty:
                    st.error("모든 지연에서 모델을 만들 수 없었습니다. 데이터 기간이나 입력 변수를 확인하세요.")
                else:
                    best = scan.idxmax()
                    st.success(f"검증 R²가 가장 높은 지연: **{best} 샘플** (R² {scan.max():.4f}) — 입력 지연에 적용했습니다.")
                    s = alt.Chart(scan.reset_index()).encode(x=alt.X("지연:Q", title="지연 (샘플 수)"),
                                                             y=alt.Y("검증 R²:Q", scale=alt.Scale(zero=False)))
                    top = s.transform_filter(f"datum['지연'] == {best}")
                    st.altair_chart(alt.layer(
                        s.mark_line(strokeWidth=2, color=SERIES[0]),
                        s.mark_point(size=64, filled=True, opacity=0, color=SERIES[0]).encode(
                            tooltip=["지연:Q", alt.Tooltip("검증 R²:Q", format=".4f")]),
                        top.mark_point(size=100, filled=True, color=SERIES[0]),
                        top.mark_text(dy=-14, color=INK).encode(text=alt.value(f"최적 {best}")),
                    ).properties(height=220), use_container_width=True)

        try:
            r = fit_soft_sensor(df, target, inputs, lag, frac, k)
        except ValueError as e:
            st.error(str(e))
            r = None  # not st.stop(): the 모델 적용 tab below must still render

    if inputs and r:
        if k:
            with st.expander("성분 수별 검증 R² (성분 수 선택 참고)"):
                st.dataframe(pd.Series(
                    {n: (r if n == k else fit_soft_sensor(df, target, inputs, lag, frac, n))["metrics"]["검증 R²"]
                     for n in range(1, len(inputs) + 1)}, name="검증 R²").rename_axis("성분 수").round(4))

        for col, (name, val) in zip(st.columns(4), r["metrics"].items()):
            col.metric(name, f"{val:.4f}")

        res = r["result"]
        split = alt.Chart(pd.DataFrame({"_t": [r["split"]]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="_t:T")
        st.subheader("실측 vs 예측 (시간)")
        trend_chart(res[["실측", "예측"]], {"실측": SERIES[0], "예측": SERIES[1]}, extra=split)
        st.caption(f"점선 = 학습/검증 경계 ({r['split']:%Y-%m-%d %H:%M}), 오른쪽이 검증 구간")

        c1, c2 = st.columns(2)
        c1.subheader("패리티 플롯")
        span = [min(res["실측"].min(), res["예측"].min()), max(res["실측"].max(), res["예측"].max())]
        c1.altair_chart(alt.layer(
            alt.Chart(pd.DataFrame({"v": span})).mark_line(color=MUTED, strokeWidth=1).encode(
                x=alt.X("v:Q", title="실측"), y=alt.Y("v:Q", title="예측")),
            alt.Chart(res.rename_axis("시간").reset_index()).mark_circle(size=64, opacity=0.5).encode(
                x=alt.X("실측:Q", scale=alt.Scale(zero=False, domain=span)),
                y=alt.Y("예측:Q", scale=alt.Scale(zero=False, domain=span)),
                color=alt.Color("구분:N", title=None, scale=alt.Scale(domain=["학습", "검증"], range=[MUTED, SERIES[1]]),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("시간:T", format=TIME_FMT), "구분:N",
                         alt.Tooltip("실측:Q", format=".5g"), alt.Tooltip("예측:Q", format=".5g")]),
        ).properties(height=360), use_container_width=True)
        c1.caption("대각선에 가까울수록 정확. 검증 점이 한쪽으로 치우치면 편향(bias)이 있다는 뜻입니다.")

        c2.subheader("변수 영향도")
        c2.dataframe(pd.concat([r["coef"], r["raw_coef"]], axis=1).round(6))
        c2.caption("표준화 계수: 입력 1 표준편차 변화당 예측 변화 (영향 크기 비교용)")

        st.subheader("모델식 (원래 단위, DCS 적용용)")
        equation(target, r["raw_intercept"], r["raw_coef"], lag)

        with st.expander("예측 결과 표"):
            st.dataframe(res)
        c1, c2 = st.columns(2)
        c1.download_button("예측 결과 CSV 다운로드", res.to_csv().encode("utf-8-sig"), f"softsensor_{target}.csv", "text/csv")
        c2.download_button("모델 저장 (JSON)", json.dumps(export_model(r, target, lag, k, rule), ensure_ascii=False, indent=2),
                           f"softsensor_{target}.json", "application/json", help="'모델 적용' 탭에서 새 데이터에 다시 적용할 수 있습니다.")

with apply:
    st.caption("소프트센서 탭에서 저장한 모델(JSON)을 지금 올린 데이터에 적용합니다. 실측값이 있으면 정확도도 계산합니다.")
    mfile = st.file_uploader("모델 파일 (JSON)", type=["json"])
    if mfile:
        try:
            model = parse_model(mfile.getvalue().decode("utf-8"))
            out, met = apply_model(df, model)
        except (ValueError, UnicodeDecodeError) as e:
            st.error(str(e))
            st.stop()

        info = {"예측 대상": model["target"], "모델": model.get("method", "-"), "입력 지연": f"{model['lag']} 샘플",
                "학습 기간": " ~ ".join(model.get("train_period", ["-"])), "학습 시 리샘플링": model.get("resample", "-"),
                "학습 시 검증 R²": f"{model.get('metrics', {}).get('검증 R²', float('nan')):.4f}", "저장 시각": model.get("created", "-")}
        st.dataframe(pd.Series(info, name="모델 정보"))
        if model.get("resample", rule) != rule:
            st.warning(f"모델은 '{model['resample']}' 간격 데이터로 학습했는데 지금은 '{rule}' 입니다. "
                       "입력 지연(샘플 수)의 시간 길이가 달라지므로 사이드바 리샘플링을 모델과 맞추세요.")
        if out.empty:
            st.error("입력 태그에 값이 있는 구간이 없어 예측할 수 없습니다.")
            st.stop()

        if met:
            for col, (name, val) in zip(st.columns(3), met.items()):
                col.metric(f"적용 {name}", f"{val:.4f}" if isinstance(val, float) else f"{val:,}")
        else:
            st.info("현재 데이터에 예측 대상의 실측값이 없거나 부족해 예측값만 표시합니다.")
        trend_chart(out, {"실측": SERIES[0], "예측": SERIES[1]} if "실측" in out else {"예측": SERIES[1]})
        equation(model["target"], model["intercept"], model["coef"], model["lag"])
        with st.expander("예측 결과 표"):
            st.dataframe(out)
        st.download_button("적용 결과 CSV 다운로드", out.to_csv().encode("utf-8-sig"),
                           f"applied_{model['target']}.csv", "text/csv")
