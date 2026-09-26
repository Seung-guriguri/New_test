from datetime import timedelta

import altair as alt
import streamlit as st

from analysis import fit_soft_sensor, load, to_timeseries


def line_chart(df):
    # st.line_chart pins the y-axis at 0, which flattens process trends into straight lines.
    long = df.rename_axis("시간").reset_index().melt("시간", var_name="태그", value_name="값")
    st.altair_chart(
        alt.Chart(long).mark_line().encode(x="시간:T", y=alt.Y("값:Q", scale=alt.Scale(zero=False)), color="태그:N").interactive(),
        use_container_width=True,
    )

st.set_page_config(page_title="증류탑 공정데이터 분석", layout="wide")
st.title("증류탑 공정데이터 분석")

file = st.sidebar.file_uploader("CSV / Excel 업로드", type=["csv", "xlsx"])
if not file:
    st.info("왼쪽에서 파일을 업로드하세요. 형식: 첫 행 헤더, 시간 컬럼 1개 + 태그별 컬럼.")
    st.stop()

raw = load(file, file.name)
time_col = st.sidebar.selectbox("시간 컬럼", raw.columns)
df = to_timeseries(raw, time_col)
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

trend, soft = st.tabs(["트렌드 · 통계", "소프트센서"])

with trend:
    tags = st.multiselect("태그", df.columns, default=list(df.columns[:4]))
    if tags:
        view = df[tags]
        norm = st.checkbox("정규화(z-score) — 단위가 다른 태그를 한 그래프에서 비교")
        line_chart((view - view.mean()) / view.std() if norm else view)
        c1, c2 = st.columns(2)
        c1.subheader("기초통계")
        c1.dataframe(view.describe().T.round(3))
        c2.subheader("상관계수")
        c2.dataframe(view.corr().round(2))

with soft:
    target = st.selectbox("예측 대상 (품질 변수: 순도, 조성 등)", df.columns, index=df.shape[1] - 1)
    inputs = st.multiselect("입력 변수 (온도, 압력, 유량 등)", [c for c in df.columns if c != target])
    c1, c2 = st.columns(2)
    lag = c1.number_input("입력 지연 (샘플 수) — 분석계 지연·dead time 보정", 0, 10_000, 0)
    frac = c2.slider("학습 비율 (앞쪽 시간 구간으로 학습, 뒤쪽으로 검증)", 0.5, 0.9, 0.7)

    if inputs:
        try:
            r = fit_soft_sensor(df, target, inputs, lag, frac)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        for col, (k, v) in zip(st.columns(4), r["metrics"].items()):
            col.metric(k, f"{v:.4f}")
        line_chart(r["result"][["실측", "예측"]])
        st.caption(f"학습/검증 경계: {r['split']}  ·  절편: {r['intercept']:.4f}")
        st.subheader("변수 영향도 (표준화 계수, 절댓값 순)")
        st.dataframe(r["coef"].round(4))
        st.download_button(
            "예측 결과 CSV 다운로드",
            r["result"].to_csv().encode("utf-8-sig"),
            f"softsensor_{target}.csv",
            "text/csv",
        )
