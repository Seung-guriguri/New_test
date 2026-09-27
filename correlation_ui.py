"""Streamlit UI for the 상관분석 tab (any numeric data, not tied to an equipment type) and the shared residual report."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import correlation as cr
from charts import MUTED, SERIES, THEME, TIME_FMT, corr_heatmap, hline, trend_chart
from equipment_ui import choice, flag, multi, num

MAX_POINTS = 5000  # scatter charts: evenly spaced rows beyond this (the statistics always use every row)


@st.cache_data(show_spinner=False, ttl="1h", max_entries=10)
def _matrix(d, method):
    return cr.corr_matrix(d, method)


@st.cache_data(show_spinner="변수 쌍 분석 중…", ttl="1h", max_entries=5)
def _pairs(d):
    return cr.pair_table(cr.thin(d, 20_000))


@st.cache_data(show_spinner=False, ttl="1h", max_entries=10)
def cached_vif(d):
    return cr.vif(d)


@st.cache_data(show_spinner="회귀 계산 중…", ttl="1h", max_entries=10)
def _biv(x, y, model, lag, hac):
    r = cr.bivariate(x, y, model, lag, hac)
    r["influence"] = cr.influence(r)
    del r["fit"]  # statsmodels results are large; everything the UI needs is extracted above
    return r


@st.cache_data(show_spinner=False, ttl="1h", max_entries=10)
def _compare(x, y, lag):
    return cr.compare_models(x, y, lag)


@st.cache_data(show_spinner="시차 상관 계산 중…", ttl="1h", max_entries=10)
def _ccf(x, y, max_lag, diff):
    return cr.ccf(x, y, max_lag, diff)


@st.cache_data(show_spinner=False, ttl="1h", max_entries=10)
def _diagnose(resid, fitted, max_lag):
    return cr.diagnose(resid, fitted, max_lag)


@st.cache_data(show_spinner="그랜저 검정 중…", ttl="1h", max_entries=10)
def _granger(x, y, max_lag, diff):
    return cr.granger(x, y, max_lag, diff)


def vif_table(v):
    """VIF values with a verdict; ∞ means the tag is computed exactly from the others (e.g. ΔP = P_bottom − P_top)."""
    verdict = np.select([np.isinf(v), v > 10, v > 5], ["⚠️ 완전 중복 (다른 태그로 정확히 계산됨)", "⚠️ 심각 (>10)", "주의 (5~10)"], "양호")
    return pd.DataFrame({"VIF": [("∞" if np.isinf(x) else f"{x:,.1f}") for x in v], "판정": verdict}, index=v.index)


def _p(v):
    return "-" if not np.isfinite(v) else "< 0.001" if v < 0.001 else f"{v:.3f}"


def _duration(td):
    s = td.total_seconds()
    for unit, sec in (("일", 86400), ("시간", 3600), ("분", 60)):
        if s >= sec:
            return f"{s / sec:.3g}{unit}"
    return f"{s:.3g}초"


def _lag_text(k, step):
    return f"{k} 샘플" + (f" (≈ {_duration(abs(k) * step)})" if step is not None and k else "")


def _step(df, seq):
    """Median sampling interval, or None for row-order data (its pseudo time axis has no real spacing)."""
    if seq or len(df) < 2:
        return None
    return df.index.to_series().diff().median()


def _default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def corr_tab(df, seq=False):
    st.caption("장치 종류와 관계없이 숫자 변수 사이의 관계를 분석합니다. 상관은 '함께 움직인다'는 뜻일 뿐 원인을 뜻하지 않으며, "
               "공정 데이터는 이웃한 시점끼리 비슷해서(자기상관) 일반 통계 교재식 p값은 지나치게 작게 나옵니다. "
               "이 탭의 p값은 그 점을 보정한 값입니다 (매뉴얼 10.27).")
    mat, biv, res, lag = st.tabs(["상관 행렬", "이변량 회귀", "잔차 분석", "시차 상관 · 선후 관계"])
    with mat:
        matrix_ui(df)
    with biv:
        fit = bivariate_ui(df, seq)
    with res:
        residual_ui(fit)
    with lag:
        lag_ui(df, seq)


# ---------- 상관 행렬 ----------

def matrix_ui(df):
    cols = list(df.columns)
    vars_ = multi(st, f"분석할 변수 (개수 제한 없음 · 전체 {len(cols)}개)", "cfg_corr_vars", cols, cols[:60])
    c = st.columns([2, 2, 1])
    method = choice(c[0], "상관계수 종류", "cfg_corr_method", list(cr.METHODS),
                    help="피어슨: 직선 관계. 스피어만: 한 방향으로 휘는 곡선도 잡음. 켄달: 이상값에 가장 강함. "
                         "편상관: 선택한 나머지 변수의 영향을 뺀 '직접' 관계 (공통 원인 때문에 생긴 가짜 상관을 걸러냄).")
    order = choice(c[1], "정렬", "cfg_corr_order", ["비슷한 변수끼리 모으기 (군집)", "선택한 순서"])
    label_min = num(c[2], "숫자 표시 |r| ≥", "cfg_corr_label", 0.7, min_value=0.0, max_value=1.0, step=0.05)
    ok, bad = cr.usable(df[vars_])
    if bad:
        st.caption(f"값이 3개 미만이거나 변하지 않아 제외: {', '.join(bad)}")
    if len(ok) < 2:
        st.info("변수를 2개 이상 고르세요.")
        return
    try:
        m = _matrix(df[ok], cr.METHODS[method])
    except ValueError as e:
        st.error(str(e))
        return
    names = cr.cluster_order(m) if order.startswith("비슷") else ok
    corr_heatmap(m, names, label_min)
    st.caption("빨강(+) = 함께 증가, 파랑(−) = 한쪽이 늘면 다른 쪽이 줄어듦, 옅을수록 관계가 약함. 칸에 마우스를 올리면 값이 보입니다."
               + (" 켄달은 계산량이 커서 2만 행을 고르게 뽑아 계산했습니다." if cr.METHODS[method] == "kendall" and len(df) > 20_000 else ""))
    with st.expander("상관계수 행렬 표"):
        st.dataframe(m.loc[names, names].round(3))
        st.download_button("행렬 CSV 다운로드", m.loc[names, names].to_csv().encode("utf-8-sig"), "correlation_matrix.csv", "text/csv")

    st.markdown("**변수 쌍 순위** (관계가 강한 순) — 행을 클릭하면 이변량 회귀·시차 상관 탭에 그 쌍이 들어갑니다")
    pairs = _pairs(df[ok])
    c = st.columns([2, 1])
    focus = choice(c[0], "변수로 거르기", "cfg_corr_focus", ["(전체)", *ok])
    _default("cfg_corr_hide_weak", True)
    hide = flag(c[1], "'관계 약함' 숨기기", "cfg_corr_hide_weak")
    view = pairs
    if focus != "(전체)":
        view = view[(view["변수 1"] == focus) | (view["변수 2"] == focus)]
    if hide:
        view = view[view["관계 유형"] != "관계 약함"]
    view = view.head(500).reset_index(drop=True)
    # Keyed by what is listed: a remembered row number must never point at a different pair after the filter changes.
    ev = st.dataframe(view, hide_index=True, on_select="rerun", selection_mode="single-row",
                      key=f"corr_pairs_{abs(hash((focus, hide, tuple(ok))))}",
                      column_config={"피어슨 r": st.column_config.NumberColumn(format="%.3f"),
                                     "스피어만 ρ": st.column_config.NumberColumn(format="%.3f"),
                                     "거리상관": st.column_config.NumberColumn(format="%.3f"),
                                     "곡선 설명력 추가": st.column_config.NumberColumn(format="%.2f"),
                                     "p값 (자기상관 보정)": st.column_config.NumberColumn(format="%.2e")})
    rows = ev.selection.rows if ev else []
    if rows and rows[0] < len(view):
        a, b = view.loc[rows[0], "변수 1"], view.loc[rows[0], "변수 2"]
        if st.session_state.get("_corr_pair_applied") != (a, b):  # apply once per click, then the user may change them
            st.session_state["_corr_pair_applied"] = (a, b)
            st.session_state.update(cfg_biv_x=a, cfg_biv_y=b, cfg_lag_x=a, cfg_lag_y=b)
        st.success(f"**{a} → {b}** 를 이변량 회귀·시차 상관 탭에 넣었습니다.")
    st.caption("거리상관: 모양과 관계없이 관련이 있으면 커지는 지표 (0 = 무관). 곡선 설명력 추가: 곡선이 직선보다 더 설명하는 분산 비율 — "
               "0.08 이상이면 '곡선' 관계로 분류합니다. 유효 n: 자기상관을 고려한 '독립 표본 수'로, 이것으로 p값을 계산했습니다. "
               "p < 0.01 이어도 |r|이 작으면 실무적 의미는 작습니다. **비단조**는 U자 곡선뿐 아니라 운전 모드(압력·부하 등)가 다른 기간이 "
               "섞여 있을 때도 나옵니다 — 이변량 회귀의 '점 색 = 시간 순서'로 확인하고, 공정단위 → 변화점 탐지로 찾은 구간을 사이드바 "
               "**기간**으로 골라 다시 보세요.")
    csv = pairs.to_csv(index=False).encode("utf-8-sig")
    st.download_button("변수 쌍 표 CSV 다운로드", csv, "correlation_pairs.csv", "text/csv")

    with st.expander("다중공선성 (VIF) — 회귀·소프트센서 입력을 고르기 전에"):
        # Expander bodies run on every rerun: compute only on request (100 variables = 100 regressions).
        if not st.toggle("VIF 계산", key="corr_vif_on"):
            st.caption("켜면 선택한 변수 전체의 VIF를 계산합니다.")
            return
        try:
            v = cached_vif(df[ok])
        except ValueError as e:
            st.warning(str(e))
        else:
            st.dataframe(vif_table(v))
            st.caption("VIF = 그 변수가 나머지 변수들로 얼마나 설명되는지 (1/(1−R²)). 10을 넘는 변수를 OLS 입력에 함께 넣으면 계수가 "
                       "불안정해집니다 (부호가 뒤집히기도 함). 그중 하나만 쓰거나 PLS를 쓰세요.")


# ---------- 이변량 회귀 ----------

def bivariate_ui(df, seq):
    cols = list(df.columns)
    if len(cols) < 2:
        st.info("숫자 변수가 2개 이상 있어야 합니다.")
        return None
    c = st.columns(2)
    x = choice(c[0], "X (설명 변수)", "cfg_biv_x", cols)
    ys = [col for col in cols if col != x]
    y = choice(c[1], "Y (반응 변수)", "cfg_biv_y", ys)
    c = st.columns([2, 1, 1, 1])
    model = choice(c[0], "모델", "cfg_biv_model", cr.MODELS)
    lag = int(num(c[1], "X 지연 (샘플)", "cfg_biv_lag", 0, min_value=0, max_value=100_000, step=1,
                  help="Y(t)를 X(t − 지연)으로 설명합니다. '시차 상관' 탭에서 찾은 지연을 넣으세요."))
    _default("cfg_biv_hac", True)
    hac = flag(c[2], "자기상관 보정 (HAC)", "cfg_biv_hac",
               help="Newey-West 표준오차. 시계열에서는 켜 두세요: 끄면 신뢰구간·p값이 실제보다 좁고 작게 나옵니다.")
    _default("cfg_biv_color", True)
    color_time = flag(c[3], "점 색 = 시간 순서", "cfg_biv_color", help="시간에 따라 관계가 이동(드리프트)하는지 보입니다.")
    try:
        r = _biv(df[x], df[y], model, lag, hac)
    except ValueError as e:
        st.error(str(e))
        return None
    r["names"] = (x, y)
    met = r["metrics"]
    k = st.columns(5)
    k[0].metric("R²", f"{met['R²']:.4f}", help="Y 변동 중 이 모델이 설명하는 비율 (원래 단위 기준)")
    k[1].metric("조정 R²", f"{met['조정 R²']:.4f}", help="항이 많은 모델에 벌점을 준 R² — 모델끼리 비교할 때 사용")
    k[2].metric("RMSE", f"{met['RMSE']:.4g}", help="예측 오차의 크기 (Y 단위)")
    k[3].metric("데이터 수", f"{met['n']:,}")
    k[4].metric("모델 p값", _p(met["모델 p값"]), help="'X와 Y는 무관하다'가 맞을 확률에 해당. 작을수록 관계가 우연이 아님"
                + (" (HAC 보정)" if hac else ""))

    xt = f"{x} (t−{lag})" if lag else x
    d = cr.thin(r["data"], MAX_POINTS).rename_axis("_t").reset_index()
    enc = dict(x=alt.X("x:Q", title=xt, scale=alt.Scale(zero=False)), y=alt.Y("y:Q", title=y, scale=alt.Scale(zero=False)))
    # Two validated series colours (orange → blue) stay visible on both themes, unlike a dark-to-light ramp.
    color = alt.Color("_t:T", title="순서" if seq else "시간", scale=alt.Scale(range=[SERIES[1], SERIES[0]], interpolate="hcl"),
                      legend=alt.Legend(orient="top", format="%Y-%m-%d" if not seq else "%H:%M")) if color_time else alt.value(SERIES[0])
    pts = alt.Chart(d).mark_circle(size=36, opacity=0.6).encode(
        **enc, color=color, tooltip=[alt.Tooltip("_t:T", title="시간", format=TIME_FMT), alt.Tooltip("x:Q", title=xt, format=".5g"),
                                     alt.Tooltip("y:Q", title=y, format=".5g")])
    g = alt.Chart(r["grid"]).encode(x="x:Q")
    ink = THEME["ink"]
    band = g.mark_area(color=ink, opacity=0.18).encode(y="신뢰 하한:Q", y2="신뢰 상한:Q")
    pi = [g.mark_line(color=ink, strokeDash=[4, 4], strokeWidth=1).encode(y=f"{b}:Q") for b in ("예측 하한", "예측 상한")]
    line = g.mark_line(color=ink, strokeWidth=2.5).encode(y="적합:Q", tooltip=[alt.Tooltip("x:Q", title=xt, format=".5g"),
                                                                                alt.Tooltip("적합:Q", format=".5g")])
    st.altair_chart(alt.layer(pts, band, *pi, line).properties(height=420), width="stretch")
    st.caption("굵은 선 = 모델, 짙은 띠 = 평균의 95% 신뢰구간 (선이 있을 법한 범위), 점선 = 95% 예측구간 (새 측정값 하나가 들어올 범위)."
               + (f" 점은 {MAX_POINTS:,}개를 고르게 뽑아 그렸고, 계산에는 전체 {met['n']:,}개를 썼습니다." if met["n"] > MAX_POINTS else "")
               + (" 색이 시간에 따라 다른 자리에 몰리면 관계 자체가 시간에 따라 이동한 것입니다 (촉매 노화·파울링·계절)." if color_time else ""))

    c = st.columns(2)
    c[0].markdown("**모델식 (원래 단위)**")
    c[0].code(cr.equation(model, r["coef"]["계수"].to_numpy(), r["center"], r["scale"], x=xt, y=y), language=None, wrap_lines=True)
    if cr._DEGREE.get(model, 1) > 1:
        c[0].caption(f"오른쪽 계수표는 계산 안정성을 위해 표준화한 z = (X − {r['center']:.6g}) / {r['scale']:.6g} 기준입니다. "
                     "최고차항의 p값이 크면 한 단계 낮은 모델로 충분합니다.")
    c[1].markdown("**계수** (95% 신뢰구간" + (", HAC 보정)" if hac else ")"))
    c[1].dataframe(r["coef"].style.format({"계수": "{:.6g}", "95% 하한": "{:.6g}", "95% 상한": "{:.6g}", "p값": _p}))

    with st.expander("모델 비교 (같은 X·Y로 6가지 모델)"):
        cmp = _compare(df[x], df[y], lag)
        fmt = {"R²": "{:.4f}", "조정 R²": "{:.4f}", "RMSE": "{:.4g}", "n": "{:,.0f}"}
        # Text cells: Streamlit shows NaN as "None" whatever the Styler's na_rep says.
        st.dataframe(cmp.assign(**{c: cmp[c].map(lambda v, f=f: "-" if pd.isna(v) else f.format(v)) for c, f in fmt.items()}))
        best = cmp["조정 R²"].astype(float)
        if best.notna().any():
            st.caption(f"조정 R²가 가장 높은 모델: **{best.idxmax()}**. 차이가 0.01 미만이면 더 단순한 모델(선형)이 낫습니다 — "
                       "고차 다항식은 데이터 범위 밖에서 급격히 틀어집니다.")
    return r


# ---------- 잔차 분석 ----------

def residual_ui(r):
    if r is None:
        st.info("'이변량 회귀' 탭에서 X와 Y를 고르면 그 모델의 잔차(실측 − 모델)를 여기서 진단합니다. "
                "소프트센서 모델의 잔차 진단은 소프트센서 탭 아래쪽에 있습니다.")
        return
    x, y = r["names"]
    xt = f"{x} (t−{r['lag']})" if r["lag"] else x
    st.caption(f"모델: **{y} = f({xt})** · {r['model']}. "
               "잔차에 패턴이 남아 있으면 모델이 놓친 정보가 있다는 뜻입니다.")
    residual_report(r["data"]["잔차"], r["data"]["적합값"])
    top, frac = r["influence"]
    with st.expander(f"영향점 — 혼자서 회귀선을 크게 움직이는 점 ({frac:.1%})"):
        if top.empty:
            st.write("쿡의 거리 4/n을 넘는 점이 없습니다.")
        else:
            st.dataframe(top.rename(columns={"x": x, "y": y}).rename_axis("시간").round(5))
            st.caption("쿡의 거리 상위 10개 (기준 4/n 초과). 계기 이상·특이 운전이면 기간 설정으로 빼고 다시 보세요. "
                       "정상 데이터라면 지우지 말고, 그 영역의 데이터가 부족하다는 신호로 보세요.")


def residual_report(resid, fitted, max_lag=40):
    """Verdict table + four diagnostic charts. resid = 실측 − 예측, both Series on the time index."""
    try:
        tab, acf_s, z = _diagnose(resid, fitted, max_lag)
    except ValueError as e:
        st.warning(str(e))
        return
    st.markdown("| 항목 | 값 | 판정 | 의미 · 조치 |\n|---|---|---|---|\n" +
                "\n".join("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in tab.to_numpy()))
    d = pd.DataFrame({"잔차": resid, "적합값": fitted}).dropna()
    c = st.columns(2)
    with c[0]:
        st.markdown("**잔차 vs 예측값** — 고르게 퍼진 띠여야 정상")
        s = cr.thin(d, MAX_POINTS)
        st.altair_chart(alt.layer(
            alt.Chart(s).mark_circle(size=24, opacity=0.5, color=SERIES[0]).encode(
                x=alt.X("적합값:Q", title="예측값", scale=alt.Scale(zero=False)), y=alt.Y("잔차:Q")),
            hline(0, "0")).properties(height=260), width="stretch")
    with c[1]:
        st.markdown("**잔차 시간 추이** — 몰려 있는 구간 = 모델이 모르는 운전")
        trend_chart(d[["잔차"]], {"잔차": SERIES[0]}, extra=hline(0, "0"), height=260)
    c = st.columns(3)
    with c[0]:
        st.markdown("**히스토그램** (표준화)")
        counts, edges = np.histogram(z.clip(-5, 5), bins=40)
        h = pd.DataFrame({"z": edges[:-1], "z2": edges[1:], "밀도": counts / counts.sum() / (edges[1] - edges[0])})
        xs = np.linspace(-4, 4, 100)
        normal = pd.DataFrame({"z": xs, "정규분포": np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi)})
        st.altair_chart(alt.layer(
            alt.Chart(h).mark_bar(color=SERIES[0], opacity=0.8, stroke=THEME["surface"], strokeWidth=0.5).encode(
                x=alt.X("z:Q", title="표준화 잔차", scale=alt.Scale(domain=[-5, 5])), x2="z2:Q",
                y=alt.Y("밀도:Q", title="밀도"), y2=alt.datum(0)),
            alt.Chart(normal).mark_line(color=THEME["ink"]).encode(x="z:Q", y="정규분포:Q"),
        ).properties(height=240), width="stretch")
    with c[1]:
        st.markdown("**Q-Q 플롯** — 대각선 위면 정규분포")
        q = cr.qq_points(z)
        lim = [float(min(q.min().min(), -3)), float(max(q.max().max(), 3))]
        st.altair_chart(alt.layer(
            alt.Chart(pd.DataFrame({"v": lim})).mark_line(color=MUTED).encode(x=alt.X("v:Q", title="이론 분위수"), y=alt.Y("v:Q", title="잔차 분위수")),
            alt.Chart(q).mark_circle(size=20, color=SERIES[0]).encode(x="이론 분위수:Q", y="잔차 분위수:Q"),
        ).properties(height=240), width="stretch")
    with c[2]:
        st.markdown("**자기상관 (ACF)** — 점선 밖 막대 = 시간 패턴")
        band = 1.96 / np.sqrt(len(d))
        a = acf_s.rename_axis("지연").reset_index()
        st.altair_chart(alt.layer(
            alt.Chart(a).mark_bar(color=SERIES[0]).encode(x=alt.X("지연:O", title="지연 (샘플)", axis=alt.Axis(labelOverlap=True)),
                                                         y=alt.Y("자기상관:Q", scale=alt.Scale(domain=[-1, 1])),
                                                         tooltip=["지연:O", alt.Tooltip("자기상관:Q", format=".3f")]),
            hline(band, ""), hline(-band, "", above=True),
        ).properties(height=240), width="stretch")


# ---------- 시차 상관 · 선후 관계 ----------

def lag_ui(df, seq):
    cols = list(df.columns)
    if len(cols) < 2:
        st.info("숫자 변수가 2개 이상 있어야 합니다.")
        return
    st.caption("X를 시간축에서 밀어 가며 Y와의 상관을 계산합니다. 봉우리의 위치가 'X가 바뀐 뒤 Y가 반응하기까지 걸리는 시간'입니다 "
               "(체류시간, 분석계 지연, dead time).")
    c = st.columns(2)
    x = choice(c[0], "X (먼저 움직일 것 같은 변수)", "cfg_lag_x", cols)
    y = choice(c[1], "Y (반응 변수)", "cfg_lag_y", [col for col in cols if col != x])
    c = st.columns(2)
    max_lag = int(num(c[0], "최대 지연 (샘플)", "cfg_lag_max", 60, min_value=1, max_value=1000, step=1))
    _default("cfg_lag_diff", True)
    diff = flag(c[1], "차분 후 계산 (권장)", "cfg_lag_diff",
                help="값 대신 '변화량'끼리 상관을 봅니다. 두 변수가 모두 천천히 드리프트하면 모든 지연에서 상관이 높게 나와 봉우리가 "
                     "묻히는데, 차분하면 그 착시가 사라집니다.")
    s = _ccf(df[x], df[y], max_lag, diff)
    if s.isna().all():
        st.warning("X와 Y가 함께 있는 구간이 부족해 계산할 수 없습니다.")
        return
    step = _step(df, seq)
    best = int(s.abs().idxmax())
    n = int(pd.concat([df[x], df[y]], axis=1).dropna().shape[0])
    band = 1.96 / np.sqrt(max(n, 1))
    a = s.rename_axis("지연").reset_index()
    base = alt.Chart(a).encode(x=alt.X("지연:Q", title="지연 (샘플) — 양수: X가 먼저"))
    top = base.transform_filter(f"datum['지연'] == {best}")
    st.altair_chart(alt.layer(
        base.mark_bar(color=SERIES[0], width=max(1, 600 // (2 * max_lag + 1))).encode(
            y=alt.Y("상관계수:Q", scale=alt.Scale(domain=[-1, 1])), tooltip=["지연:Q", alt.Tooltip("상관계수:Q", format=".3f")]),
        top.mark_point(size=100, filled=True, color=SERIES[1]).encode(y="상관계수:Q"),
        top.mark_text(dy=-14 if s[best] >= 0 else 16, color=THEME["ink"]).encode(y="상관계수:Q", text=alt.value(f"최대 {best}")),
        hline(band, ""), hline(-band, "", above=True),
    ).properties(height=280), width="stretch")
    r = float(s[best])
    if best > 0:
        msg = f"**X({x})가 Y({y})보다 {_lag_text(best, step)} 앞서** 움직입니다 (r = {r:+.3f})."
    elif best < 0:
        msg = f"**Y({y})가 X({x})보다 {_lag_text(-best, step)} 먼저** 움직입니다 (r = {r:+.3f}). X와 Y를 바꿔 보세요."
    else:
        msg = f"두 변수가 **같은 시점**에 가장 강하게 함께 움직입니다 (r = {r:+.3f})."
    st.markdown(msg)
    st.caption(f"점선 = ±{band:.3f} (관계가 없을 때 우연히 나올 수 있는 범위의 근사값). 최대값이 점선 근처면 뚜렷한 지연 관계가 없는 것입니다."
               + (" 행 순서 데이터라 지연은 샘플(행) 수로만 표시합니다." if seq else ""))

    def use_lag():
        st.session_state.update(cfg_biv_x=x if best >= 0 else y, cfg_biv_y=y if best >= 0 else x, cfg_biv_lag=abs(best))

    st.button(f"이 지연({abs(best)} 샘플)으로 이변량 회귀에 적용", on_click=use_lag,
              help="이변량 회귀 탭의 X·Y·X 지연을 이 결과로 바꿉니다. 소프트센서의 '입력 지연'에도 같은 값을 쓸 수 있습니다.")

    with st.expander("선후 관계 검정 (그랜저 인과)"):
        st.caption("'X의 과거 값이 Y 자신의 과거만으로 하는 예측을 더 좋게 만드는가'를 F검정합니다. 양방향을 모두 보고, "
                   "한쪽만 유의하면 그 방향의 선후 관계가 있다는 근거가 됩니다. 물리적 인과의 증명은 아니며, 두 변수를 함께 움직이는 제3의 "
                   "변수가 있으면 둘 다 유의하게 나옵니다.")
        c = st.columns([3, 1])
        g = int(num(c[0], "검정할 최대 지연 (샘플, 30 이하)", "cfg_granger_lag", min(10, max_lag), min_value=1, max_value=30, step=1))
        if not c[1].toggle("검정 실행", key="granger_on"):  # expander bodies run on every rerun: only on request
            return
        try:
            fwd, back = _granger(df[x], df[y], g, diff), _granger(df[y], df[x], g, diff)
        except ValueError as e:
            st.warning(str(e))
            return
        out = pd.DataFrame({f"{x} → {y}": fwd, f"{y} → {x}": back})
        st.dataframe(out.style.format(_p))
        # Bonferroni over the lags tested: the smallest of g p-values is small by chance alone.
        sig = {k: min(float(v.min()) * g, 1.0) < 0.01 for k, v in out.items()}
        verdict = {(True, False): f"**{x} → {y}** 방향의 선후 관계가 유의합니다.", (False, True): f"**{y} → {x}** 방향만 유의합니다.",
                   (True, True): "양방향 모두 유의합니다 — 되먹임(제어 루프) 또는 공통 원인이 있을 가능성이 큽니다.",
                   (False, False): "어느 방향도 유의하지 않습니다."}[tuple(sig.values())]
        st.markdown(verdict)
        st.caption("판정 기준: 지연별 p값 중 최솟값 × 검정한 지연 수 < 0.01 (여러 번 검정한 효과 보정). "
                   + (f"계산량 때문에 최근 {cr.GRANGER_ROWS:,}행으로 검정했습니다. " if n > cr.GRANGER_ROWS else "")
                   + ("차분한 데이터로 검정했습니다." if diff else "차분하지 않은 데이터는 추세 때문에 거짓 유의가 잘 나옵니다 — 차분을 켜 두세요."))
