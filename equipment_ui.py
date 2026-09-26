"""Streamlit UI for the 설비 KPI and 공정단위 tabs. Widget keys starting with cfg_ are saved in the settings file."""
import json

import numpy as np
import pandas as pd
import streamlit as st

from charts import MUTED, SERIES, bar_chart, hline, shade, trend_chart
from equipment import (FLOW_UNITS, TEMP_UNITS, balance, default_bands, heat_exchanger, intensity, pca_contrib, pca_fit,
                       pca_score, projection, reactor, segments, steady_mask, trend_per_day)

NONE = "(없음)"


# ---------- widget helpers (state lives in st.session_state so the settings file can restore it) ----------

def pick(col, label, key, options):
    opts = [NONE] + list(options)
    if st.session_state.get(key) not in opts:
        st.session_state[key] = NONE
    v = col.selectbox(label, opts, key=key)
    return None if v == NONE else v


def choice(col, label, key, options, radio=False, **kw):
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return (col.radio(label, options, key=key, horizontal=True, **kw) if radio
            else col.selectbox(label, options, key=key, **kw))


def multi(col, label, key, options, default, **kw):
    v = st.session_state.get(key, default)
    keep = [t for t in (v if isinstance(v, list) else default) if t in options]
    if not keep and v:  # every saved tag belongs to another dataset: start from the defaults, not an empty chart
        keep = [t for t in default if t in options]
    st.session_state[key] = keep
    return col.multiselect(label, options, key=key, **kw)


def num(col, label, key, default, **kw):
    try:  # a hand-edited settings file may hold 50 for 50.0 or "4.18"; number_input needs the exact type
        st.session_state[key] = type(default)(st.session_state.get(key, default))
    except (TypeError, ValueError):
        st.session_state[key] = default
    return col.number_input(label, key=key, **kw)


def flag(col, label, key, **kw):
    st.session_state[key] = st.session_state.get(key) is True
    return col.checkbox(label, key=key, **kw)


def table(name, columns, column_config, default_rows):
    """Editable table whose rows are saved in the settings file."""
    st.session_state.setdefault(f"tbl_{name}", default_rows)
    ed = st.data_editor(pd.DataFrame(st.session_state[f"tbl_{name}"], columns=columns), column_config=column_config,
                        num_rows="dynamic", hide_index=True, use_container_width=True,
                        key=f"ed_{name}_{st.session_state.get('tbl_ver', 0)}")
    st.session_state[f"tblout_{name}"] = ed.to_dict("records")
    return ed


def tag_col(label, cols):
    return st.column_config.SelectboxColumn(label, options=list(cols), required=True)


def with_trend(s, name):
    """Series + its fitted linear trend for trend_chart, plus (slope per day, fitted value at the last valid time)."""
    slope, last = trend_per_day(s)
    d = pd.DataFrame({name: s})
    if slope is not None:
        v = s.dropna()
        days = (v.index - v.index[0]).total_seconds() / 86400
        d["추세"] = pd.Series(last + slope * (days - days[-1]), index=v.index)
    return d, slope, last


def projection_text(s, limit, what):
    days = projection(s, limit)
    if days == 0:
        return f"추세값이 이미 {what}을(를) 넘었습니다."
    if days is None:
        return f"현재 추세로는 10년 안에 {what}에 도달하지 않습니다."
    return f"추세대로면 약 **{days:,.0f}일 후** {what}에 도달합니다 ({(s.index[-1] + pd.Timedelta(days=days)):%Y-%m-%d})."


# ---------- settings file ----------

def config_sidebar(columns):
    exp = st.sidebar.expander("설비 설정 파일 (태그 연결·물성값)")
    with exp:
        up = st.file_uploader("설정 불러오기 (JSON)", type=["json"], key="upload_cfg")
        if up and st.session_state.get("loaded_cfg_id") != up.file_id:
            st.session_state["loaded_cfg_id"] = up.file_id
            try:
                cfg = parse_config(up.getvalue().decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                st.error(str(e))
            else:
                st.session_state.update(cfg["widgets"])
                st.session_state.update({f"tbl_{k}": v for k, v in cfg["tables"].items()})
                st.session_state["tbl_ver"] = st.session_state.get("tbl_ver", 0) + 1
                named = [v for k, v in cfg["widgets"].items() if k.endswith("_tag")] + \
                        [t for v in cfg["widgets"].values() if isinstance(v, list) for t in v]
                missing = sorted({t for t in named if t not in columns and t != NONE})
                st.success("설정을 불러왔습니다.")
                if missing:
                    st.warning(f"현재 데이터에 없는 태그는 연결하지 않았습니다: {', '.join(missing)}")
    return exp


def config_save(exp):
    """Call after every tab has rendered, so table edits made in this run are included."""
    widgets = {k: v for k, v in st.session_state.items() if k.startswith("cfg_") and _scalar_or_names(v)}
    tables = {k[7:]: v for k, v in st.session_state.items() if k.startswith("tblout_")}
    exp.download_button("현재 설정 저장", json.dumps({"version": 1, "kind": "equipment_config", "widgets": widgets, "tables": tables},
                                                  ensure_ascii=False, indent=2, default=str),
                        "equipment_config.json", "application/json")


def _scalar_or_names(v):
    return isinstance(v, (str, int, float, bool)) or (isinstance(v, list) and all(isinstance(x, str) for x in v))


def parse_config(text):
    """Plain JSON with scalar widget values and table rows only; nothing in it can execute."""
    try:
        c = json.loads(text)
        ok = (isinstance(c, dict) and c.get("kind") == "equipment_config" and isinstance(c["widgets"], dict)
              and isinstance(c["tables"], dict)
              and all(k.startswith("cfg_") and _scalar_or_names(v) for k, v in c["widgets"].items())
              and all(isinstance(rows, list) and all(isinstance(r, dict) for r in rows) for rows in c["tables"].values())
              and all(isinstance(v, (str, int, float, bool, type(None)))
                      for rows in c["tables"].values() for r in rows for v in r.values()))
    except (ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        raise ValueError("이 프로그램에서 저장한 설비 설정 파일(JSON)이 아니거나 내용이 손상되었습니다.")
    return c


# ---------- 설비 KPI ----------

def kpi_tab(df):
    """Returns KPI columns (prefixed [HX]/[RX]) for every configured equipment."""
    st.caption("설비 유형별로 태그를 연결하면 성능 지표(KPI)를 계산합니다. 계산한 KPI는 다른 탭에서도 일반 태그처럼 분석할 수 있습니다.")
    hx_tab, rx_tab = st.tabs(["열교환기", "반응기"])
    outs = []
    with hx_tab:
        outs.append(hx_ui(df))
    with rx_tab:
        outs.append(rx_ui(df))
    outs = [o for o in outs if o is not None and not o.empty]
    return pd.concat(outs, axis=1) if outs else pd.DataFrame(index=df.index)


def hx_ui(df):
    cols = list(df.columns)
    st.markdown("**① 온도·유량 태그 연결**")
    tags, units = {}, {}
    hot, cold = st.columns(2)
    for side, roles in ((hot, [("Th_in", "고온측 입구 온도", "T"), ("Th_out", "고온측 출구 온도", "T"), ("m_hot", "고온측 유량", "F")]),
                        (cold, [("Tc_in", "저온측 입구 온도", "T"), ("Tc_out", "저온측 출구 온도", "T"), ("m_cold", "저온측 유량", "F")])):
        for role, label, kind in roles:
            a, b = side.columns([3, 1])
            tags[role] = pick(a, label, f"cfg_hx_{role}_tag", cols)
            units[role] = choice(b, "단위", f"cfg_hx_{role}_unit", list(TEMP_UNITS) if kind == "T" else FLOW_UNITS)

    st.markdown("**② 물성값·설비 정보**")
    c = st.columns(4)
    k = {"cp_hot": num(c[0], "고온측 비열 [kJ/kg·K]", "cfg_hx_cp_hot", 4.18, min_value=0.01, help="kcal/kg·℃ 값 × 4.187"),
         "cp_cold": num(c[1], "저온측 비열 [kJ/kg·K]", "cfg_hx_cp_cold", 4.18, min_value=0.01),
         "rho_hot": num(c[2], "고온측 밀도 [kg/m³]", "cfg_hx_rho_hot", 0.0, min_value=0.0, help="유량이 m³/h 일 때만 필요 (0 = 미사용)"),
         "rho_cold": num(c[3], "저온측 밀도 [kg/m³]", "cfg_hx_rho_cold", 0.0, min_value=0.0, help="유량이 m³/h 일 때만 필요 (0 = 미사용)")}
    c = st.columns(4)
    k["area"] = num(c[0], "전열 면적 [m²]", "cfg_hx_area", 0.0, min_value=0.0, help="0 = 미입력 (U 대신 UA만 계산)")
    k["F"] = num(c[1], "LMTD 보정계수 F", "cfg_hx_F", 1.0, min_value=0.1, max_value=1.0, help="순수 향류/병류 = 1, 다패스 쉘앤튜브 = 0.8~0.95")
    k["u_clean"] = num(c[2], "청정 U [W/m²K]", "cfg_hx_u_clean", 0.0, min_value=0.0,
                       help="설계값 또는 세정 직후 값. 0 = 데이터 상위 5% U를 청정 기준으로 사용")
    rf_max = num(c[3], "파울링 한계 [m²K/kW]", "cfg_hx_rf_max", 0.0, min_value=0.0, help="세정 기준. 0 = 미사용")
    c = st.columns(2)
    k["counter"] = choice(c[0], "흐름 방식", "cfg_hx_flow", ["향류", "병류"], radio=True) == "향류"
    k["basis"] = choice(c[1], "U 계산에 쓸 전열량 (양쪽 유량이 있을 때)", "cfg_hx_basis", ["평균", "고온측", "저온측"], radio=True)

    if not all(tags[r] for r in ("Th_in", "Th_out", "Tc_in", "Tc_out")) or not (tags["m_hot"] or tags["m_cold"]):
        st.info("입·출구 온도 4개와 유량(고온측 또는 저온측) 1개 이상을 연결하면 결과가 나옵니다.")
        return None
    try:
        out = heat_exchanger(df, tags, units, {**k, "u_clean": None})
    except ValueError as e:
        st.error(str(e))
        return None
    if "U [W/m²K]" in out:
        uc = k["u_clean"] or float(out["U [W/m²K]"].quantile(0.95))
        out["파울링 저항 [m²K/kW]"] = (1 / out["U [W/m²K]"] - 1 / uc) * 1000

    st.markdown("**③ 결과**")
    ucol = "U [W/m²K]" if "U [W/m²K]" in out else "UA [kW/K]"
    u_tr, u_slope, u_last = with_trend(out[ucol], ucol)
    q_cols = [col for col in out if "전열량" in col]
    m = st.columns(4)
    m[0].metric("평균 전열량 [kW]", f"{out[q_cols].mean(axis=1).mean():,.1f}")
    m[1].metric(f"최근 {ucol} (추세)", f"{u_last:,.1f}" if u_last is not None else "-")
    if u_slope is not None:
        m[2].metric(f"{ucol.split()[0]} 변화율 [%/일]", f"{u_slope / out[ucol].mean() * 100:+.3f}")
    if "열수지 차이 [%]" in out:
        m[3].metric("평균 열수지 차이 [%]", f"{out['열수지 차이 [%]'].mean():+.1f}")

    trend_chart(out[q_cols], dict(zip(q_cols, SERIES)), height=240)
    trend_chart(u_tr, {ucol: SERIES[0], "추세": MUTED} if "추세" in u_tr else {ucol: SERIES[0]}, height=240)
    if "파울링 저항 [m²K/kW]" in out:
        rf = out["파울링 저항 [m²K/kW]"]
        rf_tr, _, _ = with_trend(rf, "파울링 저항 [m²K/kW]")
        trend_chart(rf_tr, {"파울링 저항 [m²K/kW]": SERIES[1], "추세": MUTED} if "추세" in rf_tr else {"파울링 저항 [m²K/kW]": SERIES[1]},
                    extra=hline(rf_max, "세정 기준") if rf_max else (), height=240)
        if rf_max:
            st.caption(projection_text(rf, rf_max, "세정 기준"))
        if not k["u_clean"]:
            st.caption(f"청정 U를 입력하지 않아 데이터 상위 5% 값({uc:,.0f} W/m²K)을 기준으로 계산했습니다.")
    if "열수지 차이 [%]" in out:
        trend_chart(out[["열수지 차이 [%]"]], {"열수지 차이 [%]": SERIES[2]}, extra=[hline(10, "+10%"), hline(-10, "−10%", above=True)], height=200)
        st.caption("고온측과 저온측 전열량은 같아야 합니다. ±10%를 자주 벗어나면 유량계·온도계나 비열 값을 점검하세요.")
    with st.expander("KPI 표로 보기"):
        st.dataframe(out)
    return out.add_prefix("[HX] ")


def rx_ui(df):
    cols = list(df.columns)
    st.markdown("**① 촉매층 온도 연결** (층마다 한 줄, 촉매 비율 = 층별 촉매량 비율)")
    c = st.columns(3)
    tu = choice(c[0], "온도 단위", "cfg_rx_temp_unit", list(TEMP_UNITS))
    wabt_max = num(c[1], "WABT 운전 한계 (EOR) [℃]", "cfg_rx_wabt_max", 0.0, min_value=0.0, help="0 = 미사용")
    t_max = num(c[2], "최고 허용 온도 [℃]", "cfg_rx_t_max", 0.0, min_value=0.0, help="0 = 미사용")
    beds = table("rx_beds", ["입구 온도", "출구 온도", "촉매 비율"],
                 {"입구 온도": tag_col("입구 온도 태그", cols), "출구 온도": tag_col("출구 온도 태그", cols),
                  "촉매 비율": st.column_config.NumberColumn("촉매 비율", min_value=0.0, default=1.0)},
                 [{"입구 온도": None, "출구 온도": None, "촉매 비율": 1.0}])
    with st.expander("② 조성 연결 (선택): 전환율·선택도·수율 계산"):
        st.caption("같은 기준(몰 또는 질량)의 유량과 분율을 연결하세요. 출구 유량이 없으면 입구 유량과 같다고 봅니다.")
        c = st.columns(3)
        comp = {"F_in": pick(c[0], "입구 총유량", "cfg_rx_F_in_tag", cols), "F_out": pick(c[0], "출구 총유량", "cfg_rx_F_out_tag", cols),
                "x_in": pick(c[1], "반응물 입구 분율", "cfg_rx_x_in_tag", cols), "x_out": pick(c[1], "반응물 출구 분율", "cfg_rx_x_out_tag", cols),
                "y_in": pick(c[2], "생성물 입구 분율", "cfg_rx_y_in_tag", cols), "y_out": pick(c[2], "생성물 출구 분율", "cfg_rx_y_out_tag", cols),
                "nu": num(c[2], "화학양론 계수 ν (반응물 1몰당 생성물 몰)", "cfg_rx_nu", 1.0, min_value=0.001)}

    rows = [(r["입구 온도"], r["출구 온도"], float(r["촉매 비율"] or 0)) for r in beds.to_dict("records")
            if r["입구 온도"] in cols and r["출구 온도"] in cols]
    rows = [r for r in rows if r[2] > 0][:7]  # ≤ 7 beds + total fits the 8 categorical slots
    if not rows:
        st.info("촉매층의 입구·출구 온도 태그를 한 줄 이상 입력하면 결과가 나옵니다.")
        return None
    out = reactor(df, rows, tu, comp)

    st.markdown("**③ 결과**")
    w_tr, slope, w_last = with_trend(out["WABT [℃]"], "WABT [℃]")
    days = projection(out["WABT [℃]"], wabt_max)
    m = st.columns(4)
    m[0].metric("현재 WABT [℃] (추세)", f"{w_last:,.1f}" if w_last is not None else "-")
    if slope is not None:
        m[1].metric("WABT 상승 속도 [℃/일]", f"{slope:+.3f}", help="같은 전환율을 유지하려고 온도를 올리는 속도 ≈ 촉매 활성 저하 속도")
    if wabt_max:
        m[2].metric("EOR 도달 예상", "이미 도달" if days == 0 else f"{(out.index[-1] + pd.Timedelta(days=days)):%Y-%m-%d}"
                    if days is not None else "10년 이상", help="현재 추세대로 WABT가 운전 한계에 도달하는 날짜")
    if "전환율 [%]" in out:
        m[3].metric("평균 전환율 [%]", f"{out['전환율 [%]'].mean():.2f}")

    trend_chart(w_tr, {"WABT [℃]": SERIES[0], "추세": MUTED} if "추세" in w_tr else {"WABT [℃]": SERIES[0]},
                extra=hline(wabt_max, "EOR 한계") if wabt_max else (), height=260)
    dt_cols = [col for col in out if "ΔT" in col]
    trend_chart(out[dt_cols], dict(zip(dt_cols, SERIES)), height=220)
    trend_chart(out[["최고 온도 [℃]"]], {"최고 온도 [℃]": SERIES[1]}, extra=hline(t_max, "최고 허용 온도") if t_max else (), height=200)
    if t_max:
        over = (out["최고 온도 [℃]"] > t_max).mean()
        st.caption(f"최고 허용 온도를 넘은 시간 비율: **{over:.1%}**")
    perf = [col for col in ("전환율 [%]", "선택도 [%]", "수율 [%]") if col in out]
    if perf:
        trend_chart(out[perf], dict(zip(perf, SERIES)), height=240)
    with st.expander("KPI 표로 보기"):
        st.dataframe(out)
    return out.add_prefix("[RX] ")


# ---------- 공정단위 ----------

def unit_tab(df):
    bal, kpi, ss, pca = st.tabs(["물질·에너지 수지", "원단위", "정상상태 구간", "이상감지 (PCA)"])
    with bal:
        balance_ui(df)
    with kpi:
        intensity_ui(df)
    with ss:
        steady_ui(df)
    with pca:
        pca_ui(df)


def balance_ui(df):
    cols = list(df.columns)
    st.caption("입력 흐름과 출력 흐름을 적으면 '입력 합 − 출력 합' 불일치를 계산합니다. 계수로 모든 흐름을 같은 단위로 맞추세요 (예: t/h → kg/h 는 1000).")
    ed = table("balance", ["태그", "구분", "계수"],
               {"태그": tag_col("태그", cols), "구분": st.column_config.SelectboxColumn("구분", options=["입력", "출력"], required=True, default="입력"),
                "계수": st.column_config.NumberColumn("계수", default=1.0)},
               [{"태그": None, "구분": "입력", "계수": 1.0}])
    tol = num(st, "허용 불일치 [%]", "cfg_bal_tol", 5.0, min_value=0.0)
    streams = [(r["태그"], r["구분"], float(r["계수"] if r["계수"] is not None else 1)) for r in ed.to_dict("records") if r["태그"] in cols]
    if not ({d for _, d, _ in streams} >= {"입력", "출력"}):
        st.info("입력 흐름과 출력 흐름을 각각 한 개 이상 입력하면 결과가 나옵니다.")
        return
    b = balance(df, streams)
    m = st.columns(3)
    m[0].metric("평균 불일치 [%]", f"{b['불일치 [%]'].mean():+.2f}")
    m[1].metric("불일치 표준편차 [%]", f"{b['불일치 [%]'].std():.2f}")
    m[2].metric("허용 초과 시간 비율", f"{(b['불일치 [%]'].abs() > tol).mean():.1%}")
    trend_chart(b[["입력 합", "출력 합"]], {"입력 합": SERIES[0], "출력 합": SERIES[1]}, height=240)
    trend_chart(b[["불일치 [%]"]], {"불일치 [%]": SERIES[2]}, extra=[hline(tol, f"+{tol:g}%"), hline(-tol, f"−{tol:g}%", above=True)], height=220)
    st.caption("평균이 0에서 일정하게 떨어져 있으면 유량계 교정 오차나 누락된 흐름, 갑자기 벌어지면 계기 고장·누설을 의심하세요.")
    with st.expander("표로 보기"):
        st.dataframe(b)


def intensity_ui(df):
    cols = list(df.columns)
    st.caption("원단위 = 에너지·유틸리티 사용량 ÷ 생산량. 계수로 사용량을 같은 단위로 환산하세요 (예: 스팀 t/h × 0.64 → Gcal/h).")
    ed = table("intensity", ["태그", "계수"],
               {"태그": tag_col("사용량 태그", cols), "계수": st.column_config.NumberColumn("환산 계수", default=1.0)},
               [{"태그": None, "계수": 1.0}])
    c = st.columns(2)
    product = pick(c[0], "생산량 태그", "cfg_int_product_tag", cols)
    period = choice(c[1], "집계 기간", "cfg_int_period", ["일", "주", "월"], radio=True)
    users = [(r["태그"], float(r["계수"] if r["계수"] is not None else 1)) for r in ed.to_dict("records") if r["태그"] in cols]
    if not users or not product:
        st.info("사용량 태그와 생산량 태그를 연결하면 결과가 나옵니다.")
        return
    per = intensity(df, users, product, {"일": "D", "주": "W", "월": "MS"}[period])
    inst = intensity(df, users, product)
    m = st.columns(3)
    m[0].metric("전체 기간 원단위", f"{(sum(df[t] * f for t, f in users).mean() / df[product].mean()):.4g}")
    m[1].metric(f"최근 {period} 원단위", f"{per['원단위'].dropna().iloc[-1]:.4g}" if per["원단위"].notna().any() else "-")
    m[2].metric(f"최저 {period} 원단위", f"{per['원단위'].min():.4g}")
    fmt = {"일": "%Y-%m-%d", "주": "%Y-%m-%d 주", "월": "%Y-%m"}[period]
    bar_chart(per["원단위"].rename(lambda t: t.strftime(fmt)), f"{period}별 원단위", horizontal=False)
    trend_chart(inst.to_frame(), {"원단위": SERIES[0]}, height=220)
    st.caption("기간별 원단위는 '기간 총사용량 ÷ 기간 총생산량' 입니다 (순간 원단위의 단순 평균이 아님). 저부하 운전 구간은 순간 원단위가 크게 튈 수 있습니다.")
    with st.expander("표로 보기"):
        st.dataframe(per)


def steady_ui(df):
    cols = list(df.columns)
    st.caption("선택한 태그가 모두 '창 길이' 동안 허용 변동폭 안에 머무는 구간을 정상상태로 봅니다.")
    c = st.columns([3, 1])
    tags = multi(c[0], "판단 기준 태그 (최대 8개)", "cfg_ss_tags", cols, [], max_selections=8)
    window = int(num(c[1], "창 길이 (샘플 수)", "cfg_ss_window", 30, min_value=3, step=1))
    if not tags:
        st.info("정상상태를 판단할 태그를 고르세요 (예: 원료 유량, 핵심 온도·압력).")
        return
    auto = default_bands(df, tags, window)
    saved = {r["태그"]: r["허용 변동폭"] for rows in (st.session_state.get("tbl_ss_bands", []), st.session_state.get("tblout_ss_bands", []))
             for r in rows if r.get("태그") in tags and r.get("허용 변동폭") is not None}  # loaded file, then current edits
    ed = st.data_editor(pd.DataFrame({"태그": tags, "허용 변동폭": [saved.get(t, round(float(auto[t]), 4)) for t in tags]}),
                        column_config={"태그": st.column_config.TextColumn(disabled=True),
                                       "허용 변동폭": st.column_config.NumberColumn("허용 변동폭 (창 안의 최대−최소)", min_value=0.0)},
                        hide_index=True, use_container_width=True,
                        key=f"ed_ss_{'|'.join(tags)}_{window}_{st.session_state.get('tbl_ver', 0)}")
    st.session_state["tblout_ss_bands"] = ed.to_dict("records")
    bands = {r["태그"]: float(r["허용 변동폭"] or 0) for r in ed.to_dict("records")}
    mask = steady_mask(df, bands, window)
    seg = segments(mask)
    m = st.columns(3)
    m[0].metric("정상상태 비율", f"{mask.mean():.1%}")
    m[1].metric("구간 수", f"{len(seg):,}")
    m[2].metric("가장 긴 구간 (샘플)", f"{seg['길이(샘플)'].max():,}" if len(seg) else "-")
    for i, t in enumerate(tags):
        trend_chart(df[[t]], {t: SERIES[i]}, extra=shade(seg, "정상상태") if len(seg) else (), height=150)
    st.caption("회색 배경 = 정상상태 구간. 허용 변동폭은 처음에 데이터에서 자동 추정한 값이며, 공정 기준에 맞게 고쳐 쓰세요.")
    if len(seg):
        with st.expander(f"구간 목록 ({len(seg)}개)과 구간별 평균"):
            means = [df.loc[s:e, tags].mean() for s, e in zip(seg["시작"], seg["끝"])]
            st.dataframe(pd.concat([seg, pd.DataFrame(means).reset_index(drop=True)], axis=1))
    active = st.session_state.get("steady_filter")
    c = st.columns(2)
    if c[0].button("이 조건으로 다른 탭 분석 (정상상태 구간만)", type="primary", disabled=not len(seg)):
        st.session_state["steady_filter"] = {"bands": bands, "window": window}
        st.rerun()
    if active and c[1].button("정상상태 필터 해제"):
        del st.session_state["steady_filter"]
        st.rerun()


def pca_ui(df):
    cols = list(df.columns)
    st.caption("정상 운전 구간으로 태그들 사이의 관계를 학습한 뒤, 그 관계에서 벗어나는 시점(T², SPE 관리 한계 초과)과 원인 태그를 찾습니다.")
    vars_ = multi(st, "감시할 태그 (3개 이상)", "cfg_pca_vars", cols, cols[:10])
    c = st.columns(4)
    frac = num(c[0], "학습 구간 (앞쪽 %)", "cfg_pca_train_pct", 50, min_value=10, max_value=90, step=5)
    ratio = num(c[1], "설명 분산 기준 [%]", "cfg_pca_ratio", 90, min_value=50, max_value=99, step=1)
    conf = float(choice(c[2], "관리 한계 신뢰수준 [%]", "cfg_pca_conf", ["99", "95", "99.9"]))
    min_len = int(num(c[3], "최소 이상 지속 (샘플)", "cfg_pca_min_len", 3, min_value=1, step=1))
    use_steady = "steady_filter" in st.session_state and flag(
        st, "학습 구간 중 정상상태 구간만 학습에 사용", "cfg_pca_use_steady", help="'정상상태 구간' 탭에서 다른 탭에 적용한 조건을 사용합니다.")
    if len(vars_) < 3:
        st.info("태그를 3개 이상 고르세요.")
        return
    X = df[vars_]
    split = X.index[int(len(X) * frac / 100) - 1]
    train = X.loc[:split]
    if use_steady:
        f = st.session_state["steady_filter"]
        if all(t in df.columns for t in f["bands"]):
            train = train[steady_mask(df, f["bands"], f["window"]).loc[:split]]
    try:
        m = pca_fit(train, ratio / 100, conf)
    except ValueError as e:
        st.error(str(e))
        return
    s = pca_score(m, X)
    test = s.loc[s.index > split]
    alarm = ((s["T²"] > m["t2_lim"]) | (s["SPE"] > m["spe_lim"])) & (s.index > split)
    seg = segments(alarm)
    seg = seg[seg["길이(샘플)"] >= min_len].reset_index(drop=True)

    k = st.columns(4)
    k[0].metric("주성분 수", m["k"], help=f"학습 데이터 변동의 {m['explained']:.1%} 설명")
    k[1].metric("감시 구간 T² 초과율", f"{(test['T²'] > m['t2_lim']).mean():.1%}", help=f"정상이면 약 {100 - conf:g}%")
    k[2].metric("감시 구간 SPE 초과율", f"{(test['SPE'] > m['spe_lim']).mean():.1%}", help=f"정상이면 약 {100 - conf:g}%")
    k[3].metric(f"이상 구간 ({min_len}샘플 이상)", f"{len(seg):,}")
    log = flag(st, "로그 스케일", "cfg_pca_log", help="큰 이상 때문에 작은 변화가 안 보일 때")
    train_band = shade(pd.DataFrame({"시작": [X.index[0]], "끝": [split]}), "학습 구간")
    trend_chart(s[["T²"]], {"T²": SERIES[0]}, extra=[train_band, hline(m["t2_lim"], f"T² 한계 ({conf:g}%)")], height=220, log=log)
    trend_chart(s[["SPE"]], {"SPE": SERIES[1]}, extra=[train_band, hline(m["spe_lim"], f"SPE 한계 ({conf:g}%)")], height=220, log=log)
    st.caption("회색 배경 = 학습 구간. **T²** 초과: 태그들이 평소 관계는 지키면서 평소보다 멀리 움직임 (운전 조건 변화). "
               "**SPE** 초과: 태그들 사이의 평소 관계가 깨짐 (계기 이상, 새로운 현상).")

    if len(seg):
        rows = []
        for sgi in seg.itertuples():
            part = s.loc[sgi.시작:sgi.끝]
            stat = "SPE" if (part["SPE"] > m["spe_lim"]).any() else "T²"
            peak = part[stat].idxmax()
            cb = pca_contrib(m, X.loc[peak])[f"{stat} 기여"]
            rows.append({"최대 시점": peak, "기준": stat, "최대값/한계": round(part[stat].max() / m["spe_lim" if stat == "SPE" else "t2_lim"], 2),
                         "주요 원인 태그": cb.abs().idxmax()})
        seg = pd.concat([seg, pd.DataFrame(rows)], axis=1).sort_values("최대값/한계", ascending=False).reset_index(drop=True)
        st.markdown("**이상 구간** (심한 순)")
        st.dataframe(seg.head(50), hide_index=True)
        options = [f"{t:%Y-%m-%d %H:%M}  ({b}, {v:.1f}배)" for t, b, v in zip(seg["최대 시점"], seg["기준"], seg["최대값/한계"])]
        idx = st.selectbox("기여도를 볼 이상 구간", range(len(options)), format_func=lambda i: options[i])
        t, stat = seg.loc[idx, "최대 시점"], seg.loc[idx, "기준"]
    else:
        valid = test["SPE"].dropna() if test["SPE"].notna().any() else s["SPE"].dropna()
        if valid.empty:
            st.warning("감시 구간에 모든 태그 값이 있는 시점이 없어 계산할 수 없습니다. 결측이 많은 태그를 빼 보세요.")
            return
        st.success("감시 구간에서 관리 한계를 넘은 이상 구간이 없습니다.")
        t, stat = valid.idxmax(), "SPE"
    cb = pca_contrib(m, X.loc[t])
    stat = choice(st, "기여도 기준", "cfg_pca_contrib_stat", ["SPE", "T²"], radio=True) if len(seg) == 0 else stat
    part = cb[f"{stat} 기여"].sort_values(key=abs, ascending=False)
    st.markdown(f"**{t:%Y-%m-%d %H:%M} 의 {stat} 기여도** (막대가 클수록 이상의 원인일 가능성이 큼)")
    bar_chart(part, f"{stat} 기여")
    st.caption("기여도가 큰 태그의 트렌드를 '트렌드 · 통계' 탭에서 함께 확인하세요. 기여도는 원인 '후보'이며, 여러 태그에 퍼질 수 있습니다.")
    with st.expander("기여도 표"):
        st.dataframe(cb.round(4))
