import io
import json
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from analysis import (LAB, apply_model, attach_lab, export_model, fit_soft_sensor, lag_scan, load, parse_model, residual_alarm,
                      status_columns, to_sequence, to_timeseries)
import datasets
from charts import MUTED, SERIES, THEME, TIME_FMT, apply_theme, bar_chart, corr_heatmap, trend_chart
from correlation import fit_quality
from correlation_ui import cached_vif, corr_tab, residual_report, vif_table
from equipment import segments, steady_mask
from equipment_ui import (KEEP, choice, config_save, config_sidebar, kpi_tab, multi, num, restore, set_state, store,
                          unit_tab)
from guide import guide_box, tab_intro
from ml import MODELS as ML_MODELS

st.set_page_config(page_title="공정 설비 데이터 분석", layout="wide")
apply_theme()


# In-memory only; ttl/max_entries bound how long uploaded plant data lingers in the server process.
@st.cache_data(show_spinner=False, ttl="1h", max_entries=3)
def read(data: bytes, name: str):
    return load(io.BytesIO(data), name)


@st.cache_data(show_spinner="시간 컬럼 해석 중…", ttl="1h", max_entries=3)
def parse(data: bytes, name: str, time_col: str):
    return to_timeseries(read(data, name), time_col)


@st.cache_data(show_spinner=False, ttl="1h", max_entries=3)
def parse_rows(data: bytes, name: str, order_col):
    return to_sequence(read(data, name), order_col)


@st.cache_data(show_spinner="예제 데이터 불러오는 중…")
def example(kind, arg=None):
    if kind == "te":
        return datasets.tennessee_eastman(arg)
    if kind == "real":
        return datasets.petrobras_real(), None
    return datasets.chelo_dataset(arg), None  # network: only reached after the user presses the download button


@st.cache_data(show_spinner="ML 모델 학습 중…", ttl="1h", max_entries=5)
def fit_ml(data, target, inputs, lag, frac, model, trials):
    from ml import fit_ml_soft_sensor  # heavy import (scikit-learn, optuna, BibMon) only when an ML model is used
    return fit_ml_soft_sensor(data, target, list(inputs), lag, frac, model, trials)


def example_sidebar():
    names = {"te": "Tennessee Eastman (벤치마크 · 고장 21종)", "real": "Petrobras 실제 공정 데이터",
             **{f"chelo:{k}": v for k, v in datasets.CHELO.items()}}
    kind = st.sidebar.selectbox("예제 데이터", list(names), format_func=names.get)
    if kind == "te":
        fault = st.sidebar.selectbox("시나리오", list(datasets.TE_FAULTS), index=1,
                                     format_func=lambda f: f"{f:02d} · {datasets.TE_FAULTS[f]}")
        df, start = example("te", fault)
        if start is not None:
            pct = df.index.get_loc(start) / len(df)
            st.sidebar.info(f"고장 시작: {start:%m-%d %H:%M} (전체의 {pct:.0%} 지점). 그 전은 정상 운전이므로 "
                            f"PCA·소프트센서 학습 구간을 {int(pct * 20) * 5}% 이하로 두면 고장 전 데이터로만 학습합니다.")
        return df, False
    if kind == "real":
        st.sidebar.caption("출처: BibMon (Petrobras) · CC BY 4.0. 태그 이름은 익명화되어 있습니다.")
        return example("real")[0], False
    name = kind.split(":")[1]
    loaded = st.session_state.setdefault("chelo_loaded", set())
    if name not in loaded:
        st.sidebar.warning("이 데이터는 인터넷에서 내려받습니다 (chelo, ~/.chelo 에 저장)."
                           + (" Kaggle 계정 인증이 필요합니다 (매뉴얼 9.7)." if name == "coal" else ""))
        if not st.sidebar.button("내려받기"):
            st.info("왼쪽의 **내려받기** 를 누르면 데이터를 가져옵니다.")
            st.stop()
    try:
        df = example("chelo", name)[0]
    except Exception as e:  # network / credentials: explain instead of a traceback
        st.error(f"내려받지 못했습니다: {type(e).__name__}: {e}")
        st.stop()
    loaded.add(name)
    st.sidebar.caption("원본에 시각이 없어(CSTR) 또는 chelo가 날짜와 결측 행을 지워서(발전소) 순서대로 가상의 시각을 붙였습니다. "
                       "행 순서만 실제이고 시간 간격은 실제가 아니므로 리샘플링·입력 지연·추세 예측 결과는 해석하지 마세요.")
    return df, True


def soft_summary(r, target, is_ml):
    """One plain-language paragraph under the soft-sensor scores."""
    m = r["metrics"]
    val, tr = m["검증 R²"], m["학습 R²"]
    out = [f"**결과 요약** — 검증 R² {val:.2f}: **{fit_quality(val)}**"
           + (f" ({target} 변동의 {val:.0%}를 설명)." if val > 0 else ".")]
    if val < 0.5 <= tr:
        out.append("학습 구간에서는 맞는데 검증 구간에서 틀립니다 → 검증 기간의 운전 조건이 학습 기간과 다르거나(운전 구간별 비교로 확인), "
                   "입력 지연이 맞지 않습니다(최적 지연 자동 탐색).")
    elif tr - val > 0.2:
        out.append("학습 R²가 검증 R²보다 훨씬 높습니다 → 과적합 의심: 입력 수를 줄이거나 PLS·선형 모델과 비교하세요.")
    if is_ml:
        out.append(f"가장 많이 쓰인 입력: **{r['importance'].idxmax()}** (순열 중요도 1위).")
    else:
        top = r["coef"].abs().idxmax()
        out.append(f"영향이 가장 큰 입력: **{top}** — 1 오르면 {target}이(가) {r['raw_coef'][top]:+.4g} 변함 "
                   f"({'같은' if r['coef'][top] > 0 else '반대'} 방향).")
    return " ".join(out)


def equation(target, intercept, coef, lag):
    terms = "\n".join(f"    {c:+.6g} × {n}" for n, c in coef.items())
    st.code(f"{target} = {intercept:.6g}\n{terms}" + (f"\n\n※ 입력은 모두 {lag} 샘플 이전 값" if lag else ""), language=None)


def show_apply():
    tab_intro("저장해 둔 모델을 새 운전데이터에 적용할 때", "'모델 만들기'에서 저장한 모델 파일(JSON)",
              "적용 R² (실측값이 있을 때), 예측값 추이")
    mfile = st.file_uploader("모델 파일 (JSON)", type=["json"], key="model_file")
    if mfile:  # kept in the session: the uploader forgets its file whenever another main tab is opened
        st.session_state["_model_json"] = (mfile.name, mfile.getvalue())
    elif "_model_json" in st.session_state:
        c1, c2 = st.columns([4, 1])
        c1.caption(f"불러온 모델: **{st.session_state['_model_json'][0]}** — 다른 모델을 쓰려면 새 파일을 올리세요.")
        if c2.button("모델 지우기"):
            del st.session_state["_model_json"]
            st.rerun()
    if "_model_json" in st.session_state:
        try:
            model = parse_model(st.session_state["_model_json"][1].decode("utf-8"))
            out, met = apply_model(df, model)
        except (ValueError, UnicodeDecodeError) as e:
            st.error(str(e))
            return

        info = {"예측 대상": model["target"], "모델": model.get("method", "-"), "입력 지연": f"{model['lag']} 샘플",
                "학습 기간": " ~ ".join(model.get("train_period", ["-"])), "학습 시 리샘플링": model.get("resample", "-"),
                "학습 시 검증 R²": f"{model.get('metrics', {}).get('검증 R²', float('nan')):.4f}", "저장 시각": model.get("created", "-")}
        st.dataframe(pd.Series(info, name="모델 정보"))
        if model.get("resample", rule) != rule:
            st.warning(f"모델은 '{model['resample']}' 간격 데이터로 학습했는데 지금은 '{rule}' 입니다. "
                       "입력 지연(샘플 수)의 시간 길이가 달라지므로 사이드바 리샘플링을 모델과 맞추세요.")
        if out.empty:
            st.error("입력 태그에 값이 있는 구간이 없어 예측할 수 없습니다.")
            return

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


st.title("공정 설비 데이터 분석")
st.caption("v3.2 — 상관·회귀 분석 중심 · 운전데이터 + 분석데이터 · 소프트센서 · 공정단위 · 설비 KPI")
guide_box()

NO_TIME = "(없음 · 행 순서대로)"
seq = False  # True: no timestamps, rows sit on a pseudo 1-minute axis (analysis.to_sequence)
source = st.sidebar.radio("데이터", ["파일 업로드", "예제 데이터"], horizontal=True)
if source == "예제 데이터":
    df, seq = example_sidebar()
else:
    file = st.sidebar.file_uploader("① 운전데이터 (DCS/PI · CSV / Excel)", type=["csv", "xlsx"])
    if not file:
        st.info("왼쪽에서 **① 운전데이터** 파일을 올리세요 (첫 행 헤더, 시간 컬럼 1개 + 태그별 컬럼). 분석값(LIMS)이 따로 있으면 이어서 "
                "**② 분석데이터** 에 올립니다. 시간 컬럼이 없는 데이터(실험·시료)도 행 순서대로 분석할 수 있습니다. "
                "데이터가 없으면 **예제 데이터** 로 먼저 써 보고, 처음이라면 위의 **📘 분석 가이드** 를 펼쳐 보세요.")
        st.stop()
    data = file.getvalue()
    try:
        raw = read(data, file.name)
    except Exception as e:  # malformed upload: show why instead of a traceback
        st.error(f"파일을 읽을 수 없습니다: {e}")
        st.stop()
    time_col = st.sidebar.selectbox("시간 컬럼", [NO_TIME, *raw.columns], index=1 if len(raw.columns) else 0,
                                    help="시간이 없는 데이터는 '(없음 · 행 순서대로)'를 고르거나 시료번호 같은 순서 컬럼을 고르세요.")
    if time_col == NO_TIME:
        df, seq = parse_rows(data, file.name, None), True
    else:
        df = parse(data, file.name, time_col)
        if df.empty and pd.to_numeric(raw[time_col], errors="coerce").notna().mean() > 0.5:
            df, seq = parse_rows(data, file.name, time_col), True  # 시료번호, batch no.: an order, not a time
            st.sidebar.info(f"'{time_col}' 은(는) 시간이 아닌 숫자라서 **순서 번호**로 사용합니다 (그 순서로 정렬, 분석 태그에서는 제외).")
    if df.empty:
        st.error(f"'{time_col}' 컬럼을 시간으로 해석할 수 없거나 숫자 태그가 없습니다." if time_col != NO_TIME else "숫자 태그가 없습니다.")
        st.stop()
    converted = {c: n for c, n in status_columns(raw).items() if n in df.columns}
    dropped = [c for c in raw.columns if c != time_col and c not in df.columns and c not in converted]
    if converted:
        st.sidebar.caption("상태 컬럼을 1/0으로 바꿔 분석에 포함: " + ", ".join(converted.values()))
    if dropped:  # say so instead of silently losing them
        st.sidebar.caption(f"숫자 값이 없어 분석에서 제외한 컬럼: {', '.join(map(str, dropped))}")
    lab_file = st.sidebar.file_uploader(
        "② 분석데이터 (선택 · LIMS 분석값)", type=["csv", "xlsx"], key="lab_file",
        help="분석시간 1열 + 항목별 열 (예: 분석시간, Component 1, Component 2 …). 각 분석값은 분석시간에 가장 가까운 운전데이터 "
             "시점에 붙고, 이름 앞에 [분석] 이 붙습니다.")
    if lab_file and seq:
        st.sidebar.warning("시간 컬럼이 없는 운전데이터에는 분석데이터를 붙일 수 없습니다.")
    elif lab_file:
        lab_data = lab_file.getvalue()
        try:
            lab_raw = read(lab_data, lab_file.name)
        except Exception as e:  # malformed upload: show why instead of a traceback
            st.sidebar.error(f"분석데이터 파일을 읽을 수 없습니다: {e}")
            lab_raw = None
        if lab_raw is not None:
            lab_cols = list(lab_raw.columns)
            if st.session_state.get("lab_time") not in lab_cols:  # a new file: guess the time column from its name
                st.session_state["lab_time"] = next((c for c in lab_cols if any(w in str(c).lower() for w in
                                                     ("분석시간", "시간", "시각", "일시", "time", "date"))), lab_cols[0])
            lab_time = st.sidebar.selectbox("분석데이터 시간 컬럼", lab_cols, key="lab_time")
            lab = parse(lab_data, lab_file.name, lab_time)
            if lab.empty:
                st.sidebar.error(f"'{lab_time}' 컬럼을 시간으로 해석할 수 없거나 숫자 항목이 없습니다.")
            else:
                df, summ = attach_lab(df, lab)
                note = (f"분석데이터 {summ['분석 건수']:,}건 중 **{summ['연결']:,}건** 을 운전데이터 시간축에 붙였습니다 "
                        f"(항목 {lab.shape[1]}개, 이름 앞에 `[분석]`).")
                if summ["연결 안 됨"]:
                    note += f" {summ['연결 안 됨']:,}건은 운전데이터 기간 밖이거나 기록 공백이라 뺐습니다."
                if summ["같은 시점에 겹쳐 평균"]:
                    note += f" 같은 시점에 겹친 {summ['같은 시점에 겹쳐 평균']:,}건은 평균했습니다."
                st.sidebar.success(note)
                st.sidebar.caption("분석시간은 실제 공정 상태보다 늦습니다. 소프트센서의 **최적 지연 자동 탐색** 이나 상관분석의 "
                                   "**시차 상관** 으로 지연을 찾으세요.")
    if seq:
        st.sidebar.caption("시간 없이 **행 순서**로 분석합니다. 그래프의 시간축은 가상 시각입니다: n번째 행 = 2000-01-01 00:00 + n분. "
                           "리샘플링은 행 묶음 평균, 기간은 행 범위로 동작하며, 추세 예측(일 단위)은 의미가 없습니다.")

labs = [c for c in df.columns if str(c).startswith(LAB)]
if labs and st.session_state.get("_labs_seen") != labs:  # new analysis items: make them the default focus and target
    st.session_state["_labs_seen"] = labs
    picked = st.session_state.get("cfg_corr_vars", st.session_state.get(KEEP + "cfg_corr_vars"))
    if picked:  # a matrix selection made before the analysis file was added would leave the new items out
        set_state(cfg_corr_vars=[*picked, *(c for c in labs if c not in picked)])
    set_state(soft_target=labs[0], cfg_corr_focus=labs[0])

rule = st.sidebar.selectbox("리샘플링(평균)", ["원본", "1min", "10min", "1h", "1D"])
if rule != "원본":
    df = df.resample(rule).mean()

lo, hi = df.index.min().to_pydatetime(), df.index.max().to_pydatetime()
if lo < hi:
    start, end = st.sidebar.slider("기간", lo, hi, (lo, hi), timedelta(minutes=1), "YYYY-MM-DD HH:mm")
    df = df.loc[start:end]
cfg_box = config_sidebar(list(df.columns))
rel, trend, soft, unit, kpi = st.tabs(["상관분석", "트렌드 · 통계", "소프트센서", "공정단위", "설비 KPI"],
                                     key="main_tab", on_change="rerun")  # only the open tab runs (see equipment_ui KEEP)

with kpi:
    df = df.join(kpi_tab(df))
df_all = df

flt = st.session_state.get("steady_filter")
if flt and all(t in df.columns for t in flt["bands"]):
    mask = steady_mask(df, flt["bands"], flt["window"])
    df = df.copy()
    df.loc[~mask] = float("nan")  # blank, don't drop: lags count samples on the original time grid
    st.sidebar.info(f"정상상태 구간만 분석 중 ({mask.mean():.0%}). 해제: 공정단위 → 정상상태 구간 탭")
st.sidebar.caption(f"{len(df):,}행 · 태그 {df.shape[1]}개")

if rel.open:
    with rel:
        corr_tab(df, seq)

if trend.open:
    with trend:
        tab_intro("데이터를 처음 받았을 때, 이벤트 전후 흐름을 볼 때", "태그 (단위가 다르면 '태그별 분리')",
                  "빈 구간·값 고착·계단 변화, 기초통계의 count·std")
        tags = multi(st, "태그 (최대 20개 · 변수가 더 많으면 '상관분석' 탭)", "trend_tags", list(df.columns), list(df.columns[:4]), max_selections=20)
        if tags:
            # A tag keeps its color while selected, so adding/removing others never repaints it. Overlaid lines can only
            # be told apart by 8 validated colors; beyond that each tag gets its own chart (slot colors repeat there).
            prev, slots = st.session_state.get("slots", {}), {}
            for t in tags:
                if t in prev and prev[t] not in slots.values():
                    slots[t] = prev[t]
            free = [i for i in range(8) if i not in slots.values()]
            slots |= {t: free.pop(0) if free else i % 8 for i, t in enumerate(tags) if t not in slots}
            st.session_state["slots"] = slots

            view = df[tags]
            mode = choice(st, "보기", "trend_mode", ["겹쳐 보기", "태그별 분리", "정규화(z-score)"], radio=True,
                          help="단위가 다른 태그는 '태그별 분리'(각자 세로축) 또는 '정규화'로 비교하세요.")
            if mode != "태그별 분리" and len(tags) > 8:
                st.info(f"한 그래프에서 색으로 구분할 수 있는 태그는 8개까지라 {len(tags)}개는 태그별로 나눠 그립니다.")
                mode = "태그별 분리"
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
            with c2:
                corr_heatmap(corr, tags)
            with c2.expander("상관계수 표"):
                st.dataframe(corr.round(2))
            c2.caption("피어슨 상관만 표시합니다. 곡선 관계·p값·회귀·잔차·시차 분석은 **상관분석** 탭에서 하세요.")

if soft.open:
    with soft:
        make, use = st.tabs(["모델 만들기", "모델 적용"])
        with make:
            tab_intro("분석값(품질)을 운전 변수로 실시간 추정하고 싶을 때", "예측 대상 = 분석값, 입력 = 상관분석에서 찾은 변수 3~6개",
                      "검증 R²와 결과 요약, 실측 vs 예측 그래프")
            raw_cols = [c for c in df.columns if not c.startswith(("[HX] ", "[RX] "))]
            if "soft_target" not in st.session_state and KEEP + "soft_target" not in st.session_state:
                st.session_state["soft_target"] = labs[0] if labs else raw_cols[-1]
            target = choice(st, "예측 대상 (품질 변수: 순도, 조성 등)", "soft_target", list(df.columns))
            inputs = multi(st, "입력 변수 (온도, 압력, 유량 등)", "soft_inputs", [c for c in df.columns if c != target], [])
            c1, c2 = st.columns([3, 1])
            method = choice(c1, "모델", "soft_method", ["OLS (선형회귀)", "PLS (부분최소제곱)", *ML_MODELS], radio=True,
                            help="선형(OLS·PLS): 빠르고 DCS용 수식을 얻음. ML(랜덤포레스트·그래디언트 부스팅·신경망): 비선형 공정에서 더 정확, "
                                   "수식·저장 없음 (매뉴얼 10.22).")
            is_ml = method in ML_MODELS
            k = (num(c2, "PLS 성분 수", "soft_k", 2, min_value=1, max_value=len(inputs))
                 if method.startswith("PLS") and inputs else None)
            trials = int(num(c2, "자동 튜닝 횟수 (0 = 끔)", "soft_trials", 0, min_value=0, max_value=100,
                             help="Optuna로 하이퍼파라미터를 이 횟수만큼 시험 (횟수만큼 느려짐)")) if is_ml else 0
            c1, c2 = st.columns(2)
            lag = num(c1, "입력 지연 (샘플 수) — 분석계 지연·dead time 보정", "lag", 0, min_value=0, max_value=10_000)
            restore("soft_frac")
            st.session_state.setdefault("soft_frac", 0.7)
            frac = store("soft_frac", c2.slider("학습 비율 (앞쪽 시간 구간으로 학습, 뒤쪽으로 검증)", 0.3, 0.9, key="soft_frac"))

            if inputs:
                with st.expander("최적 지연 자동 탐색" + (" (빠른 선형 모델로 탐색한 뒤 ML 모델에 적용)" if is_ml else "")):
                    max_lag = num(st, "탐색할 최대 지연 (샘플 수)", "soft_max_lag", 60, min_value=1, max_value=2000)
                    key = (target, tuple(inputs), frac, k, max_lag, int(pd.util.hash_pandas_object(df[[target, *inputs]]).sum()))

                    def run_scan():
                        scan = lag_scan(df, target, inputs, max_lag, frac, k)
                        st.session_state["scan"] = (key, scan)
                        if not scan.empty:
                            set_state(lag=int(scan.idxmax()))

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
                                top.mark_text(dy=-14, color=THEME["ink"]).encode(text=alt.value(f"최적 {best}")),
                            ).properties(height=220), width="stretch")

                try:
                    r = (fit_ml(df[[target, *inputs]], target, tuple(inputs), int(lag), frac, method, trials) if is_ml
                         else fit_soft_sensor(df, target, inputs, lag, frac, k))
                except ValueError as e:
                    st.error(str(e))
                    r = None  # not st.stop(): the 모델 적용 sub-tab must still render
                except Exception as e:  # third-party ML stack (BibMon/scikit-learn/Optuna): report instead of crashing the page
                    if not is_ml:
                        raise
                    st.error(f"ML 모델 학습 중 오류가 났습니다: {type(e).__name__}: {e}")
                    r = None

            if inputs and r:
                if k:
                    with st.expander("성분 수별 검증 R² (성분 수 선택 참고)"):
                        st.dataframe(pd.Series(
                            {n: (r if n == k else fit_soft_sensor(df, target, inputs, lag, frac, n))["metrics"]["검증 R²"]
                             for n in range(1, len(inputs) + 1)}, name="검증 R²").rename_axis("성분 수").round(4))

                for col, (name, val) in zip(st.columns(4), r["metrics"].items()):
                    col.metric(name, f"{val:.4f}")
                st.markdown(soft_summary(r, target, is_ml))

                res = r["result"]
                split = alt.Chart(pd.DataFrame({"_t": [r["split"]]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="_t:T")
                limit, alarm = residual_alarm(res, limit=r.get("alarm_limit"))
                alarm = alarm & (res["구분"] == "검증")
                marks = alt.Chart(res[alarm].rename_axis("_t").reset_index()).mark_point(
                    shape="triangle-down", size=90, filled=True, color="#d03b3b").encode(  # status 'critical' + shape + label below
                    x="_t:T", y="실측:Q", tooltip=[alt.Tooltip("_t:T", title="잔차 알람", format=TIME_FMT),
                                                 alt.Tooltip("실측:Q", format=".5g"), alt.Tooltip("예측:Q", format=".5g")])
                st.subheader("실측 vs 예측 (시간)")
                trend_chart(res[["실측", "예측"]], {"실측": SERIES[0], "예측": SERIES[1]}, extra=[split, marks])
                n_alarm, n_val = int(alarm.sum()), int((res["구분"] == "검증").sum())
                basis = "학습에 쓰지 않은 데이터로 잰 잔차의 99% 수준" if is_ml else "학습 잔차의 99% 수준"
                st.caption(f"점선 = 학습/검증 경계 ({r['split']:%Y-%m-%d %H:%M}), 오른쪽이 검증 구간.  "
                           f"▼ 잔차 알람 = |실측 − 예측| > {limit:.4g} ({basis}): 검증 구간 {n_alarm}회 ({n_alarm / max(n_val, 1):.1%}). "
                           "정상이면 약 1%이며, 몰려서 나오면 분석계·계기 이상이나 모델이 모르는 운전 변화입니다.")
                if n_alarm:
                    seg = segments(alarm)
                    with st.expander(f"잔차 알람 구간 ({len(seg)}개)"):
                        st.dataframe(seg, hide_index=True)

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
                ).properties(height=360), width="stretch")
                c1.caption("대각선에 가까울수록 정확. 검증 점이 한쪽으로 치우치면 편향(bias)이 있다는 뜻입니다.")

                c2.subheader("변수 영향도")
                if is_ml:
                    with c2:
                        bar_chart(r["importance"], "순열 중요도 (R² 감소량)")
                    c2.caption("순열 중요도: 그 입력의 값을 무작위로 섞었을 때 학습 데이터 정확도(R²)가 떨어지는 정도. 0 근처면 거의 쓰이지 않는 입력.")
                    if r["tuned"]:
                        c2.markdown("**자동 튜닝 결과**")
                        c2.dataframe(pd.Series(r["tuned"], name="값").astype(str))
                else:
                    c2.dataframe(pd.concat([r["coef"], r["raw_coef"]], axis=1).round(6))
                    c2.caption("표준화 계수: 입력 1 표준편차 변화당 예측 변화 (영향 크기 비교용)")

                with st.expander("잔차 진단 (검증 구간) — 모델이 놓친 패턴이 남아 있는지"):
                    val = res[res["구분"] == "검증"]
                    residual_report(val["실측"] - val["예측"], val["예측"])
                    st.caption("잔차 = 실측 − 예측. 자기상관이 크면 입력 지연·누락 변수를, 잔차 시간 추이에 계단이나 기울기가 있으면 "
                               "운전 조건 변화·계기 드리프트를 의심하고 모델을 다시 학습하세요.")
                if len(inputs) >= 2:
                    with st.expander("입력 변수 다중공선성 (VIF)"):
                        try:
                            v = cached_vif(df[inputs])
                        except ValueError as e:
                            st.warning(str(e))
                        else:
                            st.dataframe(vif_table(v))
                            if (v > 10).any():
                                st.warning(f"VIF > 10: {', '.join(v[v > 10].index)} — 서로 거의 같은 정보를 담은 입력입니다. "
                                           + ("OLS 계수의 크기·부호를 믿기 어려우니 PLS를 쓰거나 하나만 남기세요." if method.startswith("OLS")
                                              else "PLS·ML은 예측에는 대체로 문제없지만, 개별 입력의 영향도 해석은 조심하세요."))
                            else:
                                st.caption("모든 입력의 VIF가 10 이하입니다 (서로 겹치는 정보가 적음).")

                if is_ml:
                    st.info("ML 모델은 수식으로 표현되지 않아 DCS 계산 블록에 옮길 수 없고, **모델 저장** 도 제공하지 않습니다. "
                            "ML 모델을 파일로 저장하는 방식은 불러올 때 코드가 실행될 수 있어 안전하지 않기 때문입니다 (매뉴얼 9.2). "
                            "DCS 적용이 필요하면 같은 입력으로 OLS/PLS 모델을 만들어 비교하세요.")
                else:
                    st.subheader("모델식 (원래 단위, DCS 적용용)")
                    equation(target, r["raw_intercept"], r["raw_coef"], lag)

                table = res.assign(잔차_알람=alarm)
                with st.expander("예측 결과 표"):
                    st.dataframe(table)
                c1, c2 = st.columns(2)
                c1.download_button("예측 결과 CSV 다운로드", table.to_csv().encode("utf-8-sig"),
                                   f"softsensor_{target}.csv", "text/csv")
                if not is_ml:
                    c2.download_button("모델 저장 (JSON)", json.dumps(export_model(r, target, lag, k, rule), ensure_ascii=False, indent=2),
                                       f"softsensor_{target}.json", "application/json", help="소프트센서 탭의 '모델 적용' 하위 탭에서 새 데이터에 다시 적용할 수 있습니다.")
        with use:
            show_apply()

if unit.open:
    with unit:
        unit_tab(df_all)

config_save(cfg_box)
