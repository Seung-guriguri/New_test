"""Equipment and process-unit performance calculations (pure pandas/numpy, no UI)."""
import warnings

import numpy as np
import pandas as pd

# Set once at import (a catch_warnings() block inside a call would race between Streamlit session threads).
warnings.filterwarnings("ignore", message="New behaviour in v1.1.5", module="ruptures")

TEMP_UNITS = {"℃": lambda x: x, "K": lambda x: x - 273.15, "℉": lambda x: (x - 32) / 1.8}
FLOW_UNITS = ["kg/h", "t/h", "kg/s", "m³/h"]  # m³/h needs a density
_KG_S = {"kg/h": 1 / 3600, "t/h": 1000 / 3600, "kg/s": 1.0}


def to_celsius(s, unit):
    return TEMP_UNITS[unit](s)


def to_kg_s(s, unit, rho=None):
    if unit == "m³/h":
        if not rho:
            raise ValueError("m³/h 유량은 밀도(kg/m³)를 입력해야 질량유량으로 바꿀 수 있습니다.")
        return s * rho / 3600
    return s * _KG_S[unit]


def lmtd(dt1, dt2):
    """Log-mean temperature difference; NaN where a terminal difference is ≤ 0 (temperature cross or bad data)."""
    dt1, dt2 = dt1.where(dt1 > 0), dt2.where(dt2 > 0)
    ratio = dt1 / dt2
    return ((dt1 - dt2) / np.log(ratio)).where((ratio - 1).abs() > 1e-6, (dt1 + dt2) / 2)


def heat_exchanger(df, tags, units, c):
    """
    tags:  Th_in, Th_out, Tc_in, Tc_out (required), m_hot, m_cold (at least one) → column names
    units: same roles → unit string
    c:     cp_hot, cp_cold [kJ/kg·K], rho_hot, rho_cold [kg/m³], area [m²], F, counter (bool),
           basis ('고온측'|'저온측'|'평균'), u_clean [W/m²·K] (optional)
    """
    T = {r: to_celsius(df[tags[r]], units.get(r, "℃")) for r in ("Th_in", "Th_out", "Tc_in", "Tc_out")}
    out = pd.DataFrame(index=df.index)
    if tags.get("m_hot"):
        out["고온측 전열량 [kW]"] = to_kg_s(df[tags["m_hot"]], units["m_hot"], c.get("rho_hot")) * c["cp_hot"] * (T["Th_in"] - T["Th_out"])
    if tags.get("m_cold"):
        out["저온측 전열량 [kW]"] = to_kg_s(df[tags["m_cold"]], units["m_cold"], c.get("rho_cold")) * c["cp_cold"] * (T["Tc_out"] - T["Tc_in"])
    if out.empty:
        raise ValueError("고온측 또는 저온측 유량 중 하나는 연결해야 전열량을 계산할 수 있습니다.")

    q_cols = list(out.columns)
    if len(q_cols) == 2:
        qh, qc = out[q_cols[0]], out[q_cols[1]]
        out["열수지 차이 [%]"] = (qh - qc) / ((qh + qc) / 2) * 100
        q = {"고온측": qh, "저온측": qc}.get(c.get("basis"), (qh + qc) / 2)
    else:
        q = out[q_cols[0]]

    dt1, dt2 = ((T["Th_in"] - T["Tc_out"], T["Th_out"] - T["Tc_in"]) if c.get("counter", True)
                else (T["Th_in"] - T["Tc_in"], T["Th_out"] - T["Tc_out"]))
    out["LMTD [℃]"] = lmtd(dt1, dt2)
    out["최소 온도차 [℃]"] = np.minimum(dt1, dt2)
    out["UA [kW/K]"] = q / (c.get("F", 1.0) * out["LMTD [℃]"])
    if c.get("area"):
        out["U [W/m²K]"] = out["UA [kW/K]"] * 1000 / c["area"]
        if c.get("u_clean"):
            out["파울링 저항 [m²K/kW]"] = (1 / out["U [W/m²K]"] - 1 / c["u_clean"]) * 1000
    return out.replace([np.inf, -np.inf], np.nan)


def reactor(df, beds, temp_unit="℃", comp=None):
    """
    beds: list of (inlet_tag, outlet_tag, catalyst_fraction)
    comp: optional F_in, F_out (total flows, same unit), x_in, x_out (reactant fraction), y_in, y_out (product
          fraction), nu (moles product per mole reactant). F_out defaults to F_in; use a consistent (molar) basis.
    """
    out = pd.DataFrame(index=df.index)
    temps, wabt, w_sum = [], 0, 0
    for i, (tin, tout, w) in enumerate(beds, 1):
        a, b = to_celsius(df[tin], temp_unit), to_celsius(df[tout], temp_unit)
        out[f"{i}층 발열 ΔT [℃]"] = b - a
        wabt, w_sum = wabt + w * (a + b) / 2, w_sum + w
        temps += [a, b]
    if len(beds) > 1:
        out["전체 발열 ΔT [℃]"] = out.filter(like="발열 ΔT").sum(axis=1, min_count=len(beds))
    out["WABT [℃]"] = wabt / w_sum
    out["최고 온도 [℃]"] = pd.concat(temps, axis=1).max(axis=1)

    if comp and comp.get("F_in") and comp.get("x_in") and comp.get("x_out"):
        f_in = df[comp["F_in"]]
        f_out = df[comp["F_out"]] if comp.get("F_out") else f_in
        fed, left = f_in * df[comp["x_in"]], f_out * df[comp["x_out"]]
        out["전환율 [%]"] = (1 - left / fed) * 100
        if comp.get("y_out"):
            formed = f_out * df[comp["y_out"]] - (f_in * df[comp["y_in"]] if comp.get("y_in") else 0)
            out["선택도 [%]"] = formed / comp.get("nu", 1.0) / (fed - left) * 100
            out["수율 [%]"] = out["전환율 [%]"] * out["선택도 [%]"] / 100
    return out.replace([np.inf, -np.inf], np.nan)


def trend_per_day(s):
    """Linear trend of a time series → (slope per day, fitted value at the last time)."""
    s = s.dropna()
    if len(s) < 3:
        return None, None
    days = (s.index - s.index[0]).total_seconds() / 86400
    slope, icpt = np.polyfit(days, s.to_numpy(float), 1)
    return slope, icpt + slope * days[-1]


def projection(s, limit):
    """Days until the trend of s reaches an upper limit: 0 = already reached, None = not within 10 years."""
    slope, last = trend_per_day(s)
    if not limit or slope is None:
        return None
    if last >= limit:
        return 0.0
    days = (limit - last) / slope if slope > 0 else None
    return days if days is not None and days <= 3650 else None


def balance(df, streams):
    """streams: list of (tag, '입력'|'출력', factor). Factors put every stream in the same unit."""
    side = {d: sum((df[t] * f for t, dd, f in streams if dd == d), pd.Series(0.0, index=df.index)) for d in ("입력", "출력")}
    out = pd.DataFrame({"입력 합": side["입력"], "출력 합": side["출력"]})
    out["불일치"] = out["입력 합"] - out["출력 합"]
    out["불일치 [%]"] = out["불일치"] / out["입력 합"] * 100
    return out.replace([np.inf, -np.inf], np.nan)


def intensity(df, consumers, product, rule=None):
    """Specific consumption Σ(tag·factor) / product. With rule, per-period ratio of means (= ratio of totals)."""
    use = sum(df[t] * f for t, f in consumers)
    if rule:
        m = pd.DataFrame({"사용량": use, "생산량": df[product]}).resample(rule).mean()
        m["원단위"] = m["사용량"] / m["생산량"]
        return m.replace([np.inf, -np.inf], np.nan)
    return (use / df[product]).replace([np.inf, -np.inf], np.nan).rename("원단위")


def default_bands(df, tags, window):
    # ponytail: 1.5 × median window range ≈ noise band if the process is steady most of the time; users tune per tag
    rng = df[tags].rolling(window).max() - df[tags].rolling(window).min()
    return (rng.median() * 1.5).to_dict()


def steady_mask(df, bands, window):
    """True where every tag stays within its band (max − min) over a window that covers the sample."""
    tags = list(bands)
    rng = df[tags].rolling(window).max() - df[tags].rolling(window).min()
    ok = (rng.le(pd.Series(bands)) & rng.notna()).all(axis=1).to_numpy(float)
    covered = pd.Series(ok[::-1]).rolling(window, min_periods=1).max().to_numpy()[::-1]
    return pd.Series(covered > 0, index=df.index)


def segments(mask):
    if not mask.any():
        return pd.DataFrame(columns=["시작", "끝", "길이(샘플)"])
    grp = (mask != mask.shift()).cumsum()[mask]
    seg = mask[mask].groupby(grp).agg(시작=lambda s: s.index[0], 끝=lambda s: s.index[-1], **{"길이(샘플)": "size"})
    return seg.reset_index(drop=True)


def pca_fit(train, var_ratio=0.9, q=99.0):
    X = train.dropna()
    if len(X) < train.shape[1] + 2:
        raise ValueError(f"학습 구간의 유효 데이터가 {len(X)}행뿐입니다. 기간을 넓히거나 변수를 줄이세요.")
    mu, sd = X.mean(), X.std()
    sd[sd == 0] = 1
    Z = ((X - mu) / sd).to_numpy()
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    lam = S ** 2 / (len(Z) - 1)
    cum = np.cumsum(lam) / lam.sum()
    k = min(int(np.searchsorted(cum, var_ratio) + 1), max(Z.shape[1] - 1, 1))  # keep a residual space for SPE
    m = {"mu": mu, "sd": sd, "P": Vt[:k].T, "lam": lam[:k], "k": k, "explained": float(cum[k - 1])}
    st = pca_score(m, X)
    m["t2_lim"], m["spe_lim"] = float(np.percentile(st["T²"], q)), float(np.percentile(st["SPE"], q))
    return m


def pca_score(m, X):
    Z = ((X[m["mu"].index] - m["mu"]) / m["sd"]).to_numpy()
    T = Z @ m["P"]
    E = Z - T @ m["P"].T
    return pd.DataFrame({"T²": (T ** 2 / m["lam"]).sum(axis=1), "SPE": (E ** 2).sum(axis=1)}, index=X.index)


def pca_contrib(m, row):
    """Per-variable contributions that sum exactly to T² and SPE for one sample."""
    z = ((row[m["mu"].index] - m["mu"]) / m["sd"]).to_numpy(float)
    t = m["P"].T @ z
    e = z - m["P"] @ t
    return pd.DataFrame({"T² 기여": z * (m["P"] @ (t / m["lam"])), "SPE 기여": e ** 2}, index=m["mu"].index)

CP_MODELS = {"평균 변화 (계단형)": "l2", "평균·산포 변화": "normal", "분포 변화 (모양 무관 · 느림)": "rbf"}


def _refine(z, bounds, model, step, min_rows):
    """Block averaging locates a change only to within a block: re-place each one on the raw rows (±2 blocks) by the
    same cost, computed in O(1) per candidate from cumulative sums (rbf keeps the block position)."""
    if step == 1 or model == "rbf":
        return bounds
    cs = np.vstack([np.zeros(z.shape[1]), np.cumsum(z, axis=0)])
    cs2 = np.vstack([np.zeros(z.shape[1]), np.cumsum(z * z, axis=0)])

    def cost(a, b):
        n, s, s2 = b - a, cs[b] - cs[a], cs2[b] - cs2[a]
        if model == "l2":
            return float((s2 - s * s / n).sum())
        return float(n * np.log(np.maximum(s2 / n - (s / n) ** 2, 1e-12)).sum())  # diagonal-covariance Gaussian

    bounds = list(bounds)
    for i in range(1, len(bounds) - 1):
        a, r, b = bounds[i - 1], bounds[i], bounds[i + 1]
        cands = range(max(a + min_rows, r - 2 * step), min(b - min_rows, r + 2 * step) + 1)
        if len(cands):
            bounds[i] = min(cands, key=lambda c: cost(a, c) + cost(c, b))
    return bounds


def change_points(df, model="l2", sensitivity=5, n_bkps=None, min_size=None, max_points=500):
    """Operating-mode changes (steps in level or spread) found with ruptures (PELT, or binary segmentation when the
    number of changes is given). Tags are standardized and long series averaged in blocks down to ≤ max_points (ruptures'
    costs are pure Python, O(n²) calls without pruning), then each change is re-placed on the raw rows, so a year of
    1-minute data stays interactive. Returns (change times, segment table, per-row segment means)."""
    import ruptures as rpt
    d = df.dropna()
    if len(d) < 20:
        raise ValueError(f"선택한 태그가 모두 값을 가진 행이 {len(d)}개뿐이라 변화점을 찾을 수 없습니다.")
    z = (d - d.mean()) / d.std().replace(0, 1)
    step = int(np.ceil(len(z) / max_points))
    blocks = z.groupby(np.arange(len(z)) // step).mean().to_numpy()
    n = len(blocks)
    ms = max(2, int(np.ceil((min_size or max(len(z) // 50, 1)) / step)))
    if n < 2 * ms:
        raise ValueError("데이터 길이에 비해 최소 구간 길이가 깁니다. 최소 구간 길이를 줄이세요.")
    algo = (rpt.Binseg if n_bkps else rpt.Pelt)(model=model, min_size=ms, jump=1).fit(blocks)
    if n_bkps:
        ends = algo.predict(n_bkps=int(min(n_bkps, n // ms - 1)))
    else:
        c = 2.0 * 2 ** ((5 - sensitivity) / 2)  # sensitivity 5 ≈ BIC; each step of 2 halves / doubles the penalty
        p = blocks.shape[1]
        if model == "l2":
            # l2 cost is in squared units: scale by the block-level noise variance, taken robustly around a rolling median
            # as wide as the shortest segment, so slow wander (autocorrelated noise) counts as noise and steps do not.
            dev = blocks - pd.DataFrame(blocks).rolling(4 * ms, center=True, min_periods=1).median().to_numpy()
            noise = (np.median(np.abs(dev - np.median(dev, axis=0)), axis=0) / 0.6745) ** 2
            # AR(1) long-run variance factor (1 + ρ)/(1 − ρ): a segment mean of autocorrelated noise wanders that much
            # more than white noise would, so without it PELT cuts every slow controller oscillation into segments.
            rho = np.array([pd.Series(dev[:, k]).autocorr(1) for k in range(p)])
            rho = np.clip(np.nan_to_num(rho), 0, 0.96)
            pen = c * max(float((noise * (1 + rho) / (1 - rho)).sum()), 1e-6) * np.log(n)
        elif model == "normal":
            pen = c / 2 * (p + p * (p + 1) / 2) * np.log(n)
        else:
            pen = c * np.log(n)
        ends = algo.predict(pen=pen)
    bounds = _refine(z.to_numpy(float), [0] + [min(e * step, len(d)) for e in ends], model, step, max(ms * step // 2, 2))
    starts, rows = bounds[:-1], bounds[1:]
    seg = pd.DataFrame([{"시작": d.index[a], "끝": d.index[b - 1], "길이(샘플)": b - a, **d.iloc[a:b].mean().to_dict()}
                        for a, b in zip(starts, rows)])
    means = pd.concat([pd.DataFrame([d.iloc[a:b].mean()] * (b - a), index=d.index[a:b]) for a, b in zip(starts, rows)])
    return [d.index[a] for a in starts[1:]], seg, means


if __name__ == "__main__":
    # Counter-flow exchanger with known duty and U; U drops by half over the period (fouling).
    n = 200
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    m_h, m_c, cp_h, cp_c, area = 10.0, 15.0, 2.5, 4.18, 50.0  # kg/s, kJ/kgK, m²
    th_in, tc_in = 150.0, 30.0
    u_true = np.linspace(800, 400, n)  # W/m²K
    # Solve outlet temps with ε-NTU (counter-flow) so the test data is physically consistent.
    ch, cc = m_h * cp_h, m_c * cp_c  # kW/K
    cmin, cmax = min(ch, cc), max(ch, cc)
    cr, ntu = cmin / cmax, u_true * area / 1000 / cmin
    eps = (1 - np.exp(-ntu * (1 - cr))) / (1 - cr * np.exp(-ntu * (1 - cr)))
    q_true = eps * cmin * (th_in - tc_in)
    df = pd.DataFrame({
        "TH_IN": th_in, "TH_OUT": th_in - q_true / ch, "TC_IN": tc_in, "TC_OUT_F": (tc_in + q_true / cc) * 1.8 + 32,
        "FH": m_h * 3.6, "FC": m_c * 3600 / 1000 * 1000 / 998,  # t/h and m³/h (ρ = 998)
    }, index=idx)
    tags = {"Th_in": "TH_IN", "Th_out": "TH_OUT", "Tc_in": "TC_IN", "Tc_out": "TC_OUT_F", "m_hot": "FH", "m_cold": "FC"}
    units = {"Th_in": "℃", "Th_out": "℃", "Tc_in": "℃", "Tc_out": "℉", "m_hot": "t/h", "m_cold": "m³/h"}
    c = {"cp_hot": cp_h, "cp_cold": cp_c, "rho_cold": 998, "area": area, "F": 1.0, "counter": True, "basis": "평균", "u_clean": 800}
    hx = heat_exchanger(df, tags, units, c)
    assert np.allclose(hx["고온측 전열량 [kW]"], q_true) and np.allclose(hx["저온측 전열량 [kW]"], q_true, rtol=1e-3)
    assert hx["열수지 차이 [%]"].abs().max() < 0.2
    assert np.allclose(hx["U [W/m²K]"], u_true, rtol=1e-3), (hx["U [W/m²K]"].iloc[[0, -1]], u_true[[0, -1]])
    assert abs(hx["파울링 저항 [m²K/kW]"].iloc[0]) < 1e-3 and abs(hx["파울링 저항 [m²K/kW]"].iloc[-1] - (1 / 400 - 1 / 800) * 1000) < 1e-3
    only_hot = heat_exchanger(df, {**tags, "m_cold": None}, units, c)
    assert "열수지 차이 [%]" not in only_hot and np.allclose(only_hot["U [W/m²K]"], u_true, rtol=1e-3)
    # Temperature cross → LMTD undefined, not a bogus number.
    assert lmtd(pd.Series([10.0, -5.0, 20.0]), pd.Series([10.0, 5.0, 10.0])).isna().tolist() == [False, True, False]
    try:
        heat_exchanger(df, tags, units, {**c, "rho_cold": None})
        raise AssertionError("m³/h without density must raise")
    except ValueError:
        pass

    # Reactor: two beds (40/60 % catalyst), known conversion/selectivity, WABT rising 0.5 ℃/day.
    idx = pd.date_range("2026-01-01", periods=24 * 30, freq="h")
    rise = 0.5 * np.arange(len(idx)) / 24
    r = pd.DataFrame({"B1I": 300 + rise, "B1O": 330 + rise, "B2I": 320 + rise, "B2O": 340 + rise,
                      "F": 100.0, "XI": 0.5, "XO": 0.1, "YO": 0.3}, index=idx)
    rx = reactor(r, [("B1I", "B1O", 0.4), ("B2I", "B2O", 0.6)], comp={"F_in": "F", "x_in": "XI", "x_out": "XO", "y_out": "YO"})
    assert np.allclose(rx["WABT [℃]"], 0.4 * 315 + 0.6 * 330 + rise) and np.allclose(rx["전체 발열 ΔT [℃]"], 50)
    assert np.allclose(rx["최고 온도 [℃]"], 340 + rise)
    assert np.allclose(rx["전환율 [%]"], 80) and np.allclose(rx["선택도 [%]"], 75) and np.allclose(rx["수율 [%]"], 60)
    slope, last = trend_per_day(rx["WABT [℃]"])
    assert abs(slope - 0.5) < 1e-6 and abs(last - rx["WABT [℃]"].iloc[-1]) < 1e-6
    w = rx["WABT [℃]"]
    assert abs(projection(w, w.iloc[-1] + 5) - 10) < 1e-6          # 5 ℃ at 0.5 ℃/day
    assert projection(w, w.iloc[0]) == 0.0                           # already past the limit
    assert projection(w - rise + rise * 1e-9, 400) is None           # near-flat trend: no overflowing date
    assert projection(-w, -w.iloc[0] + 1) is None and projection(w, 0) is None
    gap = w.copy(); gap.iloc[-5:] = np.nan
    assert abs(trend_per_day(gap)[1] - w.iloc[-6]) < 1e-6            # fitted last ignores trailing gaps
    rk = reactor(r.assign(B1I=r.B1I * 1.8 + 32, B1O=r.B1O * 1.8 + 32, B2I=r.B2I * 1.8 + 32, B2O=r.B2O * 1.8 + 32),
                 [("B1I", "B1O", 1), ("B2I", "B2O", 1)], "℉")
    assert np.allclose(rk["WABT [℃]"], (315 + 330) / 2 + rise) and "전환율 [%]" not in rk

    # Balance and specific consumption.
    u = pd.DataFrame({"FEED": [100.0, 100.0], "P1": [60.0, 60.0], "P2": [39.0, 40.0],
                      "STEAM": [10.0, 20.0], "POWER": [500.0, 500.0], "PROD": [50.0, 100.0]},
                     index=pd.date_range("2026-01-01", periods=2, freq="12h"))
    b = balance(u, [("FEED", "입력", 1), ("P1", "출력", 1), ("P2", "출력", 1)])
    assert b["불일치 [%]"].tolist() == [1.0, 0.0]
    kpi = intensity(u, [("STEAM", 1.0), ("POWER", 0.01)], "PROD")
    assert np.allclose(kpi, [15 / 50, 25 / 100])
    daily = intensity(u, [("STEAM", 1.0), ("POWER", 0.01)], "PROD", "D")
    assert np.allclose(daily["원단위"], (15 + 25) / 2 / 75)  # ratio of totals, not mean of ratios

    # Steady state: flat, ramp, flat → two steady segments.
    x = np.r_[np.full(50, 10.0), np.linspace(10, 20, 30), np.full(50, 20.0)] + np.random.default_rng(1).normal(0, 0.05, 130)
    sd_df = pd.DataFrame({"T": x}, index=pd.date_range("2026-01-01", periods=130, freq="min"))
    mask = steady_mask(sd_df, {"T": 0.5}, 10)
    seg = segments(mask)
    assert len(seg) == 2 and seg["길이(샘플)"].sum() >= 95 and not mask.iloc[60:75].any(), seg
    assert 0.1 < default_bands(sd_df, ["T"], 10)["T"] < 1.0
    assert segments(pd.Series(False, index=sd_df.index)).empty

    # PCA: A and B move together in training; later B breaks away → SPE alarm with B as top contributor.
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 600)
    X = pd.DataFrame({"A": a, "B": 2 * a + rng.normal(0, 0.1, 600), "C": rng.normal(0, 1, 600)},
                     index=pd.date_range("2026-01-01", periods=600, freq="min"))
    X.loc[X.index[500:], "B"] += 3
    m = pca_fit(X.iloc[:400], 0.9)
    s = pca_score(m, X)
    assert m["k"] == 2 and (s["SPE"].iloc[:400] > m["spe_lim"]).mean() <= 0.011
    assert (s["SPE"].iloc[500:] > m["spe_lim"]).mean() > 0.9 and (s["SPE"].iloc[400:500] > m["spe_lim"]).mean() < 0.05
    cb = pca_contrib(m, X.iloc[550])
    assert np.isclose(cb["SPE 기여"].sum(), s["SPE"].iloc[550]) and np.isclose(cb["T² 기여"].sum(), s["T²"].iloc[550])
    assert cb["SPE 기여"].idxmax() in ("A", "B")
    try:
        pca_fit(X.iloc[:3])
        raise AssertionError("too little training data must raise")
    except ValueError:
        pass

    # Change points: two level steps in A (and B following) are found where they are, at default sensitivity.
    rng = np.random.default_rng(3)
    lvl = np.r_[np.zeros(3000), np.full(4000, 1.5), np.full(3000, -1.0)]
    cp = pd.DataFrame({"A": lvl + rng.normal(0, 0.5, 10000), "B": 2 * lvl + rng.normal(0, 1, 10000), "C": rng.normal(0, 1, 10000)},
                      index=pd.date_range("2026-01-01", periods=10000, freq="min"))
    times, seg, means = change_points(cp)
    assert len(times) == 2 and all(abs(cp.index.get_loc(t) - e) <= 10 for t, e in zip(times, (3000, 7000))), times
    assert np.allclose(seg["A"], [0, 1.5, -1], atol=0.05) and seg["길이(샘플)"].sum() == 10000 and means.shape == (10000, 3)
    assert change_points(cp[["C"]])[0] == []                          # pure noise: no change
    assert len(change_points(cp, n_bkps=1)[0]) == 1
    var = pd.DataFrame({"V": np.r_[rng.normal(0, 1, 2000), rng.normal(0, 4, 2000)]}, index=cp.index[:4000])
    vt = change_points(var, "normal")[0]
    assert len(vt) == 1 and abs(var.index.get_loc(vt[0]) - 2000) <= 20, vt
    assert len(change_points(cp, "rbf")[0]) == 2
    print("ok")
