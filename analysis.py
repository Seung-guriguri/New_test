import numpy as np
import pandas as pd


def load(file, name: str) -> pd.DataFrame:
    if name.lower().endswith(".xlsx"):
        return pd.read_excel(file)
    try:
        return pd.read_csv(file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="cp949")


def to_timeseries(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Empty result means time_col is not a usable time column."""
    col = df[time_col]
    if not pd.api.types.is_datetime64_any_dtype(col) and pd.to_numeric(col, errors="coerce").notna().mean() > 0.5:
        return pd.DataFrame()  # tag values like 50.02 or 2025 would parse as 1970 epoch times or bare years
    try:
        t = pd.to_datetime(col, errors="coerce", format="mixed")
    except ValueError:  # mixed UTC offsets, e.g. an export spanning a DST switch
        t = pd.to_datetime(col, errors="coerce", format="mixed", utc=True)
    if t.dt.tz is not None:
        t = t.dt.tz_localize(None)
    if t.notna().mean() < 0.9:
        return pd.DataFrame()

    df = df.drop(columns=time_col).set_index(t.rename(time_col))
    df = df[df.index.notna()].sort_index()
    df = df.drop(columns=df.select_dtypes(["datetime", "datetimetz"]).columns)
    # DCS/PI exports mix in strings like "Bad" or "I/O Timeout"; treat them as missing.
    return df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")


def _scores(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(), float(np.sqrt(((y - p) ** 2).mean()))


def _pls1(X, y, k):
    """NIPALS PLS1 on standardized X. Returns regression coefficients for centered y."""
    X, y = X.copy(), y - y.mean()
    x0 = np.linalg.norm(X)
    W, P, q = [], [], []
    for _ in range(k):
        w = X.T @ y
        # Stop once X is exhausted: further components would only fit floating-point noise.
        if np.linalg.norm(X) < 1e-6 * x0 or np.linalg.norm(w) < 1e-12 * x0:
            break
        w /= np.linalg.norm(w)
        t = X @ w
        p, c = X.T @ t / (t @ t), y @ t / (t @ t)
        X, y = X - np.outer(t, p), y - c * t
        W.append(w), P.append(p), q.append(c)
    if not W:
        return np.zeros(X.shape[1])
    W, P = np.array(W).T, np.array(P).T
    return W @ np.linalg.solve(P.T @ W, np.array(q))


def fit_soft_sensor(df, target, inputs, lag=0, train_frac=0.7, n_components=None):
    """n_components=None → OLS, int → PLS with that many latent components."""
    data = pd.concat([df[inputs].shift(lag), df[target]], axis=1).dropna()
    n_train = int(len(data) * train_frac)
    if n_train < len(inputs) + 2 or n_train == len(data):
        raise ValueError(f"결측 제거 후 데이터가 {len(data)}행뿐이라 학습/검증이 불가능합니다.")

    X, y = data[inputs].to_numpy(float), data[target].to_numpy(float)
    if y[:n_train].std() == 0 or y[n_train:].std() == 0:
        raise ValueError("예측 대상 값이 학습 또는 검증 구간에서 변하지 않아 모델을 만들거나 평가할 수 없습니다. 기간·학습 비율을 바꿔 보세요.")
    mu, sd = X[:n_train].mean(0), X[:n_train].std(0)
    sd[sd == 0] = 1
    Zs = (X - mu) / sd
    if n_components:
        coef = np.r_[y[:n_train].mean(), _pls1(Zs[:n_train], y[:n_train], n_components)]
    else:
        coef = np.linalg.lstsq(np.c_[np.ones(n_train), Zs[:n_train]], y[:n_train], rcond=None)[0]
    pred = coef[0] + Zs @ coef[1:]

    (r2_tr, rmse_tr), (r2_te, rmse_te) = _scores(y[:n_train], pred[:n_train]), _scores(y[n_train:], pred[n_train:])
    raw = coef[1:] / sd
    return {
        "result": pd.DataFrame(
            {"실측": y, "예측": pred, "구분": ["학습"] * n_train + ["검증"] * (len(y) - n_train)},
            index=data.index,
        ),
        "metrics": {"학습 R²": r2_tr, "검증 R²": r2_te, "학습 RMSE": rmse_tr, "검증 RMSE": rmse_te},
        "coef": pd.Series(coef[1:], index=inputs, name="표준화 계수").sort_values(key=abs, ascending=False),
        # Original-unit form for DCS: y = raw_intercept + Σ raw_coef[i] * x_i(t - lag)
        "raw_coef": pd.Series(raw, index=inputs, name="계수(원래 단위)"),
        "raw_intercept": coef[0] - raw @ mu,
        "split": data.index[n_train],
    }


def lag_scan(df, target, inputs, max_lag, train_frac=0.7, n_components=None) -> pd.Series:
    scores = {}
    for lag in range(max_lag + 1):
        try:
            scores[lag] = fit_soft_sensor(df, target, inputs, lag, train_frac, n_components)["metrics"]["검증 R²"]
        except ValueError:
            pass
    return pd.Series(scores, name="검증 R²", dtype=float).rename_axis("지연")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=500, freq="min")
    x1, x2 = rng.normal(100, 5, 500), rng.normal(2, 0.1, 500)
    df = pd.DataFrame({"T1": x1, "P1": x2}, index=idx)
    df["purity"] = 0.2 * df["T1"].shift(3) - 10 * df["P1"].shift(3) + 50 + rng.normal(0, 0.01, 500)

    r = fit_soft_sensor(df, "purity", ["T1", "P1"], lag=3)
    assert r["metrics"]["검증 R²"] > 0.99, r["metrics"]
    assert fit_soft_sensor(df, "purity", ["T1", "P1"], lag=0)["metrics"]["검증 R²"] < 0.5
    assert lag_scan(df, "purity", ["T1", "P1"], 10).idxmax() == 3

    # Original-unit equation reproduces the model and recovers the true coefficients.
    x = df[["T1", "P1"]].shift(3).loc[r["result"].index]
    assert np.allclose(r["raw_intercept"] + x.to_numpy() @ r["raw_coef"].to_numpy(), r["result"]["예측"])
    assert np.allclose(r["raw_coef"], [0.2, -10], atol=0.01) and abs(r["raw_intercept"] - 50) < 0.1

    # PLS with all components reproduces OLS; redundant sensors don't blow up even at k = number of inputs.
    pls = fit_soft_sensor(df, "purity", ["T1", "P1"], lag=3, n_components=2)
    assert np.allclose(pls["result"]["예측"], r["result"]["예측"])
    df["T2"] = df["T1"] + rng.normal(0, 1e-6, 500)
    for k in (2, 3):
        col = fit_soft_sensor(df, "purity", ["T1", "T2", "P1"], lag=3, n_components=k)
        assert col["metrics"]["검증 R²"] > 0.99 and col["coef"].abs().max() < 10, (k, col["coef"])

    flat = df.assign(purity=1.0)
    for k in (None, 2):
        try:
            fit_soft_sensor(flat, "purity", ["T1", "P1"], n_components=k)
            raise AssertionError("constant target must raise ValueError")
        except ValueError:
            pass

    raw = pd.DataFrame({"time": ["2026-01-01 00:01", "2026-01-01 00:00", "bad"], "T": ["1.5", "Bad", "3"]})
    assert to_timeseries(raw, "time").empty  # 1 of 3 timestamps unparseable → below 90 %
    raw = pd.DataFrame({
        "time": ["2026-01-01 00:00:01.5", "2026-01-01 00:00:00", "2026-01-01 00:00:02"],
        "lab_time": pd.to_datetime(["2026-01-01"] * 3),
        "T": ["1.5", "Bad", "3"],
    })
    ts = to_timeseries(raw, "time")
    assert len(ts) == 3 and list(ts.columns) == ["T"] and ts["T"].isna().tolist() == [True, False, False]
    dst = pd.DataFrame({"t": ["2026-03-29 01:00+01:00", "2026-03-29 03:00+02:00"], "T": [1, 2]})
    assert len(to_timeseries(dst, "t")) == 2
    assert to_timeseries(df.reset_index(), "T1").empty
    assert to_timeseries(pd.DataFrame({"t": ["2025", "2026", "Bad"], "T": [1, 2, 3]}), "t").empty
    print("ok")
