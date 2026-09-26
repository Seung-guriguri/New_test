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
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    # DCS/PI exports mix in strings like "Bad" or "I/O Timeout"; treat them as missing.
    return df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")


def _r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def _rmse(y, p):
    return float(np.sqrt(((y - p) ** 2).mean()))


def fit_soft_sensor(df, target, inputs, lag=0, train_frac=0.7):
    # ponytail: plain OLS on standardized inputs; switch to PLS/Ridge if correlated tray temps make coefficients unstable
    data = pd.concat([df[inputs].shift(lag), df[target]], axis=1).dropna()
    n_train = int(len(data) * train_frac)
    if n_train < len(inputs) + 2 or n_train == len(data):
        raise ValueError(f"결측 제거 후 데이터가 {len(data)}행뿐이라 학습/검증이 불가능합니다.")

    X, y = data[inputs].to_numpy(float), data[target].to_numpy(float)
    mu, sd = X[:n_train].mean(0), X[:n_train].std(0)
    sd[sd == 0] = 1
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    coef = np.linalg.lstsq(Z[:n_train], y[:n_train], rcond=None)[0]
    pred = Z @ coef

    tr, te = slice(0, n_train), slice(n_train, None)
    return {
        "result": pd.DataFrame(
            {"실측": y, "예측": pred, "구분": ["학습"] * n_train + ["검증"] * (len(y) - n_train)},
            index=data.index,
        ),
        "metrics": {
            "학습 R²": _r2(y[tr], pred[tr]),
            "검증 R²": _r2(y[te], pred[te]),
            "학습 RMSE": _rmse(y[tr], pred[tr]),
            "검증 RMSE": _rmse(y[te], pred[te]),
        },
        "coef": pd.Series(coef[1:], index=inputs, name="표준화 계수").sort_values(key=abs, ascending=False),
        "intercept": coef[0],
        "split": data.index[n_train],
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=500, freq="min")
    x1, x2 = rng.normal(100, 5, 500), rng.normal(2, 0.1, 500)
    df = pd.DataFrame({"T1": x1, "P1": x2}, index=idx)
    df["purity"] = 0.2 * df["T1"].shift(3) - 10 * df["P1"].shift(3) + 50 + rng.normal(0, 0.01, 500)

    r = fit_soft_sensor(df, "purity", ["T1", "P1"], lag=3)
    assert r["metrics"]["검증 R²"] > 0.99, r["metrics"]
    assert fit_soft_sensor(df, "purity", ["T1", "P1"], lag=0)["metrics"]["검증 R²"] < 0.5

    raw = pd.DataFrame({"time": ["2026-01-01 00:01", "2026-01-01 00:00", "bad"], "T": ["1.5", "Bad", "3"]})
    ts = to_timeseries(raw, "time")
    assert list(ts.index.minute) == [0, 1] and ts["T"].isna().tolist() == [True, False]
    print("ok")
