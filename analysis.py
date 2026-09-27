import json

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


# Two-state text found in DCS/LIMS exports. Anything else (e.g. arbitrary labels "A"/"B") stays text and is dropped,
# because which state should be 1 would be a guess.
STATUS = {"yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0, "on": 1, "off": 0, "run": 1, "running": 1,
          "stop": 0, "stopped": 0, "open": 1, "opened": 1, "close": 0, "closed": 0, "start": 1, "started": 1,
          "예": 1, "아니오": 0, "가동": 1, "운전": 1, "정지": 0, "사용": 1, "미사용": 0, "켜짐": 1, "꺼짐": 0}


def status_columns(df: pd.DataFrame) -> dict:
    """Text columns whose every value is one of STATUS → {column: new name saying which state is 1},
    e.g. {"Preheating": "Preheating [Yes=1 · No=0]"}."""
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or s.notna().sum() == 0:
            continue
        v = s.dropna().astype(str).str.strip()
        if not v.str.lower().isin(STATUS).all():
            continue
        seen = {}
        for word in v.unique():  # first spelling seen for each state, 1 before 0
            seen.setdefault(STATUS[word.lower()], word)
        out[c] = f"{c} [" + " · ".join(f"{seen[val]}={val}" for val in sorted(seen, reverse=True)) + "]"
    return out


def encode_status(df: pd.DataFrame) -> pd.DataFrame:
    """Yes/No, On/Off, 가동/정지 … → 1/0 under the name status_columns() gives, so such columns can be analysed."""
    names = status_columns(df)
    if not names:
        return df
    df = df.copy()
    for c in names:
        df[c] = df[c].astype(str).str.strip().str.lower().map(STATUS).where(df[c].notna())
    return df.rename(columns=names)


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
    df = encode_status(df.drop(columns=df.select_dtypes(["datetime", "datetimetz"]).columns))
    # DCS/PI exports mix in strings like "Bad" or "I/O Timeout"; treat them as missing.
    return df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")


SEQ_START = pd.Timestamp("2000-01-01")


def to_sequence(df: pd.DataFrame, order_col=None) -> pd.DataFrame:
    """Data without timestamps (lab samples, batches, test runs): rows keep their order, sorted by order_col if given,
    on a 1-minute pseudo time axis so that every tab still works — row n sits at 2000-01-01 00:00 + n minutes."""
    if order_col is not None:
        key = pd.to_numeric(df[order_col], errors="coerce")
        df = df.assign(_key=key if key.notna().mean() > 0.5 else df[order_col].astype(str))
        df = df.sort_values("_key", kind="stable").drop(columns=["_key", order_col])
    df = encode_status(df.drop(columns=df.select_dtypes(["datetime", "datetimetz"]).columns))
    out = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    out.index = pd.date_range(SEQ_START, periods=len(out), freq="min", name="순서")
    return out


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


def prepare_xy(df, target, inputs, lag, train_frac):
    """Inputs shifted by lag and aligned with the target; time-ordered train/validation split. Shared by every model."""
    data = pd.concat([df[inputs].shift(lag), df[target]], axis=1).dropna()
    n_train = int(len(data) * train_frac)
    if n_train < len(inputs) + 2 or n_train == len(data):
        raise ValueError(f"결측 제거 후 데이터가 {len(data)}행뿐이라 학습/검증이 불가능합니다.")
    y = data[target].to_numpy(float)
    if y[:n_train].std() == 0 or y[n_train:].std() == 0:
        raise ValueError("예측 대상 값이 학습 또는 검증 구간에서 변하지 않아 모델을 만들거나 평가할 수 없습니다. 기간·학습 비율을 바꿔 보세요.")
    return data, n_train


def residual_alarm(result, conf=0.99, limit=None):
    """|measured − predicted| above a limit (BibMon's SPE-limit idea, model-agnostic). Default limit: the training
    residuals' conf-quantile, fine for OLS/PLS; ML models pass an out-of-sample limit instead (ml.fit_ml_soft_sensor)."""
    res = (result["실측"] - result["예측"]).abs()
    if limit is None:
        limit = float(res[result["구분"] == "학습"].quantile(conf))
    return limit, res > limit


def fit_soft_sensor(df, target, inputs, lag=0, train_frac=0.7, n_components=None):
    """n_components=None → OLS, int → PLS with that many latent components."""
    data, n_train = prepare_xy(df, target, inputs, lag, train_frac)
    X, y = data[inputs].to_numpy(float), data[target].to_numpy(float)
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


def export_model(r, target, lag, n_components, resample) -> dict:
    return {
        "version": 1,
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "target": target,
        "method": f"PLS({n_components})" if n_components else "OLS",
        "resample": resample,
        "lag": int(lag),
        "intercept": float(r["raw_intercept"]),
        "coef": {k: float(v) for k, v in r["raw_coef"].items()},
        "train_period": [str(r["result"].index[0]), str(r["split"])],
        "metrics": {k: float(v) for k, v in r["metrics"].items()},
    }


def parse_model(text: str) -> dict:
    """Validate an uploaded model file; plain JSON only, so nothing in it can execute."""
    try:
        m = json.loads(text)
        ok = (isinstance(m, dict) and isinstance(m["target"], str) and isinstance(m["lag"], int) and m["lag"] >= 0
              and isinstance(m["intercept"], (int, float)) and isinstance(m["coef"], dict) and m["coef"]
              and all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in m["coef"].items()))
    except (ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        raise ValueError("이 프로그램에서 저장한 모델 파일(JSON)이 아니거나 내용이 손상되었습니다.")
    return m


def apply_model(df, model):
    """Returns (DataFrame[실측?, 예측], metrics or None)."""
    inputs = list(model["coef"])
    missing = [c for c in inputs if c not in df.columns]
    if missing:
        raise ValueError(f"현재 데이터에 모델 입력 태그가 없습니다: {', '.join(missing)}")
    out = pd.DataFrame({"예측": model["intercept"] + df[inputs].shift(model["lag"]) @ pd.Series(model["coef"])})
    out = out.dropna()
    if model["target"] not in df.columns:
        return out, None
    out.insert(0, "실측", df[model["target"]])
    both = out.dropna()
    if len(both) < 2 or both["실측"].std() == 0:
        return out, None
    r2, rmse = _scores(both["실측"].to_numpy(), both["예측"].to_numpy())
    return out, {"R²": r2, "RMSE": rmse, "비교 가능 행": len(both)}


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

    # Saved model → JSON → reapplied reproduces the fit's predictions and scores.
    m = parse_model(json.dumps(export_model(r, "purity", 3, None, "원본")))
    out, met = apply_model(df, m)
    assert np.allclose(out.loc[r["result"].index, "예측"], r["result"]["예측"]) and met["R²"] > 0.99
    assert apply_model(df.drop(columns="purity"), m)[1] is None
    for bad in ["not json", "[]", '{"target": "x"}', json.dumps({**m, "coef": {"T1": "__import__"}}), json.dumps({**m, "lag": -1})]:
        try:
            parse_model(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    try:
        apply_model(df.drop(columns="T1"), m)
        raise AssertionError("missing input tag must raise")
    except ValueError as e:
        assert "T1" in str(e)

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
    lab = pd.DataFrame({"no": [3, 1, 2], "cat": ["1.5", "2", "Bad"], "memo": ["a", "b", "c"]})
    seq = to_sequence(lab, "no")
    assert list(seq.columns) == ["cat"] and seq["cat"].isna().tolist() == [False, True, False]  # sorted by no, "Bad" → NaN
    assert seq["cat"].iloc[0] == 2.0 and seq["cat"].iloc[2] == 1.5 and seq.index[1] == SEQ_START + pd.Timedelta(minutes=1)
    assert list(to_sequence(lab).columns) == ["no", "cat"]
    st_raw = pd.DataFrame({"t": ["2026-01-01 00:00", "2026-01-01 00:01", "2026-01-01 00:02", "2026-01-01 00:03"],
                           "Heater": ["Yes", " no", None, "YES"], "Pump": ["가동", "정지", "가동", "가동"],
                           "Grade": ["A", "B", "A", "B"], "Flag": ["No", "No", "No", "Bad"], "T": [1, 2, 3, 4]})
    assert status_columns(st_raw) == {"Heater": "Heater [Yes=1 · no=0]", "Pump": "Pump [가동=1 · 정지=0]"}
    ts = to_timeseries(st_raw, "t")
    assert list(ts.columns) == ["Heater [Yes=1 · no=0]", "Pump [가동=1 · 정지=0]", "T"]
    assert ts.iloc[:, 0].tolist()[:2] == [1.0, 0.0] and np.isnan(ts.iloc[2, 0]) and ts.iloc[:, 1].tolist() == [1, 0, 1, 1]
    assert "Grade" not in ts and not any(c.startswith("Flag") for c in ts)  # arbitrary labels / unknown words stay out
    assert list(to_sequence(st_raw.drop(columns="t")).columns)[:2] == list(ts.columns)[:2]
    print("ok")
