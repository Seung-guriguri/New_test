"""ML soft sensors on top of BibMon's scikit-learn wrapper (imported lazily: it pulls in scikit-learn, optuna, …)."""
import warnings

import numpy as np
import pandas as pd

# Set once at import: warnings.catch_warnings() inside a fit would race between Streamlit session threads.
warnings.filterwarnings("ignore", module="bibmon")
warnings.filterwarnings("ignore", category=FutureWarning, module="optuna")

from analysis import _scores, prepare_xy

MODELS = ["랜덤포레스트", "그래디언트 부스팅", "신경망 (MLP)"]

# Tuning spaces use only Optuna 'int'/'categorical': BibMon's 'uniform'/'loguniform' branches call Optuna APIs that newer
# releases deprecate.
TUNING = {
    "랜덤포레스트": ([("n_estimators", "int", [50, 300]), ("max_depth", "int", [3, 20]), ("min_samples_leaf", "int", [1, 20])]),
    "그래디언트 부스팅": ([("n_estimators", "int", [50, 400]), ("max_depth", "int", [2, 6]),
                   ("learning_rate", "categorical", [0.02, 0.05, 0.1, 0.2])]),
    "신경망 (MLP)": ([("alpha", "categorical", [1e-5, 1e-4, 1e-3, 1e-2]),
                   ("learning_rate_init", "categorical", [1e-4, 1e-3, 1e-2])]),
}


def _regressor(name, seed):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    # n_jobs=2: one viewer's fit must not take every core of a shared server.
    return {"랜덤포레스트": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=2, random_state=seed),
            "그래디언트 부스팅": lambda: GradientBoostingRegressor(random_state=seed),
            "신경망 (MLP)": lambda: MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, early_stopping=True, random_state=seed),
            }[name]()


def _bibmon_fit(regressor, X_tr, Y_tr, X_te, Y_te, space=None, trials=0):
    import bibmon
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    m = bibmon.sklearnRegressor(regressor)
    # Inputs are already aligned and NaN-free; BibMon's default 'remove_frozen_variables' could silently drop columns.
    m.pre_train(X_tr, Y_tr, f_pp=["normalize"], f_pp_test=["normalize"])
    if trials:
        # Called directly: BibMon's fit(tune=True) dispatches to a missing self.tuning method.
        m.hyperparameter_tuning(pd.DataFrame({"possibilities": [p for _, _, p in space], "type": [t for _, t, _ in space]},
                                             index=[n for n, _, _ in space]), n_trials=int(trials))
    m.train(lim_conf=0.99)
    m.predict(X_te, Y_te)
    return m


def fit_ml_soft_sensor(df, target, inputs, lag=0, train_frac=0.7, model="랜덤포레스트", tune_trials=0, seed=0):
    """Same contract as analysis.fit_soft_sensor (result/metrics/split) plus importance, tuned params and alarm_limit."""
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance

    data, n_train = prepare_xy(df, target, inputs, lag, train_frac)
    X, Y = data[inputs], data[[target]]
    frozen = [c for c in inputs if X[c].iloc[:n_train].std() == 0]
    if frozen:  # BibMon's normalize divides by the training std with no zero guard
        raise ValueError(f"학습 구간에서 값이 변하지 않는 입력이 있어 ML 모델을 만들 수 없습니다: {', '.join(frozen)} — 입력에서 빼 주세요.")
    m = _bibmon_fit(_regressor(model, seed), X.iloc[:n_train], Y.iloc[:n_train], X.iloc[n_train:], Y.iloc[n_train:],
                    TUNING[model], tune_trials)

    # Residual-alarm limit from data the model has not seen: ML fits its own training rows far tighter than new data,
    # so an in-sample limit would flag healthy validation points. Refit on the first 80 % of training, score the rest.
    n_fit = int(n_train * 0.8)
    cal = _bibmon_fit(clone(m.regressor), X.iloc[:n_fit], Y.iloc[:n_fit], X.iloc[n_fit:n_train], Y.iloc[n_fit:n_train])
    cal_res = (Y.iloc[n_fit:n_train].to_numpy(float).ravel() - cal.Y_test_pred_orig.to_numpy(float).ravel())

    imp = permutation_importance(m.regressor, m.X_train.values, m.Y_train.values.ravel(), n_repeats=5, random_state=seed)
    y = Y[target].to_numpy(float)
    pred = np.r_[m.Y_train_pred_orig.to_numpy(float).ravel(), m.Y_test_pred_orig.to_numpy(float).ravel()]
    (r2_tr, rmse_tr), (r2_te, rmse_te) = _scores(y[:n_train], pred[:n_train]), _scores(y[n_train:], pred[n_train:])
    return {
        "result": pd.DataFrame({"실측": y, "예측": pred, "구분": ["학습"] * n_train + ["검증"] * (len(y) - n_train)}, index=data.index),
        "metrics": {"학습 R²": r2_tr, "검증 R²": r2_te, "학습 RMSE": rmse_tr, "검증 RMSE": rmse_te},
        "importance": pd.Series(imp.importances_mean, index=inputs, name="중요도").sort_values(ascending=False),
        "tuned": {n: m.regressor.get_params()[n] for n, _, _ in TUNING[model]} if tune_trials else {},
        "alarm_limit": float(np.quantile(np.abs(cal_res), 0.99)),
        "split": data.index[n_train],
    }


if __name__ == "__main__":
    from analysis import fit_soft_sensor, residual_alarm

    # Nonlinear plant: y depends on x1² and an x2 threshold — OLS can't capture it, tree models should.
    rng = np.random.default_rng(0)
    n = 1500
    idx = pd.date_range("2026-01-01", periods=n, freq="min")
    x1, x2, x3 = rng.uniform(-2, 2, n), rng.uniform(0, 1, n), rng.normal(0, 1, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}, index=idx)
    df["y"] = 2 * df["x1"].shift(2) ** 2 + 3 * (df["x2"].shift(2) > 0.5) + rng.normal(0, 0.1, n)
    ols = fit_soft_sensor(df, "y", ["x1", "x2", "x3"], lag=2)
    rf = fit_ml_soft_sensor(df, "y", ["x1", "x2", "x3"], lag=2, model="랜덤포레스트")
    print("OLS 검증 R²", round(ols["metrics"]["검증 R²"], 3), "| RF 검증 R²", round(rf["metrics"]["검증 R²"], 3))
    assert ols["metrics"]["검증 R²"] < 0.5 < 0.95 < rf["metrics"]["검증 R²"]
    assert rf["importance"].index[-1] == "x3" and rf["importance"]["x3"] < 0.05      # pure noise input ranks last
    assert rf["result"].index.equals(ols["result"].index) and rf["split"] == ols["split"]  # same rows, same split
    for name in ("그래디언트 부스팅", "신경망 (MLP)"):
        r = fit_ml_soft_sensor(df, "y", ["x1", "x2", "x3"], lag=2, model=name)
        print(name, "검증 R²", round(r["metrics"]["검증 R²"], 3))
        assert r["metrics"]["검증 R²"] > 0.9
    tuned = fit_ml_soft_sensor(df, "y", ["x1", "x2", "x3"], lag=2, model="그래디언트 부스팅", tune_trials=5)
    assert set(tuned["tuned"]) == {"n_estimators", "max_depth", "learning_rate"} and tuned["metrics"]["검증 R²"] > 0.9, tuned["tuned"]

    # Residual alarm with the out-of-sample limit: healthy validation data stays near 1 %, a sensor offset is flagged.
    val = rf["result"]["구분"] == "검증"
    in_sample = residual_alarm(rf["result"])[1][val].mean()
    oos = residual_alarm(rf["result"], limit=rf["alarm_limit"])[1][val].mean()
    print(f"healthy validation alarm rate: in-sample limit {in_sample:.1%} → out-of-sample limit {oos:.1%}")
    assert oos < 0.04 and oos < in_sample
    bad = df.copy()
    bad.loc[bad.index[1300:1350], "y"] += 3
    r = fit_ml_soft_sensor(bad, "y", ["x1", "x2", "x3"], lag=2)
    lim, alarm = residual_alarm(r["result"], limit=r["alarm_limit"])
    assert alarm.loc[bad.index[1300:1350]].mean() > 0.9

    frozen = df.assign(x3=1.0)
    try:
        fit_ml_soft_sensor(frozen, "y", ["x1", "x2", "x3"], lag=2)
        raise AssertionError("constant input must raise ValueError")
    except ValueError as e:
        assert "x3" in str(e)
    a, b = (fit_ml_soft_sensor(df, "y", ["x1", "x2", "x3"], lag=2)["importance"] for _ in range(2))
    assert list(a.index) == list(b.index) and np.allclose(a, b, atol=1e-6)  # seeded: same ranking on every refit
    print("ok")
