"""Equipment-agnostic correlation, bivariate regression and residual diagnostics (statsmodels/scipy; no UI).

Process data are autocorrelated: consecutive samples are not independent, so textbook p-values are far too small.
Every p-value here therefore uses an autocorrelation-corrected sample size or HAC (Newey-West) standard errors.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

# VIF of an exactly derived tag (ΔP = P_bottom − P_top) is infinite by definition; reported as ∞, not warned about.
# Set once at import, after statsmodels' own "always" filters: catch_warnings() in a call would race between sessions.
warnings.filterwarnings("ignore", category=SingularMatrixWarning)
warnings.filterwarnings("ignore", message="The design matrix is poorly conditioned")

METHODS = {"피어슨 (직선 관계)": "pearson", "스피어만 (순위 · 단조 곡선)": "spearman", "켄달 (순위 · 이상값에 강함)": "kendall",
           "편상관 (나머지 변수 영향 제거)": "partial"}
MODELS = ["선형", "2차 다항식", "3차 다항식", "로그 (y = a + b·ln x)", "지수 (y = a·e^(bx))", "거듭제곱 (y = a·x^b)"]
_DEGREE = {"선형": 1, "2차 다항식": 2, "3차 다항식": 3}


def usable(df):
    """Columns a correlation can use: ≥ 3 values and not constant. Returns (usable columns, excluded columns)."""
    ok = [c for c in df.columns if df[c].count() >= 3 and df[c].std() > 0]
    return ok, [c for c in df.columns if c not in ok]


def thin(df, n):
    """Evenly spaced rows (keeps time order and coverage) when there are more than n."""
    return df if len(df) <= n else df.iloc[np.linspace(0, len(df) - 1, n).astype(int)]


def corr_matrix(df, method):
    if method == "partial":
        d = df.dropna()
        if len(d) < df.shape[1] + 3:
            raise ValueError(f"모든 변수에 값이 있는 행이 {len(d)}개뿐이라 편상관을 계산할 수 없습니다. 결측이 많은 변수를 빼 보세요.")
        prec = np.linalg.pinv(d.corr().to_numpy())
        dd = np.sqrt(np.abs(np.diag(prec)))
        pc = -prec / np.outer(dd, dd)
        np.fill_diagonal(pc, 1.0)
        return pd.DataFrame(pc, df.columns, df.columns).clip(-1, 1)
    # Kendall is O(n log n) per pair in scipy; 20k evenly spaced rows keep a 50-variable matrix interactive.
    return (thin(df, 20_000) if method == "kendall" else df).corr(method=method)


def cluster_order(c):
    """Variables reordered so that strongly related ones sit together (average-linkage on 1 − |r|)."""
    if len(c) < 3:
        return list(c.columns)
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    d = 1 - c.abs().fillna(0).to_numpy()
    d = ((d + d.T) / 2).clip(0)
    np.fill_diagonal(d, 0)
    return list(c.columns[leaves_list(linkage(squareform(d, checks=False), "average"))])


def dcor(x, y):
    """Distance correlation: 0 only when x and y are independent, so it also catches U-shapes that r misses."""
    def centered(a):
        d = np.abs(a[:, None] - a[None, :])
        return d - d.mean(0) - d.mean(1)[:, None] + d.mean()
    A, B = centered(np.asarray(x, float)), centered(np.asarray(y, float))
    vx, vy = (A * A).mean(), (B * B).mean()
    return float(np.sqrt(max((A * B).mean(), 0) / np.sqrt(vx * vy))) if vx > 0 and vy > 0 else float("nan")


def n_effective(x, y):
    """Bartlett's correction: n·(1 − ρx·ρy)/(1 + ρx·ρy), ρ = lag-1 autocorrelations. 10 000 slowly drifting samples
    may carry the information of only a few dozen independent ones."""
    n = len(x)
    rho = x.autocorr(1) * y.autocorr(1)
    if not np.isfinite(rho):
        return float(n)
    return float(np.clip(n * (1 - rho) / (1 + rho), 3, n))


def r_pvalue(r, n):
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1:
        return 0.0 if np.isfinite(r) and abs(r) >= 1 else float("nan")
    t = r * np.sqrt((n - 2) / (1 - r * r))
    return float(2 * stats.t.sf(abs(t), n - 2))


def relation(r, dc, gain, monotone):
    if max(abs(r), dc) < 0.3:
        return "관계 약함"
    if gain >= 0.08:
        # Not only U-shapes: two operating modes (e.g. a high-pressure period) mixed in one scatter look the same.
        return "단조 곡선 (비선형)" if monotone else "비단조 (U자 곡선 또는 운전 모드 혼재)"
    return "직선에 가까움"


def _eta2(df, bins=10):
    """eta[i, j] = correlation ratio η² of column j on the deciles of column i (share of j's variance a free-form curve
    of i explains; rows where both exist), and whether that curve of decile means is monotonic. Group sums are one
    matrix product per column (one-hot deciles × values), so 100 columns take well under a second."""
    Y = df.to_numpy(float)
    have = ~np.isnan(Y)
    Y0 = np.where(have, Y, 0.0)
    p = Y.shape[1]
    eta, mono = np.full((p, p), np.nan), np.ones((p, p), bool)
    for i in range(p):
        ok = have[:, i]
        if ok.sum() < 3 * bins:
            continue
        levels, codes = np.unique(Y[ok, i], return_inverse=True)
        if len(levels) <= bins:  # few distinct values (1/0 status, set points): group by value, not arbitrary tie order
            q, nb = codes, len(levels)
        else:
            q, nb = np.argsort(np.argsort(Y[ok, i], kind="stable")) * bins // ok.sum(), bins  # decile of each row
        onehot = np.eye(nb)[q]                                     # rows × groups
        h, y0 = have[ok].astype(float), Y0[ok]
        cnt, tot = onehot.T @ h, onehot.T @ y0                     # bins × columns
        with np.errstate(divide="ignore", invalid="ignore"):
            means = tot / cnt
            fitted = means[q]
            mean_all = y0.sum(0) / h.sum(0)
            sse = (np.where(h > 0, y0 - fitted, 0) ** 2).sum(0)
            sst = (np.where(h > 0, y0 - mean_all, 0) ** 2).sum(0)
            eta[i] = 1 - sse / sst
        # Spearman of the decile means against decile order (ranks via double argsort; NaN means sort last).
        rk = np.argsort(np.argsort(np.nan_to_num(means, nan=np.inf), axis=0), axis=0) - (nb - 1) / 2
        order = np.arange(nb) - (nb - 1) / 2
        mono[i] = np.abs(order @ rk / (order @ order)) >= 0.9 if nb > 2 else True  # two points are always monotonic
    return eta, mono


def _dcor(df, rows=300):
    """Distance-correlation matrix on ≤ rows evenly spaced rows. Columns complete in that sample share one matrix
    product of their centred distance matrices; pairs involving gaps fall back to pairwise-complete rows."""
    complete = df.dropna()
    s = thin(complete if len(complete) >= rows else df, rows)  # complete rows keep every pair on the fast path
    p = s.shape[1]
    out = np.full((p, p), np.nan)
    full = [j for j, c in enumerate(s.columns) if s[c].notna().all()]

    def centered(a):
        d = np.abs(a[:, None] - a[None, :])
        return d - d.mean(0) - d.mean(1)[:, None] + d.mean()

    if full:
        F = np.stack([centered(s.iloc[:, j].to_numpy(float)).ravel() for j in full]).astype(np.float32)
        G = (F @ F.T).astype(float) / F.shape[1]
        d = np.sqrt(np.clip(np.diag(G), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[np.ix_(full, full)] = np.sqrt(np.clip(G, 0, None) / np.outer(d, d))
    done = np.zeros((p, p), bool)
    done[np.ix_(full, full)] = True  # NaN here means a constant column: no pairwise retry
    for i in range(p):
        for j in range(i + 1, p):
            if not done[i, j]:
                # From the full data, not the thinned rows: a sparse column (lab results) barely overlaps an evenly
                # spaced sample, and distance correlation on a handful of points is large even for pure noise.
                sub = thin(df.iloc[:, [i, j]].dropna(), rows)
                if len(sub) >= 20:
                    out[i, j] = out[j, i] = dcor(sub.iloc[:, 0], sub.iloc[:, 1])
    return out


def pair_table(df, dcor_rows=300, bins=10):
    """Every variable pair: Pearson, Spearman, distance correlation, curvature gain, relation type and an
    autocorrelation-corrected p. Matrix-at-a-time, so 100 variables (≈ 5 000 pairs) take seconds, not minutes."""
    cols = list(df.columns)
    X = df.astype(float)
    have = X.notna().to_numpy(float)
    n = have.T @ have
    r = X.corr().to_numpy()
    rho = X.corr(method="spearman").to_numpy()
    dc = _dcor(X, dcor_rows)
    eta, mono = _eta2(X, bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        gain_ij = eta - r ** 2 - (bins - 1) / n  # curve of j on i's deciles; minus η²'s small-sample upward bias
    ac = np.array([X[c].autocorr(1) for c in cols])
    ac = np.where(np.isfinite(ac), ac, 0)
    rr = np.outer(ac, ac)
    n_eff = np.clip(n * (1 - rr) / (1 + rr), 3, n)
    i, j = np.triu_indices(len(cols), 1)
    keep = (n[i, j] >= 5) & np.isfinite(r[i, j])
    i, j = i[keep], j[keep]
    fwd, back = gain_ij[i, j], gain_ij[j, i]
    use_fwd = np.nan_to_num(fwd, nan=-1) >= np.nan_to_num(back, nan=-1)
    gain = np.clip(np.nan_to_num(np.where(use_fwd, fwd, back)), 0, None)
    # Count a curve only when it beats the straight line beyond chance: F-test of η² over r² on the effective n
    # (a few dozen lab results make a 10-bin curve look 0.1 better than a line on pure noise).
    ne = n_eff[i, j]
    with np.errstate(invalid="ignore", divide="ignore"):
        extra = gain + (bins - 1) / n[i, j]
        f = (extra / (bins - 2)) / np.clip(1 - extra - r[i, j] ** 2, 1e-9, None) * (ne - bins)
        curve_p = np.where(ne > bins + 2, stats.f.sf(f, bins - 2, np.clip(ne - bins, 1, None)), 1.0)
    gain = np.where(curve_p < 0.01, gain, 0.0)
    monotone = np.where(use_fwd, mono[i, j], mono[j, i])
    out = pd.DataFrame({
        "변수 1": np.array(cols, dtype=object)[i], "변수 2": np.array(cols, dtype=object)[j], "n": n[i, j].astype(int),
        "피어슨 r": r[i, j], "스피어만 ρ": rho[i, j], "거리상관": dc[i, j],
        "곡선 설명력 추가": np.where(n[i, j] >= 3 * bins, gain, 0.0),
    })
    out["관계 유형"] = [relation(a, b if np.isfinite(b) else 0, c, m)
                     for a, b, c, m in zip(out["피어슨 r"], out["거리상관"], out["곡선 설명력 추가"], monotone)]
    out["유효 n"] = np.round(n_eff[i, j]).astype(int)
    out["p값 (자기상관 보정)"] = [r_pvalue(a, b) for a, b in zip(out["피어슨 r"], n_eff[i, j])]
    strength = np.fmax(np.fmax(out["피어슨 r"].abs(), out["스피어만 ρ"].abs()), out["거리상관"])
    return out.assign(_s=strength).sort_values("_s", ascending=False, kind="stable").drop(columns="_s").reset_index(drop=True)


def vif(df):
    """Variance inflation factor per variable (rows with every value only). > 10: that variable is nearly a linear
    combination of the others, so OLS coefficients become unstable."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    d = df.dropna()
    if len(d) < df.shape[1] + 2:
        raise ValueError(f"모든 변수에 값이 있는 행이 {len(d)}개뿐이라 VIF를 계산할 수 없습니다.")
    X = sm.add_constant(d.to_numpy(float), has_constant="add")
    with np.errstate(divide="ignore", invalid="ignore"):
        v = [variance_inflation_factor(X, i + 1) for i in range(d.shape[1])]
    v = pd.Series(v, index=d.columns, name="VIF").replace(np.nan, np.inf)
    return v.where(v < 1e6, np.inf).sort_values(ascending=False)  # ≥ 10⁶ is round-off of an exact identity


# ---------- plain-language sentences for people without a statistics background ----------

def strength(r):
    a = abs(r)
    return "강한" if a >= 0.7 else "중간 정도의" if a >= 0.4 else "약한" if a >= 0.2 else "거의 없는"


def r_phrase(x, y, r, kind="직선에 가까움"):
    """'X 가 오르면 Y 도 오름' style sentence for one pair."""
    if abs(r) < 0.2:
        if kind.startswith(("단조", "비단조")):
            return (f"**{x}** — 직선 관계는 거의 없지만 (r = {r:+.2f}) 곡선 관계가 있음: 올라갔다 내려오는 모양이거나 "
                    "운전 모드가 섞였을 수 있음 → 이변량 회귀(2차식)·운전 구간별 비교로 확인")
        return f"**{x}** — 뚜렷한 관계 없음 (r = {r:+.2f})"
    move = "함께 오름" if r > 0 else "반대로 내려감"
    s = f"**{x}** — {strength(r)} {'양' if r > 0 else '음'}의 관계 (r = {r:+.2f}: {x}이(가) 오르면 {y}은(는) {move})"
    if kind.startswith("단조"):
        s += ", 곡선 모양 → 이변량 회귀에서 로그·지수·2차식을 비교해 보세요"
    elif kind.startswith("비단조"):
        s += ", 올라갔다 내려오는 곡선이거나 운전 모드가 섞였을 수 있음 → 이변량 회귀(2차식)·운전 구간별 비교로 확인"
    return s


def fit_quality(r2):
    if r2 >= 0.9:
        return "매우 좋음"
    if r2 >= 0.7:
        return "좋음"
    if r2 >= 0.5:
        return "보통"
    if r2 >= 0:
        return "부족"
    return "사용 불가 (평균값으로 찍는 것보다 못함)"



def pair_summary(pairs, focus=None, lab_prefix=None, top=3):
    """Plain-language lines for the pair table: the strongest partners of `focus`, or the strongest pairs overall."""
    strong = pairs[pairs["관계 유형"] != "관계 약함"]
    if focus:
        mine = strong[(strong["변수 1"] == focus) | (strong["변수 2"] == focus)]
        others = mine["변수 2"].where(mine["변수 1"] == focus, mine["변수 1"])
        lab = bool(lab_prefix) and str(focus).startswith(lab_prefix)
        if lab:  # for an analysis item the question is which operating variable moves with it, not other analyses
            mine, others = mine[~others.astype(str).str.startswith(lab_prefix)], others[~others.astype(str).str.startswith(lab_prefix)]
        if mine.empty:
            return [f"**{focus}** 와(과) 뚜렷하게 함께 움직이는 {'운전변수' if lab else '변수'}가 없습니다. "
                    "시차가 있을 수 있으니 '시차 상관' 탭도 확인하세요."]
        lines = [f"**{focus}** 와(과) 관계가 강한 {'운전변수' if lab else '변수'}:"]
        for other, (_, row) in zip(others.head(top), mine.head(top).iterrows()):
            lines.append("- " + r_phrase(other, focus, row["피어슨 r"], row["관계 유형"]))
        return lines
    if strong.empty:
        return ["뚜렷한 관계가 있는 변수 쌍이 없습니다. 기간·정상상태 필터를 바꾸거나 '시차 상관' 탭에서 지연 관계를 확인하세요."]
    lines = ["관계가 가장 강한 변수 쌍:"]
    for _, row in strong.head(top).iterrows():
        a, b, r = row["변수 1"], row["변수 2"], row["피어슨 r"]
        lines.append(f"- **{a}** ↔ **{b}** — {strength(r)} {'양' if r > 0 else '음'}의 관계 (r = {r:+.2f}, {row['관계 유형']})")
    mixed = strong["관계 유형"].str.startswith("비단조").sum()
    if mixed >= max(2, len(strong) // 4):
        lines.append(f"- '비단조' 쌍이 {mixed}개입니다. 운전 모드(부하·압력 등)가 섞였을 가능성이 크니 '운전 구간별 비교' 탭을 확인하세요.")
    if lab_prefix and any(str(c).startswith(lab_prefix) for c in pd.concat([pairs["변수 1"], pairs["변수 2"]])):
        lines.append(f"- 분석데이터가 있습니다. 위 **변수로 거르기**에서 `{lab_prefix}…` 항목을 고르면 그 분석값과 관계된 운전변수만 봅니다.")
    return lines


def biv_summary(r, x, y):
    """One-paragraph reading of a bivariate fit: effect size in the user's units and fit quality in words."""
    met = r["metrics"]
    r2 = met["R²"]
    xt = f"{r['lag']}샘플 전의 {x}" if r["lag"] else x
    if r["model"] == "선형":
        b = float(r["coef"].loc["x", "계수"])
        s = f"**{xt}** 이(가) 1 오르면 **{y}** 은(는) 평균 **{b:+.4g}** 변합니다. "
    else:
        s = f"**{r['model']}** 모델: {xt} 에 따라 {y} 의 변화 폭이 달라지는 곡선이라 기울기가 X 위치마다 다릅니다 (그래프의 선 참고). "
    s += f"설명력 R² = {r2:.2f} → **{fit_quality(r2)}** ({y} 변동의 {max(r2, 0):.0%}를 설명)"
    p = met["모델 p값"]
    if np.isfinite(p) and p >= 0.01:
        s += ". 다만 p값이 커서 우연일 가능성을 배제할 수 없습니다"
    return s + ". 관계는 원인을 뜻하지 않습니다."

# ---------- operating modes: overall vs within-mode relations ----------

def segment_labels(index, starts):
    """Segment number (1 … k) of each timestamp for segments beginning at the sorted `starts`."""
    pos = np.searchsorted(pd.DatetimeIndex(starts).values, pd.DatetimeIndex(index).values, side="right")
    return pd.Series(np.clip(pos, 1, None), index=index)


def within_corr(df, groups):
    """Correlation of each row's deviation from its own mode's mean: the relation inside the modes, with the
    mode-to-mode level differences (pressure regime, load, before/after a revamp) taken out."""
    g = groups.reindex(df.index)
    d, g = df[g.notna()], g[g.notna()]
    return (d - d.groupby(g).transform("mean")).corr()


def mode_verdict(r_all, r_in):
    a, w = abs(r_all), abs(r_in)
    if not (np.isfinite(a) and np.isfinite(w)):
        return "-"
    if a >= 0.3 and w >= 0.3 and np.sign(r_all) != np.sign(r_in):
        return "방향 반대 (심슨의 역설)"
    if w - a >= 0.3:
        return "모드 차이가 관계를 가림"
    if a - w >= 0.3:
        return "모드 차이가 만든 상관"
    return "비슷함"


def mode_table(df, groups):
    """Every pair: r over all rows vs r inside the modes, sorted by how much the modes change the picture."""
    overall, within = df.corr().to_numpy(), within_corr(df, groups).to_numpy()
    cols = np.array(df.columns, dtype=object)
    i, j = np.triu_indices(len(cols), 1)
    out = pd.DataFrame({"변수 1": cols[i], "변수 2": cols[j], "전체 r": overall[i, j], "구간 안 r": within[i, j]})
    out["해석"] = [mode_verdict(a, w) for a, w in zip(out["전체 r"], out["구간 안 r"])]
    gap = (out["구간 안 r"] - out["전체 r"]).abs()
    return out.assign(_g=gap).sort_values("_g", ascending=False, kind="stable").drop(columns="_g").reset_index(drop=True)


def segment_stats(x, y, groups):
    """Per mode: rows, period, means, r (autocorrelation-corrected p) and the fitted straight line y = a + b·x."""
    d = pd.DataFrame({"x": x, "y": y, "g": groups.reindex(x.index)}).dropna()
    rows = []
    for g, part in d.groupby("g", sort=False):
        if len(part) < 5 or part["x"].std() == 0 or part["y"].std() == 0:
            rows.append({"구간": g, "행 수": len(part), "시작": part.index[0], "끝": part.index[-1]})
            continue
        b, a = np.polyfit(part["x"], part["y"], 1)
        r = part["x"].corr(part["y"])
        rows.append({"구간": g, "행 수": len(part), "시작": part.index[0], "끝": part.index[-1], "X 평균": part["x"].mean(),
                     "Y 평균": part["y"].mean(), "r": r, "기울기": b, "절편": a,
                     "p값 (자기상관 보정)": r_pvalue(r, n_effective(part["x"], part["y"]))})
    return pd.DataFrame(rows)


# ---------- bivariate regression ----------

def _check_domain(model, x, y):
    if model.startswith(("로그", "거듭제곱")) and (x <= 0).any():
        raise ValueError("로그·거듭제곱 모델은 X가 모두 0보다 커야 합니다.")
    if model.startswith(("지수", "거듭제곱")) and (y <= 0).any():
        raise ValueError("지수·거듭제곱 모델은 Y가 모두 0보다 커야 합니다.")


def _design(model, x, center, scale):
    import statsmodels.api as sm
    if _DEGREE.get(model, 1) > 1:  # centred and scaled powers: raw x³ at x ≈ 350 would make the fit numerically ill-conditioned
        z = (np.asarray(x, float) - center) / scale
        return sm.add_constant(np.column_stack([z ** k for k in range(1, _DEGREE[model] + 1)]), has_constant="add")
    xx = np.log(x) if model.startswith(("로그", "거듭제곱")) else np.asarray(x, float)
    return sm.add_constant(np.asarray(xx, float), has_constant="add")


def _log_y(model):
    return model.startswith(("지수", "거듭제곱"))


def _fmt(v):
    return f"{v:.6g}"


def _signed(v):
    return f"{'−' if v < 0 else '+'} {abs(v):.6g}"


def equation(model, params, center, scale, x="x", y="y"):
    if _DEGREE.get(model, 1) > 1:
        # Convert the centred/scaled fit back to powers of x: z = (x − center)/scale.
        poly = np.polynomial.Polynomial(params, domain=[center - scale, center + scale], window=[-1, 1]).convert()
        terms = [_signed(c) + ("" if k == 0 else f" × {x}" + ("" if k == 1 else f"^{k}")) for k, c in enumerate(poly.coef)]
        return f"{y} = " + " ".join(terms).removeprefix("+ ")
    a, b = params
    if model.startswith("로그"):
        return f"{y} = {_fmt(a)} {_signed(b)} × ln({x})"
    if model.startswith("지수"):
        return f"{y} = {_fmt(np.exp(a))} × exp({_fmt(b)} × {x})"
    if model.startswith("거듭제곱"):
        return f"{y} = {_fmt(np.exp(a))} × {x}^{_fmt(b)}"
    return f"{y} = {_fmt(a)} {_signed(b)} × {x}"


def bivariate(x, y, model="선형", lag=0, hac=True, grid_points=200):
    """y(t) = f(x(t − lag)). R², RMSE and residuals are in Y's original units for every model (log-y models are fitted
    in log space and back-transformed), so the model-comparison table is like-for-like."""
    import statsmodels.api as sm
    d = pd.concat([x.shift(lag).rename("x"), y.rename("y")], axis=1).dropna()
    n_par = _DEGREE.get(model, 1) + 1
    if len(d) < n_par + 3:
        raise ValueError(f"X와 Y가 함께 있는 행이 {len(d)}개뿐이라 회귀할 수 없습니다.")
    if d["x"].std() == 0 or d["y"].std() == 0:
        raise ValueError("X 또는 Y 값이 변하지 않아 회귀할 수 없습니다.")
    _check_domain(model, d["x"], d["y"])
    center, scale = float(d["x"].mean()), float(d["x"].std())
    X = _design(model, d["x"], center, scale)
    yt = np.log(d["y"]) if _log_y(model) else d["y"]
    n = len(d)
    if hac:
        fit = sm.OLS(yt.to_numpy(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, int(4 * (n / 100) ** (2 / 9)))})
    else:
        fit = sm.OLS(yt.to_numpy(float), X).fit()

    back = np.exp if _log_y(model) else (lambda v: v)
    d["적합값"] = back(fit.fittedvalues)
    d["잔차"] = d["y"] - d["적합값"]
    sse, sst = float((d["잔차"] ** 2).sum()), float(((d["y"] - d["y"].mean()) ** 2).sum())
    r2 = 1 - sse / sst
    gx = np.linspace(d["x"].min(), d["x"].max(), grid_points)
    frame = fit.get_prediction(_design(model, gx, center, scale)).summary_frame(alpha=0.05)
    grid = pd.DataFrame({"x": gx, **{k: back(frame[c].to_numpy()) for k, c in
                                      (("적합", "mean"), ("신뢰 하한", "mean_ci_lower"), ("신뢰 상한", "mean_ci_upper"),
                                       ("예측 하한", "obs_ci_lower"), ("예측 상한", "obs_ci_upper"))}})
    names = ["절편"] + (["z", "z²", "z³"][:_DEGREE[model]] if _DEGREE.get(model, 1) > 1
                      else ["ln x" if model.startswith(("로그", "거듭제곱")) else "x"])
    ci = fit.conf_int(0.05)
    coef = pd.DataFrame({"계수": fit.params, "95% 하한": ci[:, 0], "95% 상한": ci[:, 1], "p값": fit.pvalues}, index=names)
    return {
        "model": model, "lag": lag, "hac": hac, "fit": fit, "data": d, "grid": grid, "coef": coef,
        "center": center, "scale": scale,
        "metrics": {"R²": r2, "조정 R²": 1 - (1 - r2) * (n - 1) / max(n - n_par, 1), "RMSE": float(np.sqrt(sse / n)),
                    "n": n, "모델 p값": float(fit.f_pvalue) if np.isfinite(fit.f_pvalue) else float("nan")},
    }


def compare_models(x, y, lag=0):
    rows = {}
    for m in MODELS:
        try:
            r = bivariate(x, y, m, lag, hac=False, grid_points=2)
            rows[m] = {k: r["metrics"][k] for k in ("R²", "조정 R²", "RMSE", "n")}
        except ValueError as e:
            rows[m] = {"R²": np.nan, "조정 R²": np.nan, "RMSE": np.nan, "n": np.nan, "비고": str(e)}
    out = pd.DataFrame(rows).T
    out["비고"] = out.get("비고", pd.Series(index=out.index, dtype=object)).fillna("")
    return out.astype({"R²": float, "조정 R²": float, "RMSE": float, "n": float})


def influence(r, top=10):
    """Largest Cook's distances (points that alone move the fitted line most). Rule of thumb: > 4/n is influential."""
    cooks = r["fit"].get_influence().cooks_distance[0]
    d = r["data"].assign(쿡의_거리=cooks)
    return d[d["쿡의_거리"] > 4 / len(d)].nlargest(top, "쿡의_거리"), float((cooks > 4 / len(d)).mean())


# ---------- residual diagnostics (bivariate regression and soft-sensor models alike) ----------

def diagnose(resid, fitted, max_lag=40):
    """resid, fitted: Series on the same (time) index. Returns (verdict table, ACF series, standardized residuals)."""
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from statsmodels.tsa.stattools import acf

    d = pd.DataFrame({"r": resid, "f": fitted}).dropna()
    n = len(d)
    if n < 10:
        raise ValueError(f"잔차가 {n}개뿐이라 진단할 수 없습니다 (10개 이상 필요).")
    r = d["r"].to_numpy(float)
    z = (r - r.mean()) / r.std() if r.std() > 0 else r * 0
    dw = float(durbin_watson(r))
    k = max(1, min(max_lag, n // 4))
    rho = acf(r, nlags=k, fft=True)[1:]
    lb_lag = max(1, min(10, n // 5))
    # Ljung-Box Q from the FFT autocorrelations (statsmodels' acorr_ljungbox takes seconds on 10⁵ residuals).
    q_lb = n * (n + 2) * float((rho[:lb_lag] ** 2 / (n - np.arange(1, lb_lag + 1))).sum())
    lb_p = float(stats.chi2.sf(q_lb, lb_lag))
    try:
        bp_p = float(het_breuschpagan(r, sm.add_constant(d["f"].to_numpy(float), has_constant="add"))[1])
    except (ValueError, np.linalg.LinAlgError):
        bp_p = float("nan")
    _, jb_p, skew, kurt = (float(v) for v in jarque_bera(r))
    out3 = float((np.abs(z) > 3).mean())
    ok = "✅ 양호"
    warn = "⚠️ 주의"
    rows = [
        {"항목": "자기상관 (Durbin-Watson)", "값": f"{dw:.2f}", "판정": ok if 1.5 <= dw <= 2.5 else warn,
         "의미 · 조치": "2 근처면 잔차가 서로 독립. 2보다 많이 작으면 오차가 시간적으로 이어짐 → 지연(dead time)·누락 변수·동적 효과. "
                    "시차 상관 탭과 입력 지연을 확인하고, 계수 p값은 HAC 보정 값을 보세요."},
        {"항목": f"자기상관 (Ljung-Box, {lb_lag}샘플까지)", "값": f"p = {lb_p:.3g}", "판정": ok if lb_p >= 0.05 else warn,
         "의미 · 조치": "p < 0.05 이면 잔차에 남은 시간 패턴이 있음. 공정 데이터는 대부분 여기에 걸리며, 그 자체로 모델이 틀렸다는 뜻은 아닙니다."},
        {"항목": "등분산 (Breusch-Pagan)", "값": f"p = {bp_p:.3g}", "판정": ok if not bp_p < 0.05 else warn,
         "의미 · 조치": "p < 0.05 이면 예측값 크기에 따라 오차 크기가 달라짐 (깔때기 모양). 로그·거듭제곱 모델을 고려하고 "
                    "예측 구간을 전 범위에 똑같이 믿지 마세요."},
        {"항목": "정규성 (Jarque-Bera)", "값": f"p = {jb_p:.3g} · 왜도 {skew:+.2f} · 첨도 {kurt:.2f}",
         "판정": ok if jb_p >= 0.05 or (abs(skew) < 0.5 and kurt < 4.5) else warn,
         "의미 · 조치": "정규분포면 왜도 0, 첨도 3. 데이터가 많으면 아주 작은 차이도 p < 0.05 가 되므로 Q-Q 플롯 모양으로 판단하세요. "
                    "꼬리가 두꺼우면(첨도 큼) 간헐적 이상값·운전 모드 혼재를 의심."},
        {"항목": "이상 잔차 (|표준화 잔차| > 3)", "값": f"{out3:.2%}", "판정": ok if out3 <= 0.01 else warn,
         "의미 · 조치": "정규분포면 약 0.3%. 많으면 계기 이상·특이 운전 구간이 섞여 있음 → 잔차 시간 그래프에서 위치 확인."},
    ]
    acf_s = pd.Series(rho, index=range(1, k + 1), name="자기상관")
    return pd.DataFrame(rows), acf_s, pd.Series(z, index=d.index, name="표준화 잔차")


def qq_points(z, n=1000):
    """Theoretical vs sample quantiles of standardized residuals (at most n points for the chart)."""
    s = np.sort(np.asarray(z, float))
    idx = np.linspace(0, len(s) - 1, min(n, len(s))).astype(int)
    p = (idx + 0.5) / len(s)
    return pd.DataFrame({"이론 분위수": stats.norm.ppf(p), "잔차 분위수": s[idx]})


# ---------- lagged relationships ----------

def ccf(x, y, max_lag, diff=False):
    """corr(x(t − k), y(t)) for k = −max_lag … max_lag. Peak at k > 0: x leads y by k samples."""
    if diff:
        x, y = x.diff(), y.diff()
    return pd.Series({k: y.corr(x.shift(k)) for k in range(-max_lag, max_lag + 1)}, name="상관계수").rename_axis("지연")


GRANGER_ROWS = 10_000


def granger(x, y, max_lag, diff=True):
    """Uses the latest GRANGER_ROWS consecutive rows (statsmodels refits 2 models per lag: 40 s on 200k rows).
    p-value per lag of 'past x adds nothing to predicting y beyond y's own past' (F-test). Small p: x helps predict
    y (Granger-causes it) — predictive precedence, not proof of physical cause."""
    from statsmodels.tsa.stattools import grangercausalitytests
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if diff:
        d = d.diff().dropna()
    d = d.iloc[-GRANGER_ROWS:]
    if len(d) < 3 * max_lag + 10:
        raise ValueError(f"X와 Y가 함께 있는 행이 {len(d)}개라 지연 {max_lag}까지 검정할 수 없습니다. 최대 지연을 줄이세요.")
    if d["x"].std() == 0 or d["y"].std() == 0:
        raise ValueError("X 또는 Y 값이 변하지 않아 검정할 수 없습니다.")
    res = grangercausalitytests(d[["y", "x"]].to_numpy(float), maxlag=max_lag)
    return pd.Series({k: float(v[0]["ssr_ftest"][1]) for k, v in res.items()}, name="p값").rename_axis("지연")


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(1)
    n = 600
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    t = rng.uniform(-10, 35, n)
    df = pd.DataFrame({"temp": t, "ice": 5 * t + rng.normal(0, 5, n), "power": (t - 15) ** 2 + rng.normal(0, 20, n),
                       "exp": np.exp(0.08 * t) * rng.lognormal(0, 0.05, n), "noise": rng.normal(0, 1, n)}, index=idx)

    # Pair table flags a U-shape that Pearson calls 'weak', and a straight line as straight.
    p = pair_table(df).set_index(["변수 1", "변수 2"])
    assert p.loc[("temp", "power"), "관계 유형"].startswith("비단조"), p.loc[("temp", "power")]
    assert abs(p.loc[("temp", "power"), "피어슨 r"]) < 0.45 and p.loc[("temp", "power"), "거리상관"] > 0.5
    assert p.loc[("temp", "exp"), "관계 유형"].startswith("단조"), p.loc[("temp", "exp")]
    assert p.loc[("temp", "ice"), "관계 유형"] == "직선에 가까움"
    assert p.loc[("temp", "noise"), "관계 유형"] == "관계 약함" and p.loc[("temp", "noise"), "p값 (자기상관 보정)"] > 0.01
    assert {tuple(v) for v in pair_table(df).iloc[:2][["변수 1", "변수 2"]].to_numpy()} == {("temp", "ice"), ("temp", "exp")}

    # Matrix-at-a-time values equal pairwise ones even with gaps in different rows per column.
    gaps = df.copy()
    gaps.iloc[::7, 0] = np.nan
    gaps.iloc[3::11, 2] = np.nan
    pg = pair_table(gaps).set_index(["변수 1", "변수 2"])
    both = gaps[["temp", "power"]].dropna()
    assert np.isclose(pg.loc[("temp", "power"), "피어슨 r"], both["temp"].corr(both["power"]))
    assert pg.loc[("temp", "power"), "n"] == len(both) and abs(pg.loc[("temp", "power"), "거리상관"] - dcor(*thin(both, 300).T.to_numpy())) < 0.05
    assert pg.loc[("temp", "power"), "관계 유형"].startswith("비단조")

    # Autocorrelation shrinks the effective sample size: two independent random walks look 'significant' otherwise.
    w = pd.DataFrame({"a": rng.normal(size=2000).cumsum(), "b": rng.normal(size=2000).cumsum()})
    assert n_effective(w["a"], w["b"]) < 200 and n_effective(df["temp"], df["noise"]) > 500

    # Partial correlation: ice and exp are related only through temp.
    m = corr_matrix(df[["temp", "ice", "exp"]], "partial")
    assert abs(m.loc["ice", "exp"]) < 0.2 < abs(df["ice"].corr(df["exp"]))
    assert cluster_order(corr_matrix(df, "pearson"))[:2] in (["temp", "ice"], ["ice", "temp"]) or \
        abs(cluster_order(corr_matrix(df, "pearson")).index("temp") - cluster_order(corr_matrix(df, "pearson")).index("ice")) == 1
    for meth in ("pearson", "spearman", "kendall"):
        assert corr_matrix(df, meth).shape == (5, 5)

    # Bivariate: quadratic beats linear on the U-shape; equations reproduce the fitted values in original units.
    cmp = compare_models(df["temp"], df["power"])
    assert cmp.loc["2차 다항식", "R²"] > 0.9 > cmp.loc["선형", "R²"] + 0.5
    lin = bivariate(df["temp"], df["ice"])
    assert abs(lin["coef"].loc["x", "계수"] - 5) < 0.1 and equation("선형", lin["coef"]["계수"], 0, 1).startswith("y = ")
    for model in ("2차 다항식", "3차 다항식"):
        r = bivariate(df["temp"], df["power"], model)
        poly = np.polynomial.Polynomial(r["coef"]["계수"].to_numpy(), domain=[r["center"] - r["scale"], r["center"] + r["scale"]],
                                        window=[-1, 1]).convert()
        assert np.allclose(poly(r["data"]["x"]), r["data"]["적합값"])
        assert equation(model, r["coef"]["계수"].to_numpy(), r["center"], r["scale"]).startswith("y = ")
    q = bivariate(df["temp"], df["power"], "2차 다항식")
    poly = np.polynomial.Polynomial(q["coef"]["계수"].to_numpy(), domain=[q["center"] - q["scale"], q["center"] + q["scale"]],
                                    window=[-1, 1]).convert()
    assert np.allclose(poly.coef, [225, -30, 1], rtol=0.1), poly.coef
    e = bivariate(df["temp"], df["exp"], "지수 (y = a·e^(bx))")
    assert abs(e["coef"].iloc[1, 0] - 0.08) < 0.005 and e["metrics"]["R²"] > 0.9
    g = e["grid"]
    assert (g["예측 하한"] <= g["신뢰 하한"]).all() and (g["신뢰 하한"] <= g["적합"]).all() and (g["적합"] <= g["신뢰 상한"]).all()
    for bad in ("로그 (y = a + b·ln x)", "거듭제곱 (y = a·x^b)"):
        assert np.isnan(compare_models(df["temp"], df["power"]).loc[bad, "R²"])  # temp ≤ 0 exists
    try:
        bivariate(df["temp"], df["temp"] * 0 + 1)
        raise AssertionError("constant y must raise")
    except ValueError:
        pass

    # Lag: y follows x by 5 samples → CCF peak at +5, HAC p-values larger than naive ones on autocorrelated data.
    x = pd.Series(rng.normal(size=1500)).rolling(5, min_periods=1).mean()
    y = x.shift(5) * 2 + rng.normal(0, 0.1, 1500)
    assert ccf(x, y, 20).idxmax() == 5 and ccf(x, y, 20, diff=True).idxmax() == 5
    assert bivariate(x, y, lag=5)["metrics"]["R²"] > 0.9 > bivariate(x, y, lag=0)["metrics"]["R²"]
    gp = granger(x, y, 6)
    assert gp.min() < 1e-6 and granger(pd.Series(rng.normal(size=1500)), y, 6).min() > 1e-3
    naive, robust = (bivariate(w["a"], w["b"], hac=h)["coef"].loc["x", "p값"] for h in (False, True))
    assert robust > naive

    # Residual diagnostics: autocorrelated residuals and heteroscedastic ones are flagged, white noise is not.
    fitted = pd.Series(np.linspace(1, 10, 1000))
    tab = diagnose(pd.Series(rng.normal(size=1000)), fitted)[0].set_index("항목")["판정"]
    assert (tab == "✅ 양호").all(), tab
    tab = diagnose(pd.Series(rng.normal(size=1000)).rolling(10, min_periods=1).mean(), fitted)[0].set_index("항목")["판정"]
    assert tab.iloc[0] == "⚠️ 주의"
    tab = diagnose(pd.Series(rng.normal(size=1000) * fitted), fitted)[0].set_index("항목")["판정"]
    assert tab["등분산 (Breusch-Pagan)"] == "⚠️ 주의"
    assert len(qq_points(rng.normal(size=5000))) == 1000
    from statsmodels.stats.diagnostic import acorr_ljungbox
    e = pd.Series(rng.normal(size=800)).rolling(3, min_periods=1).mean()
    lb = float(acorr_ljungbox(e.to_numpy(), lags=[10])["lb_pvalue"].iloc[0])
    ours = float(diagnose(e, fitted.iloc[:800])[0].set_index("항목").loc["자기상관 (Ljung-Box, 10샘플까지)", "값"].split("= ")[1])
    assert np.isclose(lb, ours, rtol=0.01, atol=1e-300), (lb, ours)

    # Influence: one gross outlier at high leverage has the largest Cook's distance.
    xo = pd.Series(np.r_[rng.normal(size=200), 8.0])
    yo = pd.Series(np.r_[xo[:200] + rng.normal(0, 0.2, 200), -8.0])
    top, frac = influence(bivariate(xo, yo, hac=False))
    assert top.index[0] == 200 and 0 < frac < 0.2

    # Operating modes: inside each mode B follows A (+), but the high mode sits lower → overall r is negative (Simpson);
    # C and D only share the mode shift → strong overall r that vanishes inside the modes.
    m = np.r_[np.zeros(300), np.ones(300)]
    A = rng.normal(0, 1, 600) + 4 * m
    modes = pd.DataFrame({"A": A, "B": A - 10 * m + rng.normal(0, 0.3, 600), "C": 5 * m + rng.normal(0, 1, 600),
                          "D": 5 * m + rng.normal(0, 1, 600)}, index=pd.date_range("2026-01-01", periods=600, freq="h"))
    lab = segment_labels(modes.index, [modes.index[0], modes.index[300]])
    assert lab.iloc[0] == 1 and lab.iloc[299] == 1 and lab.iloc[300] == 2 and lab.iloc[-1] == 2
    mt = mode_table(modes, lab).set_index(["변수 1", "변수 2"])
    assert mt.loc[("A", "B"), "전체 r"] < -0.3 and mt.loc[("A", "B"), "구간 안 r"] > 0.9
    assert mt.loc[("A", "B"), "해석"] == "방향 반대 (심슨의 역설)" and mt.loc[("C", "D"), "해석"] == "모드 차이가 만든 상관"
    sst = segment_stats(modes["A"], modes["B"], lab.map({1: "저", 2: "고"}))
    assert list(sst["구간"]) == ["저", "고"] and (sst["r"] > 0.9).all() and np.allclose(sst["기울기"], 1, atol=0.05)
    # Discrete x (1/0 status) is grouped by value: a time trend inside each state must not look like a curve.
    trend = pd.DataFrame({"s": np.tile([0.0, 1.0], 300)}).assign(y=lambda d: np.arange(600.0) + 400 * d.s)
    assert pair_table(trend).loc[0, "관계 유형"] == "직선에 가까움", pair_table(trend)

    assert strength(-0.8) == "강한" and strength(0.5) == "중간 정도의" and strength(0.1) == "거의 없는"
    assert "반대로 내려감" in r_phrase("온도", "순도", -0.9) and "곡선" in r_phrase("온도", "수요", -0.3, "비단조 (U자)")
    pt = pd.DataFrame({"변수 1": ["a", "a", "b"], "변수 2": ["[분석] q", "b", "c"], "피어슨 r": [-0.9, 0.5, 0.1],
                       "관계 유형": ["직선에 가까움", "비단조 (U자)", "관계 약함"]})
    assert "반대로 내려감" in pair_summary(pt, "[분석] q")[1] and "없습니다" in pair_summary(pt, "c")[0]
    pt2 = pd.concat([pt, pd.DataFrame({"변수 1": ["[분석] p"], "변수 2": ["[분석] q"], "피어슨 r": [0.95], "관계 유형": ["직선에 가까움"]})])
    assert not any("[분석] p" in l for l in pair_summary(pt2, "[분석] q", "[분석] "))  # other analyses left out
    assert any("[분석] p" in l for l in pair_summary(pt2, "[분석] q"))
    assert any("[분석]" in l for l in pair_summary(pt, None, "[분석] "))
    assert "평균" in biv_summary(lin, "temp", "ice") and "매우 좋음" in biv_summary(lin, "temp", "ice")
    assert fit_quality(0.95) == "매우 좋음" and fit_quality(0.6) == "보통" and fit_quality(-0.2).startswith("사용 불가")

    v = vif(pd.DataFrame({"a": t, "b": t * 2 + rng.normal(0, 0.01, n), "c": rng.normal(size=n)}))
    assert v["a"] > 100 and v["c"] < 2
    exact = vif(pd.DataFrame({"top": t, "btm": t * 0.5 + rng.normal(0, 1, n), "c": rng.normal(size=n)}).assign(dp=lambda d: d.btm - d.top))
    assert np.isinf(exact[["top", "btm", "dp"]]).all() and exact["c"] < 2
    print("ok")
