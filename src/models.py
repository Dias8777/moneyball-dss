"""
ML-часть:
  1. Модель развития: Δoverall в следующей версии FIFA (временная валидация, без утечки).
  2. Модель цены: как FIFA формирует стоимость (и прогноз стоимости перепродажи).
  3. Кластеризация игровых стилей внутри позиции.
  4. Поиск похожих игроков (дешёвые аналоги).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (adjusted_rand_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data import GROWTH_FEATURES

RS = C.RANDOM_STATE
DECLINE = -2   # «провал»: рейтинг упал на 2+ пункта за год


# ---------------------------------------------------------------------------
# 1. Модель развития
# ---------------------------------------------------------------------------
def _hgb_reg(**kw):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=50, random_state=RS, **kw)


def _reg_metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p), "Spearman": spearmanr(y, p).statistic}


def evaluate_growth(p: pd.DataFrame) -> dict:
    """
    Временное разбиение: train = FIFA 15–20 (цели 16–21), val = FIFA 21 (->22), test = FIFA 22 (->23).
    Модель никогда не видит будущих версий относительно тех, на которых её проверяют.
    """
    X, y = p[GROWTH_FEATURES], p.delta
    tr, va, te = p.fifa_version <= 20, p.fifa_version == 21, p.fifa_version == 22
    trva = tr | va

    age_bucket = lambda d: d.age.clip(16, 38)
    age_mean = p[trva].groupby(age_bucket(p[trva])).delta.mean()
    preds = {
        "Без изменений (Δ=0)": np.zeros(te.sum()),
        "Среднее по возрасту": age_bucket(p[te]).map(age_mean).fillna(0).values,
        "Ridge (линейная)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(1.0)).fit(X[trva], y[trva]).predict(X[te]),
    }
    # подбор числа итераций на валидации, затем переобучение на train+val
    best = min([200, 400, 800], key=lambda it: mean_absolute_error(
        y[va], _hgb_reg().set_params(max_iter=it).fit(X[tr], y[tr]).predict(X[va])))
    hgb = _hgb_reg().set_params(max_iter=best).fit(X[trva], y[trva])
    preds["Gradient Boosting"] = hgb.predict(X[te])
    reg = pd.DataFrame({k: _reg_metrics(y[te], v) for k, v in preds.items()}).T

    # риск провала (классификация)
    yb = (y <= DECLINE).astype(int)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(p.loc[trva, ["age"]], yb[trva])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50,
                                         random_state=RS).fit(X[trva], yb[trva])
    p_clf = clf.predict_proba(X[te])[:, 1]
    cls = {"base_rate": float(yb[te].mean()),
           "AUC: только возраст": roc_auc_score(yb[te], lr.predict_proba(p.loc[te, ["age"]])[:, 1]),
           "AUC: Gradient Boosting": roc_auc_score(yb[te], p_clf)}
    # калибровка по децилям
    dec = pd.qcut(p_clf, 10, labels=False, duplicates="drop")
    calib = pd.DataFrame({"pred": p_clf, "true": yb[te].values, "dec": dec}).groupby("dec").mean()

    # интервалы (квантили 10% / 90%)
    lo = _hgb_reg(loss="quantile", quantile=0.1).fit(X[trva], y[trva]).predict(X[te])
    hi = _hgb_reg(loss="quantile", quantile=0.9).fit(X[trva], y[trva]).predict(X[te])
    cover = float(((y[te] >= lo) & (y[te] <= hi)).mean())

    # ошибка по возрастам: где модель полезнее всего
    err = pd.DataFrame({"age": p.loc[te, "age"].values, "abs_err": np.abs(y[te] - preds["Gradient Boosting"]),
                        "abs_err_base": np.abs(y[te] - preds["Среднее по возрасту"])})
    err["age_band"] = pd.cut(err.age, [15, 20, 23, 26, 29, 32, 45],
                             labels=["≤20", "21–23", "24–26", "27–29", "30–32", "33+"])
    by_age = err.groupby("age_band", observed=True)[["abs_err", "abs_err_base"]].mean()

    imp = pd.Series(_permutation_importance(hgb, X[te], y[te]), index=GROWTH_FEATURES).sort_values(ascending=False)

    test_df = p.loc[te, ["player_id", "short_name", "age", "overall", "delta", "group"]].assign(
        pred=preds["Gradient Boosting"], lo=lo, hi=hi, p_decline=p_clf)
    return {"regression": reg, "classification": cls, "calibration": calib, "coverage_80": cover,
            "by_age": by_age, "importance": imp, "best_iter": best, "test_df": test_df,
            "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}}


def _permutation_importance(model, X, y, n_rep=3):
    rng = np.random.default_rng(RS)
    base = mean_absolute_error(y, model.predict(X))
    out = []
    for c in X.columns:
        s = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            s.append(mean_absolute_error(y, model.predict(Xp)) - base)
        out.append(np.mean(s))
    return out


def fit_growth_final(p: pd.DataFrame):
    """Финальные модели на всех парах до FIFA 22->23; применяются к FIFA 23 (прогноз на 2023/24)."""
    X, y = p[GROWTH_FEATURES], p.delta
    return {
        "mean": _hgb_reg().fit(X, y),
        "lo": _hgb_reg(loss="quantile", quantile=0.1).fit(X, y),
        "hi": _hgb_reg(loss="quantile", quantile=0.9).fit(X, y),
        "risk": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50,
                                               random_state=RS).fit(X, (y <= DECLINE).astype(int)),
    }


def predict_growth(models, cur: pd.DataFrame) -> pd.DataFrame:
    X = cur[GROWTH_FEATURES]
    return pd.DataFrame({
        "pred_delta": models["mean"].predict(X),
        "pred_lo": models["lo"].predict(X),
        "pred_hi": models["hi"].predict(X),
        "p_decline": models["risk"].predict_proba(X)[:, 1],
    }, index=cur.index)


def league_move_effect(p: pd.DataFrame, to_league=C.LEAGUE, min_n=15) -> pd.DataFrame:
    """
    Скрытая закономерность: что происходит с рейтингом игрока после перехода в АПЛ из лиги X,
    сверх того, что ожидалось по его возрасту и данным (out-of-fold остатки модели).
    """
    X, y = p[GROWTH_FEATURES], p.delta
    oof = np.zeros(len(p))
    for tr, te in GroupKFold(5).split(X, y, p.player_id):
        oof[te] = _hgb_reg().set_params(max_iter=200).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    p = p.assign(resid=y - oof)
    mv = p[(p.league_next == to_league) & (p.league_name != to_league)]
    stay = p[(p.league_next == to_league) & (p.league_name == to_league)]
    t = mv.groupby("league_name").agg(n=("resid", "size"), resid=("resid", "mean"),
                                      delta=("delta", "mean")).query("n >= @min_n")
    t.loc["(остались в АПЛ)"] = [len(stay), stay.resid.mean(), stay.delta.mean()]
    return t.sort_values("resid")


# ---------------------------------------------------------------------------
# 2. Модель цены
# ---------------------------------------------------------------------------
VALUE_FEATURES = ["overall", "potential", "pot_gap", "age", "international_reputation",
                  "contract_years_left", "league_level", "is_GK", "is_CB", "is_FB",
                  "is_DM", "is_CM", "is_W", "is_ST"]


def evaluate_value(cur: pd.DataFrame) -> pd.DataFrame:
    X, y = cur[VALUE_FEATURES], cur.log_value
    res = {}
    for name, mk, cols in [("Линейная: только overall", lambda: LinearRegression(), ["overall"]),
                           ("Линейная: все признаки", lambda: make_pipeline(SimpleImputer(strategy="median"), LinearRegression()), VALUE_FEATURES),
                           ("Gradient Boosting", lambda: _hgb_reg(), VALUE_FEATURES)]:
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=RS).split(X):
            pred[te] = mk().fit(X.iloc[tr][cols], y.iloc[tr]).predict(X.iloc[te][cols])
        ape = np.abs(np.expm1(pred) - cur.value_eur) / cur.value_eur
        res[name] = {"R2 (log)": r2_score(y, pred), "Медиана ошибки, %": 100 * np.median(ape)}
    return pd.DataFrame(res).T


def fit_value(cur: pd.DataFrame):
    return _hgb_reg().fit(cur[VALUE_FEATURES], cur.log_value)


def project_value(model, cur: pd.DataFrame, delta: pd.Series, years=1,
                  contract_left=C.CONTRACT_YEARS) -> np.ndarray:
    """Стоимость через `years` лет при прогнозном рейтинге и новом контракте клуба."""
    f = cur[VALUE_FEATURES].copy()
    f["overall"] = (cur.overall + delta).clip(upper=cur.potential.clip(lower=cur.overall))
    f["pot_gap"] = cur.potential - f.overall
    f["age"] = cur.age + years
    f["contract_years_left"] = max(contract_left - years, 0)
    return np.expm1(model.predict(f))


# ---------------------------------------------------------------------------
# 3. Кластеризация стилей
# ---------------------------------------------------------------------------
def _top_attrs(center: np.ndarray, attrs: list, k=3) -> str:
    idx = np.argsort(center)[::-1][:k]
    short = lambda a: a.split("_", 1)[1].replace("_", " ")
    return " / ".join(short(attrs[i]) for i in idx)


def cluster_styles(cur: pd.DataFrame, ks=range(2, 7)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    KMeans внутри каждой позиционной группы на стандартизованных атрибутах,
    где каждый игрок нормирован на свой средний уровень (иначе кластеры = «сильные/слабые», а не стиль).
    Выбор k — по silhouette, устойчивость — ARI по бутстрэп-подвыборкам.
    """
    labels = pd.Series(index=cur.index, dtype=object)
    names = pd.Series(index=cur.index, dtype=object)
    summary = []
    for g in C.FORMATION:
        m = cur.group == g
        a = C.attrs_for(g)
        Z = style_matrix(cur.loc[m, a])
        rng = np.random.default_rng(RS)
        samp = rng.choice(len(Z), size=min(3000, len(Z)), replace=False)
        sil = {k: silhouette_score(Z[samp], KMeans(k, n_init=5, random_state=RS).fit_predict(Z[samp]))
               for k in ks}
        k = max(sil, key=sil.get)
        km = KMeans(k, n_init=10, random_state=RS).fit(Z)
        aris = []
        for s in range(5):
            idx = np.random.default_rng(s).choice(len(Z), int(0.8 * len(Z)), replace=False)
            km2 = KMeans(k, n_init=5, random_state=s).fit(Z[idx])
            aris.append(adjusted_rand_score(km.labels_[idx], km2.labels_))
        cl_names = {i: _top_attrs(c, a) for i, c in enumerate(km.cluster_centers_)}
        labels[m] = [f"{g}-{l}" for l in km.labels_]
        names[m] = [cl_names[l] for l in km.labels_]
        for i in range(k):
            summary.append({"group": g, "cluster": f"{g}-{i}", "style": cl_names[i],
                            "n": int((km.labels_ == i).sum()), "k": k,
                            "silhouette": sil[k], "ARI_bootstrap": float(np.mean(aris))})
    return pd.DataFrame({"cluster": labels, "style": names}), pd.DataFrame(summary)


def style_matrix(A: pd.DataFrame) -> np.ndarray:
    """Профиль стиля: атрибуты минус средний уровень игрока, затем z-score по столбцам."""
    v = A.values.astype(float)
    v = v - v.mean(1, keepdims=True)
    return (v - v.mean(0)) / (v.std(0) + 1e-9)


# ---------------------------------------------------------------------------
# 4. Похожие игроки
# ---------------------------------------------------------------------------
def similar(cur: pd.DataFrame, player_id: int, n=10, max_value=None, max_age=None) -> pd.DataFrame:
    """Косинусная близость стилевых профилей внутри позиционной группы."""
    me = cur[cur.player_id == player_id].iloc[0]
    pool = cur[cur.group == me.group]
    a = C.attrs_for(me.group)
    Z = style_matrix(pool[a])
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    i = np.where(pool.player_id.values == player_id)[0][0]
    sim = Z @ Z[i]
    out = pool.assign(similarity=sim)
    out = out[(out.player_id != player_id) & ~out.league_name.isin(C.UNLICENSED_LEAGUES)]
    if max_value is not None:
        out = out[out.value_eur <= max_value]
    if max_age is not None:
        out = out[out.age <= max_age]
    return out.sort_values("similarity", ascending=False).head(n)


def evaluate_similarity(df: pd.DataFrame, cur: pd.DataFrame, n_per_group=300, k=10) -> pd.DataFrame:
    """
    Проверка поиска похожих без «разметки похожести» (её не существует):
      * Re-ID: берём профиль игрока из FIFA 22 и ищем его среди всех игроков группы в FIFA 23.
        Если метрика ловит реальный стиль, а не шум, сам игрок окажется среди ближайших.
      * Та же точная позиция и рабочая нога среди 10 соседей (эти поля в признаки не входят).
    Сравниваем: стилевой профиль (наш метод), сырые атрибуты, «ближайший overall» и случайный выбор.
    """
    rng = np.random.default_rng(RS)
    prev = df[df.fifa_version == C.CURRENT_VERSION - 1]
    rows = []
    for g in C.FORMATION:
        pool = cur[cur.group == g].reset_index(drop=True)
        a = C.attrs_for(g)
        both = prev[(prev.group == g) & prev.player_id.isin(pool.player_id)]
        q = both.sample(min(n_per_group, len(both)), random_state=RS)
        target = pool.reset_index().set_index("player_id").loc[q.player_id, "index"].values
        A = pd.concat([pool[a], q[a]]).astype(float)
        reps = {
            "Стиль (cosine, наш метод)": style_matrix(A),
            "Сырые атрибуты (cosine)": ((A - A.mean()) / A.std()).values,
        }
        for name, Z in reps.items():
            Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
            P, Q = Z[:len(pool)], Z[len(pool):]
            sims = Q @ P.T
            ranks = (sims > sims[np.arange(len(q)), target][:, None]).sum(1)
            nn = np.argsort(-sims, axis=1)[:, 1:k + 1]
            rows += _neigh_rows(name, g, ranks, nn, pool, target)
        ov = pool.overall.values
        d = np.abs(q.overall.values[:, None] - ov[None, :]) + rng.random((len(q), len(pool))) * 0.01
        ranks = (d < d[np.arange(len(q)), target][:, None]).sum(1)
        rows += _neigh_rows("Ближайший overall", g, ranks, np.argsort(d, axis=1)[:, 1:k + 1], pool, target)
        rnd = np.array([rng.choice(len(pool), k, replace=False) for _ in range(len(q))])
        rows += _neigh_rows("Случайно", g, rng.integers(0, len(pool), len(q)), rnd, pool, target)
    r = pd.DataFrame(rows)
    out = r.groupby("method")[["hit@1", "hit@10", "median_rank_pct"]].agg(
        {"hit@1": "mean", "hit@10": "mean", "median_rank_pct": "median"})
    mixed = r[r.group.isin(["FB", "W", "CM", "ST"])].groupby("method")[["same_position", "same_foot"]].mean()
    return out.join(mixed).sort_values("hit@10", ascending=False)


def _neigh_rows(name, g, ranks, nn, pool, target):
    pos, foot = pool.primary_pos.values, pool.preferred_foot.values
    return [{"method": name, "group": g, "hit@1": r == 0, "hit@10": r < 10,
             "median_rank_pct": 100 * r / len(pool),
             "same_position": (pos[n] == pos[t]).mean(), "same_foot": (foot[n] == foot[t]).mean()}
            for r, n, t in zip(ranks, nn, target)]
