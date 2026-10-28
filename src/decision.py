"""
Decision Intelligence: стоимость сделки, прирост качества, MCDA-ранжирование (WSM, TOPSIS),
оптимизация набора трансферов (MILP), анализ чувствительности.
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from . import config as C


# ---------------------------------------------------------------------------
# Кандидаты
# ---------------------------------------------------------------------------
def incumbents(st: pd.DataFrame) -> pd.Series:
    """Уровень самого слабого стартового игрока группы в следующем сезоне — его и заменяем."""
    s = st[st.role == "старт"].assign(next_=lambda d: d.club_rating + d.pred_delta)
    return s.groupby("group").next_.min()


def backups(st: pd.DataFrame) -> pd.Series:
    """Уровень первого запасного группы в следующем сезоне (0, если запасного нет)."""
    s = st[st.role == "ротация"].assign(next_=lambda d: d.club_rating + d.pred_delta)
    return s.groupby("group").next_.max().reindex(list(C.FORMATION)).fillna(0)


def club_strength(cur: pd.DataFrame, top_n=18) -> pd.Series:
    return cur.sort_values("overall", ascending=False).groupby("club_name").head(top_n) \
              .groupby("club_name").overall.mean()


def availability(cur: pd.DataFrame, club=C.CLUB, margin=2.0, wage_mult=2.0) -> pd.Series:
    """
    Правила реалистичности (без них система советует купить звёзд у грандов):
      * клуб сильнее нашего (> +margin по среднему рейтингу) не продаёт игроков основы;
      * зарплата игрока не больше wage_mult × максимальной зарплаты в нашем клубе.
    """
    s = club_strength(cur)
    stronger = cur.club_name.map(s) > s[club] + margin
    starter = ~cur.club_position.isin(["SUB", "RES"])
    max_wage = cur[cur.club_name == club].wage_eur.max()
    return ~(stronger & starter) & (cur.wage_eur <= wage_mult * max_wage)


def build_candidates(cur: pd.DataFrame, st: pd.DataFrame, club=C.CLUB, max_age=29,
                     fee_premium=C.FEE_PREMIUM, contract_years=C.CONTRACT_YEARS,
                     realistic=True) -> pd.DataFrame:
    club_wage = cur[cur.club_name == club].wage_eur.median()
    c = cur[(cur.club_name != club) & (cur.age <= max_age) & (cur.club_loaned_from.isna())
            & ~cur.league_name.isin(C.UNLICENSED_LEAGUES)].copy()
    c["available"] = availability(cur, club).loc[c.index]
    if realistic:
        c = c[c.available]
    c["fee"] = c.value_eur * fee_premium
    c["wage_new"] = np.maximum(c.wage_eur, C.WAGE_FLOOR_SHARE * club_wage)
    c["total_cost"] = c.fee + c.wage_new * 52 * contract_years
    c["rating_next"] = c.club_rating + c.pred_delta
    inc = incumbents(st)
    c["incumbent_next"] = c.group.map(inc)
    c["gain"] = (c.rating_next - c.incumbent_next).clip(lower=0)
    # для позиций без замены ценен и сильный запасной: половина прироста над текущим дублёром
    c["depth_gain"] = 0.5 * (c.rating_next - c.group.map(backups(st))).clip(lower=0)
    c["net_value"] = c.resale_2y - c.fee
    return c


def hidden_gems(cand: pd.DataFrame, groups, n=10) -> pd.DataFrame:
    """
    Недооценённые: по overall игрок НЕ лучше нашего стартера (простой рейтинг его отсеет),
    но с учётом стиля и прогноза развития он — улучшение. Сортировка по приросту на € затрат.
    """
    g = cand[cand.group.isin(groups) & (cand.gain > 0)].copy()
    g = g[g.overall + g.pred_delta <= g.incumbent_next]
    g["gain_per_10m"] = g.gain / (g.total_cost / 1e7)
    return g.sort_values("gain_per_10m", ascending=False).head(n)


# ---------------------------------------------------------------------------
# MCDA
# ---------------------------------------------------------------------------
CRITERIA = {
    #  колонка:          (название,                       тип)
    "rating_next":   ("Уровень в следующем сезоне",      "benefit"),
    "fit_pct":       ("Соответствие стилю, перцентиль",  "benefit"),
    "pred_delta":    ("Ожидаемый рост рейтинга",          "benefit"),
    "p_decline":     ("Риск спада",                       "cost"),
    "total_cost":    ("Полная стоимость сделки",          "cost"),
    "net_value":     ("Стоимость перепродажи − цена",     "benefit"),
}
DEFAULT_WEIGHTS = {"rating_next": 0.30, "fit_pct": 0.20, "pred_delta": 0.10,
                   "p_decline": 0.10, "total_cost": 0.20, "net_value": 0.10}


def wsm(df: pd.DataFrame, w=DEFAULT_WEIGHTS) -> pd.Series:
    """Min-max нормализация (1 = лучшее), затем взвешенная сумма."""
    s = 0
    for c, wt in w.items():
        x = df[c].astype(float)
        r = (x - x.min()) / (x.max() - x.min() + 1e-12)
        s = s + wt * (r if CRITERIA[c][1] == "benefit" else 1 - r)
    return s / sum(w.values())


def topsis(df: pd.DataFrame, w=DEFAULT_WEIGHTS) -> pd.Series:
    cols = list(w)
    X = df[cols].astype(float).values
    X = X - X.min(0)                                   # сдвиг: net_value бывает отрицательной
    R = X / (np.sqrt((X ** 2).sum(0)) + 1e-12)
    V = R * np.array([w[c] for c in cols]) / sum(w.values())
    ben = np.array([CRITERIA[c][1] == "benefit" for c in cols])
    a_pos = np.where(ben, V.max(0), V.min(0))
    a_neg = np.where(ben, V.min(0), V.max(0))
    dp = np.sqrt(((V - a_pos) ** 2).sum(1))
    dn = np.sqrt(((V - a_neg) ** 2).sum(1))
    return pd.Series(dn / (dp + dn + 1e-12), index=df.index)


def rank_group(cand: pd.DataFrame, group: str, budget=C.BUDGET_FEE_EUR, w=DEFAULT_WEIGHTS,
               only_upgrades=True, n=15) -> pd.DataFrame:
    d = cand[(cand.group == group) & (cand.fee <= budget)]
    if only_upgrades:
        d = d[d.gain > 0]
    if d.empty:
        return d
    d = d.assign(wsm=wsm(d, w), topsis=topsis(d, w))
    return d.sort_values("topsis", ascending=False).head(n)


def weight_sensitivity(cand: pd.DataFrame, group: str, n_sim=300, conc=30, top=5, seed=0, **kw):
    """Случайно возмущаем веса (Дирихле вокруг базовых) — как часто игрок остаётся в топ-5?"""
    base = rank_group(cand, group, n=200, **kw)
    if base.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    keys = list(DEFAULT_WEIGHTS)
    alpha = np.array([DEFAULT_WEIGHTS[k] for k in keys]) * conc
    hits = pd.Series(0, index=base.index)
    for _ in range(n_sim):
        w = dict(zip(keys, rng.dirichlet(alpha)))
        s = topsis(base, w)
        hits[s.nlargest(top).index] += 1
    out = base[["short_name", "club_name", "age", "overall", "fee"]].assign(top5_share=hits / n_sim)
    return out.sort_values("top5_share", ascending=False).head(10)


# ---------------------------------------------------------------------------
# Оптимизация набора трансферов
# ---------------------------------------------------------------------------
def optimize(cand: pd.DataFrame, budget=C.BUDGET_FEE_EUR, wage_budget=C.BUDGET_WAGE_WEEK_EUR,
             groups=None, max_signings=3, per_group=1, beta=0.5, risk_aversion=1.0,
             sales_income=0.0, depth_groups=()) -> pd.DataFrame:
    """
    MILP:  max Σ xᵢ·uᵢ
           uᵢ = gainᵢ·(1 − risk_aversion·p_declineᵢ) + beta·net_valueᵢ / €10M
           (для позиций из depth_groups gainᵢ = max(gain, depth_gain))
    при    Σ feeᵢ·xᵢ ≤ budget + sales_income
           Σ wage_newᵢ·xᵢ ≤ wage_budget
           Σ_{i∈g} xᵢ ≤ per_group  (новичок заменяет слабейшего стартового игрока группы)
           Σ xᵢ ≤ max_signings,  xᵢ ∈ {0,1}
    """
    d = cand[cand.fee <= budget + sales_income].copy()
    if groups is not None:
        d = d[d.group.isin(groups)]
    deep = d.group.isin(depth_groups)
    d.loc[deep, "gain"] = np.maximum(d.loc[deep, "gain"], d.loc[deep, "depth_gain"])
    d = d[d.gain > 0]
    if d.empty:
        return d
    u = d.gain * (1 - risk_aversion * d.p_decline) + beta * d.net_value / 1e7
    d = d[u > 0]
    u = u[u > 0]
    if d.empty:
        return d
    gs = sorted(d.group.unique())
    A = [d.fee.values, d.wage_new.values, np.ones(len(d))]
    ub = [budget + sales_income, wage_budget, max_signings]
    for g in gs:
        A.append((d.group == g).astype(float).values)
        ub.append(per_group)
    res = milp(c=-u.values, constraints=LinearConstraint(np.vstack(A), -np.inf, ub),
               integrality=np.ones(len(d)), bounds=Bounds(0, 1))
    if res.x is None:
        return d.iloc[0:0]
    pick = d[res.x > 0.5].assign(utility=u[res.x > 0.5])
    return pick.sort_values("utility", ascending=False)


def budget_scenarios(cand: pd.DataFrame, groups, factors=(0.5, 0.8, 1.0, 1.2), **kw) -> pd.DataFrame:
    rows = []
    for f in factors:
        p = optimize(cand, budget=C.BUDGET_FEE_EUR * f, groups=groups, **kw)
        rows.append({"budget_share": f, "budget_eur": C.BUDGET_FEE_EUR * f,
                     "players": ", ".join(f"{r.short_name} ({r.group})" for r in p.itertuples()),
                     "fee": p.fee.sum(), "gain": p.gain.sum(), "net_value": p.net_value.sum(),
                     "utility": p.utility.sum() if len(p) else 0})
    return pd.DataFrame(rows)
