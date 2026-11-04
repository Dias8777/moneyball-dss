"""
Аудит состава: профиль стиля клуба, соответствие роли (role fit),
сравнение с клубами-ориентирами по позициям, правила продажи/аренды.
"""
import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Стиль клуба (data-driven)
# ---------------------------------------------------------------------------
def squad(snap: pd.DataFrame, club=None) -> pd.DataFrame:
    club = club or C.CLUB
    return snap[snap.club_name == club].copy()


def style_profile(snap: pd.DataFrame, club=None, league=None, top_n=18) -> pd.Series:
    """
    Средний z-score игроков клуба относительно игроков ТОЙ ЖЕ позиционной группы в лиге.
    Сравнение внутри позиции убирает эффект состава (у кого больше защитников — у того «выше heading»).
    """
    club, league = club or C.CLUB, league or C.LEAGUE
    lg = snap[(snap.league_name == league) & (snap.group != "GK")]
    attrs = C.OUTFIELD_ATTRS
    mu = lg.groupby("group")[attrs].transform("mean")
    sd = lg.groupby("group")[attrs].transform("std")
    z = (lg[attrs] - mu) / sd
    z["club_name"] = lg.club_name
    z["overall"] = lg.overall
    z = z.sort_values("overall", ascending=False).groupby("club_name").head(top_n)
    return z[z.club_name == club][attrs].mean().sort_values(ascending=False)


def league_style_matrix(snap: pd.DataFrame, league=None) -> pd.DataFrame:
    league = league or C.LEAGUE
    clubs = snap[snap.league_name == league].club_name.unique()
    return pd.DataFrame({c: style_profile(snap, c, league) for c in clubs}).T


def tilted_weights(profile: pd.Series, tilt=C.STYLE_TILT) -> dict:
    """Базовые веса ролей, сдвинутые в сторону того, что клуб ценит сильнее лиги."""
    out = {}
    for g, w in C.ROLE_WEIGHTS.items():
        w2 = {}
        for a, v in w.items():
            t = float(np.clip(profile.get(a, 0.0), -1, 1)) if g != "GK" else 0.0
            w2[a] = v * (1 + tilt * t)
        if g != "GK":
            for a, p in profile.items():
                if a not in w2 and p > 0.3:          # «фирменные» качества клуба
                    w2[a] = tilt * float(p)
        out[g] = w2
    return out


# ---------------------------------------------------------------------------
# Role fit
# ---------------------------------------------------------------------------
def role_fit(snap: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    fit_z: взвешенный z-score атрибутов роли относительно игроков 1-го дивизиона той же группы.
    fit_pct: перцентиль внутри группы (0–100).
    fit_rating: fit_z в шкале рейтинга FIFA — средний игрок топ-лиги группы ~ его средний overall.
    """
    out = snap.copy()
    out["fit_z"] = np.nan
    for g, w in weights.items():
        m = out.group == g
        ref = out[m & (out.league_level == 1)]
        a = list(w)
        wv = np.array([w[k] for k in a])
        z = (out.loc[m, a] - ref[a].mean()) / ref[a].std()
        out.loc[m, "fit_z"] = (z.values * wv).sum(1) / wv.sum()
        out.loc[m, "fit_pct"] = out.loc[m, "fit_z"].rank(pct=True) * 100
        # калибровка в шкалу overall: линейная регрессия overall ~ fit_z по топ-лигам
        k, b = np.polyfit(out.loc[m & (out.league_level == 1), "fit_z"],
                          out.loc[m & (out.league_level == 1), "overall"], 1)
        out.loc[m, "fit_rating"] = b + k * out.loc[m, "fit_z"]
    # итоговая «полезность для клуба»: общий класс + соответствие стилю
    out["club_rating"] = 0.6 * out.overall + 0.4 * out.fit_rating
    return out


# ---------------------------------------------------------------------------
# Сравнение позиций с остальной лигой
# ---------------------------------------------------------------------------
def starters(club_df: pd.DataFrame, col="club_rating") -> pd.DataFrame:
    """Лучшие игроки по группам в количестве мест в схеме (+1 запасной для глубины)."""
    rows = []
    for g, n in C.FORMATION.items():
        sub = club_df[club_df.group == g].sort_values(col, ascending=False).reset_index(drop=True)
        for i, r in sub.iterrows():
            rows.append({**r.to_dict(), "depth_rank": i + 1,
                         "role": "старт" if i < n else ("ротация" if i < n + 1 else "резерв")})
    return pd.DataFrame(rows)


def league_position_table(fit: pd.DataFrame, leagues=None, col="club_rating") -> pd.DataFrame:
    """Для каждого клуба лиг-ориентиров и группы — средний рейтинг стартовых игроков и место."""
    leagues = leagues or C.BENCH_LEAGUES
    lg = fit[fit.league_name.isin(leagues) | (fit.club_name == C.CLUB)]
    recs = []
    for club, cdf in lg.groupby("club_name"):
        for g, n in C.FORMATION.items():
            top = cdf[cdf.group == g].nlargest(n + 1, col)[col].tolist()
            start = np.mean(top[:n]) if len(top) >= n else (np.mean(top) if top else np.nan)
            depth = top[n] if len(top) > n else np.nan
            recs.append({"club": club, "group": g, "start": start, "depth": depth})
    t = pd.DataFrame(recs)
    t["rank"] = t.groupby("group").start.rank(ascending=False, method="min")
    return t


def needs(table: pd.DataFrame, club=None, top_k=None) -> pd.DataFrame:
    """
    Слабые места: разрыв между стартовым уровнем клуба и уровнем цели — top_k-го по силе
    соперника (Брентфорд: 6-й клуб АПЛ; Ливерпуль: лучший соперник; ПСЖ: 3-й клуб топ-5 лиг).
    """
    club, top_k = club or C.CLUB, top_k or C.TARGET_RANK
    rows = []
    for g in C.FORMATION:
        t = table[table.group == g].sort_values("start", ascending=False)
        bench = t[t.club != club].start.iloc[top_k - 1]
        me = t[t.club == club].iloc[0]
        rows.append({"group": g, "name": C.GROUP_NAMES[g], "club_start": me.start,
                     "club_depth": me.depth, "league_rank": int(me["rank"]),
                     "target": bench, "gap": bench - me.start,
                     # глубина: насколько запасной хуже стартера (травма/дисквалификация = провал)
                     "depth_drop": me.start - me.depth,
                     "league_depth_drop": (t.start - t.depth).median()})
    return pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)


def need_groups(need: pd.DataFrame, top_gap=3, depth_mult=2.0) -> list:
    """Нужда = 3 позиции с наибольшим отставанием от цели + позиции, где нет замены
    (запасной хуже стартера более чем вдвое сильнее, чем в среднем по лиге)."""
    by_gap = need.group.head(top_gap).tolist()
    by_depth = need[need.depth_drop > depth_mult * need.league_depth_drop].group.tolist()
    return by_gap + [g for g in by_depth if g not in by_gap]


# ---------------------------------------------------------------------------
# Экспертные правила (база знаний продукционного типа): продать / аренда / оставить
# ---------------------------------------------------------------------------
def squad_actions(st: pd.DataFrame) -> pd.DataFrame:
    """st — результат starters() + колонки прогноза pred_delta (из модели развития)."""
    acts, why = [], []
    for _, r in st.iterrows():
        pd_ = r.get("pred_delta", 0.0)
        if r.age >= 30 and r.contract_years_left <= 1:
            acts.append("Продать / не продлевать")
            why.append(f"{r.age} лет, контракт до {int(r.club_contract_valid_until_year)}: стоимость ещё есть, дальше будет падать")
        elif r.role == "резерв" and r.value_eur >= 3e6 and r.age >= 23 and r.pot_gap < 6:
            acts.append("Продать")
            why.append(f"не в основе, но стоит €{r.value_eur/1e6:.1f}M — деньги работают лучше в слабой позиции")
        elif r.age <= 21 and r.pot_gap >= 8 and r.role != "старт":
            acts.append("Аренда для развития")
            why.append(f"потенциал +{int(r.pot_gap)}, нужен игровой опыт")
        elif r.role == "старт" and pd_ <= -1.5:
            acts.append("Готовить замену")
            why.append(f"модель ждёт спада рейтинга ({pd_:+.1f})")
        else:
            acts.append("Оставить")
            why.append("")
    st = st.copy()
    st["action"] = acts
    st["reason"] = why
    return st
