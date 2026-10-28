"""
Загрузка и очистка FIFA 15–23 (male_players legacy, Kaggle: stefanoleone992).
Одна строка = игрок × версия FIFA. Отсюда строим:
  * snapshot(23)  — текущий рынок;
  * pairs()       — панель (FIFA N -> FIFA N+1) для прогноза развития без утечки.
"""
import numpy as np
import pandas as pd

from . import config as C

KEEP = [
    "player_id", "fifa_version", "short_name", "long_name", "player_positions",
    "overall", "potential", "value_eur", "wage_eur", "age", "height_cm", "weight_kg",
    "league_id", "league_name", "league_level", "club_name", "club_position", "club_loaned_from",
    "club_contract_valid_until_year", "nationality_name", "preferred_foot",
    "weak_foot", "skill_moves", "international_reputation", "work_rate",
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
] + sorted(set(C.OUTFIELD_ATTRS + C.GK_ATTRS))


LEAGUE_ALIASES = {
    "Bundesliga #80": "Bundesliga (Austria)", "Premier League #332": "Premier League (Ukraine)",
    "Premier League #347": "Premier League (South Africa)", "Premier League #67": "Premier League (Russia)",
    "Primera Division #2019": "Primera Division (Venezuela)", "Primera Division #2020": "Primera Division (Peru)",
    "Primera Division #338": "Primera Division (Uruguay)", "Pro League #2013": "Pro League (UAE)",
    "Serie A #7": "Serie A (Brazil)", "Super League #2012": "Super League (China)",
    "Super League #2149": "Super League (India)", "Super League #63": "Super League (Greece)",
}


def load_raw(path=C.DATA_RAW) -> pd.DataFrame:
    return pd.read_csv(path, usecols=KEEP, low_memory=False)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Возвращает очищенную таблицу и отчёт о качестве данных."""
    rep = {"rows_raw": len(df)}
    rep["duplicates_player_version"] = int(df.duplicated(["player_id", "fifa_version"]).sum())
    df = df.drop_duplicates(["player_id", "fifa_version"]).copy()

    # Без клуба или стоимости — свободные агенты/ошибки выгрузки: для рынка бесполезны
    rep["missing_value_or_club"] = int((df.value_eur.isna() | df.club_name.isna()).sum())
    df = df[df.value_eur.notna() & df.club_name.notna() & (df.value_eur > 0)]

    # Одинаковые названия у разных лиг («Premier League»: Англия, Россия, Украина, ЮАР).
    # Основное название оставляем самой крупной лиге, остальным добавляем id.
    main_id = df.groupby("league_name").league_id.agg(lambda s: s.value_counts().idxmax())
    clash = df.league_id != df.league_name.map(main_id)
    rep["league_name_clashes"] = int(clash.sum())
    df.loc[clash, "league_name"] = df.league_name + " #" + df.league_id.astype("Int64").astype(str)
    df["league_name"] = df.league_name.replace(LEAGUE_ALIASES)

    df["primary_pos"] = df.player_positions.str.split(",").str[0].str.strip()
    df["group"] = df.primary_pos.map(C.POS_TO_GROUP)
    rep["unknown_position"] = int(df.group.isna().sum())
    df = df[df.group.notna()]

    # Сезон FIFA N начинается в году 2000+N-1 (FIFA 23 -> сентябрь 2022)
    season_start = 2000 + df.fifa_version - 1
    df["contract_years_left"] = (df.club_contract_valid_until_year - season_start).clip(0, 6)
    df["pot_gap"] = df.potential - df.overall
    df["log_value"] = np.log1p(df.value_eur)
    df["log_wage"] = np.log1p(df.wage_eur.fillna(0))
    df["on_loan"] = df.club_loaned_from.notna().astype(int)

    # Атрибуты вратарей для полевых и наоборот иногда пустые/бессмысленные — заполняем медианой группы
    attrs = sorted(set(C.OUTFIELD_ATTRS + C.GK_ATTRS))
    rep["missing_attrs_cells"] = int(df[attrs].isna().sum().sum())
    df[attrs] = df.groupby(["fifa_version", "group"])[attrs].transform(lambda s: s.fillna(s.median()))

    # Выбросы: стоимость сильно скошена — работаем с логарифмом, саму стоимость не режем
    rep["value_skew"] = float(df.value_eur.skew())
    rep["rows_clean"] = len(df)
    rep["by_version"] = df.fifa_version.value_counts().sort_index().to_dict()
    return df.reset_index(drop=True), rep


def load(cache=True) -> tuple[pd.DataFrame, dict]:
    if cache and C.DATA_CACHE.exists():
        df = pd.read_parquet(C.DATA_CACHE)
        return df, {}
    df, rep = clean(load_raw())
    df.to_parquet(C.DATA_CACHE, index=False)
    return df, rep


def snapshot(df: pd.DataFrame, version=C.CURRENT_VERSION) -> pd.DataFrame:
    return df[df.fifa_version == version].reset_index(drop=True)


GROWTH_FEATURES = [
    "age", "overall", "potential", "pot_gap", "international_reputation",
    "league_level", "log_value", "log_wage", "contract_years_left", "on_loan",
    "skill_moves", "weak_foot", "height_cm", "movement_reactions", "mentality_composure",
    "prev_delta", "is_GK", "is_CB", "is_FB", "is_DM", "is_CM", "is_W", "is_ST",
]


def add_group_dummies(df: pd.DataFrame) -> pd.DataFrame:
    for g in C.FORMATION:
        df[f"is_{g}"] = (df.group == g).astype(int)
    return df


def pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Панель: признаки в версии N, цель — изменение рейтинга в N+1.
    prev_delta = overall_N - overall_{N-1}: известно в момент N, поэтому не утечка.
    Внимание: в панель попадают только игроки, оставшиеся в игре (survivorship bias).
    """
    base = df[["player_id", "fifa_version", "overall", "value_eur", "club_name", "league_name"]]
    nxt = base.rename(columns={"overall": "overall_next", "value_eur": "value_next",
                               "club_name": "club_next", "league_name": "league_next"})
    nxt = nxt.assign(fifa_version=nxt.fifa_version - 1)
    prv = base[["player_id", "fifa_version", "overall"]].rename(columns={"overall": "overall_prev"})
    prv = prv.assign(fifa_version=prv.fifa_version + 1)

    p = df.merge(nxt, on=["player_id", "fifa_version"], how="inner")
    p = p.merge(prv, on=["player_id", "fifa_version"], how="left")
    p["prev_delta"] = (p.overall - p.overall_prev).fillna(0)
    p["delta"] = p.overall_next - p.overall
    p["moved_club"] = (p.club_next != p.club_name).astype(int)
    return add_group_dummies(p)


def current_features(df: pd.DataFrame, version=C.CURRENT_VERSION) -> pd.DataFrame:
    """Срез FIFA 23 с теми же признаками, что в панели (для прогноза на сезон 2023/24)."""
    cur = snapshot(df, version)
    prv = df[df.fifa_version == version - 1][["player_id", "overall"]].rename(columns={"overall": "overall_prev"})
    cur = cur.merge(prv, on="player_id", how="left")
    cur["prev_delta"] = (cur.overall - cur.overall_prev).fillna(0)
    return add_group_dummies(cur)
