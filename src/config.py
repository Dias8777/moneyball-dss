"""
Глобальные настройки проекта: пути, клуб, тактическая схема, бюджет, роли.
Все допущения собраны здесь, чтобы их было легко показать и изменить на защите.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "male_players_legacy.csv"
DATA_CACHE = ROOT / "data" / "players_clean.parquet"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

# ---------------------------------------------------------------------------
# Контекст решения
# ---------------------------------------------------------------------------
CLUB = "Brentford"
LEAGUE = "Premier League"
CURRENT_VERSION = 23          # FIFA 23 = состояние на сентябрь 2022 (сезон 2022/23)
FIRST_VERSION = 15

# Схема 4-3-3: сколько стартовых мест в каждой позиционной группе
FORMATION = {"GK": 1, "CB": 2, "FB": 2, "DM": 1, "CM": 2, "W": 2, "ST": 1}
GROUP_NAMES = {
    "GK": "Вратарь", "CB": "Центральный защитник", "FB": "Крайний защитник",
    "DM": "Опорный полузащитник", "CM": "Центральный полузащитник",
    "W": "Вингер", "ST": "Нападающий",
}

# Первая позиция в player_positions -> позиционная группа
POS_TO_GROUP = {
    "GK": "GK", "CB": "CB",
    "RB": "FB", "LB": "FB", "RWB": "FB", "LWB": "FB",
    "CDM": "DM", "CM": "CM", "CAM": "CM",
    "LW": "W", "RW": "W", "LM": "W", "RM": "W",
    "ST": "ST", "CF": "ST",
}

# ---------------------------------------------------------------------------
# Финансовые допущения (FIFA-стоимость ≠ реальная трансферная цена!)
# ---------------------------------------------------------------------------
BUDGET_FEE_EUR = 60_000_000        # бюджет на трансферы (допущение)
BUDGET_WAGE_WEEK_EUR = 200_000     # лимит суммарной новой недельной зарплаты (допущение)
FEE_PREMIUM = 1.3                  # клубы АПЛ переплачивают относительно оценки (допущение)
CONTRACT_YEARS = 4                 # длина нового контракта
WAGE_FLOOR_SHARE = 0.8             # новичок получает не меньше 80% медианы зарплат клуба

# ---------------------------------------------------------------------------
# Признаки
# ---------------------------------------------------------------------------
OUTFIELD_ATTRS = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve",
    "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
    "power_stamina", "power_strength", "power_long_shots", "mentality_aggression",
    "mentality_interceptions", "mentality_positioning", "mentality_vision",
    "mentality_penalties", "mentality_composure", "defending_marking_awareness",
    "defending_standing_tackle", "defending_sliding_tackle",
]
GK_ATTRS = [
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes", "movement_reactions",
    "attacking_short_passing", "skill_long_passing", "mentality_composure",
]

def attrs_for(group: str) -> list:
    return GK_ATTRS if group == "GK" else OUTFIELD_ATTRS

# ---------------------------------------------------------------------------
# Тактическая модель: базовые веса ролей (экспертные, 1..3).
# Брентфорд Томаса Франка: прессинг, прямая игра, стандарты, борьба в воздухе.
# Эти веса затем корректируются data-driven профилем стиля клуба (audit.style_profile).
# ---------------------------------------------------------------------------
ROLE_WEIGHTS = {
    "GK": {"goalkeeping_reflexes": 3, "goalkeeping_diving": 3, "goalkeeping_positioning": 2,
           "goalkeeping_handling": 2, "goalkeeping_kicking": 2, "skill_long_passing": 2,
           "mentality_composure": 1},
    "CB": {"defending_marking_awareness": 3, "defending_standing_tackle": 3,
           "attacking_heading_accuracy": 3, "power_jumping": 2, "power_strength": 2,
           "mentality_interceptions": 2, "defending_sliding_tackle": 1,
           "attacking_short_passing": 1, "mentality_composure": 1},
    "FB": {"movement_sprint_speed": 2, "movement_acceleration": 2, "power_stamina": 3,
           "attacking_crossing": 3, "defending_standing_tackle": 2,
           "defending_marking_awareness": 1, "mentality_interceptions": 1,
           "attacking_short_passing": 1},
    "DM": {"mentality_interceptions": 3, "defending_standing_tackle": 3,
           "attacking_short_passing": 2, "skill_long_passing": 2, "power_stamina": 2,
           "power_strength": 1, "mentality_composure": 1, "mentality_aggression": 2},
    "CM": {"power_stamina": 3, "attacking_short_passing": 2, "mentality_vision": 2,
           "skill_ball_control": 2, "mentality_aggression": 2, "power_long_shots": 1,
           "mentality_interceptions": 1, "skill_fk_accuracy": 1},
    "W":  {"movement_sprint_speed": 3, "movement_acceleration": 2, "skill_dribbling": 2,
           "attacking_finishing": 2, "attacking_crossing": 2, "power_stamina": 2,
           "movement_agility": 1, "mentality_positioning": 1},
    "ST": {"attacking_finishing": 3, "mentality_positioning": 3,
           "attacking_heading_accuracy": 2, "power_strength": 2, "power_jumping": 1,
           "power_shot_power": 1, "mentality_composure": 2, "mentality_penalties": 1},
}
STYLE_TILT = 0.5   # насколько профиль стиля клуба сдвигает веса ролей

# Лиги без лицензии EA: у игроков вымышленные имена (в Бразилии топ-игрок «Палмейраса» — Ronaldo Cabrais).
# Для обучения моделей рейтинги пригодны, но покупать «несуществующих» игроков нельзя.
UNLICENSED_LEAGUES = ["Serie A (Brazil)"]

RANDOM_STATE = 42
