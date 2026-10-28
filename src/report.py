"""
Скаутский отчёт по кандидату.
  * template_report — детерминированный шаблон (работает всегда, без сети);
  * llm_report      — LLM (Claude) пишет текст ТОЛЬКО по переданным фактам (grounded generation).
LLM не принимает решение и не добавляет чисел: решение уже принято моделью и оптимизатором,
LLM лишь объясняет его спортивному директору человеческим языком.
"""
import json

import pandas as pd

from . import config as C

MODEL = "claude-opus-5-5"

FACT_COLS = ["short_name", "long_name", "age", "club_name", "league_name", "primary_pos",
             "player_positions", "overall", "potential", "fit_pct", "style", "pred_delta",
             "pred_lo", "pred_hi", "p_decline", "value_eur", "fee", "wage_new", "resale_2y",
             "gain", "contract_years_left", "preferred_foot", "height_cm"]


def facts(row: pd.Series, profile: pd.Series = None, analogs: pd.DataFrame = None) -> dict:
    f = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in FACT_COLS if k in row}
    for k in ("fit_pct", "pred_delta", "pred_lo", "pred_hi", "p_decline", "gain"):
        if k in f:
            f[k] = round(float(f[k]), 2)
    attrs = C.attrs_for(row.group)
    f["top_attributes"] = {a: int(row[a]) for a in sorted(attrs, key=lambda a: -row[a])[:6]}
    role = list(C.ROLE_WEIGHTS[row.group])          # слабые места ищем среди атрибутов роли
    f["weak_attributes"] = {a: int(row[a]) for a in sorted(role, key=lambda a: row[a])[:3]}
    f["position_need"] = C.GROUP_NAMES[row.group]
    if profile is not None:
        f["club_style_top"] = {a: round(float(v), 2) for a, v in profile.head(5).items()}
    if analogs is not None and len(analogs):
        f["style_analogs"] = analogs[["short_name", "club_name", "value_eur"]].head(3).to_dict("records")
    return f


def template_report(f: dict) -> str:
    m = lambda x: f"€{x / 1e6:.1f}M"
    risk = "низкий" if f["p_decline"] < 0.1 else ("умеренный" if f["p_decline"] < 0.2 else "высокий")
    top = ", ".join(a.split("_", 1)[1].replace("_", " ") for a in list(f["top_attributes"])[:4])
    weak = ", ".join(a.split("_", 1)[1].replace("_", " ") for a in f["weak_attributes"])
    return (
        f"**{f['short_name']}** ({f['age']} лет, {f['club_name']}, {f['league_name']}), "
        f"позиция {f['primary_pos']}. Закрывает потребность «{f['position_need']}».\n\n"
        f"* Класс: overall {f['overall']}, потенциал {f['potential']}. "
        f"Соответствие стилю {C.CLUB}: {f['fit_pct']:.0f}-й перцентиль. Стиль: {f.get('style', '—')}.\n"
        f"* Сильные стороны: {top}. Зоны роста: {weak}.\n"
        f"* Прогноз на сезон: {f['pred_delta']:+.1f} к рейтингу "
        f"(80%-интервал {f['pred_lo']:+.1f}…{f['pred_hi']:+.1f}); риск спада {f['p_decline']:.0%} — {risk}.\n"
        f"* Деньги: оценка трансфера {m(f['fee'])}, зарплата ~€{f['wage_new'] / 1e3:.0f}k/нед, "
        f"прогноз стоимости через 2 года {m(f['resale_2y'])}.\n"
        f"* Прирост к текущему стартовому составу: {f['gain']:+.1f} пункта клубного рейтинга."
    )


SYSTEM = (
    "Ты — аналитик скаутского отдела футбольного клуба {club}. Пиши по-русски, кратко и по делу, "
    "для спортивного директора. Используй ТОЛЬКО факты из JSON. Не добавляй чисел, клубов, "
    "турниров или событий, которых нет в JSON. Если чего-то не хватает для вывода — так и скажи. "
    "Все рейтинги — оценки игры EA FIFA 23, а не реальная статистика матчей; стоимость — оценка FIFA, "
    "а не реальная трансферная цена. Структура: 1) кто это и какую проблему закрывает; "
    "2) почему подходит стилю клуба; 3) риски; 4) рекомендация (покупать / наблюдать / отказаться) "
    "с одним предложением обоснования. Не больше 180 слов."
)


def llm_report(f: dict) -> tuple[str, str]:
    """Возвращает (текст, источник). При любой ошибке — шаблон, чтобы демо не ломалось."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=2000,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM.format(club=C.CLUB),
            messages=[{"role": "user", "content":
                       "Факты о кандидате (JSON):\n" + json.dumps(f, ensure_ascii=False, default=str)}],
        )
        if resp.stop_reason == "refusal":
            return template_report(f), "шаблон (LLM отказалась отвечать)"
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return (text, f"LLM ({resp.model})") if text else (template_report(f), "шаблон (пустой ответ LLM)")
    except Exception as e:                       # нет ключа, нет сети, нет пакета и т.п.
        return template_report(f), f"шаблон (LLM недоступна: {type(e).__name__})"
