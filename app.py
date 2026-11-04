"""
Moneyball DSS — Streamlit-интерфейс для спортивного директора.
Запуск: streamlit run app.py   (сначала python run_pipeline.py)
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import audit, config as C, decision, models, report

st.set_page_config(page_title="Moneyball DSS", page_icon="⚽", layout="wide")

st.sidebar.title("⚽ Moneyball DSS")
club_key = st.sidebar.selectbox("Клуб", list(C.CLUBS), format_func=lambda k: C.CLUBS[k]["name"])
C.use(club_key)

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#c3c2b7"
M = lambda x: f"€{x / 1e6:,.1f}M"


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
@st.cache_data
def load(key):
    cur = pd.read_parquet(C.OUT / "players_23.parquet")
    squad = pd.read_csv(C.OUT / "squad.csv")
    need = pd.read_csv(C.OUT / "needs.csv")
    profile = pd.read_csv(C.OUT / "style_profile.csv", index_col=0).z
    return cur, squad, need, profile


@st.cache_data
def candidates(key, max_age, fee_premium, realistic):
    cur, squad, _, _ = load(key)
    return decision.build_candidates(cur, squad, max_age=max_age, fee_premium=fee_premium,
                                     realistic=realistic)


try:
    cur, squad, need, profile = load(club_key)
except FileNotFoundError:
    st.error(f"Нет результатов для клуба. Выполните `python run_pipeline.py --club {club_key}`.")
    st.stop()

need_groups = audit.need_groups(need)
core_groups = need.group.head(3).tolist()
depth_groups = [g for g in need_groups if g not in core_groups]
CORE = dict(core_groups=core_groups, core_min=1)

st.sidebar.caption(f"{C.CLUB} · {C.LEAGUE} · FIFA 23 (сентябрь 2022)  \n{C.CLUBS[club_key]['note']}")
page = st.sidebar.radio("Раздел", ["Обзор", "Аудит состава", "Кандидаты (MCDA)", "Похожие игроки",
                                   "Оптимизатор трансферов", "Проверка качества"])
st.sidebar.divider()
st.sidebar.subheader("Допущения")
max_age = st.sidebar.slider("Максимальный возраст покупки", 18, 33, 29)
fee_premium = st.sidebar.slider("Премия к оценке FIFA (реальные цены выше)", 1.0, 2.0, C.FEE_PREMIUM, 0.05)
realistic = st.sidebar.checkbox("Только реалистичные цели", True,
                                help="Сильные клубы не продают игроков основы; зарплата ≤ 2× максимума клуба")
cand = candidates(club_key, max_age, fee_premium, realistic)

SHOW = {"short_name": "Игрок", "club_name": "Клуб", "league_name": "Лига", "age": "Возраст",
        "overall": "Overall", "potential": "Потенциал", "fit_pct": "Стиль, %", "style": "Тип",
        "pred_delta": "Прогноз Δ", "p_decline": "Риск спада", "fee": "Цена", "wage_new": "Зарплата/нед",
        "gain": "Прирост", "net_value": "Перепродажа − цена"}


def table(d, cols=None, extra=None):
    cols = cols or [c for c in SHOW if c in d]
    out = d[cols + (extra or [])].rename(columns=SHOW)
    fmt = {"Цена": M, "Перепродажа − цена": M, "Зарплата/нед": lambda x: f"€{x / 1e3:,.0f}k",
           "Стиль, %": "{:.0f}", "Прогноз Δ": "{:+.1f}", "Риск спада": "{:.0%}", "Прирост": "{:+.2f}"}
    st.dataframe(out.style.format({k: v for k, v in fmt.items() if k in out}, na_rep="—"),
                 use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
if page == "Обзор":
    st.title(f"Moneyball 2.0: кого купить {C.CLUB_DAT}")
    st.markdown(
        f"> **Вопрос:** каких игроков {C.CLUB_RU} может купить в пределах бюджета, чтобы закрыть "
        "слабые позиции, — чтобы они подходили под стиль команды, росли в цене и не провалились?\n\n"
        f"**Цель клуба:** {C.GOAL}; ориентир по позициям — {C.TARGET_LABEL}.  \n"
        "**Пользователь:** спортивный директор. **Решение:** список покупок, продаж и аренд "
        "на трансферное окно.")
    c = st.columns(4)
    c[0].metric("Игроков в базе (FIFA 23)", f"{len(cur):,}".replace(",", " "))
    c[1].metric("Бюджет (допущение)", M(C.BUDGET_FEE_EUR))
    c[2].metric("Слабейшая позиция", need.iloc[0]["name"],
                help=f"Отставание от цели ({C.TARGET_LABEL}): {need.iloc[0].gap:.1f} пункта")
    plan = decision.optimize(cand, groups=need_groups, depth_groups=depth_groups, max_signings=4, **CORE)
    c[3].metric("Рекомендовано покупок", len(plan), M(plan.fee.sum()))
    st.subheader("Рекомендуемый план (базовые настройки)")
    table(plan)
    st.markdown(
        "**Как работает система**\n"
        f"1. **Аудит:** стиль клуба из данных, сравнение каждой позиции с клубами-ориентирами ({', '.join(C.BENCH_LEAGUES)}).\n"
        "2. **Role fit:** оценка игрока под требования роли, сдвинутые в сторону стиля клуба.\n"
        "3. **Модель развития:** прогноз рейтинга на следующий сезон, риск спада, 80%-интервал.\n"
        "4. **Похожие игроки:** дешёвые аналоги дорогих игроков по стилевому профилю.\n"
        "5. **MCDA (WSM/TOPSIS):** ранжирование с весами, которые задаёт директор.\n"
        "6. **MILP-оптимизатор:** лучший набор покупок при бюджете и лимите зарплат.")

# ---------------------------------------------------------------------------
elif page == "Аудит состава":
    st.title(f"Аудит состава {C.CLUB}")
    a, b = st.columns(2)
    with a:
        st.subheader("Стиль клуба (из данных)")
        p = pd.concat([profile.head(7), profile.tail(5)]).sort_values()
        fig = go.Figure(go.Bar(x=p.values, y=[i.split("_", 1)[1].replace("_", " ") for i in p.index],
                               orientation="h", marker_color=[BLUE if v > 0 else GREY for v in p.values],
                               hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="z-score к лиге (внутри позиции)")
        st.plotly_chart(fig, use_container_width=True)
        nm = lambda idx: ", ".join(i.split("_", 1)[1].replace("_", " ") for i in idx)
        st.caption(f"Сильнее лиги: {nm(profile.head(4).index)}. Слабее лиги: {nm(profile.tail(3).index)}. "
                   "Эти особенности усиливают веса ролей при оценке кандидатов.")
    with b:
        st.subheader("Позиции против клубов-ориентиров")
        n = need.rename(columns={"name": "Позиция", "league_rank": "Место среди ориентиров", "club_start": "Старт",
                                 "target": f"Цель ({C.TARGET_LABEL})", "gap": "Отставание",
                                 "depth_drop": "Провал при замене", "league_depth_drop": "…медиана"})
        st.dataframe(n.drop(columns=["group", "club_depth"]).style.format(precision=1),
                     hide_index=True, use_container_width=True)
        st.info(f"Потребности: **{', '.join(C.GROUP_NAMES[g] for g in need_groups)}**. "
                f"Первые три — по отставанию от цели (план обязан закрыть хотя бы одну), остальные — из-за слабой замены.")
    st.subheader("Состав и рекомендации экспертных правил")
    s = squad[["short_name", "group", "age", "overall", "club_rating", "role", "contract_years_left",
               "value_eur", "action", "reason"]].copy()
    s["value_eur"] = s.value_eur.map(M)
    s = s.rename(columns={"short_name": "Игрок", "group": "Группа", "age": "Возраст", "club_rating": "Рейтинг клуба",
                          "role": "Роль", "contract_years_left": "Лет контракта", "value_eur": "Стоимость",
                          "action": "Действие", "reason": "Почему"})
    s = s.rename(columns={"overall": "Overall"})
    st.dataframe(s.style.format({"Рейтинг клуба": "{:.1f}", "Лет контракта": "{:.0f}"}),
                 hide_index=True, use_container_width=True, height=420)

# ---------------------------------------------------------------------------
elif page == "Кандидаты (MCDA)":
    st.title("Ранжирование кандидатов")
    c1, c2 = st.columns([1, 2])
    with c1:
        g = st.selectbox("Позиция", list(C.FORMATION), index=list(C.FORMATION).index(need_groups[0]),
                         format_func=lambda x: C.GROUP_NAMES[x] + (" ⚠️" if x in need_groups else ""))
        bmax = int(C.BUDGET_FEE_EUR / 1e6)
        budget = st.slider("Максимальная цена, €M", 1, bmax, int(bmax * 2 / 3)) * 1e6
        only_up = st.checkbox("Только те, кто усиливает старт", True)
        st.markdown("**Веса критериев**")
        w = {k: st.slider(v[0], 0.0, 1.0, decision.DEFAULT_WEIGHTS[k], 0.05, key=k)
             for k, v in decision.CRITERIA.items()}
    if sum(w.values()) == 0:
        st.error("Сумма весов равна 0.")
        st.stop()
    r = decision.rank_group(cand, g, budget=budget, w=w, only_upgrades=only_up, n=25)
    with c2:
        if r.empty:
            st.warning("Нет кандидатов при этих ограничениях.")
            st.stop()
        fig = px.scatter(r, x="fee", y="rating_next", size="fit_pct", color="topsis",
                         color_continuous_scale="Blues", hover_name="short_name",
                         hover_data={"club_name": True, "age": True, "fee": ":,.0f", "topsis": ":.3f"},
                         labels={"fee": "Цена, €", "rating_next": "Уровень в следующем сезоне", "topsis": "TOPSIS"})
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    table(r, extra=["wsm", "topsis"])
    with st.expander("Устойчивость к весам: как часто игрок в топ-5 при случайных весах"):
        sens = decision.weight_sensitivity(cand, g, budget=budget, only_upgrades=only_up)
        st.dataframe(sens.style.format({"fee": M, "top5_share": "{:.0%}"}), hide_index=True)
    st.subheader("Скаутский отчёт")
    who = st.selectbox("Кандидат", r.short_name + " — " + r.club_name)
    row = r.iloc[(r.short_name + " — " + r.club_name).tolist().index(who)]
    analogs = models.similar(cur, row.player_id, n=3, max_value=row.value_eur * 0.7)
    f = report.facts(row, profile, analogs)
    use_llm = st.toggle("Текст от LLM (Claude, нужен ANTHROPIC_API_KEY)", False)
    if use_llm:
        with st.spinner("LLM пишет отчёт только по фактам из таблицы…"):
            txt, src = report.llm_report(f)
    else:
        txt, src = report.template_report(f), "шаблон"
    st.markdown(txt)
    st.caption(f"Источник текста: {src}. Решение принимает модель и оптимизатор, LLM только объясняет.")
    with st.expander("Факты, переданные в отчёт (JSON)"):
        st.json(f)

# ---------------------------------------------------------------------------
elif page == "Похожие игроки":
    st.title("Дешёвые аналоги дорогих игроков")
    stars = cur.sort_values("value_eur", ascending=False)
    label = stars.short_name + " — " + stars.club_name + " (" + stars.primary_pos + ")"
    who = st.selectbox("Эталонный игрок", label.head(3000).tolist())
    me = stars.iloc[label.tolist().index(who)]
    c = st.columns(3)
    share = c[0].slider("Не дороже, % от эталона", 5, 100, 25)
    age = c[1].slider("Возраст до", 18, 35, 26)
    n = c[2].slider("Сколько показать", 5, 30, 10)
    res = models.similar(cur, me.player_id, n=n, max_value=me.value_eur * share / 100, max_age=age)
    st.caption(f"{me.short_name}: overall {me.overall}, {M(me.value_eur)}, тип «{me['style']}»")
    if res.empty:
        st.warning("Никого не нашлось — ослабьте фильтры.")
        st.stop()
    table(res, ["short_name", "club_name", "league_name", "age", "overall", "potential", "style", "pred_delta"],
          extra=["value_eur", "similarity"])
    attrs = list(C.ROLE_WEIGHTS[me.group])
    fig = go.Figure()
    for r_, color in [(me, BLUE), (res.iloc[0], ORANGE)]:
        fig.add_trace(go.Scatterpolar(r=[r_[a] for a in attrs] + [r_[attrs[0]]],
                                      theta=[a.split("_", 1)[1].replace("_", " ") for a in attrs + attrs[:1]],
                                      name=f"{r_.short_name} ({M(r_.value_eur)})", line_color=color))
    fig.update_layout(height=420, polar=dict(radialaxis=dict(range=[30, 99])), margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
elif page == "Оптимизатор трансферов":
    st.title("Оптимизатор трансферного окна (MILP)")
    st.latex(r"\max \sum_i x_i\big[\text{gain}_i(1-\rho\,p^{\text{спад}}_i) + \beta\,\tfrac{\text{resale}_i-\text{fee}_i}{10^7}\big]"
             r"\quad\text{s.t.}\ \sum fee_i x_i \le B,\ \sum wage_i x_i \le W,\ \le 1\ \text{на позицию}")
    c = st.columns(3)
    budget = c[0].slider("Трансферный бюджет, €M", 0, int(C.BUDGET_FEE_EUR / 1e6 * 2.5),
                         int(C.BUDGET_FEE_EUR / 1e6), key=f"b_{club_key}") * 1e6
    wage = c[1].slider("Лимит новых зарплат, €k/нед", 20, int(C.BUDGET_WAGE_WEEK_EUR / 1e3 * 2.5),
                       int(C.BUDGET_WAGE_WEEK_EUR / 1e3), key=f"w_{club_key}") * 1e3
    k = c[2].slider("Максимум покупок", 1, 6, 4)
    c = st.columns(3)
    beta = c[0].slider("β: вес перепродажи", 0.0, 2.0, 0.5, 0.1)
    rho = c[1].slider("ρ: неприятие риска", 0.0, 3.0, 1.0, 0.1)
    groups = c[2].multiselect("Позиции", list(C.FORMATION), need_groups, format_func=lambda x: C.GROUP_NAMES[x])
    sells = squad[squad.action.str.startswith("Продать")]
    chosen = st.multiselect("Продать (деньги добавятся к бюджету)", sells.short_name.tolist(), sells.short_name.tolist(),
                            key=f"s_{club_key}")
    income = sells[sells.short_name.isin(chosen)].value_eur.sum()
    plan = decision.optimize(cand, budget=budget, wage_budget=wage, groups=groups or None,
                             max_signings=k, beta=beta, risk_aversion=rho, sales_income=income,
                             depth_groups=[g for g in depth_groups if g in groups], **CORE)
    m = st.columns(4)
    m[0].metric("Доступно", M(budget + income), f"+{M(income)} от продаж" if income else None)
    m[1].metric("Потрачено", M(plan.fee.sum()) if len(plan) else "€0")
    m[2].metric("Прирост состава", f"+{plan.gain.sum():.1f} пункта" if len(plan) else "0")
    m[3].metric("Стоимость через 2 года − цена", M(plan.net_value.sum()) if len(plan) else "€0")
    if plan.empty:
        st.warning("Оптимизатор ничего не выбрал: увеличьте бюджет или ослабьте ограничения.")
    else:
        table(plan, extra=["utility"])
        st.subheader("Уровень стартовых позиций: до и после")
        before = squad[squad.role == "старт"].groupby("group").club_rating.mean()
        after = before.copy()
        for r_ in plan.itertuples():
            s_ = squad[(squad.group == r_.group) & (squad.role == "старт")].club_rating.sort_values()
            vals = s_.values.copy()
            vals[0] = max(vals[0], r_.club_rating)
            after[r_.group] = vals.mean()
        fig = go.Figure([go.Bar(name="Сейчас", x=[C.GROUP_NAMES[g] for g in before.index], y=before, marker_color=GREY),
                         go.Bar(name="После покупок", x=[C.GROUP_NAMES[g] for g in after.index], y=after, marker_color=BLUE)])
        lo = min(before.min(), after.min())
        fig.update_layout(barmode="group", height=340, yaxis_range=[lo - 5, max(after.max(), before.max()) + 2],
                          margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Сценарии бюджета")
    sc = decision.budget_scenarios(cand, groups or need_groups, depth_groups=depth_groups, max_signings=k,
                                   beta=beta, risk_aversion=rho, **CORE)
    st.dataframe(sc.style.format({"budget_share": "{:.0%}", "budget_eur": M, "fee": M, "net_value": M,
                                  "gain": "{:.1f}", "utility": "{:.1f}"}), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
elif page == "Проверка качества":
    st.title("Почему результатам можно доверять (и где нельзя)")
    t = st.tabs(["Модель развития", "Модель цены", "Похожие игроки", f"Переходы в {C.LEAGUE}", "Кластеры"])
    with t[0]:
        st.dataframe(pd.read_csv(C.OUT / "growth_metrics.csv", index_col=0).style.format(precision=3))
        st.image(str(C.FIG / "04_growth_metrics.png"))
        st.image(str(C.FIG / "05_growth_by_age.png"))
        st.image(str(C.FIG / "07_risk_calibration.png"), width=420)
    with t[1]:
        st.dataframe(pd.read_csv(C.OUT / "value_metrics.csv", index_col=0).style.format(precision=3))
        st.image(str(C.FIG / "08_value_formula.png"))
        st.warning("R² ≈ 0.999: стоимость в FIFA — детерминированная формула EA. Поэтому «недооценённость» "
                   "по остаткам этой модели бессмысленна — мы ищем ценность через стиль и прогноз развития.")
    with t[2]:
        st.dataframe(pd.read_csv(C.OUT / "similarity_eval.csv", index_col=0).style.format(precision=3))
        st.image(str(C.FIG / "11_similarity_reid.png"))
    with t[3]:
        st.image(str(C.FIG / "09_league_moves.png"))
    with t[4]:
        st.dataframe(pd.read_csv(C.OUT / "clusters.csv").style.format(precision=3), hide_index=True)
        st.image(str(C.FIG / "10_clusters_cm.png"))
