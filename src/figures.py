"""
Статичные графики (PNG) для README, ноутбука и презентации.
Палитра: фиксированный порядок категориальных цветов, один оттенок для величин,
серый — для «контекста», оранжевый — только для выделения нашего клуба / выбора.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CONTEXT = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 110,
})

SHORT = lambda a: a.split("_", 1)[1].replace("_", " ") if "_" in a else a


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(C.FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def style_profile(profile: pd.Series):
    p = pd.concat([profile.head(7), profile.tail(5)]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh([SHORT(a) for a in p.index], p.values, color=[BLUE if v > 0 else CONTEXT for v in p.values],
            height=0.6)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_xlabel(f"z-score относительно игроков той же позиции ({C.LEAGUE})")
    ax.set_title(f"Стиль {C.CLUB}: чем игроки отличаются от лиги", pad=10)
    ax.grid(axis="y", visible=False)
    _save(fig, "01_style_profile")


def needs(need: pd.DataFrame):
    d = need.sort_values("gap")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(d))
    ax.barh(y, d.gap, color=BLUE, height=0.55)
    ax.set_yticks(y, [f"{n}  (место {r})" for n, r in zip(d.name, d.league_rank)])
    for yi, g in zip(y, d.gap):
        ax.text(g + 0.1, yi, f"{g:.1f}", va="center", color=INK2, fontsize=9)
    ax.set_xlabel(f"Отставание стартового уровня от цели ({C.TARGET_LABEL}), пункты рейтинга")
    ax.set_title(f"Где {C.CLUB_DAT} не хватает качества", pad=10)
    ax.grid(axis="y", visible=False)
    _save(fig, "02_needs")


def depth(need: pd.DataFrame):
    d = need.set_index("name")[["depth_drop", "league_depth_drop"]].sort_values("depth_drop")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(d))
    ax.barh(y + 0.16, d.depth_drop, height=0.3, color=ORANGE, label=C.CLUB)
    ax.barh(y - 0.16, d.league_depth_drop, height=0.3, color=CONTEXT, label="Медиана клубов-ориентиров")
    ax.set_yticks(y, d.index)
    ax.set_xlabel("Насколько первый запасной слабее стартера, пункты")
    ax.set_title("Глубина состава: где нет замены", pad=10)
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, "03_depth")


def growth_metrics(reg: pd.DataFrame, cls: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    r = reg.MAE.sort_values(ascending=False)
    axes[0].barh(r.index, r.values, color=[BLUE if "Boost" in i else CONTEXT for i in r.index], height=0.55)
    for i, v in enumerate(r.values):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", color=INK2, fontsize=9)
    axes[0].set_xlim(1.4, r.max() + 0.12)
    axes[0].set_title("Прогноз Δoverall: MAE на тесте (меньше — лучше)")
    axes[0].grid(axis="y", visible=False)
    a = {"Только возраст": cls["AUC: только возраст"], "Gradient Boosting": cls["AUC: Gradient Boosting"]}
    axes[1].barh(list(a), list(a.values()), color=[CONTEXT, BLUE], height=0.5)
    for i, v in enumerate(a.values()):
        axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", color=INK2, fontsize=9)
    axes[1].set_xlim(0.5, 0.82)
    axes[1].axvline(0.5, color=BASE, lw=1)
    axes[1].set_title("Риск спада (−2 и больше): ROC-AUC")
    axes[1].grid(axis="y", visible=False)
    fig.suptitle("Временная валидация: обучение FIFA 15–21, тест FIFA 22 → 23", x=0.01, ha="left",
                 color=INK2, fontsize=10)
    _save(fig, "04_growth_metrics")


def growth_by_age(by_age: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(by_age))
    ax.bar(x - 0.18, by_age.abs_err_base, 0.34, color=CONTEXT, label="Среднее по возрасту")
    ax.bar(x + 0.18, by_age.abs_err, 0.34, color=BLUE, label="Gradient Boosting")
    ax.set_xticks(x, by_age.index)
    ax.set_xlabel("Возраст")
    ax.set_ylabel("MAE, пункты")
    ax.set_title("Ошибка прогноза по возрасту: у молодых неопределённость выше", pad=10)
    ax.legend()
    ax.grid(axis="x", visible=False)
    _save(fig, "05_growth_by_age")


def importance(imp: pd.Series):
    d = imp.head(10).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(d.index, d.values, color=BLUE, height=0.55)
    ax.set_xlabel("Рост MAE при перемешивании признака")
    ax.set_title("Что определяет развитие игрока (permutation importance)", pad=10)
    ax.grid(axis="y", visible=False)
    _save(fig, "06_growth_importance")


def calibration(cal: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, cal.pred.max()], [0, cal.pred.max()], color=BASE, lw=1, ls="--", label="Идеал")
    ax.plot(cal.pred, cal.true, color=BLUE, lw=2, marker="o", ms=6, label="Модель")
    ax.set_xlabel("Предсказанная вероятность спада")
    ax.set_ylabel("Фактическая доля спадов")
    ax.set_title("Калибровка риска (децили)", pad=10)
    ax.legend()
    _save(fig, "07_risk_calibration")


def value_formula(cur: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    s = cur.sample(min(4000, len(cur)), random_state=0)
    sc = ax.scatter(s.overall, s.value_eur / 1e6, c=s.age, cmap="Blues_r", s=8, alpha=0.8, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("Overall")
    ax.set_ylabel("Стоимость FIFA, €M (лог)")
    ax.set_title("Стоимость в FIFA — почти формула от рейтинга и возраста", pad=10)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Возраст")
    _save(fig, "08_value_formula")


def league_moves(moves: pd.DataFrame):
    d = moves.sort_values("resid")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    col = [ORANGE if "остались" in i else BLUE for i in d.index]
    ax.barh([f"{i}  (n={int(n)})" for i, n in zip(d.index, d.n)], d.resid, color=col, height=0.6)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_xlabel("Средний остаток Δoverall после перехода (факт − прогноз)")
    ax.set_title(f"Что происходит с рейтингом после перехода в {C.LEAGUE}", pad=10)
    ax.grid(axis="y", visible=False)
    _save(fig, "09_league_moves")


def clusters(cur: pd.DataFrame, group="CM"):
    from sklearn.decomposition import PCA
    from .models import style_matrix
    d = cur[cur.group == group].reset_index(drop=True)
    Z = style_matrix(d[C.attrs_for(group)])
    xy = PCA(2, random_state=0).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(7.4, 5))
    for i, (cl, sub) in enumerate(d.groupby("cluster")):
        ax.scatter(xy[sub.index, 0], xy[sub.index, 1], s=6, alpha=0.5, lw=0, color=SERIES[i],
                   label=sub["style"].iloc[0])
    b = d[d.club_name == C.CLUB]
    ax.scatter(xy[b.index, 0], xy[b.index, 1], s=60, color=INK, edgecolor=SURF, lw=1.5, zorder=5,
               label=f"Игроки {C.CLUB}")
    for i, r in b.nlargest(4, "overall").iterrows():
        ax.annotate(r.short_name, xy[i], xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Стили центральных полузащитников (KMeans, проекция PCA)", pad=10)
    ax.legend(fontsize=8, markerscale=2, loc="best")
    _save(fig, "10_clusters_cm")


def similarity(sim: pd.DataFrame):
    d = sim["hit@10"].sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(d.index, d.values * 100, color=[BLUE if "наш" in i else CONTEXT for i in d.index], height=0.55)
    for i, v in enumerate(d.values):
        ax.text(v * 100 + 1, i, f"{v:.0%}", va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% игроков, найденных среди 10 ближайших через год (FIFA 22 → 23)")
    ax.set_title("Поиск похожих: узнаёт ли метрика того же игрока год спустя", pad=10)
    ax.grid(axis="y", visible=False)
    _save(fig, "11_similarity_reid")


def candidates(cand: pd.DataFrame, plan: pd.DataFrame, groups):
    d = cand[cand.group.isin(groups) & (cand.gain > 0) & (cand.fee <= 1.2 * C.BUDGET_FEE_EUR)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, g in enumerate(groups):
        s = d[d.group == g]
        ax.scatter(s.fee / 1e6, s.gain, s=10, alpha=0.55, lw=0, color=SERIES[i], label=C.GROUP_NAMES[g])
    ax.scatter(plan.fee / 1e6, plan.gain, s=110, facecolor="none", edgecolor=INK, lw=1.8, zorder=5,
               label="Выбор оптимизатора")
    for r in plan.itertuples():
        ax.annotate(r.short_name, (r.fee / 1e6, r.gain), xytext=(7, 4), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xlabel("Оценка трансфера, €M (с премией к оценке FIFA)")
    ax.set_ylabel("Прирост к стартовому составу, пункты")
    ax.set_title("Кандидаты: прирост качества против цены", pad=10)
    ax.legend(fontsize=8, loc="upper left", markerscale=1.5)
    _save(fig, "12_candidates")


def budget(scen: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = [f"{int(b * 100)}%\n€{e / 1e6:.0f}M" for b, e in zip(scen.budget_share, scen.budget_eur)]
    ax.bar(x, scen.gain, color=[ORANGE if b == 1 else BLUE for b in scen.budget_share], width=0.55)
    for i, (g, p) in enumerate(zip(scen.gain, scen.players)):
        ax.text(i, g + 0.2, f"{g:.1f}", ha="center", color=INK2, fontsize=9)
    ax.set_ylabel("Суммарный прирост, пункты")
    ax.set_title("Сценарии бюджета: что даёт каждый следующий евро", pad=10)
    ax.grid(axis="x", visible=False)
    _save(fig, "13_budget_scenarios")


def make_all(cur, need, profile, growth_eval, moves, sim_eval, cand, plan, scen, groups):
    C.FIG.mkdir(parents=True, exist_ok=True)
    style_profile(profile)
    needs(need)
    depth(need)
    growth_metrics(growth_eval["regression"], growth_eval["classification"])
    growth_by_age(growth_eval["by_age"])
    importance(growth_eval["importance"])
    calibration(growth_eval["calibration"])
    value_formula(cur)
    league_moves(moves)
    clusters(cur)
    similarity(sim_eval)
    candidates(cand, plan, groups)
    budget(scen)
