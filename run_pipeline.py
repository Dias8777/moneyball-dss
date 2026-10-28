"""
Полный пайплайн: данные -> аудит -> модели -> решения. Результаты сохраняются в outputs/.
Запуск:  python run_pipeline.py
"""
import json
import time

import pandas as pd

from src import audit, config as C, data, decision, figures, models


def main():
    t0 = time.time()
    C.OUT.mkdir(exist_ok=True)
    C.FIG.mkdir(exist_ok=True)
    log = lambda m: print(f"[{time.time() - t0:5.1f}s] {m}")

    # --- 1. Данные ---------------------------------------------------------
    df, quality = data.clean(data.load_raw())
    df.to_parquet(C.DATA_CACHE, index=False)
    log(f"данные: {quality['rows_raw']} -> {quality['rows_clean']} строк")
    cur = data.current_features(df)
    pairs = data.pairs(df)
    log(f"FIFA 23: {len(cur)} игроков; панель N->N+1: {len(pairs)} пар")

    # --- 2. Стиль клуба и role fit ----------------------------------------
    profile = audit.style_profile(cur)
    weights = audit.tilted_weights(profile)
    cur = audit.role_fit(cur, weights)
    log("стиль: " + ", ".join(f"{a} {v:+.2f}" for a, v in profile.head(5).items()))

    # --- 3. Модель развития -----------------------------------------------
    growth_eval = models.evaluate_growth(pairs)
    log("развитие (test FIFA 22->23):\n" + growth_eval["regression"].round(3).to_string())
    log(f"риск спада: {growth_eval['classification']}")
    log(f"покрытие 80%-интервала: {growth_eval['coverage_80']:.3f}")
    gm = models.fit_growth_final(pairs)
    cur = cur.join(models.predict_growth(gm, cur))
    moves = models.league_move_effect(pairs)
    log("переход в АПЛ (остаток):\n" + moves.round(2).to_string())

    # --- 4. Модель цены ---------------------------------------------------
    value_eval = models.evaluate_value(cur)
    log("цена:\n" + value_eval.round(3).to_string())
    vm = models.fit_value(cur)
    cur["resale_2y"] = models.project_value(vm, cur, 2 * cur.pred_delta, years=2)

    # --- 5. Стили ---------------------------------------------------------
    cl, cl_summary = models.cluster_styles(cur)
    cur = cur.join(cl)
    sim_eval = models.evaluate_similarity(df, cur)
    log("похожие игроки:\n" + sim_eval.round(3).to_string())

    # --- 6. Аудит Брентфорда ---------------------------------------------
    table = audit.league_position_table(cur)
    need = audit.needs(table)
    st = audit.squad_actions(audit.starters(audit.squad(cur)))
    log("потребности:\n" + need.round(2).to_string())

    # --- 7. Решения -------------------------------------------------------
    need_groups = audit.need_groups(need)
    cand = decision.build_candidates(cur, st)
    depth_groups = [g for g in need_groups if g not in need.group.head(3).tolist()]
    plan = decision.optimize(cand, groups=need_groups, depth_groups=depth_groups, max_signings=4)
    scen = decision.budget_scenarios(cand, need_groups, depth_groups=depth_groups, max_signings=4)
    gems = decision.hidden_gems(cand, need_groups)
    ranks = {g: decision.rank_group(cand, g) for g in need_groups}
    sens = {g: decision.weight_sensitivity(cand, g) for g in need_groups}
    show = ["short_name", "club_name", "league_name", "group", "age", "overall", "fit_pct",
            "pred_delta", "p_decline", "fee", "wage_new", "gain", "net_value"]
    log("действия по составу:\n" + st[st.action != "Оставить"][["short_name", "group", "age", "role", "action", "reason"]].to_string())
    log("план:\n" + plan[show + ["utility"]].round(2).to_string())
    log("сценарии бюджета:\n" + scen.to_string())
    log("скрытые таланты:\n" + gems[show].head(6).round(2).to_string())

    # --- Сохранение -------------------------------------------------------
    cur.to_parquet(C.OUT / "players_23.parquet", index=False)
    pairs[["player_id", "fifa_version", "short_name", "age", "overall", "delta", "group",
           "league_name", "league_next"]].to_parquet(C.OUT / "pairs.parquet", index=False)
    st.to_csv(C.OUT / "brentford_squad.csv", index=False)
    need.to_csv(C.OUT / "needs.csv", index=False)
    table.to_csv(C.OUT / "league_position_table.csv", index=False)
    profile.rename("z").to_csv(C.OUT / "style_profile.csv")
    audit.league_style_matrix(cur).to_csv(C.OUT / "league_style_matrix.csv")
    cl_summary.to_csv(C.OUT / "clusters.csv", index=False)
    moves.to_csv(C.OUT / "league_move_effect.csv")
    growth_eval["regression"].to_csv(C.OUT / "growth_metrics.csv")
    growth_eval["by_age"].to_csv(C.OUT / "growth_by_age.csv")
    growth_eval["importance"].rename("mae_increase").to_csv(C.OUT / "growth_importance.csv")
    growth_eval["calibration"].to_csv(C.OUT / "risk_calibration.csv")
    growth_eval["test_df"].to_parquet(C.OUT / "growth_test.parquet", index=False)
    value_eval.to_csv(C.OUT / "value_metrics.csv")
    sim_eval.to_csv(C.OUT / "similarity_eval.csv")
    plan.to_csv(C.OUT / "transfer_plan.csv", index=False)
    scen.to_csv(C.OUT / "budget_scenarios.csv", index=False)
    gems.to_csv(C.OUT / "hidden_gems.csv", index=False)
    pd.concat(ranks, names=["need"]).to_csv(C.OUT / "rankings.csv")
    pd.concat(sens, names=["need"]).to_csv(C.OUT / "weight_sensitivity.csv")
    json.dump({"quality": quality, "weights": weights, "need_groups": need_groups,
               "depth_groups": depth_groups,
               "growth": {"classification": growth_eval["classification"],
                          "coverage_80": growth_eval["coverage_80"], "n": growth_eval["n"],
                          "best_iter": growth_eval["best_iter"]}},
              open(C.OUT / "metrics.json", "w"), ensure_ascii=False, indent=2, default=float)
    figures.make_all(cur, need, profile, growth_eval, moves, sim_eval, cand, plan, scen, need_groups)
    log("графики сохранены в outputs/figures")
    log("готово")


if __name__ == "__main__":
    main()
