"""Step 8: calibrate and test the game simulator on real games.

Game model: home margin = h + b * (home net rating - away net rating) * pace / 100 + noise,
noise ~ Normal(0, sigma). h = home-court advantage in points, sigma = game-to-game spread.
  1. fit h, b, sigma for every season (shows how home court has changed)
  2. fit on 2006-07 to 2015-16, test win probabilities on 2016-17 to 2025-26 games:
     Brier score, log loss, reliability (games called at 70% should be won about 70% of the time)
  3. season coverage: simulate every test team-season 4,000 times on its real schedule and check how
     often its real win total lands in the 90% interval. Done three ways: with the team's real net rating
     (tests the game model), with the team model's held-out prediction from step 7 (tests the pipeline),
     and with that prediction plus the model's own error drawn each season (the honest interval)
  4. save h, b, sigma for the target season, used by step 10
Outputs: results/calibration_by_season.csv, results/calibration_test.csv, results/sim_params.json,
         figures/figS4_calibration.png, figures/figS5_coverage.png (supplementary)
Run:  python3 08_calibrate.py
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
import config
from common import load, style, save, INK, INK2, MUTED, ACCENT, COL_W, FULL_W
import matplotlib.pyplot as plt

TEST = range(max(config.FIRST_SEASON + 5, config.LAST_SEASON - 9), config.LAST_SEASON + 1)   # last 10 seasons
TRAIN = range(max(config.FIRST_SEASON, TEST.start - 10), TEST.start)                      # the 10 before


def game_table():
    G = load("games")
    T = load("team_seasons")[["season", "team", "nrtg", "pace", "wins", "losses"]]
    G = G.merge(T.add_prefix("h_"), left_on=["season", "home"], right_on=["h_season", "h_team"])
    G = G.merge(T.add_prefix("a_"), left_on=["season", "away"], right_on=["a_season", "a_team"])
    G["d"] = (G.h_nrtg - G.a_nrtg) * (G.h_pace + G.a_pace) / 200
    return G, T


def fit(g):
    A = np.column_stack([np.ones(len(g)), g.d])
    coef, *_ = np.linalg.lstsq(A, g.margin, rcond=None)
    resid = g.margin - A @ coef
    return coef[0], coef[1], resid.std(ddof=2)


def simulate_wins(team, season, rating, G, h, b, sigma, rng, n=4000):
    g = G[(G.season == season) & ((G.home == team) | (G.away == team))]
    home = (g.home == team).values
    opp = np.where(home, g.a_nrtg, g.h_nrtg)
    pace = (g.h_pace + g.a_pace).values / 200
    rating = np.broadcast_to(np.asarray(rating, dtype=float).reshape(-1, 1), (n, 1))
    mu = np.where(home, h, -h) + b * (rating - opp) * pace       # (n seasons, games)
    p = norm.cdf(mu / sigma)
    u = rng.random(p.shape)
    return (u < p).sum(axis=1)


def main():
    style()
    G, T = game_table()
    by = pd.DataFrame([(s, *fit(g), len(g), (g.margin > 0).mean()) for s, g in G.groupby("season")],
                      columns=["season", "home_court", "slope", "sigma", "games", "home_win_pct"])
    by.to_csv(config.RESULTS / "calibration_by_season.csv", index=False)
    print(by.round(2).to_string(index=False))

    h, b, sigma = fit(G[G.season.isin(TRAIN)])
    te = G[G.season.isin(TEST)].copy()
    te["p"] = norm.cdf((h + b * te.d) / sigma)
    te["won"] = (te.margin > 0).astype(int)
    brier = np.mean((te.p - te.won) ** 2)
    logloss = -np.mean(te.won * np.log(te.p) + (1 - te.won) * np.log(1 - te.p))
    base = G[G.season.isin(TRAIN)].margin.gt(0).mean()
    brier0 = np.mean((base - te.won) ** 2)
    print(f"\ntrained on {TRAIN.start-1}-{TRAIN.stop-1}: h {h:.2f}, b {b:.3f}, sigma {sigma:.2f}")
    print(f"test {len(te)} games: Brier {brier:.4f} (home-team-only baseline {brier0:.4f}), log loss {logloss:.4f}")

    te["bin"] = pd.cut(te.p, np.linspace(0, 1, 11))
    rel = te.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("won", "mean"), n=("won", "size")).reset_index()

    # season coverage on test seasons
    rng = np.random.default_rng(config.SEED)
    F = pd.read_csv(config.RESULTS / "team_model_inputs.csv")
    global RMSE
    RMSE = float(np.sqrt(np.mean((F.nrtg - F.cv_ridge) ** 2)))
    cov = []
    for r in T[T.season.isin(TEST)].itertuples():
        hs, bs, ss = by.set_index("season").loc[r.season, ["home_court", "slope", "sigma"]]
        w_true = simulate_wins(r.team, r.season, r.nrtg, G, hs, bs, ss, rng, 4000)
        pr = F[(F.season == r.season) & (F.team == r.team)].cv_ridge
        if len(pr) == 0:
            continue
        w_pred = simulate_wins(r.team, r.season, float(pr.iloc[0]), G, hs, bs, ss, rng, 4000)
        # same, plus the team model's own error: the rating is drawn around the prediction each season
        w_unc = simulate_wins(r.team, r.season, float(pr.iloc[0]) + rng.normal(0, RMSE, 4000), G, hs, bs, ss, rng, 4000)
        cov.append({"season": r.season, "team": r.team, "wins": r.wins,
                    "true_lo": np.percentile(w_true, 5), "true_hi": np.percentile(w_true, 95),
                    "pred_mean": w_pred.mean(), "pred_lo": np.percentile(w_pred, 5), "pred_hi": np.percentile(w_pred, 95),
                    "unc_lo": np.percentile(w_unc, 5), "unc_hi": np.percentile(w_unc, 95)})
    C = pd.DataFrame(cov)
    C["in_true"] = C.wins.between(C.true_lo, C.true_hi)
    C["in_pred"] = C.wins.between(C.pred_lo, C.pred_hi)
    C["in_unc"] = C.wins.between(C.unc_lo, C.unc_hi)
    C.to_csv(config.RESULTS / "calibration_test.csv", index=False)
    print(f"90% win interval coverage, {len(C)} test team-seasons: game model {C.in_true.mean():.1%}, "
          f"pipeline, game noise only {C.in_pred.mean():.1%}, pipeline + model error {C.in_unc.mean():.1%}")

    tgt = by.set_index("season").loc[config.TARGET_SEASON]
    params = {"season": config.TARGET_SEASON, "home_court": float(tgt.home_court), "slope": float(tgt.slope),
              "sigma": float(tgt.sigma), "test_brier": float(brier), "test_logloss": float(logloss),
              "coverage_game_model": float(C.in_true.mean()), "coverage_pipeline": float(C.in_pred.mean()),
              "coverage_pipeline_with_model_error": float(C.in_unc.mean()), "model_rmse": RMSE}
    json.dump(params, open(config.RESULTS / "sim_params.json", "w", encoding="utf-8"), indent=1)
    print("target-season parameters:", {k: round(v, 3) for k, v in params.items()})

    # Supplementary: reliability of game probabilities, and home court by season
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.2))
    a1.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    a1.scatter(rel.pred, rel.actual, s=np.clip(rel.n / 40, 4, 40), color=ACCENT, zorder=3)
    a1.set_xlabel(f"예측 홈 승률 (검증 경기 {len(te):,}개)")
    a1.set_ylabel("실제 홈팀 승리 비율")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1)
    a2.plot(by.season, by.home_court, color=ACCENT, lw=1.2, label="홈 코트 이점 (점)")
    a2.plot(by.season, by.sigma, color=INK2, lw=1.2, label="경기별 득실차 표준편차 σ (점)")
    a2.set_xlabel("시즌 (종료 연도)")
    a2.legend(loc="center left")
    a2.set_ylim(0, by.sigma.max() * 1.1)
    fig.tight_layout()
    save(fig, "figS4_calibration.png")

    # Supplementary: season win totals vs the 90% interval of the full pipeline
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    hit = C.in_unc
    for m, c, lab in [(hit, MUTED, "90% 구간 안"), (~hit, ACCENT, "구간 밖")]:
        ax.errorbar(C.pred_mean[m], C.wins[m], xerr=[C.pred_mean[m] - C.unc_lo[m], C.unc_hi[m] - C.pred_mean[m]],
                    fmt="o", ms=1.8, color=c, ecolor=c, elinewidth=0.4, alpha=0.8, label=lab)
    ax.plot([0, 82], [0, 82], color=INK2, lw=0.6, ls="--")
    ax.set_xlabel("선수 기록으로 시뮬레이션한 승수 (평균, 90% 구간)")
    ax.set_ylabel("실제 승수")
    ax.legend(loc="upper left")
    save(fig, "figS5_coverage.png")


if __name__ == "__main__":
    main()
