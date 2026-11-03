"""Step 7: from players to a team. Fit and test team-rating models on real team-seasons.

Training data: every team-season 1979-80 to 2025-26 (about 1,300). Each player's talent for season t is his
minutes-weighted average over seasons t-2, t-1, t+1, t+2 (season t left out, so the team's own result
does not leak into its features). Shares = minutes in season t / (team minutes / 5).

Also estimates k, the usage-efficiency trade-off: how many points of ORtg (vs league) a player loses per
point of usage he adds, from season-to-season changes of the same player.

Outputs:
  results/team_model_cv.csv           held-out error of each model on real teams
  results/team_model_inputs.csv       the training table
  results/model_params.json           k and the ridge model's training ranges
  results/ridge_model.pkl             the fitted ridge model
  results/team_model_summary.json
  figures/fig6_team_model.png (paper Fig. 6), figures/figS2_models.png, figures/figS3_out_of_range.png
Run:  python3 07_team_model.py
"""
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import config
from common import load, style, save, season_label, INK, INK2, MUTED, ACCENT, COL_W, FULL_W
from teammodel import features, fit_ridge, five_frame, FIT_FEATURES, PLAYER_COLS
import matplotlib.pyplot as plt


def usage_pairs(S):
    """Consecutive seasons of the same player, 1000+ minutes each: change in usage and in relative ORtg."""
    a = S[S.mp >= 1000][["player_id", "season", "usg_pct", "rel_ortg"]].dropna()
    b = a.assign(season=a.season + 1)
    m = a.merge(b, on=["player_id", "season"], suffixes=("", "_prev"))
    return pd.DataFrame({"player_id": m.player_id, "du": m.usg_pct - m.usg_pct_prev, "dr": m.rel_ortg - m.rel_ortg_prev})


def usage_tradeoff(S, n_boot=config.N_BOOTSTRAP):
    """Within-player slope of relative ORtg on usage, plus a bootstrap over players (whole careers resampled)."""
    m = usage_pairs(S)
    slope = np.polyfit(m.du, m.dr, 1)[0]
    rng = np.random.default_rng(config.SEED)
    groups = {pid: g for pid, g in m.groupby("player_id")}
    ids = np.array(list(groups))
    boots = []
    for _ in range(n_boot):
        pick = pd.concat([groups[i] for i in rng.choice(ids, len(ids))])
        boots.append(np.polyfit(pick.du, pick.dr, 1)[0])
    return slope, len(m), np.array(boots)


def talent_table(S):
    """For each player-season t: average of his seasons t-2..t+2 excluding t (250+ minutes each)."""
    s = S[S.mp >= 250][["player_id", "season", "mp"] + PLAYER_COLS].dropna(subset=["bpm"])
    rows = []
    for off in (-2, -1, 1, 2):
        rows.append(s.assign(target=s.season - off))
    nb = pd.concat(rows)
    vals = nb[PLAYER_COLS].fillna(nb[PLAYER_COLS].median())
    nb[PLAYER_COLS] = vals.mul(nb.mp, axis=0)
    g = nb.groupby(["player_id", "target"])
    out = g[PLAYER_COLS].sum().div(g.mp.sum(), axis=0)
    out["mp_nb"] = g.mp.sum()
    return out.reset_index().rename(columns={"target": "season"})


def build(S, P, T, k):
    tal = talent_table(S)
    roster = P[~P["agg"]][["player_id", "season", "team", "mp"]]
    r = roster.merge(tal, on=["player_id", "season"], how="left")
    # players with no neighbouring seasons: fringe level of that season
    fringe = S[(S.mp >= 250) & (S.mp < 1000)].groupby("season")[PLAYER_COLS].mean()
    miss = r.bpm.isna()
    for c in PLAYER_COLS:
        r.loc[miss, c] = r.loc[miss, "season"].map(fringe[c])
        r[c] = r[c].fillna(r[c].median())
    r["share"] = r.mp / (r.groupby(["season", "team"]).mp.transform("sum") / 5)
    F = features(r, ["season", "team"], k=k)
    return F.merge(T[["season", "team", "nrtg"]], on=["season", "team"])


def cv_predict(F, kind):
    """Leave-one-season-out predictions on real teams."""
    pred = np.empty(len(F))
    for s in F.season.unique():
        tr, te = F.season != s, F.season == s
        if kind == "ridge":
            m = fit_ridge(F[tr], F.nrtg[tr])
            pred[te.values] = m.predict(F.loc[te, FIT_FEATURES])
        elif kind == "boosting":
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, random_state=0)
            m.fit(F.loc[tr, FIT_FEATURES], F.nrtg[tr])
            pred[te.values] = m.predict(F.loc[te, FIT_FEATURES])
        else:   # one-variable calibration of a formula: additive or usage model
            col = "bpm_sum" if kind == "additive" else "usage_model"
            A = np.column_stack([np.ones(tr.sum()), F.loc[tr, col]])
            coef, *_ = np.linalg.lstsq(A, F.nrtg[tr], rcond=None)
            pred[te.values] = coef[0] + coef[1] * F.loc[te, col]
    return pred


def main():
    style()
    S = pd.read_csv(config.RESULTS / "player_season_one.csv")
    P = load("player_seasons")
    T = load("team_seasons")
    pr = pd.read_csv(config.RESULTS / "primes.csv")

    slope, n, boots = usage_tradeoff(S)
    k = max(0.0, -slope)
    kb = np.clip(-boots, 0, None)
    pd.DataFrame({"k": kb}).to_csv(config.RESULTS / "k_bootstrap.csv", index=False)
    print(f"usage-efficiency trade-off: {slope:+.2f} ORtg points per usage point ({n} player season pairs) -> k = {k:.2f}, "
          f"bootstrap 90% interval {np.percentile(kb, 5):.2f} to {np.percentile(kb, 95):.2f}")

    F = build(S, P, T, k)
    F.to_csv(config.RESULTS / "team_model_inputs.csv", index=False)
    print(f"{len(F)} team-seasons for training")

    res, preds = [], {}
    labels = {"additive": "Additive (sum of BPM, calibrated)", "usage": "One-ball usage model (calibrated)",
              "ridge": "Ridge on fit features", "boosting": "Gradient boosting on fit features"}
    for kind in labels:
        p = cv_predict(F, kind)
        preds[kind] = p
        res.append({"model": labels[kind], "kind": kind, "rmse": np.sqrt(np.mean((F.nrtg - p) ** 2))})
    R = pd.DataFrame(res)
    R.to_csv(config.RESULTS / "team_model_cv.csv", index=False)
    for kind, p in preds.items():
        F[f"cv_{kind}"] = p
    F.to_csv(config.RESULTS / "team_model_inputs.csv", index=False)
    print(R.round(3).to_string(index=False))
    a, b = np.linalg.lstsq(np.column_stack([np.ones(len(F)), F.usage_model]), F.nrtg, rcond=None)[0]
    a, b = float(a), float(b)
    print(f"one-ball usage model on real teams: net rating = {a:+.2f} + {b:.2f} x formula (1.00 = on scale, no refit needed)")
    best = F.loc[F.nrtg.idxmax()]
    print(f"best real team in the data: {best.team} {int(best.season)}, net rating {best.nrtg:+.1f}")

    ridge = fit_ridge(F, F.nrtg)
    coefs = dict(zip(FIT_FEATURES, ridge[-1].coef_ / ridge[0].scale_))
    print("ridge coefficients per unit of each feature:", {a: round(float(b), 3) for a, b in coefs.items()})
    with open(config.RESULTS / "ridge_model.pkl", "wb") as f:
        pickle.dump(ridge, f)
    boost = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, random_state=0)
    boost.fit(F[FIT_FEATURES], F.nrtg)
    with open(config.RESULTS / "boost_model.pkl", "wb") as f:
        pickle.dump(boost, f)

    # how far outside real teams the best-players five sits
    top = pr[pr["rank"] <= 5]
    five = features(five_frame(top), "team", k=k).iloc[0]
    rng = {c: {"mean": float(F[c].mean()), "sd": float(F[c].std()), "min": float(F[c].min()),
               "max": float(F[c].max()), "five": float(five[c])} for c in FIT_FEATURES + ["bpm_sum", "usage_model"]}
    json.dump({"k": k, "usage_slope": slope, "ranges": rng}, open(config.RESULTS / "model_params.json", "w", encoding="utf-8"), indent=1)
    print("\nbest-players five vs real teams (standard deviations from the mean of real teams):")
    for c in ["bpm_sum", "usg_load", "drb_load", "ast_load", "spacing", "rim"]:
        z = (rng[c]["five"] - rng[c]["mean"]) / rng[c]["sd"]
        print(f"  {c:10s} five {rng[c]['five']:7.1f}  real teams {rng[c]['min']:7.1f} to {rng[c]['max']:7.1f}  -> {z:+.1f} SD")
    p_add = five.bpm_sum
    p_use = five.usage_model
    p_ridge = float(ridge.predict(pd.DataFrame([five[FIT_FEATURES]]))[0])
    p_boost = float(boost.predict(pd.DataFrame([five[FIT_FEATURES]]))[0])
    print(f"\nbest-players five predicted net rating: additive {p_add:+.1f}, usage model {p_use:+.1f}, "
          f"ridge {p_ridge:+.1f}, boosting {p_boost:+.1f} (best real team in data {F.nrtg.max():+.1f})")

    json.dump({"n_team_seasons": int(len(F)), "best_real": {"team": best.team, "season": int(best.season),
               "nrtg": float(best.nrtg)}, "five": {"additive": float(p_add), "usage_model": float(p_use),
               "ridge": float(p_ridge), "boosting": float(p_boost)},
               "usage_calibration": {"a": float(a), "b": float(b)},
               "r2_usage": float(1 - np.sum((F.nrtg - F.cv_usage) ** 2) / np.sum((F.nrtg - F.nrtg.mean()) ** 2))},
              open(config.RESULTS / "team_model_summary.json", "w", encoding="utf-8"), indent=1)

    # Fig. 6: held-out predictions of the main (one-ball) model vs real net rating
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.scatter(F.cv_usage, F.nrtg, s=4, color=MUTED, alpha=0.7, linewidths=0)
    lim = [min(F.cv_usage.min(), F.nrtg.min()) - 1, max(F.cv_usage.max(), F.nrtg.max()) + 1]
    ax.plot(lim, lim, color=INK2, lw=0.7, ls="--")
    ax.scatter([best.cv_usage], [best.nrtg], s=14, color=ACCENT, zorder=3)
    ax.annotate(f"{best.team} {season_label(best.season)}", (best.cv_usage, best.nrtg), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=6.2, color=ACCENT)
    r2 = 1 - np.sum((F.nrtg - F.cv_usage) ** 2) / np.sum((F.nrtg - F.nrtg.mean()) ** 2)
    ax.text(0.03, 0.95, f"RMSE {R.set_index('kind').rmse['usage']:.2f}, $R^2$ {r2:.2f}\n팀-시즌 {len(F):,}개",
            transform=ax.transAxes, va="top", fontsize=6.5, color=INK)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("모델 예측 넷레이팅 (해당 시즌 제외 교차검증)")
    ax.set_ylabel("실제 넷레이팅")
    save(fig, "fig6_team_model.png")

    # Supplementary: models agree on real teams but split on the all-time five
    order = ["additive", "usage", "ridge", "boosting"]
    short = {"additive": "합산 (BPM 합)", "usage": "원볼 사용률 모델", "ridge": "릿지 회귀", "boosting": "그래디언트 부스팅"}
    vals = {"additive": p_add, "usage": p_use, "ridge": p_ridge, "boosting": p_boost}
    rm = R.set_index("kind").rmse
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 1.9))
    a1.barh(range(4), [rm[o] for o in order][::-1], color=MUTED, height=0.6)
    a2.barh(range(4), [vals[o] for o in order][::-1], color=ACCENT, height=0.6)
    for i, o in enumerate(order[::-1]):
        a1.text(rm[o] + 0.04, i, f"{rm[o]:.2f}", va="center", fontsize=6.5)
        a2.text(vals[o] + 0.8, i, f"{vals[o]:+.1f}", va="center", fontsize=6.5)
    for a in (a1, a2):
        a.set_yticks(range(4), [short[o] for o in order][::-1])
        a.grid(axis="y", visible=False)
    a1.set_xlim(0, rm.max() * 1.25)
    a1.set_xlabel("실제 팀 예측 오차 (RMSE)")
    a2.axvline(best.nrtg, color=INK2, lw=0.7, ls="--")
    a2.set_xlim(0, max(vals.values()) * 1.2)
    a2.set_xlabel(f"올타임 5인 예측 넷레이팅 (점선: 역대 최고 실제 팀 {best.nrtg:+.1f})")
    fig.tight_layout()
    save(fig, "figS2_models.png")

    # Supplementary: how far the all-time five sits outside real teams
    feats = [("bpm_sum", "BPM 합"), ("usg_load", "사용률 합 (100 = 공 하나)"), ("ast_load", "어시스트 점유 합"),
             ("drb_load", "수비 리바운드 점유 합"), ("spacing", "3점 시도 비율 (리그 대비)"), ("rim", "블록률 합")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for i, (c, lab) in enumerate(feats[::-1]):
        z = (F[c] - F[c].mean()) / F[c].std()
        zf = (rng[c]["five"] - rng[c]["mean"]) / rng[c]["sd"]
        ax.hlines(i, z.min(), z.max(), color=MUTED, lw=5, alpha=0.6)
        ax.scatter([zf], [i], color=ACCENT, s=14, zorder=3)
        ax.text(zf + 0.3, i, f"{rng[c]['five']:.0f}", va="center", fontsize=6.2, color=INK)
    ax.set_yticks(range(len(feats)), [l for _, l in feats[::-1]], fontsize=6.5)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel("실제 팀 평균 대비 표준편차 (회색: 실제 팀 전체 범위)")
    ax.grid(axis="y", visible=False)
    save(fig, "figS3_out_of_range.png")


if __name__ == "__main__":
    main()
