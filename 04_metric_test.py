"""Step 4: which player metric predicts team results best? (retrodiction test)

For every team-season t (1980-81 on): take the players who actually played for the team in season t,
look up each player's metric from season t-1, and sum it weighted by his share of the team's minutes
(share = minutes / (team minutes / 5), so the five floor slots add to 5).
Then predict team net rating in season t from that sum with a one-variable regression,
fit with leave-one-season-out cross-validation. Lower held-out RMSE = better metric.
Players with fewer than 250 minutes the season before (rookies, returns) get the fringe-player level.
Same-season BPM is not tested: Basketball-Reference scales BPM so a team's players add up to its own
net rating, which makes a same-season test circular.
Outputs: results/metric_test.csv, results/metric_test_summary.json, figures/fig2_metric_rmse.png (paper Fig. 2)
Run:  python3 04_metric_test.py
"""
import json
import numpy as np
import pandas as pd
import config
from common import load, style, save, INK, INK2, MUTED, ACCENT, COL_W
import matplotlib.pyplot as plt

METRICS = {
    "BPM": ["bpm"],
    "OBPM + DBPM (가중치 2개)": ["obpm", "dbpm"],
    "WS/48": ["ws_per_48"],
    "PER": ["per"],
    "ORtg (리그 대비)": ["rel_ortg"],
    "박스스코어 합 (PTS+REB+AST+STL+BLK)": ["box5"],
}
BASELINE = "직전 시즌 팀 넷레이팅 (선수 정보 없음)"


def build():
    P = load("player_seasons")
    S = pd.read_csv(config.RESULTS / "player_season_one.csv")
    S["box5"] = S[["pts_per_g", "trb_per_g", "ast_per_g", "stl_per_g", "blk_per_g"]].sum(axis=1)
    T = load("team_seasons")
    cols = sorted({c for v in METRICS.values() for c in v})
    rows = []
    for t in range(config.FIRST_SEASON + 1, config.LAST_SEASON + 1):
        roster = P[(P.season == t) & ~P["agg"]][["player_id", "team", "mp"]]
        prev = S[S.season == t - 1].set_index("player_id")
        fringe = prev[(prev.mp >= 250) & (prev.mp < 1000)]
        repl = {c: np.average(fringe[c].fillna(fringe[c].median()), weights=fringe.mp) for c in cols}
        r = roster.join(prev[cols + ["mp"]].rename(columns={"mp": "mp_prev"}), on="player_id")
        low = r.mp_prev.isna() | (r.mp_prev < 250)
        for c in cols:
            r.loc[low | r[c].isna(), c] = repl[c]
        team_mp = r.groupby("team").mp.transform("sum")
        r["share"] = r.mp / (team_mp / 5)
        agg = r.assign(**{c: r[c] * r.share for c in cols}).groupby("team")[cols].sum()
        agg["season"] = t
        rows.append(agg.reset_index())
    X = pd.concat(rows, ignore_index=True)
    X = X.merge(T[["season", "team", "nrtg"]], on=["season", "team"])
    prev_team = T[["season", "team", "nrtg"]].assign(season=T.season + 1).rename(columns={"nrtg": "nrtg_prev"})
    X = X.merge(prev_team, on=["season", "team"], how="left")
    return X


def loso_rmse(X, feats):
    pred = np.empty(len(X))
    for s in X.season.unique():
        tr, te = X.season != s, X.season == s
        A = np.column_stack([np.ones(tr.sum())] + [X.loc[tr, f] for f in feats])
        coef, *_ = np.linalg.lstsq(A, X.loc[tr, "nrtg"], rcond=None)
        B = np.column_stack([np.ones(te.sum())] + [X.loc[te, f] for f in feats])
        pred[te.values] = B @ coef
    err = X.nrtg - pred
    return np.sqrt(np.mean(err ** 2)), 1 - np.sum(err ** 2) / np.sum((X.nrtg - X.nrtg.mean()) ** 2)


def main():
    style()
    X = build()
    X = X.dropna(subset=["nrtg_prev"]).reset_index(drop=True)   # same team-seasons for every metric
    X.to_csv(config.RESULTS / "metric_test_inputs.csv", index=False)
    res = []
    for name, feats in METRICS.items():
        rmse, r2 = loso_rmse(X, feats)
        res.append({"metric": name, "rmse": rmse, "r2": r2})
    rmse, r2 = loso_rmse(X, ["nrtg_prev"])
    res.append({"metric": BASELINE, "rmse": rmse, "r2": r2})
    R = pd.DataFrame(res).sort_values("rmse").reset_index(drop=True)
    R.to_csv(config.RESULTS / "metric_test.csv", index=False)
    print(f"{len(X)} team-seasons, {X.season.min()} to {X.season.max()} (year the season ends); "
          "held-out RMSE of team net rating (points per 100)")
    print(R.round(3).to_string(index=False))

    json.dump({"n_team_seasons": int(len(X)), "first": int(X.season.min()), "last": int(X.season.max())},
              open(config.RESULTS / "metric_test_summary.json", "w", encoding="utf-8"), indent=1)

    # Fig. 2: held-out error by metric, the no-player baseline as a reference line
    P = R[R.metric != BASELINE].iloc[::-1]
    base = float(R[R.metric == BASELINE].rmse.iloc[0])
    best = R[R.metric != BASELINE].metric.iloc[0]
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    ax.barh(range(len(P)), P.rmse, color=[ACCENT if m == best else MUTED for m in P.metric], height=0.62)
    for i, r in enumerate(P.itertuples()):
        x = r.rmse + 0.04 if abs(r.rmse - base) > 0.25 or r.rmse > base else base + 0.04
        ax.text(x, i, f"{r.rmse:.2f}", va="center", fontsize=6.5, color=INK)
    ax.axvline(base, color=INK2, lw=0.8, ls="--")
    ax.text(base, len(P) - 0.35, f"선수 정보 없이 직전 시즌 팀 성적: {base:.2f}", ha="right", va="bottom",
            fontsize=6, color=INK2)
    ax.set_yticks(range(len(P)), P.metric, fontsize=6.5)
    ax.set_xlim(0, R.rmse.max() * 1.18)
    ax.set_xlabel("다음 시즌 팀 넷레이팅 예측 오차 (RMSE, 낮을수록 정확)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_metric_rmse.png")


if __name__ == "__main__":
    main()
