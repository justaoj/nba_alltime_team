"""Step 3: player primes and the best-player ranking.

A prime = the player's best run of N consecutive qualifying seasons (one missed season allowed inside),
scored by minutes-weighted BPM. A qualifying season = 1500+ minutes, scaled down for short seasons.
Outputs:
  results/player_season_one.csv   one row per player-season (multi-team total for traded players)
  results/primes.csv              one row per player: 3-season prime stats, positions, 1- and 5-season peaks
  results/primes_by_decade.csv
  figures/fig5_top_primes.png (paper Fig. 5), figures/figS1_window_sensitivity.png (supplementary)
Run:  python3 03_primes.py
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import config
from common import load, style, save, season_label, INK, INK2, MUTED, ACCENT, COL_W
import matplotlib.pyplot as plt

STATS = ["bpm", "obpm", "dbpm", "per", "ws_per_48", "usg_pct", "rel_ortg", "ts_pct", "rel_ts", "orb_pct",
         "drb_pct", "ast_pct", "stl_pct", "blk_pct", "tov_pct", "fg3a_per_fga_pct", "rel_3pa", "fg3_pct",
         "rel_3p", "ft_pct", "mp_per_g", "pts_per_g", "trb_per_g", "ast_per_g", "stl_per_g", "blk_per_g"]


def one_row_per_season(P):
    """Multi-team total row for traded players, else the single team row; adds league-relative stats."""
    single = P[~P["agg"]].copy()
    # league averages, weighted by possessions used (usage x minutes) or by minutes
    w = single.usg_pct.fillna(0) * single.mp
    lg = pd.DataFrame({
        "lg_ortg": single.assign(w=w).dropna(subset=["off_rtg"]).groupby("season").apply(
            lambda g: np.average(g.off_rtg, weights=g.w)),
        "lg_ts": single.dropna(subset=["ts_pct"]).groupby("season").apply(
            lambda g: np.average(g.ts_pct, weights=g.mp * g.usg_pct.fillna(0) + 1e-9)),
        "lg_3pa": single.dropna(subset=["fg3a_per_fga_pct"]).groupby("season").apply(
            lambda g: np.average(g.fg3a_per_fga_pct, weights=g.mp)),
        "lg_3p": single.dropna(subset=["fg3_pct"]).assign(w3=lambda d: d.fg3a_per_poss * d.mp).groupby("season").apply(
            lambda g: np.average(g.fg3_pct, weights=g.w3 + 1e-9)),
    })
    traded = set(map(tuple, P[P["agg"]][["season", "player_id"]].values))
    keep = P["agg"] | ~P.apply(lambda r: (r.season, r.player_id) in traded, axis=1)
    S = P[keep].merge(lg, left_on="season", right_index=True)
    S["rel_ortg"] = S.off_rtg - S.lg_ortg
    S["rel_ts"] = (S.ts_pct - S.lg_ts) * 100
    S["rel_3pa"] = S.fg3a_per_fga_pct - S.lg_3pa
    S["rel_3p"] = (S.fg3_pct - S.lg_3p) * 100
    S["min_mp"] = S.season.map(lambda y: config.MIN_MINUTES_FULL_SEASON * config.SEASON_GAMES.get(y, 82) / 82)
    S["qualified"] = S.mp >= S.min_mp
    S["pos_list"] = S.pos.fillna("").str.split("-")
    return S, lg


def wavg(g, col):
    v = g[col]
    ok = v.notna()
    return np.average(v[ok], weights=g.mp[ok]) if ok.any() else np.nan


def best_window(g, n):
    """Best run of n qualifying seasons spanning at most n+1 seasons. Returns (bpm, rows) or None."""
    best = None
    for i in range(len(g) - n + 1):
        w = g.iloc[i:i + n]
        if w.season.iloc[-1] - w.season.iloc[0] > n:
            continue
        v = wavg(w, "bpm")
        if best is None or v > best[0]:
            best = (v, w)
    return best


def top15_table(t):
    """The top-15 prime ranking drawn as a table: rank, player, prime, position, BPM, OBPM, DBPM."""
    cols = [("순위", 0.02, "left"), ("선수", 0.08, "left"), ("전성기 (3년 기준)", 0.405, "left"), ("포지션", 0.62, "left"),
            ("BPM", 0.785, "right"), ("OBPM", 0.893, "right"), ("DBPM", 1.0, "right")]
    n = len(t)
    fig, ax = plt.subplots(figsize=(COL_W * 1.65, 0.3 + 0.165 * (n + 1)))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(n + 0.6, -0.9)
    for name, x, ha in cols:
        ax.text(x, -0.3, name, ha=ha, va="center", fontsize=7, color=INK, fontweight="bold")
    ax.hlines([-0.9, 0.25], 0, 1, color=INK, lw=0.8)
    for i, r in enumerate(t.itertuples(), start=1):
        y = i - 0.25
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f1f0ec", zorder=0, lw=0)
        vals = [f"{r.rank}", r.name, r.prime.replace(" to ", "~"), r.prime_pos,
                f"{r.bpm:+.1f}", f"{r.obpm:+.1f}", f"{r.dbpm:+.1f}"]
        for (name, x, ha), v in zip(cols, vals):
            ax.text(x, y, v, ha=ha, va="center", fontsize=6.8, color=INK,
                    fontweight="bold" if name == "BPM" else "normal")
    ax.hlines(n + 0.25, 0, 1, color=INK, lw=0.8)
    ax.text(0, n + 0.55, "BPM = OBPM + DBPM (100포세션당 득점, 평균 선수 대비). 전성기 (3년 기준) = 가장 좋은 연속 3시즌의 출전시간 가중 평균.\n"
            "표시 값이 같으면 반올림 전 값으로 정렬. 출처: Basketball-Reference, 1979-80~2025-26.",
            fontsize=5.6, color=INK2, va="top", linespacing=1.4)
    save(fig, "table_top15_primes.png")


def main():
    style()
    P = load("player_seasons")
    S, lg = one_row_per_season(P)
    S.drop(columns=["pos_list"]).to_csv(config.RESULTS / "player_season_one.csv", index=False)
    lg.to_csv(config.RESULTS / "league_averages.csv")
    Q = S[S.qualified].sort_values(["player_id", "season"])
    career_pos = Q.groupby("player_id").pos_list.apply(
        lambda s: "-".join(p for p in config.POSITIONS if any(p in x for x in s)))

    recs = []
    for pid, g in Q.groupby("player_id"):
        g = g.reset_index(drop=True)
        b3 = best_window(g, config.PRIME_WINDOW)
        if b3 is None:
            continue
        v, w = b3
        b1, b5 = best_window(g, 1), best_window(g, 5)
        rec = {"player_id": pid, "name": w.name.iloc[0],
               "first": int(w.season.iloc[0]), "last": int(w.season.iloc[-1]),
               "prime": f"{season_label(w.season.iloc[0])} to {season_label(w.season.iloc[-1])}",
               "prime_pos": "-".join(p for p in config.POSITIONS if any(p in x for x in w.pos_list)),
               "career_pos": career_pos[pid],
               "bpm_1": b1[0], "bpm_5": b5[0] if b5 else np.nan, "mp_prime": w.mp.sum()}
        rec.update({s: wavg(w, s) for s in STATS})
        recs.append(rec)
    pr = pd.DataFrame(recs).sort_values("bpm", ascending=False).reset_index(drop=True)
    pr["rank"] = np.arange(1, len(pr) + 1)
    pr.to_csv(config.RESULTS / "primes.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"{len(pr)} players with a {config.PRIME_WINDOW}-season prime")
    print("\nTop 15 primes (minutes-weighted BPM)")
    print(pr.head(15)[["rank", "name", "prime", "prime_pos", "bpm", "obpm", "dbpm", "usg_pct", "bpm_1", "bpm_5"]]
          .round(1).to_string(index=False))
    print("\nTop 3 per position (prime positions)")
    for p in config.POSITIONS:
        top = pr[pr.prime_pos.str.split("-").apply(lambda s: p in s)].head(3)
        print(f"  {p}: " + "; ".join(f"{r['name']} {r.bpm:+.1f}" for _, r in top.iterrows()))

    # window sensitivity on the top 50
    top50 = pr.head(50).dropna(subset=["bpm_5"])
    rho13 = spearmanr(top50.bpm_1, top50.bpm).correlation
    rho35 = spearmanr(top50.bpm, top50.bpm_5).correlation
    print(f"\nSpearman rank correlation on the top 50: 1 vs 3 seasons {rho13:.2f}, 3 vs 5 seasons {rho35:.2f}")
    dec = (pr.head(50).first // 10 * 10).value_counts().sort_index()
    print("Top-50 primes by decade the prime starts in:", {int(a): int(b) for a, b in dec.items()})
    pd.DataFrame({"decade": dec.index, "top50_primes": dec.values}).to_csv(config.RESULTS / "primes_by_decade.csv", index=False)
    json.dump({"n_players": int(len(pr)), "rho_1_3": float(rho13), "rho_3_5": float(rho35)},
              open(config.RESULTS / "primes_summary.json", "w", encoding="utf-8"), indent=1)

    # Fig. 5: top 15 primes
    t = pr.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(COL_W, 3.1))
    colors = [ACCENT if i == len(t) - 1 else MUTED for i in range(len(t))]
    ax.barh(range(len(t)), t.bpm, color=colors, height=0.66)
    for i, r in enumerate(t.itertuples()):
        ax.text(r.bpm + 0.15, i, f"{r.bpm:+.1f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(t)), [f"{r.name} ({r.prime_pos}, {r.prime.replace(' to ', '~')})" for r in t.itertuples()],
                  fontsize=6.2)
    ax.set_xlim(0, t.bpm.max() * 1.13)
    ax.set_xlabel("전성기 BPM (3년 기준, 평균 선수 대비, 100포세션당 득점)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig5_top_primes.png")

    # Table figure: top 15 primes with the BPM split (figures/table_top15_primes.png)
    top15_table(pr.head(15))

    # Supplementary: window sensitivity for the top 15
    t = pr.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(COL_W, 3.1))
    y = np.arange(len(t))
    lo = t[["bpm_1", "bpm", "bpm_5"]].min(axis=1)
    hi = t[["bpm_1", "bpm", "bpm_5"]].max(axis=1)
    ax.hlines(y, lo, hi, color=MUTED, lw=1)
    ax.scatter(t.bpm_5, y, s=12, color=INK2, label="5시즌", zorder=3)
    ax.scatter(t.bpm, y, s=16, color=ACCENT, label="3시즌 (사용)", zorder=4)
    ax.scatter(t.bpm_1, y, s=12, facecolor="white", edgecolor=INK2, label="1시즌", zorder=3)
    ax.set_yticks(y, t.name, fontsize=6.5)
    ax.set_xlabel(f"BPM (상위 50명 순위상관: 1~3시즌 {rho13:.2f}, 3~5시즌 {rho35:.2f})")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "figS1_window_sensitivity.png")


if __name__ == "__main__":
    main()
