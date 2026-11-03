"""Extra (not part of the paper): the best five when each player counts at his single best season
instead of his best 3-season prime.

Same method as steps 09 and 10, with a 1-season window:
  pool    each player's best qualifying season by BPM (1500+ minutes, scaled for short seasons);
          the top POOL players (60, as in step 09)
  search  every G-G-F-F-C five from the pool (lineups.py; slot = the position he was listed at that season),
          scored by the same team models from step 07
  pick    S1 = highest one-ball usage model rating (the main model); A1 = highest BPM sum, for reference
  sim     the step-10 Monte Carlo for S1 and, for comparison, the 3-season best five (P3):
          N_SIMS single games and N_SIMS best-of-7 series against each dynasty (neutral court);
          N_SIMS league seasons in the target season with the five in the worst team's place,
          rating fixed and with model uncertainty; S1 also under the other team models.
Outputs: results/single_season_pool.csv, results/single_season_candidates.csv, results/single_season_h2h.csv,
         results/single_season_league.csv, results/single_season_summary.json,
         figures/fig9_single_season.png
Needs steps 02-10 first (run_all.py). Changes no paper file.
Run:  python3 12_single_season.py
"""
import _venv
_venv.ensure()                      # re-runs this script inside .venv if needed
import importlib
import json
import pickle
import time
import numpy as np
import pandas as pd
import config
from common import style, save, season_label, INK, INK2, MUTED, ACCENT, FULL_W
from lineups import slot_groups, valid_fives, assign_slots
from teammodel import PLAYER_COLS
import matplotlib.pyplot as plt

search = importlib.import_module("09_search")      # score(), POOL
sim = importlib.import_module("10_simulate")       # ratings, game model, League

MAIN = sim.MAIN


def season_pool():
    """One row per player: his best qualifying season by BPM, in the same columns as primes.csv."""
    S = pd.read_csv(config.RESULTS / "player_season_one.csv")
    Q = S[S.qualified].copy()
    Q[["rel_3pa", "fg3_pct"]] = Q[["rel_3pa", "fg3_pct"]].fillna(0)
    best = Q.sort_values("bpm", ascending=False).drop_duplicates("player_id").reset_index(drop=True)
    best = best.fillna(best.median(numeric_only=True))
    best["prime_pos"] = best.pos.fillna("").str.split("-").apply(
        lambda s: "-".join(p for p in config.POSITIONS if p in s))
    best["prime"] = best.season.map(season_label)
    best["rank"] = np.arange(1, len(best) + 1)
    return best


def pct(p):
    """Percent with 2 decimals; never rounds a probability below 1 up to 100%."""
    s = f"{p:.2%}"
    return ">99.99%" if s == "100.00%" and p < 1 else s


def h2h_rows(lab, R, draws, opps, lg_pace, P, U_game, U_series):
    rows = []
    for o in opps:
        pf = (o["pace"] + lg_pace) / 200
        p = float(sim.p_win(R - o["nrtg"], P, 0.0, pf))
        pd_ = sim.p_win(draws - o["nrtg"], P, 0.0, pf)
        rows.append({"label": lab, "opponent": f"{season_label(o['season'])} {o['team']}",
                     "opp_record": o["record"], "opp_nrtg": o["nrtg"], "rating": R,
                     "game_win_mc": float(np.mean(U_game < p)), "game_win_exact": p,
                     "game_lo": float(np.percentile(pd_, 5)), "game_hi": float(np.percentile(pd_, 95)),
                     "series_win_mc": float(np.mean((U_series < p).sum(1) >= 4)),
                     "series_win_exact": float(sim.p_series_neutral(p)),
                     "series_lo": float(np.percentile(sim.p_series_neutral(pd_), 5)),
                     "series_hi": float(np.percentile(sim.p_series_neutral(pd_), 95))})
    return rows


def main():
    style()
    t0 = time.time()
    rng = np.random.default_rng(config.SEED)
    N = config.N_SIMS
    P = json.load(open(config.RESULTS / "sim_params.json", encoding="utf-8"))
    k = json.load(open(config.RESULTS / "model_params.json", encoding="utf-8"))["k"]
    k_boot = pd.read_csv(config.RESULTS / "k_bootstrap.csv").k.values
    rmse = pd.read_csv(config.RESULTS / "team_model_cv.csv").set_index("kind").rmse["usage"]
    ridge = pickle.load(open(config.RESULTS / "ridge_model.pkl", "rb"))
    boost = pickle.load(open(config.RESULTS / "boost_model.pkl", "rb"))
    pr = pd.read_csv(config.RESULTS / "primes.csv")
    pr[["rel_3pa", "fg3_pct"]] = pr[["rel_3pa", "fg3_pct"]].fillna(0)
    pd.set_option("display.width", 220)

    # ---------- pool: each player's best single season ----------
    best = season_pool()
    best.to_csv(config.RESULTS / "single_season_pool.csv", index=False)
    print("Top 15 single seasons (one per player, minutes-weighted BPM)")
    print(best.head(15)[["rank", "name", "prime", "prime_pos", "bpm", "obpm", "dbpm", "usg_pct"]]
          .round(1).to_string(index=False))

    # ---------- search ----------
    pool = best.head(search.POOL).reset_index(drop=True)
    combos = valid_fives([slot_groups(p) for p in pool.prime_pos])
    print(f"\npool of {len(pool)} single seasons -> {len(combos):,} valid G-G-F-F-C fives")
    F = search.score(pool, combos, ridge, boost, k)
    picks = {"S1": ("usage_model", "1시즌 최적 조합 (원볼 모델)"), "A1": ("bpm_sum", "1시즌 최고 선수 (BPM 합)")}
    cand = []
    for lab, (col, desc) in picks.items():
        j = F[col].idxmax()
        players = pool.iloc[combos[j]]
        order, slots = assign_slots(players.prime_pos.tolist())
        print(f"{lab} {desc}: " + " | ".join(players.name.iloc[order]) +
              f"   usage model {F.usage_model[j]:+.1f}, BPM sum {F.bpm_sum[j]:+.1f}")
        for k_, (slot, i) in enumerate(zip(slots, order)):
            p = players.iloc[i]
            cand.append({"label": lab, "description": desc, "order": k_, "slot": slot, "player_id": p.player_id,
                         "name": p["name"], "season": p.prime, **{c: p[c] for c in PLAYER_COLS}})
    C = pd.DataFrame(cand).sort_values(["label", "order"], ascending=[False, True])
    C.to_csv(config.RESULTS / "single_season_candidates.csv", index=False)

    # ---------- the two fives to simulate: S1 and the 3-season best (P3) ----------
    C3 = pd.read_csv(config.RESULTS / "candidates.csv")
    best3 = pd.read_csv(config.RESULTS / "ratings_by_model.csv", index_col=0)[MAIN].idxmax()
    d3 = C3[C3.label == best3].sort_values("order")[["slot", "player_id", "name"]].merge(pr[["player_id", "prime"] + PLAYER_COLS],
                                                                   on="player_id")
    d3["label"] = "P3"
    d1 = C[C.label == "S1"].copy()
    d = pd.concat([d1[["label", "slot", "player_id", "name"] + PLAYER_COLS],
                   d3[["label", "slot", "player_id", "name"] + PLAYER_COLS]], ignore_index=True)
    d["share"] = 1.0
    ratings = sim.candidate_ratings(d, k, ridge, boost)
    draws = sim.rating_draws(d, k_boot, rmse, ridge, boost, rng)
    print("\nnet rating by model (S1 = best single seasons, P3 = best 3-season primes):")
    print(ratings.round(1).to_string())

    # ---------- head-to-head vs the dynasties, neutral court ----------
    T = sim.load("team_seasons")
    lg_pace = T[T.season == config.TARGET_SEASON].pace.mean()
    opps = []
    for s, tm in config.DYNASTIES:
        r = T[(T.season == s) & (T.team == tm)].iloc[0]
        opps.append({"season": s, "team": tm, "record": f"{int(r.wins)}-{int(r.losses)}",
                     "nrtg": r.nrtg, "pace": r.pace})
    U_game = rng.random(N)                                   # same random numbers for both fives
    U_series = rng.random((N, 7))
    H = pd.DataFrame([row for lab in ["S1", "P3"] for row in
                      h2h_rows(lab, ratings.loc[lab, MAIN], draws[lab].values, opps, lg_pace, P, U_game, U_series)])
    H.to_csv(config.RESULTS / "single_season_h2h.csv", index=False)
    print("\nhead-to-head, neutral court (single game / best-of-7):")
    hp = H.pivot(index="opponent", columns="label", values=["game_win_mc", "series_win_mc"])
    print(hp.round(3).to_string())

    # ---------- league season + playoffs ----------
    lg = sim.League(config.TARGET_SEASON, config.REPLACE_TEAM, P)
    print(f"\nleague: {season_label(config.TARGET_SEASON)}, the five replaces {lg.replaced}; "
          f"{N:,} seasons per run")
    L = []

    def record(lab, mode, R, res, fw):
        L.append({"label": lab, "mode": mode, "rating": R, "mean_wins": fw.mean(),
                  "wins_p5": np.percentile(fw, 5), "wins_p95": np.percentile(fw, 95),
                  "p_73plus": np.mean(fw >= 73), "p_82_0": np.mean(fw == 82), "p_top_seed": res["top_seed"][lg.five],
                  "p_playoffs": res["made"][lg.five], "p_finals": res["finals"][lg.five],
                  "p_champion": res["champ"][lg.five]})
        print(f"  {lab} {mode:16s} rating {R:+5.1f}  wins {fw.mean():5.1f} [{np.percentile(fw, 5):.0f}-"
              f"{np.percentile(fw, 95):.0f}]  P(82-0) {np.mean(fw == 82):6.2%}  P(title) {res['champ'][lg.five]:6.1%}"
              f"  ({time.time() - t0:.0f} s)")

    for lab in ["S1", "P3"]:
        R = ratings.loc[lab, MAIN]
        res, fw = sim.run_league(lg, lambda m, R=R: np.full(m, R), N, rng)
        record(lab, "fixed", R, res, fw)
        res, fw = sim.run_league(lg, lambda m, v=draws[lab].values: rng.choice(v, m), N, rng)
        record(lab, "uncertain", R, res, fw)
    for m in [x for x in sim.MODEL_NAMES if x != MAIN]:
        R = ratings.loc["S1", m]
        res, fw = sim.run_league(lg, lambda n_, R=R: np.full(n_, R), N, rng)
        record("S1", f"model:{m}", R, res, fw)
    LF = pd.DataFrame(L)
    LF.to_csv(config.RESULTS / "single_season_league.csv", index=False)

    n_games = 2 * len(opps) * N * 2
    n_seasons = len(L) * N
    summary = {"pool": int(len(pool)), "n_fives": int(len(combos)), "n_sims": N,
               "S1": list(d1.name), "S1_seasons": list(d1.season), "P3": list(d3.name),
               "ratings": ratings.round(3).to_dict(orient="index"),
               "h2h_simulated_games_and_series": n_games, "league_seasons_simulated": n_seasons,
               "runtime_s": time.time() - t0}
    json.dump(summary, open(config.RESULTS / "single_season_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n{n_games:,} head-to-head games and series, {n_seasons:,} league seasons in {time.time() - t0:.0f} s")

    # ---------- Fig. 9 ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.5), gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) the single-season five: that season's BPM, and the same player's 3-season prime BPM
    t = d1.merge(pr[["player_id", "bpm"]].rename(columns={"bpm": "bpm3"}), on="player_id", how="left")
    t = t.iloc[::-1].reset_index(drop=True)                  # PG on top
    y = np.arange(len(t))
    a1.barh(y, t.bpm, color=ACCENT, height=0.6, label="그 시즌 BPM")
    a1.scatter(t.bpm3, y, s=18, facecolor="white", edgecolor=INK, lw=0.9, zorder=3, label="같은 선수의 전성기 BPM (3년 기준)")
    for i, r in t.iterrows():
        a1.text(r.bpm + 0.25, i, f"{r.bpm:+.1f}", va="center", fontsize=6.3, color=INK)
    a1.set_yticks(y, [f"{r.slot}  {r['name']} ({r.season})" for _, r in t.iterrows()], fontsize=6.5)
    a1.set_xlim(0, t.bpm.max() * 1.18)
    a1.set_xlabel(f"BPM (100포세션당, 평균 선수 대비) · 팀 레이팅 {ratings.loc['S1', MAIN]:+.1f}")
    a1.legend(loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, fontsize=6)
    a1.grid(axis="y", visible=False)

    # (b) single-game win % against each dynasty: single-season five vs 3-season five
    h1 = H[H.label == "S1"].iloc[::-1].reset_index(drop=True)
    h3 = H[H.label == "P3"].iloc[::-1].reset_index(drop=True)
    y = np.arange(len(h1))
    lf = LF.set_index(["label", "mode"])
    for h, dy, lab, kw in [(h3, -0.18, "P3", dict(facecolor="white", edgecolor=INK2, lw=0.9)),
                           (h1, 0.18, "S1", dict(color=ACCENT))]:
        r_ = lf.loc[(lab, "fixed")]
        name = "1시즌 5인" if lab == "S1" else "전성기 5인 (3년 기준, 본문)"
        a2.hlines(y + dy, h.game_lo * 100, h.game_hi * 100, color=MUTED, lw=1.6)
        a2.scatter(h.game_win_mc * 100, y + dy, s=18, zorder=3, **kw,
                   label=f"{name} ({r_.rating:+.1f}): 평균 {r_.mean_wins:.1f}승, 82-0 {r_.p_82_0:.0%}, 우승 {pct(r_.p_champion)}")
    for i, r in h1.iterrows():
        a2.text(101, i + 0.18, f"{r.game_win_mc:.0%}", va="center", fontsize=6, color=INK)
        a2.text(101, i - 0.18, f"{h3.game_win_mc[i]:.0%}", va="center", fontsize=6, color=INK2)
    a2.set_yticks(y, [f"{r.opponent} ({r.opp_record}, {r.opp_nrtg:+.1f})" for r in h1.itertuples()], fontsize=6.3)
    a2.set_xlim(70, 106)
    a2.set_xlabel("단일 경기 승률 (%), 중립 경기장, 막대 = 90% 불확실성 구간")
    a2.legend(loc="upper center", bbox_to_anchor=(0.4, -0.24), ncol=1, fontsize=6)
    a2.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig9_single_season.png")


if __name__ == "__main__":
    main()
