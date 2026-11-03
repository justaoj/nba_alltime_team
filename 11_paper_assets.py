"""Step 11: turn the results into paper assets.

  paper/figures/   the 7 result figures used in the paper (Fig. 2-8; Fig. 1 is drawn in LaTeX)
  paper/tables/    table1_data.tex, table2_jordan.tex, table3_candidates.tex
  paper/numbers.tex  \\newcommand macros for every number quoted in the text, so the paper
                     updates itself after a re-run (write \\BestTitleOdds instead of typing 99.9%)
Run:  python3 11_paper_assets.py
"""
import json
import shutil
import pandas as pd
import config
from common import season_label

FIGS = ["fig2_metric_rmse.png", "fig3_awards.png", "fig4_jordan.png", "fig5_top_primes.png",
        "fig6_team_model.png", "fig7_head_to_head.png", "fig8_season_title.png"]
R = config.RESULTS


def j(name):
    return json.load(open(R / name, encoding="utf-8"))


def pct(x, d=1):
    return f"{x * 100:.{d}f}\\%"


def comma(n):
    return f"{int(n):,}".replace(",", "{,}")


def tex_escape(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def main():
    (config.PAPER / "figures").mkdir(parents=True, exist_ok=True)
    (config.PAPER / "tables").mkdir(parents=True, exist_ok=True)
    for f in FIGS:
        shutil.copy(config.FIGURES / f, config.PAPER / "figures" / f)
    print(f"copied {len(FIGS)} figures to paper/figures")

    P = pd.read_csv(config.DATA / "player_seasons.csv")
    T = pd.read_csv(config.DATA / "team_seasons.csv")
    G = pd.read_csv(config.DATA / "games.csv")
    pr = pd.read_csv(R / "primes.csv")
    mt = pd.read_csv(R / "metric_test.csv").set_index("metric")
    aw = pd.read_csv(R / "metric_awards.csv").set_index("column")
    cj = pd.read_csv(R / "case_jordan.csv")
    cjs = pd.read_csv(R / "case_jordan_seasons.csv")
    tm = j("team_model_summary.json")
    cv = pd.read_csv(R / "team_model_cv.csv").set_index("kind")
    mp = j("model_params.json")
    sp = j("sim_params.json")
    ss = j("search_summary.json")
    sim = j("simulation_summary.json")
    cand = pd.read_csv(R / "candidates.csv")
    h2h = pd.read_csv(R / "h2h.csv")
    lf = pd.read_csv(R / "league_five.csv")
    ps = j("primes_summary.json")
    n_pages = len(list(config.RAW.glob("*.html")))
    best = sim["best"]

    # ---------- Table I: data ----------
    rows = [("다운로드한 웹페이지", comma(n_pages), "시즌당 약 15쪽"),
            ("선수-시즌 행", comma(len(P)), "트레이드 선수는 팀별 + 합계 행"),
            ("전성기(3년 기준) 선수", comma(ps["n_players"]), "시즌당 1{,}500분 이상, 연속 3시즌"),
            ("팀-시즌", comma(len(T)), f"{season_label(T.season.min())}--{season_label(T.season.max())}"),
            ("정규시즌 경기", comma(len(G)), "플레이인, 플레이오프, 컵 결승 제외")]
    body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)
    (config.PAPER / "tables" / "table1_data.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{데이터 요약 (출처: Basketball-Reference)}\n\\label{tab:data}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lrl@{}}\n\\toprule\n항목 & 개수 & 비고 \\\\\n\\midrule\n" + body +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")

    # ---------- Table II: Jordan ----------
    body = "\n".join(f"{tex_escape(r.label)} & {tex_escape(r.jordan_txt)} & {tex_escape(r.league_txt)} & {r.percentile:.0f} \\\\"
                     for r in cj.itertuples())
    (config.PAPER / "tables" / "table2_jordan.tex").write_text(
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{마이클 조던 {season_label(config.CASE_SEASONS[0])}--{season_label(config.CASE_SEASONS[-1])} "
        "3시즌 기록과 리그 평균 (출전 시간 가중 평균)}\n\\label{tab:jordan}\n\\footnotesize\n"
        "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n지표 & 조던 & 리그 평균 & 백분위 \\\\\n\\midrule\n" + body +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")

    # ---------- Table III: candidate fives ----------
    labels = sorted(cand.label.unique())
    desc = cand.drop_duplicates("label").set_index("label").description
    bulls = [o for o in h2h.opponent.unique() if o.endswith("CHI")]
    lines = []
    for lab in labels:
        c = cand[cand.label == lab].sort_values("order")
        names = " & ".join(tex_escape(n.split(" ", 1)[-1]) for n in c.name)
        h = h2h[h2h.label == lab]
        f = lf[(lf.label == lab) & (lf["mode"] == "fixed")].iloc[0]
        u = lf[(lf.label == lab) & (lf["mode"] == "uncertain")].iloc[0]
        vs_bulls = h[h.opponent.isin(bulls)].series_win_mc.iloc[0] if bulls else float("nan")
        lines.append(f"{lab} & {names} & {f.rating:+.1f} & {pct(h.game_win_mc.mean())} & {pct(vs_bulls)} & "
                     f"{f.mean_wins:.1f} [{f.wins_p5:.0f}--{f.wins_p95:.0f}] & {pct(f.p_champion)} ({pct(u.p_champion)}) \\\\")
    notes = "; ".join(f"{l}: {tex_escape(desc[l])}" for l in labels)
    (config.PAPER / "tables" / "table3_candidates.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{후보 라인업과 시뮬레이션 결과 (경기·시리즈·시즌 각 "
        f"{comma(sim['n_sims'])}회)}}\n\\label{{tab:candidates}}\n\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}clllllrrrrr@{}}\n\\toprule\n"
        "팀 & G & G & F & F & C & 넷레이팅 & \\shortstack{왕조 상대\\\\경기 승률} & "
        "\\shortstack{1995--96 불스\\\\상대 시리즈} & \\shortstack{정규시즌 승수\\\\{[}90\\% 구간{]}} & "
        "\\shortstack{우승 확률\\\\(불확실성 포함)} \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\n\\vspace{2pt}\n"
        f"\\parbox{{0.95\\textwidth}}{{\\scriptsize {notes}. G·F·C = 가드·포워드·센터 (올-NBA 팀과 같은 구성). 넷레이팅은 원볼 사용률 모델 예측. 정규시즌·우승 확률은 "
        f"{season_label(config.TARGET_SEASON)} 리그에서 {tex_escape(sim['replaced_name'])} 자리를 대신한 결과이며, "
        "괄호 안은 모델 불확실성을 포함한 값.}\n\\end{table*}\n", encoding="utf-8")

    # ---------- numbers.tex ----------
    best_names = ", ".join(cand[cand.label == best].sort_values("order").name)
    hb = h2h[h2h.label == best]
    fb = lf[(lf.label == best) & (lf["mode"] == "fixed")].iloc[0]
    ub = lf[(lf.label == best) & (lf["mode"] == "uncertain")].iloc[0]
    boost = lf[(lf.label == best) & (lf["mode"] == "model:boosting")]
    jb = cj.set_index("stat")
    macros = {
        "FirstSeason": season_label(config.FIRST_SEASON), "LastSeason": season_label(config.LAST_SEASON),
        "NSeasons": str(config.LAST_SEASON - config.FIRST_SEASON + 1), "NPages": comma(n_pages),
        "NPlayerRows": comma(len(P)), "NTeamSeasons": comma(len(T)), "NGames": comma(len(G)),
        "NPrimePlayers": comma(ps["n_players"]), "RhoOneThree": f"{ps['rho_1_3']:.2f}", "RhoThreeFive": f"{ps['rho_3_5']:.2f}",
        "RmseBPM": f"{mt.loc['BPM', 'rmse']:.2f}", "RmseWS": f"{mt.loc['WS/48', 'rmse']:.2f}",
        "RmsePER": f"{mt.loc['PER', 'rmse']:.2f}",
        "RmseBox": f"{mt.loc[[m for m in mt.index if m.startswith('박스')][0], 'rmse']:.2f}",
        "RmseBaseline": f"{mt.loc[[m for m in mt.index if m.startswith('직전')][0], 'rmse']:.2f}",
        "MvpTopThreeBPM": pct(aw.loc["bpm", "mvp_top3"], 0), "NbaTopTenBPM": pct(aw.loc["bpm", "nba1_top10"], 0),
        "MvpTopThreePTS": pct(aw.loc["pts_per_g", "mvp_top3"], 0),
        "JordanBPM": f"{jb.loc['bpm', 'jordan']:+.1f}", "JordanPTS": f"{jb.loc['pts_per_g', 'jordan']:.1f}",
        "JordanTS": f"{jb.loc['ts_pct', 'jordan']:.3f}", "JordanLeagueTS": f"{jb.loc['ts_pct', 'league_avg']:.3f}",
        "JordanBPMRanks": ", ".join(str(r) for r in cjs.bpm_rank),
        "TopPrime": tex_escape(pr.iloc[0]["name"]), "TopPrimeBPM": f"{pr.iloc[0].bpm:+.1f}",
        "KUsage": f"{mp['k']:.2f}", "RmseTeamModel": f"{cv.loc['usage', 'rmse']:.2f}",
        "RTwoTeamModel": f"{tm['r2_usage']:.2f}", "RmseRidge": f"{cv.loc['ridge', 'rmse']:.2f}",
        "BestRealTeam": f"{tm['best_real']['team']} {season_label(tm['best_real']['season'])}",
        "BestRealNR": f"{tm['best_real']['nrtg']:+.1f}",
        "HomeCourt": f"{sp['home_court']:.2f}", "SigmaGame": f"{sp['sigma']:.2f}", "BrierTest": f"{sp['test_brier']:.3f}",
        "Coverage": pct(sp["coverage_pipeline_with_model_error"]),
        "NFives": comma(ss["n_fives"]), "PriceOfPositions": f"{ss['price_of_positions']:.1f}",
        "NSims": comma(sim["n_sims"]), "NLeagueSeasons": comma(sim["league_seasons_simulated"]),
        "NLeagueGames": comma(sim["league_games_simulated"]), "NHtoH": comma(sim["h2h_simulated_games_and_series"]),
        "ReplacedTeam": tex_escape(sim["replaced_name"]),
        "BestLabel": best, "BestFive": tex_escape(best_names), "BestRating": f"{fb.rating:+.1f}",
        "BestMeanWins": f"{fb.mean_wins:.1f}", "BestTitleOdds": pct(fb.p_champion),
        "BestTitleOddsUnc": pct(ub.p_champion),
        "BestMinGameWin": pct(hb.game_win_mc.min(), 0), "BestMaxGameWin": pct(hb.game_win_mc.max(), 0),
        "BestVsBullsGame": pct(hb[hb.opponent.str.endswith("CHI")].game_win_mc.iloc[0], 0),
        "BoostingTitleOdds": pct(boost.p_champion.iloc[0]) if len(boost) else "--",
    }
    text = "% generated by 11_paper_assets.py - do not edit; re-run the pipeline instead\n" + "\n".join(
        f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n"
    (config.PAPER / "numbers.tex").write_text(text, encoding="utf-8")
    print(f"wrote paper/tables (3 tables) and paper/numbers.tex ({len(macros)} numbers)")


if __name__ == "__main__":
    main()
