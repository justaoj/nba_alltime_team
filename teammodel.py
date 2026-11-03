"""Team-rating models shared by steps 5, 7 and 8.

A team (or a five) is a set of players, each with a floor share: share = minutes / (team minutes / 5),
so the five floor slots add to 5. For the all-time five every share is 1 (48 minutes each).

Models (all give net rating, points per 100 possessions vs an average team):
  additive    sum(share x BPM)                                   ignores fit
  ridge       ridge regression on fit features, trained on real team-seasons
  usage       structural one-ball model: usage rescaled to 100%, each player's efficiency moves
              k points of ORtg per point of usage he gives up (or must take on), defense = sum(share x DBPM)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIT_FEATURES = ["obpm_sum", "dbpm_sum", "usg_load", "drb_load", "ast_load", "spacing", "rim"]
PLAYER_COLS = ["bpm", "obpm", "dbpm", "usg_pct", "rel_ortg", "drb_pct", "ast_pct", "rel_3pa", "blk_pct"]


def features(df, key, k=0.0):
    """df: one row per player with PLAYER_COLS + 'share'; key: column(s) identifying the team.
    Returns one row per team with the model features."""
    d = df.copy()
    s = d.share
    d["bpm_sum"] = s * d.bpm
    d["obpm_sum"] = s * d.obpm
    d["dbpm_sum"] = s * d.dbpm
    d["usg_load"] = s * d.usg_pct
    d["drb_load"] = s * d.drb_pct
    d["ast_load"] = s * d.ast_pct
    d["spacing"] = s * d.rel_3pa * 100
    d["rim"] = s * d.blk_pct
    g = d.groupby(key)
    F = g[["bpm_sum", "obpm_sum", "dbpm_sum", "usg_load", "drb_load", "ast_load", "spacing", "rim"]].sum()
    # structural usage model
    d["u"] = d.usg_pct
    tot = g.apply(lambda x: (x.share * x.u).sum())
    d = d.join(tot.rename("u_tot"), on=key)
    d["u_new"] = d.u * 100 / d.u_tot                      # usage after rescaling to 100%
    d["ortg_new"] = d.rel_ortg - k * (d.u_new - d.u)      # efficiency moves against usage
    d["poss"] = d.share * d.u_new
    off = d.groupby(key).apply(lambda x: np.average(x.ortg_new, weights=x.poss))
    F["usage_off"] = off
    F["usage_model"] = off + F.dbpm_sum
    return F.reset_index()


def fit_ridge(F, y):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    m.fit(F[FIT_FEATURES], y)
    return m


def five_frame(players, label="five"):
    """players: DataFrame of 5 prime rows -> frame ready for features(), all shares = 1."""
    d = players[["name"] + PLAYER_COLS].copy()
    d["share"] = 1.0
    d["team"] = label
    return d
