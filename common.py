"""Helpers shared by the step scripts: HTML table parsing, loading tables, figure style."""
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from bs4 import BeautifulSoup, Comment
import config

# ---------- HTML parsing ----------

def undo_latin1(text):
    """Repair a page that was decoded as ISO-8859-1 before it was saved (requests' default when the
    server sends no charset): 'Jokić' had become 'JokiÄ\x87'. Re-decoding restores the original
    exactly. A page that is already correct fails the round trip and comes back unchanged."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text


def read_table(path, table_id):
    """Rows of one Basketball-Reference table as a list of dicts keyed by data-stat.
    Also looks inside HTML comments, where the site hides some tables."""
    html = undo_latin1(path.read_text(encoding="utf-8"))
    soup = BeautifulSoup(html, "lxml")
    t = soup.find("table", id=table_id)
    if t is None:
        for c in soup.find_all(string=lambda s: isinstance(s, Comment) and table_id in s):
            t = BeautifulSoup(c, "lxml").find("table", id=table_id)
            if t is not None:
                break
    if t is None:
        return []
    out = []
    for tr in t.find("tbody").find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        rec = {}
        for cell in tr.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if not stat:
                continue
            rec[stat] = cell.get_text(strip=True)
            if cell.get("data-append-csv"):
                rec[stat + "_id"] = cell["data-append-csv"]
            a = cell.find("a")
            if a and a.get("href"):
                m = re.search(r"/teams/([A-Z0-9]{3})/", a["href"])
                if m:
                    rec[stat + "_abbr"] = m.group(1)
        out.append(rec)
    return out


def to_num(df, skip=()):
    for c in df.columns:
        if c in skip:
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().any() and conv.notna().sum() >= 0.5 * df[c].notna().sum():
            df[c] = conv
    return df

# ---------- loading ----------

def load(name):
    return pd.read_csv(config.DATA / f"{name}.csv")


def season_label(year):
    """2026 -> '2025-26'"""
    year = int(year)
    return f"{year - 1}-{str(year)[2:]}"

# ---------- figure style: paper column width, grey first, one blue accent ----------

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#b9b8b2"
GRID = "#e6e5e1"
ACCENT = "#2a78d6"
COL_W = 3.5            # IEEE column width, inches
FULL_W = 7.16          # IEEE text width (two columns), inches

KOREAN_FONTS = ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic", "AppleGothic", "Noto Sans KR"]
# Malgun Gothic has no glyphs for letters like ć and č (Jokić, Dončić). Matplotlib takes each missing
# glyph from the next font in the list, so a full Latin font goes right after the Korean one.
LATIN_FALLBACK = ["Segoe UI", "DejaVu Sans"]


def style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    korean = [f for f in KOREAN_FONTS if f in installed][:1]       # one Korean font only
    family = korean + [f for f in LATIN_FALLBACK if f in installed or f == "DejaVu Sans"]
    if not korean:
        warnings.warn("No Korean font found: Korean labels will not render. Install 'Malgun Gothic' or 'Noto Sans CJK KR'.")
    plt.rcParams.update({
        "font.family": family, "axes.unicode_minus": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "text.color": INK, "legend.frameon": False,
    })


def save(fig, name):
    config.FIGURES.mkdir(exist_ok=True)
    fig.savefig(config.FIGURES / name)
    plt.close(fig)
    print(f"  wrote figures/{name}")
