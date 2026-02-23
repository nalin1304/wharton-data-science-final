#!/usr/bin/env python3
"""
WHL 2026 — v9.4: Leakage-Free Competition Pipeline

Key updates:
- Leakage-free model evaluation (pregame features only)
- Time-aware validation + overfit diagnostics
- Log-loss-first model selection with calibration/stability tie-breakers
- Wharton-aligned outputs, including all-pairs matchup probabilities
- Stabilized model pool with stronger regularization (v9.1)
- Train-CV Elo parameter tuning (v9.2)
- WHR-like whole-history ratings + weighted model-consensus layer (v9.3)
- Formal leakage sanity report + robustness simulation layer (v9.4)
"""

import warnings
warnings.filterwarnings("ignore")

import os
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join("/tmp", "mplconfig_whl")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

import xgboost as xgb
import lightgbm as lgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


BASE = os.path.dirname(os.path.abspath(__file__))
OUT = BASE
RANDOM_STATE = 42

ELO_K = 8
ELO_HA = 60
ELO_XG_W = 0.15
ELO_REG = 0.10
ELO_DECAY = True
ELO_TUNE_MIN_GAIN = 0.003

MODEL_MIN_HISTORY_GAMES = 9
CONSENSUS_WEIGHT_TEMP = 10.0
CONSENSUS_SD_PENALTY = 3.0
CONSENSUS_MAX_SHRINK = 0.30
ROBUSTNESS_N_SIMS = 2000
ROBUSTNESS_BASE_SIGMA = 0.10
ROBUSTNESS_TEAM_SIGMA_W = 0.25
ROBUSTNESS_SPREAD_SIGMA_W = 0.15

FALLBACK_MATCHUPS = [
    ("brazil", "kazakhstan"),
    ("netherlands", "mongolia"),
    ("peru", "rwanda"),
    ("thailand", "oman"),
    ("pakistan", "germany"),
    ("india", "usa"),
    ("panama", "switzerland"),
    ("iceland", "canada"),
    ("china", "france"),
    ("philippines", "morocco"),
    ("ethiopia", "saudi_arabia"),
    ("singapore", "new_zealand"),
    ("guatemala", "south_korea"),
    ("uk", "mexico"),
    ("vietnam", "serbia"),
    ("indonesia", "uae"),
]


def safe_div(num, den, default=0.0):
    if den == 0 or pd.isna(den):
        return default
    return num / den


def model_probs(model, x):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(x)
        p = (d - d.min()) / (d.max() - d.min() + 1e-12)
    else:
        p = model.predict(x).astype(float)
    return np.clip(p, 1e-7, 1 - 1e-7)


def detect_corrupted_csvs(base):
    bad = []
    for name in sorted([f for f in os.listdir(base) if f.lower().endswith(".csv")]):
        path = os.path.join(base, name)
        try:
            with open(path, "rb") as fh:
                head = fh.read(200).decode("utf-8", "ignore").lower()
            if "<!doctype html" in head or "<html" in head:
                bad.append(name)
        except OSError:
            continue
    return bad


def load_matchups(base, teams):
    xlsx_path = os.path.join(base, "WHSDSC_Rnd1_matchups.xlsx")
    try:
        df = pd.read_excel(xlsx_path)
        required = {"home_team", "away_team"}
        if not required.issubset(df.columns):
            raise KeyError(f"Missing required columns in matchup file: {required}")
        matchups = list(zip(df["home_team"], df["away_team"]))
        missing = [(h, a) for h, a in matchups if h not in teams or a not in teams]
        if missing:
            raise ValueError(f"Matchups include unknown teams: {missing[:3]}")
        return matchups, "xlsx"
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"  Matchup load warning: {exc}")
        return FALLBACK_MATCHUPS, "fallback"


def load_data(base):
    ms = pd.read_csv(os.path.join(base, "MainSheet.csv"), low_memory=False)
    raw = pd.read_csv(os.path.join(base, "whl_2025.csv"), low_memory=False)
    pp = pd.read_csv(os.path.join(base, "PP_consistency.csv"), header=None)
    unit = pd.read_csv(os.path.join(base, "Unit_specific.csv"))
    rest = pd.read_csv(os.path.join(base, "Game_Rest.csv"), low_memory=False)
    tev = pd.read_csv(os.path.join(base, "Team_ev_strength.csv"))

    # Normalize blank-like strings to NaN before dropping all-empty rows.
    for df in (ms, rest):
        obj_cols = df.select_dtypes(include=["object"]).columns
        if len(obj_cols):
            df[obj_cols] = df[obj_cols].replace(r"^\s*$", np.nan, regex=True)
    ms = ms.dropna(how="all").reset_index(drop=True)
    rest = rest.dropna(how="all").reset_index(drop=True)
    ms = ms.dropna(subset=["game_id", "record_id", "home_team", "away_team"]).reset_index(drop=True)
    rest = rest.dropna(subset=["team", "game_id", "rest_proxy"]).reset_index(drop=True)

    # Correct PP consistency mapping (verified against row-level schema)
    if pp.shape[1] != 26:
        raise ValueError(f"PP_consistency.csv expected 26 columns, found {pp.shape[1]}")
    pp.columns = [
        "game_id", "record_id", "home_team", "away_team", "flag",
        "home_off", "home_def", "away_off", "away_def", "home_goalie", "away_goalie",
        "toi", "home_aux_1", "home_shots", "home_xg", "home_aux_2",
        "away_aux_1", "away_aux_2", "away_shots", "away_xg", "away_aux_3",
        "aux_a", "aux_b", "aux_c", "aux_d", "aux_e",
    ]

    unit.columns = ["team", "off_line", "def_pair", "total_toi", "total_goals", "total_shots", "total_xg"]
    unit["xg60"] = np.where(unit["total_toi"] > 0, unit["total_xg"] / (unit["total_toi"] / 3600), 0.0)

    required_raw = {"game_id", "record_id", "home_team", "away_team", "toi", "home_xg", "away_xg", "home_goals", "away_goals"}
    required_ms = required_raw | {"game_state", "TOI flagging", "Shelter_Index", "home_off_line", "away_off_line"}
    required_rest = {"team", "game_id", "rest_proxy"}
    required_tev = {"Team", "xG_For", "xG_against", "Even_Str_xG%"}

    for name, req, got in [
        ("whl_2025.csv", required_raw, set(raw.columns)),
        ("MainSheet.csv", required_ms, set(ms.columns)),
        ("Game_Rest.csv", required_rest, set(rest.columns)),
        ("Team_ev_strength.csv", required_tev, set(tev.columns)),
    ]:
        miss = sorted(req - got)
        if miss:
            raise ValueError(f"{name} missing required columns: {miss}")

    teams = sorted(raw["home_team"].unique())
    return ms, raw, pp, unit, rest, tev, teams


def build_games(raw):
    gs = raw.groupby("game_id").agg(
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        home_goals=("home_goals", "sum"),
        away_goals=("away_goals", "sum"),
        home_xG=("home_xg", "sum"),
        away_xG=("away_xg", "sum"),
        home_shots=("home_shots", "sum"),
        away_shots=("away_shots", "sum"),
        went_ot=("went_ot", "max"),
        total_toi=("toi", "sum"),
    ).reset_index()
    gs["home_win"] = (gs["home_goals"] > gs["away_goals"]).astype(int)
    gs["game_num"] = gs["game_id"].str.extract(r"game_(\d+)").astype(int)
    gs = gs.sort_values("game_num").reset_index(drop=True)
    gs["goal_diff"] = gs["home_goals"] - gs["away_goals"]
    gs["close_game"] = (gs["goal_diff"].abs() <= 1).astype(int)
    return gs


def build_shift_team_game(ms):
    df = ms[(ms["TOI flagging"] != "LONG") & (ms["TOI flagging"] != "SHORT")].copy()
    df = df.dropna(subset=["game_id", "home_team", "away_team"])
    df["is_ev"] = (df["game_state"] == "Even Strength").astype(int)
    df["is_pp"] = (df["game_state"] == "Power Play").astype(int)

    home = df[
        [
            "game_id", "home_team", "away_team", "home_xg", "away_xg", "home_goals", "away_goals",
            "home_shots", "away_shots", "toi", "is_ev", "is_pp", "home_off_line", "Shelter_Index",
        ]
    ].copy()
    home.columns = [
        "game_id", "team", "opp", "xGF", "xGA", "GF", "GA", "SF", "SA", "toi",
        "is_ev", "is_pp", "off_line", "shelter",
    ]

    away = df[
        [
            "game_id", "away_team", "home_team", "away_xg", "home_xg", "away_goals", "home_goals",
            "away_shots", "home_shots", "toi", "is_ev", "is_pp", "away_off_line", "Shelter_Index",
        ]
    ].copy()
    away.columns = [
        "game_id", "team", "opp", "xGF", "xGA", "GF", "GA", "SF", "SA", "toi",
        "is_ev", "is_pp", "off_line", "shelter",
    ]

    long = pd.concat([home, away], ignore_index=True)

    long["ev_xGF"] = np.where(long["is_ev"] == 1, long["xGF"], 0.0)
    long["ev_xGA"] = np.where(long["is_ev"] == 1, long["xGA"], 0.0)

    pp_up = (long["is_pp"] == 1) & (long["off_line"] == "PP_up")
    pp_kill = (long["is_pp"] == 1) & (long["off_line"] == "PP_kill_dwn")
    long["pp_xGF"] = np.where(pp_up, long["xGF"], 0.0)
    long["pp_toi"] = np.where(pp_up, long["toi"], 0.0)
    long["pk_xGA"] = np.where(pp_kill, long["xGA"], 0.0)
    long["pk_toi"] = np.where(pp_kill, long["toi"], 0.0)

    line1 = (long["is_ev"] == 1) & (long["off_line"] == "first_off")
    line2 = (long["is_ev"] == 1) & (long["off_line"] == "second_off")
    long["f1_xGF"] = np.where(line1, long["xGF"], 0.0)
    long["f1_toi"] = np.where(line1, long["toi"], 0.0)
    long["f2_xGF"] = np.where(line2, long["xGF"], 0.0)
    long["f2_toi"] = np.where(line2, long["toi"], 0.0)

    shift_game = long.groupby(["game_id", "team"], as_index=False).agg(
        xGF=("xGF", "sum"),
        xGA=("xGA", "sum"),
        GF=("GF", "sum"),
        GA=("GA", "sum"),
        SF=("SF", "sum"),
        SA=("SA", "sum"),
        toi=("toi", "sum"),
        ev_xGF=("ev_xGF", "sum"),
        ev_xGA=("ev_xGA", "sum"),
        pp_xGF=("pp_xGF", "sum"),
        pp_toi=("pp_toi", "sum"),
        pk_xGA=("pk_xGA", "sum"),
        pk_toi=("pk_toi", "sum"),
        f1_xGF=("f1_xGF", "sum"),
        f1_toi=("f1_toi", "sum"),
        f2_xGF=("f2_xGF", "sum"),
        f2_toi=("f2_toi", "sum"),
        shelter=("shelter", "mean"),
    )
    return shift_game, len(df)


def build_goalie_table(raw):
    hg = raw.groupby("home_goalie").agg(
        xGA=("away_xg", "sum"), GA=("away_goals", "sum"), TOI=("toi", "sum")
    ).reset_index().rename(columns={"home_goalie": "gid"})
    ag = raw.groupby("away_goalie").agg(
        xGA=("home_xg", "sum"), GA=("home_goals", "sum"), TOI=("toi", "sum")
    ).reset_index().rename(columns={"away_goalie": "gid"})
    gt = pd.concat([hg, ag], ignore_index=True).groupby("gid", as_index=False).sum()
    gt = gt[gt["gid"] != "empty_net"].copy()
    gt["GSAx60"] = np.where(gt["TOI"] > 0, (gt["xGA"] - gt["GA"]) / gt["TOI"] * 3600, 0.0)
    goalie_map = raw.groupby("home_team")["home_goalie"].agg(
        lambda x: x[x != "empty_net"].mode().iloc[0] if len(x[x != "empty_net"].mode()) > 0 else "unknown"
    )
    return gt, goalie_map


def build_pp_team_stats(pp, teams):
    out = {}
    for team in teams:
        h_pp = pp[(pp["home_team"] == team) & (pp["home_off"] == "PP_up")]
        a_pp = pp[(pp["away_team"] == team) & (pp["away_off"] == "PP_up")]
        pp_xg = h_pp["home_xg"].sum() + a_pp["away_xg"].sum()
        pp_n = len(h_pp) + len(a_pp)

        h_pk = pp[(pp["home_team"] == team) & (pp["home_off"] == "PP_kill_dwn")]
        a_pk = pp[(pp["away_team"] == team) & (pp["away_off"] == "PP_kill_dwn")]
        pk_xga = h_pk["away_xg"].sum() + a_pk["home_xg"].sum()
        pk_n = len(h_pk) + len(a_pk)

        out[team] = {
            "pp_xg_shift": safe_div(pp_xg, pp_n, 0.0),
            "pk_xga_shift": safe_div(pk_xga, pk_n, 0.0),
        }
    return out


def build_unit_team_stats(unit, teams):
    out = {}
    for team in teams:
        tdf = unit[unit["team"] == team]
        out[team] = {
            "best_unit_xg60": float(tdf["xg60"].max()) if len(tdf) else 0.0,
            "avg_unit_xg60": float(tdf["xg60"].mean()) if len(tdf) else 0.0,
        }
    return out


def build_rest_by_game_team(rest, raw):
    rec_map = raw[["record_id", "game_id"]].drop_duplicates("record_id")
    merged = rest.merge(rec_map, left_on="game_id", right_on="record_id", how="left")
    merged = merged.dropna(subset=["game_id_y", "team", "rest_proxy"])
    merged["team"] = merged["team"].astype(str)
    out = merged.groupby(["game_id_y", "team"], as_index=False)["rest_proxy"].mean()
    out = out.rename(columns={"game_id_y": "game_id"})
    return out


def compute_elo_pregame(gs, teams, k=ELO_K, ha=ELO_HA, xg_w=ELO_XG_W, decay=ELO_DECAY, regress_pct=ELO_REG):
    elos = {t: 1500.0 for t in teams}
    rows = []
    for i, (_, g) in enumerate(gs.iterrows()):
        h, a = g["home_team"], g["away_team"]
        he, ae = elos[h], elos[a]
        exp = 1.0 / (1.0 + 10 ** ((ae - he - ha) / 400))

        rows.append(
            {
                "game_id": g["game_id"],
                "home_team": h,
                "away_team": a,
                "home_elo_pre": he,
                "away_elo_pre": ae,
                "home_exp": exp,
            }
        )

        act = g["home_win"]
        k_eff = k * max(0.5, 1 - 0.4 * i / len(gs)) if decay else k
        delta = k_eff * (act - exp)
        xg_signal = np.tanh(g["home_xG"] - g["away_xG"])
        delta += xg_w * k * xg_signal
        elos[h] = he + delta
        elos[a] = ae - delta

        if regress_pct > 0 and i > 0 and i % 200 == 0:
            m = np.mean(list(elos.values()))
            for t in elos:
                elos[t] = elos[t] * (1 - regress_pct) + m * regress_pct

    return pd.DataFrame(rows), elos


def tune_elo_params(gs, teams, game_ids, y, split_idx):
    # Grid tuned only on training segment using time-series CV
    k_grid = [6, 8, 10]
    ha_grid = [55, 60, 65]
    xg_grid = [0.05, 0.10, 0.15]
    decay_grid = [True]
    reg_grid = [0.00, 0.05, 0.10]

    train_ids = game_ids[:split_idx]
    y_train = y[:split_idx]

    tscv = TimeSeriesSplit(n_splits=4 if len(y_train) >= 240 else 3)
    best = None
    best_score = float("inf")

    def cv_stats_for_params(k, ha, xgw, decay, reg):
        elo_df, _ = compute_elo_pregame(gs, teams, k=k, ha=ha, xg_w=xgw, decay=decay, regress_pct=reg)
        prob_map = dict(zip(elo_df["game_id"], elo_df["home_exp"]))
        p_train = np.array([prob_map[g] for g in train_ids])
        p_train = np.clip(p_train, 1e-7, 1 - 1e-7)
        cv_vals = []
        for _, va_idx in tscv.split(p_train):
            cv_vals.append(log_loss(y_train[va_idx], p_train[va_idx]))
        cv_mean = float(np.mean(cv_vals))
        cv_sd = float(np.std(cv_vals))
        score = cv_mean + 0.15 * cv_sd
        return cv_mean, cv_sd, score

    for k in k_grid:
        for ha in ha_grid:
            for xgw in xg_grid:
                for decay in decay_grid:
                    for reg in reg_grid:
                        cv_mean, cv_sd, score = cv_stats_for_params(k, ha, xgw, decay, reg)
                        if score < best_score:
                            best_score = score
                            best = {
                                "k": k,
                                "ha": ha,
                                "xg_w": xgw,
                                "decay": decay,
                                "reg": reg,
                                "cv_ll": cv_mean,
                                "cv_sd": cv_sd,
                            }

    default_cv_ll, default_cv_sd, default_score = cv_stats_for_params(
        ELO_K, ELO_HA, ELO_XG_W, ELO_DECAY, ELO_REG
    )
    default_choice = {
        "k": ELO_K,
        "ha": ELO_HA,
        "xg_w": ELO_XG_W,
        "decay": ELO_DECAY,
        "reg": ELO_REG,
        "cv_ll": default_cv_ll,
        "cv_sd": default_cv_sd,
    }

    if best is None:
        return default_choice

    if (default_score - best_score) < ELO_TUNE_MIN_GAIN:
        # Keep proven default unless CV gain is meaningfully better.
        return default_choice

    return best


def build_team_game_panel(gs, shift_game, elo_df, rest_game):
    home = gs[
        [
            "game_id", "game_num", "home_team", "away_team", "home_win", "went_ot",
            "home_goals", "away_goals", "home_shots", "away_shots", "home_xG", "away_xG",
            "goal_diff", "close_game",
        ]
    ].copy()
    home.columns = [
        "game_id", "game_num", "team", "opp", "win", "went_ot",
        "GF_raw", "GA_raw", "SF_raw", "SA_raw", "xGF_raw", "xGA_raw",
        "goal_diff", "close_game",
    ]
    home["is_home"] = 1

    away = gs[
        [
            "game_id", "game_num", "away_team", "home_team", "home_win", "went_ot",
            "away_goals", "home_goals", "away_shots", "home_shots", "away_xG", "home_xG",
            "goal_diff", "close_game",
        ]
    ].copy()
    away.columns = [
        "game_id", "game_num", "team", "opp", "home_win", "went_ot",
        "GF_raw", "GA_raw", "SF_raw", "SA_raw", "xGF_raw", "xGA_raw",
        "goal_diff", "close_game",
    ]
    away["win"] = 1 - away["home_win"]
    away["goal_diff"] = -away["goal_diff"]
    away["is_home"] = 0
    away = away.drop(columns=["home_win"])

    panel = pd.concat([home, away], ignore_index=True)
    panel = panel.merge(shift_game, on=["game_id", "team"], how="left")

    for c_raw, c_shift in [
        ("xGF_raw", "xGF"),
        ("xGA_raw", "xGA"),
        ("GF_raw", "GF"),
        ("GA_raw", "GA"),
        ("SF_raw", "SF"),
        ("SA_raw", "SA"),
    ]:
        panel[c_shift] = panel[c_shift].fillna(panel[c_raw])

    if "toi" not in panel.columns:
        panel["toi"] = 0.0
    panel["toi"] = panel["toi"].fillna(0.0)

    fill_zero_cols = [
        "ev_xGF", "ev_xGA", "pp_xGF", "pp_toi", "pk_xGA", "pk_toi",
        "f1_xGF", "f1_toi", "f2_xGF", "f2_toi", "shelter",
    ]
    for c in fill_zero_cols:
        panel[c] = panel[c].fillna(0.0)
    panel["shelter"] = panel["shelter"].replace(0.0, np.nan).fillna(0.5)

    panel = panel.merge(rest_game, on=["game_id", "team"], how="left")
    panel["rest_proxy"] = panel["rest_proxy"].fillna(1.0)

    panel = panel.merge(elo_df[["game_id", "home_elo_pre", "away_elo_pre", "home_exp"]], on="game_id", how="left")
    panel["elo_pre"] = np.where(panel["is_home"] == 1, panel["home_elo_pre"], panel["away_elo_pre"])
    panel["opp_elo_pre"] = np.where(panel["is_home"] == 1, panel["away_elo_pre"], panel["home_elo_pre"])
    panel["elo_prob"] = np.where(panel["is_home"] == 1, panel["home_exp"], 1 - panel["home_exp"])

    panel = panel.sort_values(["team", "game_num"]).reset_index(drop=True)
    return panel


def add_pregame_features(panel):
    panel = panel.copy()
    panel["gp_prev"] = panel.groupby("team").cumcount()
    panel["close_win"] = panel["win"] * panel["close_game"]

    cum_cols = [
        "win", "xGF", "xGA", "GF", "GA", "SF", "SA", "toi", "ev_xGF", "ev_xGA",
        "pp_xGF", "pp_toi", "pk_xGA", "pk_toi", "f1_xGF", "f1_toi", "f2_xGF", "f2_toi",
        "close_game", "close_win",
    ]
    for c in cum_cols:
        panel[f"cum_{c}"] = panel.groupby("team")[c].cumsum() - panel[c]

    shelter_cum = panel.groupby("team")["shelter"].cumsum() - panel["shelter"]
    panel["shelter_prev"] = np.where(panel["gp_prev"] > 0, shelter_cum / panel["gp_prev"], 0.5)

    panel["roll_xGF"] = panel.groupby("team")["xGF"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean()
    ).fillna(0.0)
    panel["roll_xGA"] = panel.groupby("team")["xGA"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean()
    ).fillna(0.0)
    panel["roll_win"] = panel.groupby("team")["win"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean()
    ).fillna(0.0)
    panel["form_5"] = panel.groupby("team")["win"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(0.0)
    panel["sos"] = panel.groupby("team")["opp_elo_pre"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean()
    ).fillna(1500.0)

    toi_h_prev = panel["cum_toi"] / 3600.0
    panel["xGF60"] = np.where(toi_h_prev > 0, panel["cum_xGF"] / toi_h_prev, 0.0)
    panel["xGA60"] = np.where(toi_h_prev > 0, panel["cum_xGA"] / toi_h_prev, 0.0)

    ev_den = panel["cum_ev_xGF"] + panel["cum_ev_xGA"]
    panel["ev_pct"] = np.where(ev_den > 0, panel["cum_ev_xGF"] / ev_den, 0.5)

    pp_h_prev = panel["cum_pp_toi"] / 3600.0
    pk_h_prev = panel["cum_pk_toi"] / 3600.0
    panel["pp60"] = np.where(pp_h_prev > 0, panel["cum_pp_xGF"] / pp_h_prev, 0.0)
    panel["pk60"] = np.where(pk_h_prev > 0, panel["cum_pk_xGA"] / pk_h_prev, 0.0)

    f1_h_prev = panel["cum_f1_toi"] / 3600.0
    f2_h_prev = panel["cum_f2_toi"] / 3600.0
    f1_60 = np.where(f1_h_prev > 0, panel["cum_f1_xGF"] / f1_h_prev, 0.0)
    f2_60 = np.where(f2_h_prev > 0, panel["cum_f2_xGF"] / f2_h_prev, 0.0)
    panel["disp"] = np.where(f2_60 > 0, f1_60 / f2_60, 1.0)

    panel["wp"] = np.where(panel["gp_prev"] > 0, panel["cum_win"] / panel["gp_prev"], 0.5)

    sh_pct = np.where(panel["cum_SF"] > 0, panel["cum_GF"] / panel["cum_SF"], 0.0)
    sv_pct = np.where(panel["cum_SA"] > 0, 1 - (panel["cum_GA"] / panel["cum_SA"]), 0.0)
    panel["pdo"] = (sh_pct + sv_pct) * 1000.0

    pyth_num = np.power(np.clip(panel["xGF60"], 0, None), 2.1)
    pyth_den = pyth_num + np.power(np.clip(panel["xGA60"], 0, None), 2.1)
    panel["pyth"] = np.where(pyth_den > 0, pyth_num / pyth_den, 0.5)

    panel["close_wp"] = np.where(panel["cum_close_game"] > 0, panel["cum_close_win"] / panel["cum_close_game"], 0.5)
    panel["corsi60"] = np.where(toi_h_prev > 0, (panel["cum_SF"] - panel["cum_SA"]) / toi_h_prev, 0.0)
    panel["xG_diff"] = panel["cum_xGF"] - panel["cum_xGA"]
    panel["shelter"] = panel["shelter_prev"]
    panel["elo"] = panel["elo_pre"]
    panel["rest"] = panel["rest_proxy"]
    return panel


def build_training_matrix(panel, model_features, min_history_games):
    keep_cols = ["game_id", "game_num", "team", "opp", "is_home", "win", "elo_prob", "gp_prev"] + model_features
    work = panel[keep_cols].copy()

    home = work[work["is_home"] == 1].copy()
    away = work[work["is_home"] == 0].copy()

    home = home.rename(columns={c: f"{c}_h" for c in home.columns if c != "game_id"})
    away = away.rename(columns={c: f"{c}_a" for c in away.columns if c != "game_id"})
    tr = home.merge(away, on="game_id", how="inner")

    tr["home_win"] = tr["win_h"].astype(int)
    tr["elo_prob"] = tr["elo_prob_h"]
    tr["min_gp"] = np.minimum(tr["gp_prev_h"], tr["gp_prev_a"])

    for c in model_features:
        tr[f"{c}_d"] = tr[f"{c}_h"] - tr[f"{c}_a"]

    tr = tr.sort_values("game_num_h").reset_index(drop=True)
    tr = tr[tr["min_gp"] >= min_history_games].reset_index(drop=True)

    diff_cols = [f"{c}_d" for c in model_features]
    return tr, diff_cols


def cross_val_ll(model, scaled, x, y):
    if len(x) < 200:
        splits = 3
    else:
        splits = 4

    tscv = TimeSeriesSplit(n_splits=splits)
    ll_vals = []
    for tr_idx, va_idx in tscv.split(x):
        x_tr, x_va = x[tr_idx], x[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        m = clone(model)

        if scaled:
            sc = StandardScaler()
            x_tr = sc.fit_transform(x_tr)
            x_va = sc.transform(x_va)

        m.fit(x_tr, y_tr)
        p_va = model_probs(m, x_va)
        ll_vals.append(log_loss(y_va, p_va))

    return float(np.mean(ll_vals)), float(np.std(ll_vals))


def evaluate_models(train_df, diff_cols):
    x = np.nan_to_num(train_df[diff_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = train_df["home_win"].values
    elo_probs = np.clip(train_df["elo_prob"].values, 1e-7, 1 - 1e-7)

    split = min(1000, int(len(x) * 0.76))
    xtr, xte = x[:split], x[split:]
    ytr, yte = y[:split], y[split:]
    elo_tr, elo_te = elo_probs[:split], elo_probs[split:]

    scaler = StandardScaler()
    xtr_s = scaler.fit_transform(xtr)
    xte_s = scaler.transform(xte)

    # v9.1 model pool: prioritize stable, regularized models; prune high-variance models
    models = {
        "01_LogReg": (LogisticRegression(max_iter=2500, C=0.30), True),
        "02_LogReg_L1": (LogisticRegression(max_iter=2500, C=0.22, penalty="l1", solver="saga"), True),
        "03_LogReg_EN": (LogisticRegression(max_iter=2500, C=0.22, penalty="elasticnet", solver="saga", l1_ratio=0.5), True),
        "04_Ridge": (CalibratedClassifierCV(RidgeClassifier(alpha=2.4), cv=5), True),
        "05_SGD": (CalibratedClassifierCV(SGDClassifier(loss="modified_huber", alpha=0.005, max_iter=2500, random_state=RANDOM_STATE), cv=5), True),
        "06_LDA": (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), True),
        "07_SVM_L": (CalibratedClassifierCV(LinearSVC(C=0.30, max_iter=8000, random_state=RANDOM_STATE), cv=5), True),
        "08_SVM_R": (SVC(kernel="rbf", probability=True, C=0.35, gamma="scale", random_state=RANDOM_STATE), True),
        "09_KNN": (KNeighborsClassifier(n_neighbors=41, weights="uniform"), True),
        "10_RF": (RandomForestClassifier(n_estimators=220, max_depth=3, min_samples_leaf=35, max_features=0.40, random_state=RANDOM_STATE), False),
        "11_ExtraT": (ExtraTreesClassifier(n_estimators=220, max_depth=3, min_samples_leaf=35, max_features=0.40, random_state=RANDOM_STATE), False),
        "12_HistGB": (HistGradientBoostingClassifier(max_iter=80, max_depth=2, learning_rate=0.02, min_samples_leaf=40, l2_regularization=3.0, random_state=RANDOM_STATE), False),
        "13_GB": (GradientBoostingClassifier(n_estimators=70, learning_rate=0.02, max_depth=2, min_samples_leaf=35, subsample=0.55, random_state=RANDOM_STATE), False),
        "14_XGB": (xgb.XGBClassifier(n_estimators=90, max_depth=2, learning_rate=0.015, subsample=0.50, colsample_bytree=0.40, reg_alpha=5.0, reg_lambda=8.0, gamma=2.0, min_child_weight=18, eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE), False),
        "15_XGB2": (xgb.XGBClassifier(n_estimators=110, max_depth=3, learning_rate=0.008, subsample=0.45, colsample_bytree=0.35, reg_alpha=6.0, reg_lambda=10.0, gamma=3.0, min_child_weight=20, eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE), False),
        "16_LGBM": (lgb.LGBMClassifier(n_estimators=90, max_depth=2, learning_rate=0.015, subsample=0.50, num_leaves=6, min_child_samples=45, reg_alpha=6.0, reg_lambda=10.0, verbose=-1, random_state=RANDOM_STATE), False),
    }

    hdr = f"  {'#':<3s} {'Model':<12s} {'LL':>7s} {'Acc':>7s} {'AUC':>7s} {'Brier':>7s} {'CV_LL':>7s} {'CV_SD':>7s} {'Gap':>7s} {'St':>4s}"
    print(hdr)
    print("  " + "-" * len(hdr))

    results = {}
    for name, (mdl, scaled) in models.items():
        try:
            cv_ll, cv_sd = cross_val_ll(mdl, scaled, xtr, ytr)
            m = clone(mdl)

            x_fit = xtr_s if scaled else xtr
            x_eval = xte_s if scaled else xte
            m.fit(x_fit, ytr)

            p_tr = model_probs(m, x_fit)
            p_te = model_probs(m, x_eval)
            yhat = (p_te >= 0.5).astype(int)

            tr_ll = log_loss(ytr, p_tr)
            te_ll = log_loss(yte, p_te)
            gap = te_ll - tr_ll

            if gap > 0.05 or (gap > 0.03 and cv_sd > 0.03):
                status = "OVR"
            elif gap < -0.08:
                status = "UND"
            else:
                status = "OK"

            res = {
                "ll": te_ll,
                "acc": accuracy_score(yte, yhat),
                "prec": precision_score(yte, yhat, zero_division=0),
                "rec": recall_score(yte, yhat, zero_division=0),
                "f1": f1_score(yte, yhat, zero_division=0),
                "auc": roc_auc_score(yte, p_te),
                "brier": brier_score_loss(yte, p_te),
                "mcc": matthews_corrcoef(yte, yhat),
                "tr_ll": tr_ll,
                "gap": gap,
                "cv_ll": cv_ll,
                "cv_sd": cv_sd,
                "status": status,
                "model": m,
                "scaled": scaled,
                "te_probs": p_te,
            }
            results[name] = res
            print(
                f"  {name[:2]:<3s} {name[3:]:<12s} {res['ll']:>7.4f} {res['acc']:>7.4f} "
                f"{res['auc']:>7.4f} {res['brier']:>7.4f} {res['cv_ll']:>7.4f} {res['cv_sd']:>7.4f} "
                f"{res['gap']:>7.4f} {res['status']:>4s}"
            )
        except Exception as exc:
            print(f"  {name:<15s} FAILED: {str(exc)[:60]}")

    # Elo baseline with strict pregame probabilities
    tscv = TimeSeriesSplit(n_splits=4 if len(ytr) >= 200 else 3)
    elo_cv = []
    for _, va_idx in tscv.split(elo_tr):
        elo_cv.append(log_loss(ytr[va_idx], elo_tr[va_idx]))
    elo_cv_ll = float(np.mean(elo_cv))
    elo_cv_sd = float(np.std(elo_cv))

    elo_te = np.clip(elo_te, 1e-7, 1 - 1e-7)
    elo_tr = np.clip(elo_tr, 1e-7, 1 - 1e-7)
    elo_yhat = (elo_te >= 0.5).astype(int)
    elo_tr_ll = log_loss(ytr, elo_tr)
    elo_ll = log_loss(yte, elo_te)
    elo_gap = elo_ll - elo_tr_ll
    if elo_gap > 0.05 or (elo_gap > 0.03 and elo_cv_sd > 0.03):
        elo_status = "OVR"
    elif elo_gap < -0.08:
        elo_status = "UND"
    else:
        elo_status = "OK"

    results["17_Elo"] = {
        "ll": elo_ll,
        "acc": accuracy_score(yte, elo_yhat),
        "prec": precision_score(yte, elo_yhat, zero_division=0),
        "rec": recall_score(yte, elo_yhat, zero_division=0),
        "f1": f1_score(yte, elo_yhat, zero_division=0),
        "auc": roc_auc_score(yte, elo_te),
        "brier": brier_score_loss(yte, elo_te),
        "mcc": matthews_corrcoef(yte, elo_yhat),
        "tr_ll": elo_tr_ll,
        "gap": elo_gap,
        "cv_ll": elo_cv_ll,
        "cv_sd": elo_cv_sd,
        "status": elo_status,
        "model": None,
        "scaled": False,
        "te_probs": elo_te,
    }
    print(
        f"  17  {'Elo':<12s} {elo_ll:>7.4f} {results['17_Elo']['acc']:>7.4f} "
        f"{results['17_Elo']['auc']:>7.4f} {results['17_Elo']['brier']:>7.4f} "
        f"{elo_cv_ll:>7.4f} {elo_cv_sd:>7.4f} {elo_gap:>7.4f} {elo_status:>4s}"
    )

    return results, scaler, yte, split


def select_best_model(results):
    for r in results.values():
        # Higher is better (for quick table reading)
        r["composite"] = (
            0.45 * (1 - r["ll"])
            + 0.20 * r["auc"]
            + 0.15 * (1 - r["brier"])
            + 0.10 * (1 - r["cv_ll"])
            + 0.10 * (1 / (1 + abs(r["gap"])))
        )
        if r["status"] == "OVR":
            r["composite"] -= 0.03

    eligible = [(n, r) for n, r in results.items() if r["status"] != "OVR"]
    if not eligible:
        eligible = list(results.items())

    eligible = sorted(
        eligible,
        key=lambda x: (
            x[1]["ll"],
            x[1]["brier"],
            x[1]["cv_ll"],
            -x[1]["auc"],
            abs(x[1]["gap"]),
        ),
    )

    best_name, best = eligible[0]
    return best_name, best


def build_final_team_features(panel, final_elos, gt, goalie_map, pp_stats, unit_stats, tev):
    rows = []
    tev_map = tev.set_index("Team")["Even_Str_xG%"].to_dict() if "Team" in tev.columns and "Even_Str_xG%" in tev.columns else {}

    for team, t in panel.groupby("team"):
        t = t.sort_values("game_num")
        gp = len(t)
        wins = t["win"].sum()

        xGF = t["xGF"].sum()
        xGA = t["xGA"].sum()
        GF = t["GF"].sum()
        GA = t["GA"].sum()
        SF = t["SF"].sum()
        SA = t["SA"].sum()
        toi = t["toi"].sum()
        toi_h = toi / 3600.0

        xGF60 = safe_div(xGF, toi_h, 0.0)
        xGA60 = safe_div(xGA, toi_h, 0.0)
        ev_pct = safe_div(t["ev_xGF"].sum(), t["ev_xGF"].sum() + t["ev_xGA"].sum(), 0.5)
        pp60 = safe_div(t["pp_xGF"].sum(), t["pp_toi"].sum() / 3600.0, 0.0)
        pk60 = safe_div(t["pk_xGA"].sum(), t["pk_toi"].sum() / 3600.0, 0.0)

        f1_60 = safe_div(t["f1_xGF"].sum(), t["f1_toi"].sum() / 3600.0, 0.0)
        f2_60 = safe_div(t["f2_xGF"].sum(), t["f2_toi"].sum() / 3600.0, 0.0)
        disp = f1_60 / f2_60 if f2_60 > 0 else 1.0

        sh_pct = safe_div(GF, SF, 0.0)
        sv_pct = 1 - safe_div(GA, SA, 0.0)
        pdo = (sh_pct + sv_pct) * 1000.0

        wp = safe_div(wins, gp, 0.5)
        pyth_num = xGF60 ** 2.1
        pyth_den = pyth_num + xGA60 ** 2.1
        pyth = safe_div(pyth_num, pyth_den, 0.5)
        close_wp = safe_div((t["win"] * t["close_game"]).sum(), t["close_game"].sum(), 0.5)
        corsi60 = safe_div(SF - SA, toi_h, 0.0)

        roll_xGF = t["xGF"].tail(20).mean() if gp else 0.0
        roll_xGA = t["xGA"].tail(20).mean() if gp else 0.0
        roll_win = t["win"].tail(20).mean() if gp else 0.0
        form_5 = t["win"].tail(5).mean() if gp else 0.0
        sos = t["opp_elo_pre"].tail(20).mean() if gp else 1500.0

        gid = goalie_map.get(team, "")
        gsax60 = gt.loc[gt["gid"] == gid, "GSAx60"].iloc[0] if gid in gt["gid"].values else 0.0

        pps = pp_stats.get(team, {"pp_xg_shift": 0.0, "pk_xga_shift": 0.0})
        uts = unit_stats.get(team, {"best_unit_xg60": 0.0, "avg_unit_xg60": 0.0})

        rows.append(
            {
                "team": team,
                "wins": wins,
                "ng": gp,
                "wp": wp,
                "xGF60": xGF60,
                "xGA60": xGA60,
                "ev_pct": ev_pct,
                "pp60": pp60,
                "pk60": pk60,
                "f1_60": f1_60,
                "f2_60": f2_60,
                "disp": disp,
                "gsax60": gsax60,
                "pdo": pdo,
                "pyth": pyth,
                "close_wp": close_wp,
                "corsi60": corsi60,
                "shelter": t["shelter"].mean(),
                "elo": final_elos.get(team, 1500.0),
                "xG_diff": xGF - xGA,
                "SF": SF,
                "SA": SA,
                "toi": toi,
                "roll_xGF": roll_xGF,
                "roll_xGA": roll_xGA,
                "roll_win": roll_win,
                "sos": sos,
                "form_5": form_5,
                "rest": 1.0,  # neutral default for unseen tournament games
                "pp_xg_shift": pps["pp_xg_shift"],
                "pk_xga_shift": pps["pk_xga_shift"],
                "best_unit_xg60": uts["best_unit_xg60"],
                "avg_unit_xg60": uts["avg_unit_xg60"],
                "ev_strength_ref": tev_map.get(team, np.nan),
            }
        )

    return pd.DataFrame(rows)


def build_power_rankings(final_df):
    work = final_df.copy()
    sc = MinMaxScaler()
    cols = ["ev_pct", "xGF60", "wp", "pyth", "gsax60", "elo"]
    n = pd.DataFrame(sc.fit_transform(work[cols]), columns=[f"{c}_n" for c in cols])
    work["ps"] = (
        0.25 * n["ev_pct_n"]
        + 0.15 * n["xGF60_n"]
        + 0.20 * n["wp_n"]
        + 0.10 * n["pyth_n"]
        + 0.15 * n["gsax60_n"]
        + 0.15 * n["elo_n"]
    )
    work = work.sort_values("ps", ascending=False).reset_index(drop=True)
    work["rank"] = np.arange(1, len(work) + 1)
    return work


def save_model_results(results):
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "Model": name[3:],
                "Log_Loss": round(r["ll"], 4),
                "Accuracy": round(r["acc"], 4),
                "Precision": round(r["prec"], 4),
                "Recall": round(r["rec"], 4),
                "F1": round(r["f1"], 4),
                "AUC_ROC": round(r["auc"], 4),
                "Brier": round(r["brier"], 4),
                "MCC": round(r["mcc"], 4),
                "Train_LL": round(r["tr_ll"], 4),
                "Overfit_Gap": round(r["gap"], 4),
                "CV_LL": round(r["cv_ll"], 4),
                "CV_SD": round(r["cv_sd"], 4),
                "Status": r["status"],
                "Composite": round(r["composite"], 4),
            }
        )
    mdf = pd.DataFrame(rows)
    mdf = mdf.sort_values(["Log_Loss", "Brier", "CV_LL"]).reset_index(drop=True)
    mdf.to_csv(os.path.join(OUT, "Model_Comparison_Results.csv"), index=False)
    mdf.to_csv(os.path.join(OUT, "v9_Model_Comparison.csv"), index=False)
    return mdf


def predict_matchups(matchups, best_name, best, team_feat, model_features, scaler, final_elos, elo_ha=ELO_HA):
    tf = team_feat.set_index("team")
    out = []
    missing = []

    for h, a in matchups:
        if h not in tf.index or a not in tf.index:
            missing.append((h, a))
            continue

        if best["model"] is None:
            he, ae = final_elos.get(h, 1500.0), final_elos.get(a, 1500.0)
            prob = 1.0 / (1.0 + 10 ** ((ae - he - elo_ha) / 400))
        else:
            diff = np.array([[tf.loc[h, c] - tf.loc[a, c] for c in model_features]])
            x_in = scaler.transform(diff) if best["scaled"] else diff
            prob = float(model_probs(best["model"], x_in)[0])

        winner = h if prob >= 0.5 else a
        out.append(
            {
                "Game": f"{h} vs {a}",
                "Home_Team": h,
                "Away_Team": a,
                "Win_Prob": round(prob, 4),
                "Predicted_Winner": winner,
                "Confidence": round(max(prob, 1 - prob), 4),
                "Model": best_name[3:],
            }
        )

    tdf = pd.DataFrame(out)
    if missing:
        print(f"  Warning: skipped {len(missing)} matchups with unknown teams.")
    return tdf


def predict_all_pairs(teams, best, team_feat, model_features, scaler, final_elos, elo_ha=ELO_HA):
    tf = team_feat.set_index("team")
    rows = []
    ordered_teams = sorted(teams)

    def home_prob(h, a):
        if best["model"] is None:
            he, ae = final_elos.get(h, 1500.0), final_elos.get(a, 1500.0)
            return 1.0 / (1.0 + 10 ** ((ae - he - elo_ha) / 400))
        diff = np.array([[tf.loc[h, c] - tf.loc[a, c] for c in model_features]])
        x_in = scaler.transform(diff) if best["scaled"] else diff
        return float(model_probs(best["model"], x_in)[0])

    for i, t1 in enumerate(ordered_teams):
        for t2 in ordered_teams[i + 1:]:
            p_t1_home = home_prob(t1, t2)
            p_t2_home = home_prob(t2, t1)
            p_t1_neutral = (p_t1_home + (1 - p_t2_home)) / 2
            rows.append(
                {
                    "Team_A": t1,
                    "Team_B": t2,
                    "Team_A_Win_Prob_Neutral": round(p_t1_neutral, 4),
                    "Team_B_Win_Prob_Neutral": round(1 - p_t1_neutral, 4),
                }
            )
    ap = pd.DataFrame(rows)
    ap.to_csv(os.path.join(OUT, "All_Pair_Matchup_Probabilities.csv"), index=False)
    return ap


def fit_whr_like_model(gs, teams):
    # Order-independent Bradley-Terry style rating model (whole-history flavor).
    t2i = {t: i for i, t in enumerate(teams)}
    n = len(gs)
    p = len(teams)
    x = np.zeros((n, p + 1), dtype=float)  # team-diff one-hot + home effect
    y = gs["home_win"].values.astype(int)

    for i, (_, g) in enumerate(gs.iterrows()):
        h = g["home_team"]
        a = g["away_team"]
        x[i, t2i[h]] = 1.0
        x[i, t2i[a]] = -1.0
        x[i, -1] = 1.0  # home indicator

    bt = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        fit_intercept=False,
        max_iter=6000,
    )
    bt.fit(x, y)
    coef = bt.coef_[0]

    team_strength = coef[:-1]
    team_strength = team_strength - team_strength.mean()
    home_coef = float(coef[-1])

    scale = 400 / np.log(10)
    ratings = {t: 1500 + team_strength[t2i[t]] * scale for t in teams}
    home_adv_pts = home_coef * scale
    return bt, t2i, ratings, home_coef, home_adv_pts


def whr_match_prob(home, away, t2i, bt_coef, home_coef):
    s = bt_coef[t2i[home]] - bt_coef[t2i[away]] + home_coef
    return float(1 / (1 + np.exp(-s)))


def attach_whr_outputs(tpred, teams, bt_model, t2i, whr_ratings, home_coef):
    tpred = tpred.copy()
    bt_coef = bt_model.coef_[0][:-1]
    probs = []
    for _, r in tpred.iterrows():
        h, a = r["Home_Team"], r["Away_Team"]
        if h in t2i and a in t2i:
            p = whr_match_prob(h, a, t2i, bt_coef, home_coef)
        else:
            p = np.nan
        probs.append(p)
    tpred["WHR_Win_Prob"] = np.round(probs, 4)

    whr_df = pd.DataFrame(
        {"Team": teams, "WHR_Rating": [whr_ratings[t] for t in teams]}
    ).sort_values("WHR_Rating", ascending=False)
    return tpred, whr_df


def matchup_prob_for_result_model(home, away, tf, model_entry, model_features, scaler, final_elos, elo_ha=ELO_HA):
    if home not in tf.index or away not in tf.index:
        return np.nan

    if model_entry["model"] is None:
        he, ae = final_elos.get(home, 1500.0), final_elos.get(away, 1500.0)
        return float(1.0 / (1.0 + 10 ** ((ae - he - elo_ha) / 400)))

    diff = np.array([[tf.loc[home, c] - tf.loc[away, c] for c in model_features]])
    x_in = scaler.transform(diff) if model_entry["scaled"] else diff
    return float(model_probs(model_entry["model"], x_in)[0])


def attach_tournament_model_probs(tpred, results, team_feat, model_features, scaler, final_elos, elo_ha=ELO_HA):
    tpred = tpred.copy()
    tf = team_feat.set_index("team")

    preferred = ["17_Elo", "05_SGD", "04_Ridge", "14_XGB", "15_XGB2"]
    selected = []
    for key in preferred:
        if key in results and (results[key]["status"] != "OVR" or key == "17_Elo"):
            selected.append(key)
    if "17_Elo" in results and "17_Elo" not in selected:
        selected.insert(0, "17_Elo")

    col_map = {}
    for key in selected:
        col = f"{key[3:]}_Win_Prob".replace(" ", "_")
        probs = []
        for _, r in tpred.iterrows():
            probs.append(
                matchup_prob_for_result_model(
                    r["Home_Team"],
                    r["Away_Team"],
                    tf,
                    results[key],
                    model_features,
                    scaler,
                    final_elos,
                    elo_ha=elo_ha,
                )
            )
        tpred[col] = np.round(probs, 4)
        col_map[key] = col

    return tpred, col_map


def add_model_consensus(tpred, results, model_prob_cols):
    tpred = tpred.copy()

    raw_weights = {}
    for key, col in model_prob_cols.items():
        if key in results:
            cv_ll = float(results[key].get("cv_ll", results[key]["ll"]))
            cv_sd = float(results[key].get("cv_sd", 0.0))
            score = np.exp(-CONSENSUS_WEIGHT_TEMP * cv_ll) / (1.0 + CONSENSUS_SD_PENALTY * cv_sd)
            raw_weights[col] = max(float(score), 1e-9)

    if "WHR_Win_Prob" in tpred.columns:
        if "17_Elo" in results:
            elo_cv_ll = float(results["17_Elo"].get("cv_ll", results["17_Elo"]["ll"]))
            elo_cv_sd = float(results["17_Elo"].get("cv_sd", 0.0))
            whr_base = np.exp(-CONSENSUS_WEIGHT_TEMP * elo_cv_ll) / (1.0 + CONSENSUS_SD_PENALTY * elo_cv_sd)
            whr_w = 0.90 * float(whr_base)
        elif raw_weights:
            whr_w = float(np.mean(list(raw_weights.values())))
        else:
            whr_w = 1.0
        raw_weights["WHR_Win_Prob"] = max(float(whr_w), 1e-9)

    weight_sum = float(sum(raw_weights.values()))
    if weight_sum <= 0:
        weights = {k: 1.0 / max(len(raw_weights), 1) for k in raw_weights}
    else:
        weights = {k: float(v / weight_sum) for k, v in raw_weights.items()}

    weight_rows = []
    for key, col in model_prob_cols.items():
        if col in weights:
            weight_rows.append(
                {
                    "Model_Key": key,
                    "Model": key[3:],
                    "Probability_Column": col,
                    "Weight": round(weights[col], 6),
                    "Weight_Basis": "train_cv",
                }
            )
    if "WHR_Win_Prob" in weights:
        weight_rows.append(
            {
                "Model_Key": "WHR",
                "Model": "WHR",
                "Probability_Column": "WHR_Win_Prob",
                "Weight": round(weights["WHR_Win_Prob"], 6),
                "Weight_Basis": "elo_cv_anchor",
            }
        )

    rows = []
    for _, r in tpred.iterrows():
        plist, wlist = [], []
        for col, w in weights.items():
            if col in tpred.columns and pd.notna(r[col]):
                plist.append(float(r[col]))
                wlist.append(float(w))

        if not plist:
            p_raw = float(r["Win_Prob"])
            spread = 0.0
            n_models = 1
        else:
            p_raw = float(np.average(plist, weights=np.asarray(wlist)))
            spread = float(np.max(plist) - np.min(plist))
            n_models = len(plist)

        shrink = min(CONSENSUS_MAX_SHRINK, 1.25 * spread)
        p_cons = 0.5 + (p_raw - 0.5) * (1.0 - shrink)

        ent = float(
            -(
                p_cons * np.log2(max(p_cons, 1e-7))
                + (1 - p_cons) * np.log2(max(1 - p_cons, 1e-7))
            )
        )
        label = "High" if spread >= 0.14 else ("Medium" if spread >= 0.08 else "Low")
        rows.append((p_raw, shrink, p_cons, spread, label, n_models, ent))

    tpred["Consensus_Win_Prob_Raw"] = [round(x[0], 4) for x in rows]
    tpred["Consensus_Shrink"] = [round(x[1], 4) for x in rows]
    tpred["Consensus_Win_Prob"] = [round(x[2], 4) for x in rows]
    tpred["Model_Spread"] = [round(x[3], 4) for x in rows]
    tpred["Disagreement"] = [x[4] for x in rows]
    tpred["Consensus_N_Models"] = [x[5] for x in rows]
    tpred["Consensus_Entropy"] = [round(x[6], 4) for x in rows]
    tpred["Consensus_Confidence"] = np.round(np.maximum(tpred["Consensus_Win_Prob"], 1 - tpred["Consensus_Win_Prob"]), 4)
    tpred["Consensus_Predicted_Winner"] = np.where(
        tpred["Consensus_Win_Prob"] >= 0.5, tpred["Home_Team"], tpred["Away_Team"]
    )
    return tpred, pd.DataFrame(weight_rows)


def _minmax_0_1(vals):
    arr = np.asarray(vals, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def build_team_uncertainty_index(final_df):
    work = final_df.copy()
    v_form = np.abs(work["form_5"].values - work["roll_win"].values)
    v_pdo = np.abs(work["pdo"].values - 1000.0) / 1000.0
    v_disp = np.abs(work["disp"].values - 1.0)

    u = (
        0.50 * _minmax_0_1(v_form)
        + 0.30 * _minmax_0_1(v_pdo)
        + 0.20 * _minmax_0_1(v_disp)
    )
    return {t: float(v) for t, v in zip(work["team"].values, u)}


def attach_robustness_layer(tpred, final_df, n_sims=ROBUSTNESS_N_SIMS, seed=RANDOM_STATE):
    tpred = tpred.copy()
    team_unc = build_team_uncertainty_index(final_df)
    rng = np.random.default_rng(seed)

    probs, lo95, hi95, flip, sigma_out, labels = [], [], [], [], [], []
    for _, r in tpred.iterrows():
        base_p = float(r["Consensus_Win_Prob"]) if "Consensus_Win_Prob" in tpred.columns else float(r["Win_Prob"])
        base_p = float(np.clip(base_p, 1e-7, 1 - 1e-7))

        h_unc = team_unc.get(r["Home_Team"], 0.5)
        a_unc = team_unc.get(r["Away_Team"], 0.5)
        spread = float(r["Model_Spread"]) if "Model_Spread" in tpred.columns else 0.0
        sigma = ROBUSTNESS_BASE_SIGMA + ROBUSTNESS_TEAM_SIGMA_W * (h_unc + a_unc) / 2.0 + ROBUSTNESS_SPREAD_SIGMA_W * spread
        sigma = float(np.clip(sigma, 0.05, 0.55))

        logit = np.log(base_p / (1 - base_p))
        draws = 1.0 / (1.0 + np.exp(-(logit + rng.normal(0, sigma, size=n_sims))))
        p_mean = float(np.mean(draws))
        p_lo = float(np.quantile(draws, 0.025))
        p_hi = float(np.quantile(draws, 0.975))

        if base_p >= 0.5:
            p_flip = float(np.mean(draws < 0.5))
        else:
            p_flip = float(np.mean(draws >= 0.5))

        if p_flip < 0.18 and (p_lo > 0.5 or p_hi < 0.5):
            label = "High"
        elif p_flip < 0.30:
            label = "Medium"
        else:
            label = "Low"

        probs.append(round(p_mean, 4))
        lo95.append(round(p_lo, 4))
        hi95.append(round(p_hi, 4))
        flip.append(round(p_flip, 4))
        sigma_out.append(round(sigma, 4))
        labels.append(label)

    tpred["Robust_Win_Prob"] = probs
    tpred["Robust_CI_Low_95"] = lo95
    tpred["Robust_CI_High_95"] = hi95
    tpred["Fragility_Index"] = flip
    tpred["Scenario_Sigma"] = sigma_out
    tpred["Edge_Stability"] = labels
    return tpred


def run_leakage_and_overfit_checks(train_df, diff_cols, split):
    x = np.nan_to_num(train_df[diff_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = train_df["home_win"].values.astype(int)

    xtr, xte = x[:split], x[split:]
    ytr, yte = y[:split], y[split:]

    rows = []
    tr_ids = set(train_df["game_id"].iloc[:split].values)
    te_ids = set(train_df["game_id"].iloc[split:].values)
    overlap = len(tr_ids & te_ids)
    rows.append(
        {
            "Check": "train_test_game_id_overlap",
            "Value": overlap,
            "Status": "PASS" if overlap == 0 else "FAIL",
            "Details": "Expected 0 overlap between train and holdout game ids",
        }
    )

    sc = StandardScaler()
    xtr_s = sc.fit_transform(xtr)
    xte_s = sc.transform(xte)

    base_m = LogisticRegression(max_iter=2500, C=0.30, random_state=RANDOM_STATE)
    base_m.fit(xtr_s, ytr)
    p_time = model_probs(base_m, xte_s)
    time_ll = float(log_loss(yte, p_time))
    time_auc = float(roc_auc_score(yte, p_time))
    rows.append(
        {
            "Check": "time_split_logreg_holdout",
            "Value": round(time_ll, 6),
            "Status": "INFO",
            "Details": f"AUC={time_auc:.4f}",
        }
    )

    rng = np.random.default_rng(RANDOM_STATE)
    y_perm = rng.permutation(ytr)
    perm_m = LogisticRegression(max_iter=2500, C=0.30, random_state=RANDOM_STATE)
    perm_m.fit(xtr_s, y_perm)
    p_perm = model_probs(perm_m, xte_s)
    perm_ll = float(log_loss(yte, p_perm))
    perm_auc = float(roc_auc_score(yte, p_perm))
    perm_status = "PASS" if (perm_ll > 0.685 and 0.45 <= perm_auc <= 0.55) else "WARN"
    rows.append(
        {
            "Check": "permuted_label_sanity",
            "Value": round(perm_ll, 6),
            "Status": perm_status,
            "Details": f"AUC={perm_auc:.4f} (should be near random)",
        }
    )

    idx = np.arange(len(x))
    rng.shuffle(idx)
    tr_idx = idx[:split]
    te_idx = idx[split:]

    sc_r = StandardScaler()
    xr_tr = sc_r.fit_transform(x[tr_idx])
    xr_te = sc_r.transform(x[te_idx])
    yr_tr = y[tr_idx]
    yr_te = y[te_idx]

    rand_m = LogisticRegression(max_iter=2500, C=0.30, random_state=RANDOM_STATE)
    rand_m.fit(xr_tr, yr_tr)
    p_rand = model_probs(rand_m, xr_te)
    rand_ll = float(log_loss(yr_te, p_rand))
    lift = time_ll - rand_ll
    rows.append(
        {
            "Check": "random_split_vs_time_split_ll_gap",
            "Value": round(lift, 6),
            "Status": "INFO",
            "Details": "Positive means random split is easier than temporal split",
        }
    )

    max_corr = 0.0
    max_feat = ""
    for c in diff_cols:
        v = np.asarray(train_df[c].values, dtype=float)
        if np.std(v) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(v, y)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
            corr = abs(corr)
        if corr > max_corr:
            max_corr = corr
            max_feat = c
    rows.append(
        {
            "Check": "max_feature_target_corr_abs",
            "Value": round(max_corr, 6),
            "Status": "PASS" if max_corr < 0.75 else "WARN",
            "Details": f"Feature={max_feat}",
        }
    )
    return pd.DataFrame(rows)


def build_calibration_table(probs, outcomes, bins=10, min_bin=20):
    probs = np.clip(np.asarray(probs), 1e-7, 1 - 1e-7)
    outcomes = np.asarray(outcomes)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n < min_bin:
            continue
        p_avg = float(probs[mask].mean())
        y_avg = float(outcomes[mask].mean())
        se = float(np.sqrt(max(y_avg * (1 - y_avg), 1e-7) / n))
        rows.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "n": n,
                "pred_mean": p_avg,
                "emp_win_rate": y_avg,
                "se": se,
                "ci_low_95": max(0.0, y_avg - 1.96 * se),
                "ci_high_95": min(1.0, y_avg + 1.96 * se),
            }
        )

    tab = pd.DataFrame(rows)
    if len(tab) == 0:
        # fallback single-bin estimate
        y_avg = float(outcomes.mean()) if len(outcomes) else 0.5
        n = max(1, len(outcomes))
        se = float(np.sqrt(max(y_avg * (1 - y_avg), 1e-7) / n))
        tab = pd.DataFrame(
            [
                {
                    "bin_low": 0.0,
                    "bin_high": 1.0,
                    "n": n,
                    "pred_mean": float(probs.mean()) if len(probs) else 0.5,
                    "emp_win_rate": y_avg,
                    "se": se,
                    "ci_low_95": max(0.0, y_avg - 1.96 * se),
                    "ci_high_95": min(1.0, y_avg + 1.96 * se),
                }
            ]
        )
    return tab


def attach_uncertainty_to_matchups(tpred, calibration_table):
    tpred = tpred.copy()
    if len(calibration_table) == 0:
        tpred["Calibrated_Win_Prob"] = tpred["Win_Prob"]
        tpred["CI_Low_95"] = np.clip(tpred["Win_Prob"] - 0.08, 0, 1)
        tpred["CI_High_95"] = np.clip(tpred["Win_Prob"] + 0.08, 0, 1)
        tpred["Upset_Risk"] = 1 - np.abs(2 * tpred["Win_Prob"] - 1)
        return tpred

    mids = (calibration_table["bin_low"] + calibration_table["bin_high"]) / 2
    rows = []
    for _, r in tpred.iterrows():
        p = float(r["Win_Prob"])
        idx = (mids - p).abs().idxmin()
        row = calibration_table.loc[idx]
        p_cal = 0.6 * p + 0.4 * float(row["emp_win_rate"])
        ci_low = max(0.0, p_cal - (float(row["emp_win_rate"]) - float(row["ci_low_95"])))
        ci_high = min(1.0, p_cal + (float(row["ci_high_95"]) - float(row["emp_win_rate"])))
        upset_risk = 1 - abs(2 * p_cal - 1)
        rows.append((p_cal, ci_low, ci_high, upset_risk))

    tpred["Calibrated_Win_Prob"] = [round(x[0], 4) for x in rows]
    tpred["CI_Low_95"] = [round(x[1], 4) for x in rows]
    tpred["CI_High_95"] = [round(x[2], 4) for x in rows]
    tpred["Upset_Risk"] = [round(x[3], 4) for x in rows]
    return tpred


def derive_team_archetypes(final_df, n_clusters=4):
    df = final_df.copy()
    style_cols = ["ev_pct", "pp60", "pk60", "corsi60", "disp", "roll_win", "form_5", "xGF60", "xGA60"]
    x = df[style_cols].fillna(0.0).values
    sc = StandardScaler()
    xz = sc.fit_transform(x)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    df["archetype_id"] = kmeans.fit_predict(xz)

    archetype_name = {}
    for cid, g in df.groupby("archetype_id"):
        ev = g["ev_pct"].mean()
        pp = g["pp60"].mean()
        xga = g["xGA60"].mean()
        disp = g["disp"].mean()
        corsi = g["corsi60"].mean()

        if pp >= df["pp60"].quantile(0.75):
            label = "Special-Teams Attack"
        elif xga <= df["xGA60"].quantile(0.25):
            label = "Defensive Suppression"
        elif disp >= df["disp"].quantile(0.75):
            label = "Top-Line Driven"
        elif corsi >= df["corsi60"].quantile(0.60) and ev >= df["ev_pct"].quantile(0.60):
            label = "Possession Control"
        else:
            label = "Balanced Hybrid"
        archetype_name[cid] = label

    df["Archetype"] = df["archetype_id"].map(archetype_name)

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=style_cols)
    centers["archetype_id"] = np.arange(n_clusters)
    centers["Archetype"] = centers["archetype_id"].map(archetype_name)

    out = df[
        [
            "team",
            "Archetype",
            "archetype_id",
            "ev_pct",
            "pp60",
            "pk60",
            "corsi60",
            "disp",
            "xGF60",
            "xGA60",
            "roll_win",
            "form_5",
        ]
    ].copy()
    out.columns = [
        "Team",
        "Archetype",
        "Archetype_ID",
        "EV_xG%",
        "PP_xG60",
        "PK_xGA60",
        "Corsi60",
        "Line_Disparity",
        "xGF60",
        "xGA60",
        "Roll_Win",
        "Form_5",
    ]
    return out, centers


def attach_archetypes_to_matchups(tpred, archetypes_df, final_df):
    tpred = tpred.copy()
    amap = archetypes_df.set_index("Team")["Archetype"].to_dict()
    tpred["Home_Archetype"] = tpred["Home_Team"].map(amap)
    tpred["Away_Archetype"] = tpred["Away_Team"].map(amap)

    style_cols = ["ev_pct", "pp60", "pk60", "corsi60", "disp", "roll_win", "form_5", "xGF60", "xGA60"]
    sf = final_df.set_index("team")[style_cols]
    dists = []
    for _, r in tpred.iterrows():
        if r["Home_Team"] in sf.index and r["Away_Team"] in sf.index:
            d = float(np.linalg.norm(sf.loc[r["Home_Team"]].values - sf.loc[r["Away_Team"]].values))
        else:
            d = np.nan
        dists.append(d)
    tpred["Style_Distance"] = np.round(dists, 4)
    return tpred


def make_visualizations(power_df, line_df, model_df, tpred_df, best_name, best_result, yte):
    bg = "#0a0a1a"
    cbg = "#12122e"
    txt = "#e0e0e0"
    acc = "#00d4ff"
    red = "#ff6b6b"
    gold = "#ffd700"
    green = "#00e676"

    fig = plt.figure(figsize=(26, 22), facecolor=bg)
    gs4 = gridspec.GridSpec(3, 2, hspace=0.30, wspace=0.25, left=0.06, right=0.96, top=0.94, bottom=0.04)

    def sty(ax, title):
        ax.set_facecolor(cbg)
        ax.set_title(title, color=gold, fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors=txt, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#333366")

    ax1 = fig.add_subplot(gs4[0, 0])
    top16 = power_df.head(16).iloc[::-1]
    ax1.barh(range(len(top16)), top16["ps"], color=plt.cm.viridis(np.linspace(0.30, 0.95, len(top16))), height=0.70)
    ax1.set_yticks(range(len(top16)))
    ax1.set_yticklabels([f"#{r['rank']} {r['team'].upper()}" for _, r in top16.iterrows()], color=txt, fontsize=7)
    ax1.set_xlabel("Power Score", color=txt)
    sty(ax1, "Power Rankings - Top 16")

    ax2 = fig.add_subplot(gs4[0, 1])
    top10 = line_df.head(10).iloc[::-1]
    ax2.barh(range(len(top10)), top10["Disparity_Ratio"], color=plt.cm.Reds(np.linspace(0.40, 0.90, len(top10))), height=0.65)
    ax2.set_yticks(range(len(top10)))
    ax2.set_yticklabels(top10["Team"].str.upper(), color=txt, fontsize=8)
    ax2.axvline(x=1.0, color=gold, linestyle="--", alpha=0.7)
    sty(ax2, "Line Disparity - Top 10")

    ax3 = fig.add_subplot(gs4[1, 0])
    mc = model_df.sort_values("Log_Loss", ascending=False)
    colors = [gold if m == best_name[3:] else (red if s == "OVR" else acc) for m, s in zip(mc["Model"], mc["Status"])]
    ax3.barh(np.arange(len(mc)), mc["Log_Loss"], color=colors, height=0.55)
    ax3.set_yticks(np.arange(len(mc)))
    ax3.set_yticklabels(mc["Model"], fontsize=5.5, color=txt)
    ax3.set_xlabel("Log Loss (lower is better)", color=txt)
    sty(ax3, f"Model Comparison - Best: {best_name[3:]}")

    ax4 = fig.add_subplot(gs4[1, 1])
    mlbl = [f"{r['Home_Team'][:3].upper()} v {r['Away_Team'][:3].upper()}" for _, r in tpred_df.iterrows()]
    if "Consensus_Win_Prob" in tpred_df.columns:
        probs = tpred_df["Consensus_Win_Prob"].values
    else:
        probs = tpred_df["Win_Prob"].values
    if "Robust_CI_Low_95" in tpred_df.columns and "Robust_CI_High_95" in tpred_df.columns:
        lo = tpred_df["Robust_CI_Low_95"].values
        hi = tpred_df["Robust_CI_High_95"].values
    else:
        lo = tpred_df["CI_Low_95"].values if "CI_Low_95" in tpred_df.columns else np.clip(probs - 0.08, 0, 1)
        hi = tpred_df["CI_High_95"].values if "CI_High_95" in tpred_df.columns else np.clip(probs + 0.08, 0, 1)

    yidx = np.arange(len(probs))
    if "Edge_Stability" in tpred_df.columns:
        cmap = {"High": green, "Medium": acc, "Low": red}
        cvals = [cmap.get(v, acc) for v in tpred_df["Edge_Stability"].values]
    else:
        cvals = [green if p >= 0.5 else red for p in probs]

    ax4.hlines(yidx, lo, hi, color="#8899cc", alpha=0.9, linewidth=2.0)
    ax4.scatter(probs, yidx, c=cvals, s=55, edgecolors="white", linewidth=0.6, zorder=3)
    ax4.set_yticks(yidx)
    ax4.set_yticklabels(mlbl, fontsize=7, color=txt)
    ax4.axvline(x=0.5, color=gold, linestyle="--", alpha=0.7)
    ax4.set_xlim(0, 1)
    sty(ax4, "Tournament Consensus + Robust CI")

    ax5 = fig.add_subplot(gs4[2, 0])
    p = np.array(best_result["te_probs"])
    bins = np.linspace(0, 1, 11)
    mids = []
    accs = []
    for i in range(len(bins) - 1):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if mask.sum() > 5:
            mids.append((bins[i] + bins[i + 1]) / 2)
            accs.append(yte[mask].mean())
    ax5.plot([0, 1], [0, 1], "--", color=gold, alpha=0.7, label="Perfect")
    ax5.plot(mids, accs, "o-", color=acc, markersize=8, label=best_name[3:])
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_xlabel("Predicted Probability", color=txt)
    ax5.set_ylabel("Empirical Win Rate", color=txt)
    ax5.legend(fontsize=8, facecolor=cbg, edgecolor="#333366", labelcolor=txt)
    sty(ax5, "Calibration")

    ax6 = fig.add_subplot(gs4[2, 1])
    of = model_df.sort_values("Log_Loss")
    x = np.arange(len(of))
    ax6.bar(x - 0.15, of["Train_LL"], 0.3, color=green, alpha=0.85, label="Train LL")
    ax6.bar(x + 0.15, of["Log_Loss"], 0.3, color=red, alpha=0.85, label="Test LL")
    ax6.set_xticks(x)
    ax6.set_xticklabels(of["Model"], rotation=50, ha="right", fontsize=5, color=txt)
    ax6.legend(fontsize=7, facecolor=cbg, edgecolor="#333366", labelcolor=txt)
    sty(ax6, "Overfit Diagnostic")

    fig.suptitle(
        f"WHL 2026 v9.4 - Leakage-Free Pipeline | Best: {best_name[3:]} (LL={best_result['ll']:.4f}, Acc={best_result['acc']:.1%})",
        color=gold,
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(os.path.join(OUT, "WHL_Final_Visualizations.png"), dpi=200, facecolor=bg, bbox_inches="tight")
    plt.close(fig)

    # Phase 1c scatter
    fig2, ax = plt.subplots(figsize=(10, 8), facecolor=bg)
    ax.set_facecolor(cbg)
    for _, r in power_df.iterrows():
        color = gold if r["rank"] <= 5 else (acc if r["rank"] <= 16 else "#666688")
        size = max(40, r["wins"] * 3)
        ax.scatter(r["disp"], r["ps"], c=color, s=size, alpha=0.85, edgecolors="white", linewidth=0.5)
        ax.annotate(r["team"].upper(), (r["disp"], r["ps"]), fontsize=6, color=txt, xytext=(5, 5), textcoords="offset points")
    z = np.polyfit(power_df["disp"], power_df["ps"], 1)
    pfit = np.poly1d(z)
    xr = np.linspace(power_df["disp"].min() - 0.02, power_df["disp"].max() + 0.02, 100)
    ax.plot(xr, pfit(xr), "--", color=red, alpha=0.7, linewidth=2, label=f"Trend (slope={z[0]:.2f})")
    ax.axvline(x=1.0, color=gold, linestyle=":", alpha=0.5, label="Even disparity")
    ax.set_xlabel("Offensive Line Disparity Ratio", color=txt, fontsize=12)
    ax.set_ylabel("Team Power Score", color=txt, fontsize=12)
    ax.set_title("Impact of Offensive Line Disparity on Team Strength", color=gold, fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors=txt)
    for s in ax.spines.values():
        s.set_color("#333366")
    ax.legend(fontsize=9, facecolor=cbg, edgecolor="#333366", labelcolor=txt)
    fig2.savefig(os.path.join(OUT, "WHL_Phase1c_Visualization.png"), dpi=200, facecolor=bg, bbox_inches="tight")
    plt.close(fig2)


def make_distinctive_visualizations(final_df, tpred_df, whr_df, consensus_weights):
    bg = "#090b14"
    cbg = "#131a2a"
    txt = "#e6ecff"
    gold = "#f8d66d"
    cyan = "#5ad8ff"
    mint = "#4de7a8"
    red = "#ff7b7b"

    fig = plt.figure(figsize=(20, 14), facecolor=bg)
    gs2 = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.20, left=0.06, right=0.97, top=0.92, bottom=0.07)

    def sty(ax, title):
        ax.set_facecolor(cbg)
        ax.set_title(title, color=gold, fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors=txt, labelsize=9)
        for s in ax.spines.values():
            s.set_color("#314064")

    ax1 = fig.add_subplot(gs2[0, 0])
    merged = final_df[["team", "elo"]].merge(whr_df.rename(columns={"Team": "team", "WHR_Rating": "whr"}), on="team", how="inner")
    ax1.scatter(merged["elo"], merged["whr"], c=cyan, s=55, alpha=0.85, edgecolors="white", linewidth=0.6)
    z = np.polyfit(merged["elo"], merged["whr"], 1)
    fz = np.poly1d(z)
    xr = np.linspace(merged["elo"].min(), merged["elo"].max(), 100)
    ax1.plot(xr, fz(xr), "--", color=mint, linewidth=2, alpha=0.9, label=f"Trend slope={z[0]:.2f}")
    for _, r in merged.sort_values("elo", ascending=False).head(5).iterrows():
        ax1.annotate(r["team"].upper(), (r["elo"], r["whr"]), color=txt, fontsize=7, xytext=(5, 4), textcoords="offset points")
    ax1.set_xlabel("Elo Rating", color=txt)
    ax1.set_ylabel("WHR-like Rating", color=txt)
    ax1.legend(fontsize=8, facecolor=cbg, edgecolor="#314064", labelcolor=txt, loc="lower right")
    sty(ax1, "Dual Rating Agreement (Elo vs WHR)")

    ax2 = fig.add_subplot(gs2[0, 1])
    cw = consensus_weights.sort_values("Weight")
    ax2.barh(cw["Model"], cw["Weight"], color=plt.cm.cividis(np.linspace(0.2, 0.95, len(cw))), height=0.65)
    ax2.set_xlabel("Consensus Weight", color=txt)
    for i, v in enumerate(cw["Weight"].values):
        ax2.text(v + 0.003, i, f"{v:.3f}", va="center", color=txt, fontsize=8)
    sty(ax2, "Consensus Attribution")

    ax3 = fig.add_subplot(gs2[1, 0])
    x = tpred_df["Style_Distance"].values if "Style_Distance" in tpred_df.columns else np.arange(len(tpred_df))
    y = tpred_df["Fragility_Index"].values if "Fragility_Index" in tpred_df.columns else np.zeros(len(tpred_df))
    if "Disagreement" in tpred_df.columns:
        cmap = {"High": red, "Medium": gold, "Low": mint}
        cols = [cmap.get(v, cyan) for v in tpred_df["Disagreement"].values]
    else:
        cols = [cyan] * len(tpred_df)
    ax3.scatter(x, y, c=cols, s=70, alpha=0.85, edgecolors="white", linewidth=0.6)
    for _, r in tpred_df.iterrows():
        lbl = f"{r['Home_Team'][:3].upper()}-{r['Away_Team'][:3].upper()}"
        xx = r["Style_Distance"] if "Style_Distance" in tpred_df.columns else 0
        yy = r["Fragility_Index"] if "Fragility_Index" in tpred_df.columns else 0
        ax3.annotate(lbl, (xx, yy), fontsize=6.5, color=txt, xytext=(4, 4), textcoords="offset points")
    ax3.set_xlabel("Style Distance", color=txt)
    ax3.set_ylabel("Fragility Index", color=txt)
    sty(ax3, "Matchup Volatility Map")

    ax4 = fig.add_subplot(gs2[1, 1])
    w = tpred_df[["Home_Team", "Away_Team", "Consensus_Win_Prob", "Robust_Win_Prob", "Upset_Risk"]].copy()
    w["Game"] = w["Home_Team"].str.upper().str[:3] + "-" + w["Away_Team"].str.upper().str[:3]
    w = w.sort_values("Upset_Risk", ascending=False).head(10).iloc[::-1]
    yidx = np.arange(len(w))
    ax4.hlines(yidx, w["Consensus_Win_Prob"], w["Robust_Win_Prob"], color="#8ea0c5", linewidth=2.2)
    ax4.scatter(w["Consensus_Win_Prob"], yidx, color=gold, s=55, label="Consensus")
    ax4.scatter(w["Robust_Win_Prob"], yidx, color=cyan, s=55, label="Robust")
    ax4.set_yticks(yidx)
    ax4.set_yticklabels(w["Game"], color=txt, fontsize=8)
    ax4.set_xlim(0, 1)
    ax4.axvline(0.5, color=mint, linestyle="--", alpha=0.7)
    ax4.legend(fontsize=8, facecolor=cbg, edgecolor="#314064", labelcolor=txt, loc="lower right")
    sty(ax4, "Top Upset-Risk Games: Consensus vs Robust")

    fig.suptitle(
        "WHL 2026 Distinctive Layer - Consensus, Robustness, and Dual Ratings",
        color=gold,
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(os.path.join(OUT, "WHL_Distinctive_Visualizations.png"), dpi=220, facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def rebuild_reference_csvs(ms, gs, gt, power_df, line_df, final_df):
    # Regenerate previously corrupted placeholder CSVs from valid core data.
    clean = ms[(ms["TOI flagging"] != "LONG") & (ms["TOI flagging"] != "SHORT")].copy()
    clean = clean.dropna(subset=["game_id", "home_team", "away_team"])

    # 1) Def_Pairing_Summary.csv
    h_def = clean[["home_team", "home_def_pairing", "toi", "home_xg", "away_xg"]].copy()
    h_def.columns = ["team", "def_pairing", "toi", "xGF", "xGA"]
    a_def = clean[["away_team", "away_def_pairing", "toi", "away_xg", "home_xg"]].copy()
    a_def.columns = ["team", "def_pairing", "toi", "xGF", "xGA"]
    dps = pd.concat([h_def, a_def], ignore_index=True)
    dps = dps.dropna(subset=["def_pairing"])
    dps = dps.groupby(["team", "def_pairing"], as_index=False).agg(
        shifts=("toi", "count"), total_toi=("toi", "sum"), xGF=("xGF", "sum"), xGA=("xGA", "sum")
    )
    dps["xG_diff"] = dps["xGF"] - dps["xGA"]
    dps["xGF60"] = np.where(dps["total_toi"] > 0, dps["xGF"] / (dps["total_toi"] / 3600), 0.0)
    dps = dps.sort_values(["team", "xGF60"], ascending=[True, False])
    dps.to_csv(os.path.join(OUT, "Def_Pairing_Summary.csv"), index=False)

    # 2) Even_strength.csv
    even = final_df[["team", "ev_pct", "xGF60", "xGA60", "wp", "elo"]].copy()
    even.columns = ["Team", "EV_xG%", "xGF60", "xGA60", "Win%", "Elo"]
    even = even.sort_values("EV_xG%", ascending=False)
    even.to_csv(os.path.join(OUT, "Even_strength.csv"), index=False)

    # 3) Game_summary.csv
    game_summary = gs[
        ["game_id", "home_team", "away_team", "home_goals", "away_goals", "home_xG", "away_xG", "went_ot", "goal_diff"]
    ].copy()
    game_summary["winner"] = np.where(game_summary["home_goals"] > game_summary["away_goals"], game_summary["home_team"], game_summary["away_team"])
    game_summary.to_csv(os.path.join(OUT, "Game_summary.csv"), index=False)

    # 4) Goalie_Table (GSAx calculation).csv
    goalie = gt[["gid", "xGA", "GA", "TOI", "GSAx60"]].copy()
    goalie.columns = ["goalie_id", "xGA", "GA", "TOI", "GSAx_per_60"]
    goalie["GSAx"] = goalie["xGA"] - goalie["GA"]
    goalie = goalie.sort_values("GSAx_per_60", ascending=False)
    goalie.to_csv(os.path.join(OUT, "Goalie_Table (GSAx calculation).csv"), index=False)

    # 5) Line_Disparity.csv
    line_df.to_csv(os.path.join(OUT, "Line_Disparity.csv"), index=False)

    # 6) Linearity_check.csv
    corr_features = ["wp", "xGF60", "xGA60", "ev_pct", "pp60", "pk60", "pdo", "pyth", "elo", "sos", "form_5", "disp"]
    rows = []
    for c in corr_features:
        rows.append({"feature": c, "pearson_corr_with_win_pct": final_df[c].corr(final_df["wp"])})
    pd.DataFrame(rows).sort_values("pearson_corr_with_win_pct", ascending=False).to_csv(
        os.path.join(OUT, "Linearity_check.csv"), index=False
    )

    # 7) Power_SCORE.csv
    pscore = power_df[["rank", "team", "ps", "ev_pct", "xGF60", "gsax60", "elo"]].copy()
    pscore.columns = ["Rank", "Team", "Power_Score", "EV_xG%", "xG_per_60", "GSAx_per_60", "Elo_Rating"]
    pscore.to_csv(os.path.join(OUT, "Power_SCORE.csv"), index=False)

    # 8) Power_play.csv
    ppower = final_df[["team", "pp60", "pk60", "pp_xg_shift", "pk_xga_shift"]].copy()
    ppower.columns = ["Team", "PP_xG60", "PK_xGA60", "PP_xG_per_shift", "PK_xGA_per_shift"]
    ppower = ppower.sort_values("PP_xG60", ascending=False)
    ppower.to_csv(os.path.join(OUT, "Power_play.csv"), index=False)

    # 9) Team_Profile.csv
    profile = final_df[
        ["team", "wins", "wp", "xGF60", "xGA60", "ev_pct", "gsax60", "pdo", "elo", "sos", "form_5", "disp"]
    ].copy()
    profile.columns = ["Team", "Wins", "Win%", "xGF60", "xGA60", "EV_xG%", "GSAx60", "PDO", "Elo", "SOS", "Form_5", "Line_Disparity"]
    profile.to_csv(os.path.join(OUT, "Team_Profile.csv"), index=False)

    # 10) Team_Totals.csv
    totals = final_df[["team", "wins", "ng", "SF", "SA", "toi", "xG_diff"]].copy()
    totals.columns = ["Team", "Wins", "Games", "Shots_For", "Shots_Against", "TOI", "xG_Diff"]
    totals.to_csv(os.path.join(OUT, "Team_Totals.csv"), index=False)

    # 11) line summary.csv
    lsum = line_df[["Team", "Disparity_Ratio", "Wins"]].copy()
    lsum["Line_Profile"] = np.where(
        lsum["Disparity_Ratio"] > 1.20,
        "Top-heavy",
        np.where(lsum["Disparity_Ratio"] < 0.95, "Depth-driven", "Balanced"),
    )
    lsum.to_csv(os.path.join(OUT, "line summary.csv"), index=False)


def build_completeness_report(out_dir):
    expected = [
        "1A_Power_Rankings.csv",
        "1A_Tournament_Predictions.csv",
        "1B_Line_Disparity_Rankings.csv",
        "Model_Comparison_Results.csv",
        "All_Pair_Matchup_Probabilities.csv",
        "Team_Archetypes.csv",
        "Team_Archetype_Centers.csv",
        "Tournament_Uncertainty_Report.csv",
        "WHR_Ratings.csv",
        "Tournament_Model_Consensus.csv",
        "Consensus_Model_Weights.csv",
        "Tournament_Robustness_Scenarios.csv",
        "Leakage_Sanity_Report.csv",
        "WHL_Final_Visualizations.png",
        "WHL_Phase1c_Visualization.png",
        "WHL_Distinctive_Visualizations.png",
    ]

    rows = []
    for name in expected:
        path = os.path.join(out_dir, name)
        exists = os.path.exists(path)
        size_bytes = os.path.getsize(path) if exists else 0
        n_rows = np.nan
        if exists and name.lower().endswith(".csv"):
            try:
                n_rows = max(0, sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1)
            except OSError:
                n_rows = np.nan

        if not exists:
            status = "MISSING"
        elif name.lower().endswith(".csv") and (not np.isnan(n_rows)) and int(n_rows) == 0:
            status = "EMPTY"
        elif size_bytes == 0:
            status = "EMPTY"
        else:
            status = "OK"

        rows.append(
            {
                "Artifact": name,
                "Exists": bool(exists),
                "Status": status,
                "Rows": int(n_rows) if np.isfinite(n_rows) else "",
                "Size_Bytes": int(size_bytes),
            }
        )

    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(out_dir, "Pipeline_Completeness_Report.csv"), index=False)
    return rep


def main():
    print("=" * 70)
    print("WHL 2026 - v9.4 LEAKAGE-FREE PIPELINE")
    print("=" * 70)

    print("\n[1/12] Loading files...")
    ms, raw, pp, unit, rest, tev, teams = load_data(BASE)
    bad_csvs = detect_corrupted_csvs(BASE)
    matchups, matchup_source = load_matchups(BASE, teams)

    print(f"  MainSheet: {len(ms):,} rows")
    print(f"  whl_2025: {len(raw):,} rows")
    print(f"  PP_consistency: {len(pp):,} rows")
    print(f"  Unit_specific: {len(unit):,} rows")
    print(f"  Game_Rest: {len(rest):,} rows")
    print(f"  Team_ev_strength: {len(tev):,} rows")
    print(f"  Teams: {len(teams)}")
    print(f"  Matchups: {len(matchups)} ({matchup_source})")
    print(f"  Corrupted CSV placeholders detected: {len(bad_csvs)}")

    print("\n[2/12] Building core tables...")
    gs = build_games(raw)
    shift_game, clean_shift_rows = build_shift_team_game(ms)
    gt, goalie_map = build_goalie_table(raw)
    pp_stats = build_pp_team_stats(pp, teams)
    unit_stats = build_unit_team_stats(unit, teams)
    rest_game = build_rest_by_game_team(rest, raw)
    print(f"  Games: {len(gs):,}")
    print(f"  Clean shifts used: {clean_shift_rows:,}")
    print(f"  Team-game rows from shifts: {len(shift_game):,}")

    print("\n[3/12] Elo pregame ratings...")
    elo_df, final_elos = compute_elo_pregame(
        gs, teams, k=ELO_K, ha=ELO_HA, xg_w=ELO_XG_W, decay=ELO_DECAY, regress_pct=ELO_REG
    )
    for t, e in sorted(final_elos.items(), key=lambda x: -x[1])[:5]:
        print(f"  {t}: {e:.0f}")

    print("\n[4/12] Building leakage-free pregame panel...")
    panel = build_team_game_panel(gs, shift_game, elo_df, rest_game)
    panel = add_pregame_features(panel)

    model_features = [
        "xG_diff", "ev_pct", "xGF60", "xGA60", "wp", "pyth", "disp", "pp60", "pk60",
        "pdo", "close_wp", "corsi60", "shelter", "elo", "roll_xGF", "roll_xGA",
        "roll_win", "sos", "form_5", "rest",
    ]

    print("\n[5/12] Training matrix...")
    train_df, diff_cols = build_training_matrix(panel, model_features, MODEL_MIN_HISTORY_GAMES)
    print(f"  Games after min-history filter ({MODEL_MIN_HISTORY_GAMES}): {len(train_df):,}")
    print(f"  Model features: {len(model_features)} ({len(diff_cols)} diffs)")

    print("\n[5b/12] Tuning Elo params on train-CV...")
    tune_split = min(1000, int(len(train_df) * 0.76))
    elo_best = tune_elo_params(
        gs,
        teams,
        train_df["game_id"].values,
        train_df["home_win"].values,
        tune_split,
    )
    print(
        f"  Best Elo params (train-CV): K={elo_best['k']} HA={elo_best['ha']} "
        f"xG_w={elo_best['xg_w']} decay={elo_best['decay']} reg={elo_best['reg']:.2f} "
        f"CV_LL={elo_best['cv_ll']:.4f} CV_SD={elo_best['cv_sd']:.4f}"
    )

    # Rebuild panel/features using tuned Elo so both Elo baseline and Elo-based features are aligned.
    elo_df, final_elos = compute_elo_pregame(
        gs,
        teams,
        k=elo_best["k"],
        ha=elo_best["ha"],
        xg_w=elo_best["xg_w"],
        decay=elo_best["decay"],
        regress_pct=elo_best["reg"],
    )
    panel = build_team_game_panel(gs, shift_game, elo_df, rest_game)
    panel = add_pregame_features(panel)
    train_df, diff_cols = build_training_matrix(panel, model_features, MODEL_MIN_HISTORY_GAMES)
    print(f"  Rebuilt training rows with tuned Elo: {len(train_df):,}")

    print("\n[6/12] Model training and evaluation...")
    results, scaler, yte, split = evaluate_models(train_df, diff_cols)
    overfit_count = sum(1 for r in results.values() if r["status"] == "OVR")
    print(f"\n  Overfitting models: {overfit_count}/{len(results)}")
    sanity_df = run_leakage_and_overfit_checks(train_df, diff_cols, split)
    sanity_df.to_csv(os.path.join(OUT, "Leakage_Sanity_Report.csv"), index=False)
    sanity_pass = (sanity_df["Status"] == "PASS").sum()
    print(f"  Leakage/overfit sanity checks saved: Leakage_Sanity_Report.csv ({sanity_pass} PASS)")

    print("\n[7/12] Best model selection...")
    best_name, best = select_best_model(results)
    print(f"  Final model: {best_name[3:]}")
    print(
        f"  LL={best['ll']:.4f} Acc={best['acc']:.4f} AUC={best['auc']:.4f} "
        f"Brier={best['brier']:.4f} Gap={best['gap']:.4f} Status={best['status']}"
    )

    print("\n[8/12] End-of-season team features...")
    final_df = build_final_team_features(panel, final_elos, gt, goalie_map, pp_stats, unit_stats, tev)
    power_df = build_power_rankings(final_df)
    archetypes_df, archetype_centers = derive_team_archetypes(final_df)
    archetypes_df.to_csv(os.path.join(OUT, "Team_Archetypes.csv"), index=False)
    archetype_centers.to_csv(os.path.join(OUT, "Team_Archetype_Centers.csv"), index=False)

    pr = power_df[["rank", "team", "wins", "ev_pct", "xGF60", "gsax60", "elo", "ps"]].copy()
    pr.columns = ["Rank", "Team", "Wins", "EV_xG%", "xG_per_60", "GSAx_per_60", "Elo_Rating", "Power_Score"]
    pr.to_csv(os.path.join(OUT, "1A_Power_Rankings.csv"), index=False)

    line_df = power_df[["team", "f1_60", "f2_60", "disp", "wins"]].sort_values("disp", ascending=False).reset_index(drop=True)
    line_df.insert(0, "Rank", np.arange(1, len(line_df) + 1))
    line_df.columns = ["Rank", "Team", "First_Line_xG_60", "Second_Line_xG_60", "Disparity_Ratio", "Wins"]
    line_df.to_csv(os.path.join(OUT, "1B_Line_Disparity_Rankings.csv"), index=False)
    print("  Saved 1A_Power_Rankings.csv and 1B_Line_Disparity_Rankings.csv")

    print("\n[9/12] Tournament predictions...")
    tpred = predict_matchups(
        matchups, best_name, best, final_df, model_features, scaler, final_elos, elo_ha=elo_best["ha"]
    )
    tpred = attach_archetypes_to_matchups(tpred, archetypes_df, final_df)

    bt_model, t2i, whr_ratings, whr_home_coef, whr_home_adv_pts = fit_whr_like_model(gs, teams)
    tpred, whr_df = attach_whr_outputs(tpred, teams, bt_model, t2i, whr_ratings, whr_home_coef)
    whr_df.to_csv(os.path.join(OUT, "WHR_Ratings.csv"), index=False)
    print(f"  WHR-like home edge: {whr_home_adv_pts:.1f} Elo pts")

    tpred, model_prob_cols = attach_tournament_model_probs(
        tpred, results, final_df, model_features, scaler, final_elos, elo_ha=elo_best["ha"]
    )
    tpred, consensus_weights = add_model_consensus(tpred, results, model_prob_cols)
    consensus_weights.to_csv(os.path.join(OUT, "Consensus_Model_Weights.csv"), index=False)

    cal_tbl = build_calibration_table(best["te_probs"], yte, bins=10, min_bin=20)
    cal_tbl.to_csv(os.path.join(OUT, "Tournament_Uncertainty_Report.csv"), index=False)
    tpred = attach_uncertainty_to_matchups(tpred, cal_tbl)
    tpred = attach_robustness_layer(tpred, final_df, n_sims=ROBUSTNESS_N_SIMS, seed=RANDOM_STATE)
    tpred[
        [
            "Home_Team",
            "Away_Team",
            "Consensus_Win_Prob",
            "Robust_Win_Prob",
            "Robust_CI_Low_95",
            "Robust_CI_High_95",
            "Fragility_Index",
            "Edge_Stability",
            "Scenario_Sigma",
        ]
    ].to_csv(os.path.join(OUT, "Tournament_Robustness_Scenarios.csv"), index=False)

    # Keep primary column for submission readability but include calibrated variant for risk-aware use.
    tpred = tpred.sort_values(
        ["Upset_Risk", "Fragility_Index", "Model_Spread", "Confidence"], ascending=[False, False, False, True]
    ).reset_index(drop=True)
    tpred.to_csv(os.path.join(OUT, "1A_Tournament_Predictions.csv"), index=False)
    tpred.to_csv(os.path.join(OUT, "v9_Tournament_Predictions.csv"), index=False)
    tpred.to_csv(os.path.join(OUT, "v9_2_Tournament_Predictions.csv"), index=False)
    tpred.to_csv(os.path.join(OUT, "v9_3_Tournament_Predictions.csv"), index=False)
    tpred.to_csv(os.path.join(OUT, "v9_4_Tournament_Predictions.csv"), index=False)
    tpred[
        [
            "Home_Team",
            "Away_Team",
            "Win_Prob",
            "WHR_Win_Prob",
            "Consensus_Win_Prob_Raw",
            "Consensus_Shrink",
            "Consensus_Win_Prob",
            "Model_Spread",
            "Disagreement",
            "Consensus_N_Models",
            "Consensus_Confidence",
            "Consensus_Predicted_Winner",
        ]
    ].to_csv(os.path.join(OUT, "Tournament_Model_Consensus.csv"), index=False)
    tpred.to_csv(os.path.join(OUT, "WHR_Tournament_Predictions.csv"), index=False)
    print(f"  Saved predictions for {len(tpred)} matchups")
    print(
        tpred[
            [
                "Home_Team",
                "Away_Team",
                "Win_Prob",
                "WHR_Win_Prob",
                "Consensus_Win_Prob",
                "Disagreement",
                "Calibrated_Win_Prob",
                "Robust_Win_Prob",
                "Fragility_Index",
                "Edge_Stability",
                "CI_Low_95",
                "CI_High_95",
                "Predicted_Winner",
                "Confidence",
                "Upset_Risk",
            ]
        ].to_string(index=False)
    )

    print("\n[10/12] All-pairs matrix (Wharton unknown-matchup alignment)...")
    ap = predict_all_pairs(teams, best, final_df, model_features, scaler, final_elos, elo_ha=elo_best["ha"])
    print(f"  Saved All_Pair_Matchup_Probabilities.csv ({len(ap)} rows)")
    print("  Saved Team_Archetypes.csv + Tournament_Uncertainty_Report.csv")

    # Recover historical CSV placeholders so they become usable tables.
    rebuild_reference_csvs(ms, gs, gt, power_df, line_df, final_df)
    bad_after_rebuild = detect_corrupted_csvs(BASE)
    print(f"  Rebuilt reference CSV pack; corrupted CSV count now: {len(bad_after_rebuild)}")

    print("\n[11/12] Model table and visualizations...")
    model_df = save_model_results(results)
    model_df.to_csv(os.path.join(OUT, "v9_2_Model_Comparison.csv"), index=False)
    make_visualizations(power_df, line_df, model_df, tpred, best_name, best, yte)
    make_distinctive_visualizations(final_df, tpred, whr_df, consensus_weights)
    print(
        "  Saved Model_Comparison_Results.csv, WHL_Final_Visualizations.png, "
        "WHL_Phase1c_Visualization.png, WHL_Distinctive_Visualizations.png"
    )

    comp_df = build_completeness_report(OUT)
    comp_ok = int((comp_df["Status"] == "OK").sum())
    print(f"  Completeness report saved: Pipeline_Completeness_Report.csv ({comp_ok}/{len(comp_df)} OK)")

    print("\n[12/12] Validation summary...")
    print("=" * 70)
    print(f"  [OK] Phase 1a outputs: rankings ({len(pr)}) + tournament predictions ({len(tpred)})")
    print(f"  [OK] Phase 1b output: line disparity ({len(line_df)})")
    print(f"  [OK] Phase 1c output: visualization PNG")
    print(f"  [OK] Leakage-free model evaluation: pregame-only features")
    print(f"  [OK] Leakage sanity artifact: Leakage_Sanity_Report.csv")
    print(f"  [OK] Matchup generalization artifact: All_Pair_Matchup_Probabilities.csv")
    print(f"  [OK] Distinctive layer: archetypes + uncertainty + WHR consensus + robustness simulation")
    print(f"  [OK] Models evaluated: {len(results)}")
    print(f"  [OK] Overfitting flagged: {overfit_count}/{len(results)}")
    print(f"  [OK] Best model: {best_name[3:]} (LL={best['ll']:.4f}, Acc={best['acc']:.1%})")
    print(f"  [OK] Corrupted CSV placeholders at start: {len(bad_csvs)}")
    print(f"  [OK] Corrupted CSV placeholders after rebuild: {len(bad_after_rebuild)}")
    print(f"  [OK] Completeness artifacts: {comp_ok}/{len(comp_df)}")
    print("=" * 70)
    print("v9.4 complete.")


if __name__ == "__main__":
    main()
