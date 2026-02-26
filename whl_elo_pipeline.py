#!/usr/bin/env python3
"""
Elo-focused pipeline for WHL Phase 1.

Runs Elo tuning, evaluates Elo and blend variants, and exports Elo-centric
rankings, matchup probabilities, and diagnostic reports.
"""

import os

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join("/tmp", "mplconfig_whl_elo")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd

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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import whl_analysis as wa


BASE = os.path.dirname(os.path.abspath(__file__))
OUT = BASE

MODEL_FEATURES = wa.MODEL_FEATURES

BLEND_WEIGHT_GRID = np.linspace(0.00, 1.00, 1001)
THREE_WAY_ELO_GRID = np.linspace(0.60, 1.00, 201)
THREE_WAY_SPLIT_GRID = np.linspace(0.00, 1.00, 201)


def evaluate_elo_holdout(train_df):
    y = train_df["home_win"].values.astype(int)
    p = np.clip(train_df["elo_prob"].values.astype(float), 1e-7, 1 - 1e-7)

    split = min(1000, int(len(train_df) * 0.76))
    ytr, yte = y[:split], y[split:]
    ptr, pte = p[:split], p[split:]

    tscv = TimeSeriesSplit(n_splits=4 if len(ytr) >= 200 else 3)
    cv_vals = []
    for _, va_idx in tscv.split(ptr):
        cv_vals.append(log_loss(ytr[va_idx], ptr[va_idx]))
    cv_ll = float(np.mean(cv_vals))
    cv_sd = float(np.std(cv_vals))

    yhat = (pte >= 0.5).astype(int)
    tr_ll = float(log_loss(ytr, ptr))
    te_ll = float(log_loss(yte, pte))
    gap = te_ll - tr_ll
    if gap > 0.05 or (gap > 0.03 and cv_sd > 0.03):
        status = "OVR"
    elif gap < -0.08:
        status = "UND"
    else:
        status = "OK"

    metrics = {
        "Model": "Elo",
        "Log_Loss": round(te_ll, 6),
        "Accuracy": round(float(accuracy_score(yte, yhat)), 6),
        "Precision": round(float(precision_score(yte, yhat, zero_division=0)), 6),
        "Recall": round(float(recall_score(yte, yhat, zero_division=0)), 6),
        "F1": round(float(f1_score(yte, yhat, zero_division=0)), 6),
        "AUC_ROC": round(float(roc_auc_score(yte, pte)), 6),
        "Brier": round(float(brier_score_loss(yte, pte)), 6),
        "MCC": round(float(matthews_corrcoef(yte, yhat)), 6),
        "Train_LL": round(tr_ll, 6),
        "Overfit_Gap": round(gap, 6),
        "CV_LL": round(cv_ll, 6),
        "CV_SD": round(cv_sd, 6),
        "Status": status,
        "Train_Rows": int(len(ytr)),
        "Test_Rows": int(len(yte)),
    }
    return metrics, yte, pte, split


def metric_row(name, y_true, probs, thr=0.5):
    probs = np.clip(np.asarray(probs, dtype=float), 1e-7, 1 - 1e-7)
    yhat = (probs >= thr).astype(int)
    return {
        "Model": name,
        "Threshold": round(float(thr), 6),
        "Log_Loss": round(float(log_loss(y_true, probs)), 6),
        "Accuracy": round(float(accuracy_score(y_true, yhat)), 6),
        "Precision": round(float(precision_score(y_true, yhat, zero_division=0)), 6),
        "Recall": round(float(recall_score(y_true, yhat, zero_division=0)), 6),
        "F1": round(float(f1_score(y_true, yhat, zero_division=0)), 6),
        "AUC_ROC": round(float(roc_auc_score(y_true, probs)), 6),
        "Brier": round(float(brier_score_loss(y_true, probs)), 6),
        "MCC": round(float(matthews_corrcoef(y_true, yhat)), 6),
    }


def fit_sgd_blend_components(train_df, diff_cols, split):
    x = np.nan_to_num(train_df[diff_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = train_df["home_win"].values.astype(int)
    xtr, xte = x[:split], x[split:]
    ytr, yte = y[:split], y[split:]

    # Out-of-fold train predictions for leakage-safe blend tuning.
    oof = np.full(len(ytr), np.nan, dtype=float)
    tscv = TimeSeriesSplit(n_splits=4 if len(ytr) >= 200 else 3)
    for tri, vai in tscv.split(xtr):
        sc = StandardScaler()
        xtri = sc.fit_transform(xtr[tri])
        xvai = sc.transform(xtr[vai])
        base = SGDClassifier(loss="modified_huber", alpha=0.005, max_iter=2500, random_state=wa.RANDOM_STATE)
        mdl = CalibratedClassifierCV(base, cv=3)
        mdl.fit(xtri, ytr[tri])
        oof[vai] = wa.model_probs(mdl, xvai)

    # Final train-fitted SGD for holdout / tournament inference.
    sc_full = StandardScaler()
    xtr_s = sc_full.fit_transform(xtr)
    xte_s = sc_full.transform(xte)
    base_full = SGDClassifier(loss="modified_huber", alpha=0.005, max_iter=2500, random_state=wa.RANDOM_STATE)
    mdl_full = CalibratedClassifierCV(base_full, cv=5)
    mdl_full.fit(xtr_s, ytr)
    pte_sgd = wa.model_probs(mdl_full, xte_s)
    return oof, pte_sgd, mdl_full, sc_full, ytr, yte


def fit_logreg_blend_components(train_df, diff_cols, split):
    x = np.nan_to_num(train_df[diff_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = train_df["home_win"].values.astype(int)
    xtr, xte = x[:split], x[split:]
    ytr = y[:split]

    oof = np.full(len(ytr), np.nan, dtype=float)
    tscv = TimeSeriesSplit(n_splits=4 if len(ytr) >= 200 else 3)
    for tri, vai in tscv.split(xtr):
        sc = StandardScaler()
        xtri = sc.fit_transform(xtr[tri])
        xvai = sc.transform(xtr[vai])
        mdl = LogisticRegression(max_iter=5000, C=0.30)
        mdl.fit(xtri, ytr[tri])
        oof[vai] = wa.model_probs(mdl, xvai)

    sc_full = StandardScaler()
    xtr_s = sc_full.fit_transform(xtr)
    xte_s = sc_full.transform(xte)
    mdl_full = LogisticRegression(max_iter=5000, C=0.30)
    mdl_full.fit(xtr_s, ytr)
    pte_lr = wa.model_probs(mdl_full, xte_s)
    return oof, pte_lr, mdl_full, sc_full


def fit_full_blend_models(train_df, diff_cols):
    """
    Refit blend component models on all leakage-safe season games for final inference.
    Evaluation and weight tuning remain train/OOF-only.
    """
    x = np.nan_to_num(train_df[diff_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = train_df["home_win"].values.astype(int)

    sc = StandardScaler()
    x_s = sc.fit_transform(x)

    sgd_base = SGDClassifier(loss="modified_huber", alpha=0.005, max_iter=2500, random_state=wa.RANDOM_STATE)
    sgd_full = CalibratedClassifierCV(sgd_base, cv=5)
    sgd_full.fit(x_s, y)

    lr_full = LogisticRegression(max_iter=5000, C=0.30)
    lr_full.fit(x_s, y)
    return sgd_full, lr_full, sc


def tune_blend_weight_for_ll(ytr, p_elo_tr, p_sgd_oof):
    valid = np.isfinite(p_sgd_oof)
    if int(valid.sum()) == 0:
        return 1.0, float("inf")

    yv = ytr[valid]
    pev = p_elo_tr[valid]
    psv = p_sgd_oof[valid]

    best_w = 1.0
    best_ll = float("inf")
    for w in BLEND_WEIGHT_GRID:
        p = np.clip(w * pev + (1 - w) * psv, 1e-7, 1 - 1e-7)
        ll = float(log_loss(yv, p))
        if ll < best_ll:
            best_ll = ll
            best_w = float(w)
    return best_w, best_ll


def tune_blend_weight_for_acc(ytr, p_elo_tr, p_aux_oof):
    valid = np.isfinite(p_aux_oof)
    if int(valid.sum()) == 0:
        return 1.0, 0.0, float("inf")

    yv = ytr[valid]
    pev = p_elo_tr[valid]
    pav = p_aux_oof[valid]

    best_w = 1.0
    best_acc = -1.0
    best_ll = float("inf")
    for w in BLEND_WEIGHT_GRID:
        p = np.clip(w * pev + (1 - w) * pav, 1e-7, 1 - 1e-7)
        acc = float(accuracy_score(yv, (p >= 0.5).astype(int)))
        ll = float(log_loss(yv, p))
        if (acc > best_acc) or (acc == best_acc and ll < best_ll):
            best_acc = acc
            best_ll = ll
            best_w = float(w)
    return best_w, best_acc, best_ll


def tune_three_way_convex_for_ll(ytr, p_elo_tr, p_sgd_oof, p_lr_oof):
    valid = np.isfinite(p_sgd_oof) & np.isfinite(p_lr_oof)
    if int(valid.sum()) == 0:
        return 1.0, 0.0, 0.0, float("inf")

    yv = ytr[valid]
    pev = np.clip(p_elo_tr[valid], 1e-7, 1 - 1e-7)
    psv = np.clip(p_sgd_oof[valid], 1e-7, 1 - 1e-7)
    plv = np.clip(p_lr_oof[valid], 1e-7, 1 - 1e-7)

    best_we, best_ws, best_wl = 1.0, 0.0, 0.0
    best_ll = float("inf")

    for we in THREE_WAY_ELO_GRID:
        rem = 1.0 - we
        ws_vec = rem * THREE_WAY_SPLIT_GRID
        wl_vec = rem * (1.0 - THREE_WAY_SPLIT_GRID)

        p_mat = np.clip(
            we * pev[None, :] + ws_vec[:, None] * psv[None, :] + wl_vec[:, None] * plv[None, :],
            1e-7,
            1 - 1e-7,
        )
        ll_vec = -np.mean(yv[None, :] * np.log(p_mat) + (1 - yv)[None, :] * np.log(1 - p_mat), axis=1)
        j = int(np.argmin(ll_vec))
        if float(ll_vec[j]) < best_ll:
            best_ll = float(ll_vec[j])
            best_we = float(we)
            best_ws = float(ws_vec[j])
            best_wl = float(wl_vec[j])

    return best_we, best_ws, best_wl, best_ll


def predict_sgd_matchups(matchups, team_feat, model_features, scaler, model):
    tf = team_feat.set_index("team")
    probs = []
    for h, a in matchups:
        if h not in tf.index or a not in tf.index:
            probs.append(np.nan)
            continue
        diff = np.array([[tf.loc[h, c] - tf.loc[a, c] for c in model_features]])
        p = float(wa.model_probs(model, scaler.transform(diff))[0])
        probs.append(p)
    return np.asarray(probs, dtype=float)


def predict_logreg_matchups(matchups, team_feat, model_features, scaler, model):
    tf = team_feat.set_index("team")
    probs = []
    for h, a in matchups:
        if h not in tf.index or a not in tf.index:
            probs.append(np.nan)
            continue
        diff = np.array([[tf.loc[h, c] - tf.loc[a, c] for c in model_features]])
        p = float(wa.model_probs(model, scaler.transform(diff))[0])
        probs.append(p)
    return np.asarray(probs, dtype=float)


def build_elo_team_table(final_df, power_df):
    pmeta = power_df[["team", "rank", "ps"]].copy()
    pmeta.columns = ["team", "power_rank", "power_score"]

    work = final_df.merge(pmeta, on="team", how="left")
    work["elo_rank"] = work["elo"].rank(ascending=False, method="min").astype(int)
    work["xG_balance_60"] = work["xGF60"] - work["xGA60"]

    out = work[
        [
            "team",
            "elo_rank",
            "elo",
            "power_rank",
            "power_score",
            "wins",
            "wp",
            "xGF60",
            "xGA60",
            "xG_balance_60",
            "xG_diff",
            "ev_pct",
            "pp60",
            "pk60",
            "gsax60",
            "pyth",
            "pdo",
            "roll_win",
            "form_5",
            "sos",
            "disp",
        ]
    ].copy()
    out.columns = [
        "Team",
        "Elo_Rank",
        "Elo_Rating",
        "Power_Rank",
        "Power_Score",
        "Wins",
        "Win_Pct",
        "xGF60",
        "xGA60",
        "xG_Balance_60",
        "xG_Diff_Total",
        "EV_xG_Pct",
        "PP_xG60",
        "PK_xGA60",
        "GSAx60",
        "Pythagorean",
        "PDO",
        "Roll_Win_20",
        "Form_5",
        "SOS",
        "Line_Disparity",
    ]
    out = out.sort_values(["Elo_Rank", "Power_Rank"]).reset_index(drop=True)
    return out


def predict_all_pairs_elo(teams, final_elos, elo_ha):
    rows = []
    ordered = sorted(teams)
    for i, t1 in enumerate(ordered):
        for t2 in ordered[i + 1 :]:
            e1 = final_elos.get(t1, 1500.0)
            e2 = final_elos.get(t2, 1500.0)
            p_t1_home = 1.0 / (1.0 + 10 ** ((e2 - e1 - elo_ha) / 400))
            p_t2_home = 1.0 / (1.0 + 10 ** ((e1 - e2 - elo_ha) / 400))
            p_t1_neutral = (p_t1_home + (1 - p_t2_home)) / 2.0
            rows.append(
                {
                    "Team_A": t1,
                    "Team_B": t2,
                    "Team_A_Win_Prob_Neutral": round(float(p_t1_neutral), 4),
                    "Team_B_Win_Prob_Neutral": round(float(1 - p_t1_neutral), 4),
                }
            )
    return pd.DataFrame(rows)


def predict_all_pairs_blend(teams, team_feat, model_features, model_scaler, model_obj, final_elos, elo_ha, w_elo):
    tf = team_feat.set_index("team")
    rows = []
    ordered = sorted(teams)
    for i, t1 in enumerate(ordered):
        for t2 in ordered[i + 1 :]:
            e1 = final_elos.get(t1, 1500.0)
            e2 = final_elos.get(t2, 1500.0)
            p_elo_t1_home = 1.0 / (1.0 + 10 ** ((e2 - e1 - elo_ha) / 400))
            p_elo_t2_home = 1.0 / (1.0 + 10 ** ((e1 - e2 - elo_ha) / 400))

            d12 = np.array([[tf.loc[t1, c] - tf.loc[t2, c] for c in model_features]])
            d21 = np.array([[tf.loc[t2, c] - tf.loc[t1, c] for c in model_features]])
            p_sgd_t1_home = float(wa.model_probs(model_obj, model_scaler.transform(d12))[0])
            p_sgd_t2_home = float(wa.model_probs(model_obj, model_scaler.transform(d21))[0])

            p_bl_t1_home = w_elo * p_elo_t1_home + (1 - w_elo) * p_sgd_t1_home
            p_bl_t2_home = w_elo * p_elo_t2_home + (1 - w_elo) * p_sgd_t2_home
            p_t1_neutral = (p_bl_t1_home + (1 - p_bl_t2_home)) / 2.0
            rows.append(
                {
                    "Team_A": t1,
                    "Team_B": t2,
                    "Team_A_Win_Prob_Neutral": round(float(p_t1_neutral), 4),
                    "Team_B_Win_Prob_Neutral": round(float(1 - p_t1_neutral), 4),
                }
            )
    return pd.DataFrame(rows)


def predict_all_pairs_three_way_blend(
    teams,
    team_feat,
    model_features,
    sgd_scaler,
    sgd_model,
    lr_scaler,
    lr_model,
    final_elos,
    elo_ha,
    w_elo,
    w_sgd,
    w_lr,
):
    tf = team_feat.set_index("team")
    rows = []
    ordered = sorted(teams)
    for i, t1 in enumerate(ordered):
        for t2 in ordered[i + 1 :]:
            e1 = final_elos.get(t1, 1500.0)
            e2 = final_elos.get(t2, 1500.0)
            p_elo_t1_home = 1.0 / (1.0 + 10 ** ((e2 - e1 - elo_ha) / 400))
            p_elo_t2_home = 1.0 / (1.0 + 10 ** ((e1 - e2 - elo_ha) / 400))

            d12 = np.array([[tf.loc[t1, c] - tf.loc[t2, c] for c in model_features]])
            d21 = np.array([[tf.loc[t2, c] - tf.loc[t1, c] for c in model_features]])
            p_sgd_t1_home = float(wa.model_probs(sgd_model, sgd_scaler.transform(d12))[0])
            p_sgd_t2_home = float(wa.model_probs(sgd_model, sgd_scaler.transform(d21))[0])
            p_lr_t1_home = float(wa.model_probs(lr_model, lr_scaler.transform(d12))[0])
            p_lr_t2_home = float(wa.model_probs(lr_model, lr_scaler.transform(d21))[0])

            p_bl_t1_home = w_elo * p_elo_t1_home + w_sgd * p_sgd_t1_home + w_lr * p_lr_t1_home
            p_bl_t2_home = w_elo * p_elo_t2_home + w_sgd * p_sgd_t2_home + w_lr * p_lr_t2_home
            p_t1_neutral = (p_bl_t1_home + (1 - p_bl_t2_home)) / 2.0
            rows.append(
                {
                    "Team_A": t1,
                    "Team_B": t2,
                    "Team_A_Win_Prob_Neutral": round(float(p_t1_neutral), 4),
                    "Team_B_Win_Prob_Neutral": round(float(1 - p_t1_neutral), 4),
                }
            )
    return pd.DataFrame(rows)


def make_elo_visualizations(team_tbl, tpred, cal_tbl, metrics):
    bg = "#0d1222"
    cbg = "#141c33"
    txt = "#e8ecff"
    gold = "#f6d06f"
    cyan = "#5dd5ff"
    mint = "#53e3ab"
    red = "#ff7f7f"

    fig = plt.figure(figsize=(20, 14), facecolor=bg)
    gs = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.22, left=0.06, right=0.97, top=0.90, bottom=0.07)

    def sty(ax, title):
        ax.set_facecolor(cbg)
        ax.set_title(title, color=gold, fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors=txt, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#2c3f66")

    ax1 = fig.add_subplot(gs[0, 0])
    top = team_tbl.head(16).iloc[::-1]
    ax1.barh(np.arange(len(top)), top["Elo_Rating"], color=plt.cm.cividis(np.linspace(0.2, 0.95, len(top))), height=0.65)
    ax1.set_yticks(np.arange(len(top)))
    ax1.set_yticklabels(top["Team"].str.upper(), color=txt)
    ax1.set_xlabel("Elo Rating", color=txt)
    sty(ax1, "Top 16 Team Elo Ratings")

    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(
        team_tbl["Elo_Rating"],
        team_tbl["Power_Score"],
        c=team_tbl["xG_Balance_60"],
        cmap="coolwarm",
        s=80,
        edgecolors="white",
        linewidth=0.5,
        alpha=0.85,
    )
    z = np.polyfit(team_tbl["Elo_Rating"], team_tbl["Power_Score"], 1)
    fit = np.poly1d(z)
    xr = np.linspace(team_tbl["Elo_Rating"].min(), team_tbl["Elo_Rating"].max(), 120)
    ax2.plot(xr, fit(xr), "--", color=mint, linewidth=2, alpha=0.9)
    cb = plt.colorbar(sc, ax=ax2, shrink=0.85)
    cb.set_label("xG Balance / 60", color=txt)
    cb.ax.yaxis.set_tick_params(color=txt)
    plt.setp(cb.ax.get_yticklabels(), color=txt)
    ax2.set_xlabel("Elo Rating", color=txt)
    ax2.set_ylabel("Power Score", color=txt)
    sty(ax2, "Elo vs Power (colored by xG Balance)")

    ax3 = fig.add_subplot(gs[1, 0])
    view = tpred.sort_values("Upset_Risk", ascending=False).head(12).iloc[::-1]
    y = np.arange(len(view))
    ax3.hlines(y, view["Robust_CI_Low_95"], view["Robust_CI_High_95"], color="#8da3d0", linewidth=2.0, alpha=0.9)
    ax3.scatter(view["Win_Prob"], y, color=gold, s=50, label="Elo")
    ax3.scatter(view["Robust_Win_Prob"], y, color=cyan, s=50, label="Robust")
    ax3.axvline(0.5, color=mint, linestyle="--", alpha=0.75)
    ax3.set_xlim(0, 1)
    ax3.set_yticks(y)
    ax3.set_yticklabels(
        [f"{r['Home_Team'][:3].upper()}-{r['Away_Team'][:3].upper()}" for _, r in view.iterrows()],
        color=txt,
        fontsize=7,
    )
    ax3.legend(fontsize=8, facecolor=cbg, edgecolor="#2c3f66", labelcolor=txt, loc="lower right")
    sty(ax3, "Top Upset-Risk Matchups (Elo + Robust Interval)")

    ax4 = fig.add_subplot(gs[1, 1])
    if len(cal_tbl):
        ax4.plot([0, 1], [0, 1], "--", color=gold, alpha=0.8, label="Perfect")
        ax4.plot(cal_tbl["pred_mean"], cal_tbl["emp_win_rate"], "o-", color=cyan, markersize=7, label="Elo")
        ax4.fill_between(cal_tbl["pred_mean"], cal_tbl["ci_low_95"], cal_tbl["ci_high_95"], color=cyan, alpha=0.2)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel("Predicted Probability", color=txt)
    ax4.set_ylabel("Empirical Win Rate", color=txt)
    ax4.legend(fontsize=8, facecolor=cbg, edgecolor="#2c3f66", labelcolor=txt, loc="lower right")
    sty(ax4, "Elo Calibration (Holdout)")

    fig.suptitle(
        f"WHL 2026 Elo-Only Pipeline | LL={metrics['Log_Loss']:.4f} Acc={metrics['Accuracy']:.1%} AUC={metrics['AUC_ROC']:.4f}",
        color=gold,
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(os.path.join(OUT, "ELO_Pipeline_Visualizations.png"), dpi=220, facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def build_elo_completeness_report():
    expected = [
        "ELO_Parameter_Selection.csv",
        "ELO_Model_Performance.csv",
        "ELO_Blend_Tuning.csv",
        "ELO_Blend_Performance.csv",
        "ELO_Model_Selection_Policy.csv",
        "ELO_Leakage_Sanity_Report.csv",
        "ELO_Power_Rankings.csv",
        "ELO_Team_Elo_xG_Power.csv",
        "ELO_Team_Profile_Full.csv",
        "ELO_Archetypes.csv",
        "ELO_Archetype_Centers.csv",
        "ELO_Tournament_Uncertainty.csv",
        "ELO_Tournament_Calibration_Selection.csv",
        "ELO_Tournament_Predictions.csv",
        "ELO_Tournament_Uncertainty_LLBlend.csv",
        "ELO_Tournament_Calibration_Selection_LLBlend.csv",
        "ELO_Tournament_Predictions_LLBlend.csv",
        "ELO_Tournament_Uncertainty_AccBlend.csv",
        "ELO_Tournament_Calibration_Selection_AccBlend.csv",
        "ELO_Tournament_Predictions_AccBlend.csv",
        "ELO_Tournament_Uncertainty_3WayBlend.csv",
        "ELO_Tournament_Calibration_Selection_3WayBlend.csv",
        "ELO_Tournament_Predictions_3WayBlend.csv",
        "ELO_All_Pair_Matchups.csv",
        "ELO_All_Pair_Matchups_LLBlend.csv",
        "ELO_All_Pair_Matchups_AccBlend.csv",
        "ELO_All_Pair_Matchups_3WayBlend.csv",
        "ELO_Pipeline_Visualizations.png",
    ]
    rows = []
    for name in expected:
        path = os.path.join(OUT, name)
        exists = os.path.exists(path)
        size_bytes = os.path.getsize(path) if exists else 0
        n_rows = ""
        if exists and name.endswith(".csv"):
            try:
                n_rows = max(0, sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1)
            except OSError:
                n_rows = ""
        if not exists:
            status = "MISSING"
        elif isinstance(n_rows, int) and n_rows == 0:
            status = "EMPTY"
        elif size_bytes == 0:
            status = "EMPTY"
        else:
            status = "OK"
        rows.append(
            {
                "Artifact": name,
                "Exists": exists,
                "Status": status,
                "Rows": n_rows,
                "Size_Bytes": int(size_bytes),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "ELO_Completeness_Report.csv"), index=False)
    return out


def main():
    print("=" * 70)
    print("WHL 2026 - ELO-ONLY PIPELINE")
    print("=" * 70)

    print("\n[1/9] Loading data...")
    ms, raw, pp, unit, rest, tev, teams = wa.load_data(BASE)
    matchups, matchup_source = wa.load_matchups(BASE, teams)
    bad_csvs = wa.detect_corrupted_csvs(BASE)
    print(f"  Teams: {len(teams)} | Matchups: {len(matchups)} ({matchup_source})")
    print(f"  Corrupted CSV placeholders detected: {len(bad_csvs)}")

    print("\n[2/9] Core tables...")
    gs = wa.build_games(raw)
    shift_game, clean_shift_rows = wa.build_shift_team_game(ms)
    gt, goalie_map = wa.build_goalie_table(raw)
    pp_stats = wa.build_pp_team_stats(pp, teams)
    unit_stats = wa.build_unit_team_stats(unit, teams)
    rest_game = wa.build_rest_by_game_team(rest, raw)
    print(f"  Games: {len(gs):,} | Clean shifts: {clean_shift_rows:,}")

    print("\n[3/9] Build pregame panel...")
    elo_df, final_elos = wa.compute_elo_pregame(gs, teams, k=wa.ELO_K, ha=wa.ELO_HA, xg_w=wa.ELO_XG_W, decay=wa.ELO_DECAY, regress_pct=wa.ELO_REG)
    panel = wa.build_team_game_panel(gs, shift_game, elo_df, rest_game)
    panel = wa.add_pregame_features(panel)
    train_df, diff_cols = wa.build_training_matrix(panel, MODEL_FEATURES, wa.MODEL_MIN_HISTORY_GAMES)
    print(f"  Training games after min-history filter: {len(train_df):,}")

    print("\n[4/9] Tune Elo params (train-CV only)...")
    tune_split = min(1000, int(len(train_df) * 0.76))
    elo_best = wa.tune_elo_params(
        gs,
        teams,
        train_df["game_id"].values,
        train_df["home_win"].values,
        tune_split,
    )
    pd.DataFrame([elo_best]).to_csv(os.path.join(OUT, "ELO_Parameter_Selection.csv"), index=False)
    print(
        f"  Selected K={elo_best['k']} HA={elo_best['ha']} xG_w={elo_best['xg_w']} "
        f"decay={elo_best['decay']} reg={elo_best['reg']:.2f} CV_LL={elo_best['cv_ll']:.4f}"
    )

    print("\n[5/9] Rebuild tuned Elo + evaluate Elo holdout...")
    elo_df, final_elos = wa.compute_elo_pregame(
        gs,
        teams,
        k=elo_best["k"],
        ha=elo_best["ha"],
        xg_w=elo_best["xg_w"],
        decay=elo_best["decay"],
        regress_pct=elo_best["reg"],
    )
    panel = wa.build_team_game_panel(gs, shift_game, elo_df, rest_game)
    panel = wa.add_pregame_features(panel)
    train_df, diff_cols = wa.build_training_matrix(panel, MODEL_FEATURES, wa.MODEL_MIN_HISTORY_GAMES)

    metrics, yte, pte, split = evaluate_elo_holdout(train_df)
    y = train_df["home_win"].values.astype(int)
    p_elo = np.clip(train_df["elo_prob"].values.astype(float), 1e-7, 1 - 1e-7)
    p_elo_tr, p_elo_te = p_elo[:split], p_elo[split:]
    ytr = y[:split]

    p_sgd_oof, p_sgd_te, sgd_model, sgd_scaler, _, _ = fit_sgd_blend_components(train_df, diff_cols, split)
    p_lr_oof, p_lr_te, lr_model, lr_scaler = fit_logreg_blend_components(train_df, diff_cols, split)

    valid_sgd = int(np.isfinite(p_sgd_oof).sum())
    valid_lr = int(np.isfinite(p_lr_oof).sum())
    valid_joint = int((np.isfinite(p_sgd_oof) & np.isfinite(p_lr_oof)).sum())

    w_ll, ll_train = tune_blend_weight_for_ll(ytr, p_elo_tr, p_sgd_oof)
    p_bl_ll_te = np.clip(w_ll * p_elo_te + (1 - w_ll) * p_sgd_te, 1e-7, 1 - 1e-7)

    w_lr, acc_train_lr, ll_train_lr = tune_blend_weight_for_acc(ytr, p_elo_tr, p_lr_oof)
    p_bl_lr_te = np.clip(w_lr * p_elo_te + (1 - w_lr) * p_lr_te, 1e-7, 1 - 1e-7)

    w3_elo, w3_sgd, w3_lr, ll_train_3way = tune_three_way_convex_for_ll(ytr, p_elo_tr, p_sgd_oof, p_lr_oof)
    p_bl_3way_te = np.clip(w3_elo * p_elo_te + w3_sgd * p_sgd_te + w3_lr * p_lr_te, 1e-7, 1 - 1e-7)

    blend_tuning = pd.DataFrame(
        [
            {
                "Blend": "Elo_SGD_Blend_LL",
                "Elo_Weight": round(w_ll, 6),
                "Aux_Model_Weight": round(1 - w_ll, 6),
                "Aux_Model": "SGD",
                "Train_Objective": "log_loss",
                "Train_LL": round(ll_train, 6),
                "Threshold": 0.5,
                "OOF_Rows_Used": valid_sgd,
            },
            {
                "Blend": "Elo_LogReg_Blend_Acc",
                "Elo_Weight": round(w_lr, 6),
                "Aux_Model_Weight": round(1 - w_lr, 6),
                "Aux_Model": "LogReg",
                "Train_Objective": "accuracy@0.5",
                "Train_Acc": round(acc_train_lr, 6),
                "Train_LL": round(ll_train_lr, 6),
                "Threshold": 0.5,
                "OOF_Rows_Used": valid_lr,
            },
            {
                "Blend": "Elo_SGD_LogReg_Convex3_LL",
                "Elo_Weight": round(w3_elo, 6),
                "Aux_Model_Weight": round(1 - w3_elo, 6),
                "Aux_Model": f"SGD:{w3_sgd:.4f}|LogReg:{w3_lr:.4f}",
                "Train_Objective": "log_loss",
                "Train_LL": round(ll_train_3way, 6),
                "Threshold": 0.5,
                "OOF_Rows_Used": valid_joint,
            },
        ]
    )
    blend_tuning.to_csv(os.path.join(OUT, "ELO_Blend_Tuning.csv"), index=False)

    row_elo = metric_row("Elo", yte, p_elo_te, 0.5)
    row_ll = metric_row("Elo_SGD_Blend_LL", yte, p_bl_ll_te, 0.5)
    row_lr = metric_row("Elo_LogReg_Blend_Acc", yte, p_bl_lr_te, 0.5)
    row_3way = metric_row("Elo_SGD_LogReg_Convex3_LL", yte, p_bl_3way_te, 0.5)
    perf = pd.DataFrame([row_elo, row_ll, row_lr, row_3way]).sort_values(
        ["Log_Loss", "Brier", "AUC_ROC"], ascending=[True, True, False]
    )
    perf.to_csv(os.path.join(OUT, "ELO_Blend_Performance.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUT, "ELO_Model_Performance.csv"), index=False)

    # Selection policy is leakage-safe: select by train/OOF objective, not holdout.
    selection_policy = pd.DataFrame(
        [
            {"Model": "Elo", "Train_Objective": "log_loss", "Train_LL": round(metrics["Train_LL"], 6), "Train_Acc": np.nan},
            {"Model": "Elo_SGD_Blend_LL", "Train_Objective": "log_loss", "Train_LL": round(ll_train, 6), "Train_Acc": np.nan},
            {
                "Model": "Elo_LogReg_Blend_Acc",
                "Train_Objective": "accuracy@0.5",
                "Train_LL": round(ll_train_lr, 6),
                "Train_Acc": round(acc_train_lr, 6),
            },
            {
                "Model": "Elo_SGD_LogReg_Convex3_LL",
                "Train_Objective": "log_loss",
                "Train_LL": round(ll_train_3way, 6),
                "Train_Acc": np.nan,
            },
        ]
    ).sort_values(["Train_LL", "Model"], ascending=[True, True])
    ll_model_name = selection_policy.iloc[0]["Model"]
    selection_policy.insert(0, "Selection_Policy", "best_train_ll_no_holdout_tuning")
    selection_policy.to_csv(os.path.join(OUT, "ELO_Model_Selection_Policy.csv"), index=False)

    sanity_df = wa.run_leakage_and_overfit_checks(train_df, diff_cols, split, raw_df=raw)
    sanity_df.to_csv(os.path.join(OUT, "ELO_Leakage_Sanity_Report.csv"), index=False)
    sgd_model_full, lr_model_full, blend_scaler_full = fit_full_blend_models(train_df, diff_cols)
    print(
        f"  Elo holdout: LL={metrics['Log_Loss']:.4f} Acc={metrics['Accuracy']:.1%} "
        f"AUC={metrics['AUC_ROC']:.4f} Status={metrics['Status']}"
    )
    print(
        f"  Blend(LL): w_elo={w_ll:.3f} LL={row_ll['Log_Loss']:.4f} "
        f"| Blend(Acc): w_elo={w_lr:.3f} Acc={row_lr['Accuracy']:.1%} "
        f"| 3Way: w=[{w3_elo:.3f},{w3_sgd:.3f},{w3_lr:.3f}] LL={row_3way['Log_Loss']:.4f}"
    )
    print("  Refit SGD/LogReg on full leakage-safe season data for tournament inference.")

    print("\n[6/9] Team Elo + xG + power outputs...")
    final_df = wa.build_final_team_features(panel, final_elos, gt, goalie_map, pp_stats, unit_stats, tev)
    power_df = wa.build_power_rankings(final_df)
    team_tbl = build_elo_team_table(final_df, power_df)
    team_tbl.to_csv(os.path.join(OUT, "ELO_Team_Elo_xG_Power.csv"), index=False)
    final_df.sort_values("elo", ascending=False).to_csv(os.path.join(OUT, "ELO_Team_Profile_Full.csv"), index=False)

    pr = power_df[["rank", "team", "wins", "ev_pct", "xGF60", "gsax60", "elo", "ps"]].copy()
    pr.columns = ["Rank", "Team", "Wins", "EV_xG%", "xG_per_60", "GSAx_per_60", "Elo_Rating", "Power_Score"]
    pr.to_csv(os.path.join(OUT, "ELO_Power_Rankings.csv"), index=False)

    archetypes_df, centers = wa.derive_team_archetypes(final_df)
    archetypes_df.to_csv(os.path.join(OUT, "ELO_Archetypes.csv"), index=False)
    centers.to_csv(os.path.join(OUT, "ELO_Archetype_Centers.csv"), index=False)
    print("  Saved ELO_Power_Rankings.csv, ELO_Team_Elo_xG_Power.csv, ELO_Team_Profile_Full.csv")

    print("\n[7/9] Elo/Blend tournament predictions...")
    elo_best_obj = {"model": None, "scaled": False}
    base_tpred = wa.predict_matchups(
        matchups,
        "17_Elo",
        elo_best_obj,
        final_df,
        MODEL_FEATURES,
        scaler=None,
        final_elos=final_elos,
        elo_ha=elo_best["ha"],
    )
    base_tpred = wa.attach_archetypes_to_matchups(base_tpred, archetypes_df, final_df)
    base_tpred["Win_Prob_Elo"] = base_tpred["Win_Prob"]

    sgd_probs = predict_sgd_matchups(matchups, final_df, MODEL_FEATURES, blend_scaler_full, sgd_model_full)
    base_tpred["Win_Prob_SGD"] = np.round(sgd_probs, 6)
    lr_probs = predict_logreg_matchups(matchups, final_df, MODEL_FEATURES, blend_scaler_full, lr_model_full)
    base_tpred["Win_Prob_LogReg"] = np.round(lr_probs, 6)
    base_tpred["Win_Prob_LLBlend"] = np.round(
        np.clip(w_ll * base_tpred["Win_Prob_Elo"].values + (1 - w_ll) * base_tpred["Win_Prob_SGD"].values, 1e-7, 1 - 1e-7),
        6,
    )
    base_tpred["Win_Prob_AccBlend"] = np.round(
        np.clip(w_lr * base_tpred["Win_Prob_Elo"].values + (1 - w_lr) * base_tpred["Win_Prob_LogReg"].values, 1e-7, 1 - 1e-7),
        6,
    )
    base_tpred["Win_Prob_3WayBlend"] = np.round(
        np.clip(
            w3_elo * base_tpred["Win_Prob_Elo"].values
            + w3_sgd * base_tpred["Win_Prob_SGD"].values
            + w3_lr * base_tpred["Win_Prob_LogReg"].values,
            1e-7,
            1 - 1e-7,
        ),
        6,
    )

    ll_col = {
        "Elo": "Win_Prob_Elo",
        "Elo_SGD_Blend_LL": "Win_Prob_LLBlend",
        "Elo_LogReg_Blend_Acc": "Win_Prob_AccBlend",
        "Elo_SGD_LogReg_Convex3_LL": "Win_Prob_3WayBlend",
    }[ll_model_name]

    def build_variant(df, prob_col, cal_probs, thr, cal_name, out_name):
        t = df.copy()
        t["Win_Prob"] = np.clip(t[prob_col].values.astype(float), 1e-7, 1 - 1e-7)
        t["Predicted_Winner"] = np.where(t["Win_Prob"] >= thr, t["Home_Team"], t["Away_Team"])
        t["Confidence"] = np.round(np.maximum(t["Win_Prob"], 1 - t["Win_Prob"]), 4)
        ctab, csel, cal_fn, cal_method = wa.build_calibration_artifacts(cal_probs, yte, bins=10, min_bin=20)
        ctab.to_csv(os.path.join(OUT, cal_name), index=False)
        csel.to_csv(os.path.join(OUT, cal_name.replace("Uncertainty", "Calibration_Selection")), index=False)
        t = wa.attach_uncertainty_to_matchups(t, ctab, calibrate_fn=cal_fn)
        t["Calibration_Method"] = cal_method
        t["Consensus_Win_Prob"] = t["Win_Prob"]
        t["Model_Spread"] = np.abs(t["Win_Prob_Elo"] - t["Win_Prob_SGD"])
        t = wa.attach_robustness_layer(t, final_df)
        t = t.sort_values(["Upset_Risk", "Fragility_Index"], ascending=[False, False]).reset_index(drop=True)
        t.to_csv(os.path.join(OUT, out_name), index=False)
        return t, ctab

    # Elo only variant
    tpred_elo, cal_elo = build_variant(
        base_tpred,
        "Win_Prob_Elo",
        p_elo_te,
        0.5,
        "ELO_Tournament_Uncertainty.csv",
        "ELO_Tournament_Predictions.csv",
    )
    # LL-optimized blend
    tpred_ll, cal_ll = build_variant(
        base_tpred,
        "Win_Prob_LLBlend",
        p_bl_ll_te,
        0.5,
        "ELO_Tournament_Uncertainty_LLBlend.csv",
        "ELO_Tournament_Predictions_LLBlend.csv",
    )
    # Accuracy-optimized blend
    tpred_acc, cal_acc = build_variant(
        base_tpred,
        "Win_Prob_AccBlend",
        p_bl_lr_te,
        0.5,
        "ELO_Tournament_Uncertainty_AccBlend.csv",
        "ELO_Tournament_Predictions_AccBlend.csv",
    )
    # 3-way convex blend
    tpred_3way, cal_3way = build_variant(
        base_tpred,
        "Win_Prob_3WayBlend",
        p_bl_3way_te,
        0.5,
        "ELO_Tournament_Uncertainty_3WayBlend.csv",
        "ELO_Tournament_Predictions_3WayBlend.csv",
    )

    # Canonical output: best LL variant
    chosen = {
        "Win_Prob_Elo": tpred_elo,
        "Win_Prob_LLBlend": tpred_ll,
        "Win_Prob_AccBlend": tpred_acc,
        "Win_Prob_3WayBlend": tpred_3way,
    }[ll_col]
    cal_choice = {
        "Win_Prob_Elo": cal_elo,
        "Win_Prob_LLBlend": cal_ll,
        "Win_Prob_AccBlend": cal_acc,
        "Win_Prob_3WayBlend": cal_3way,
    }[ll_col]
    chosen.to_csv(os.path.join(OUT, "ELO_Tournament_Predictions.csv"), index=False)
    print(
        f"  Saved ELO_Tournament_Predictions.csv ({len(chosen)} games) | "
        f"primary={ll_model_name} ({ll_col})"
    )

    print("\n[8/9] Elo/Blend all-pairs + visualizations...")
    all_pairs = predict_all_pairs_elo(teams, final_elos, elo_best["ha"])
    all_pairs.to_csv(os.path.join(OUT, "ELO_All_Pair_Matchups.csv"), index=False)
    all_pairs_ll = predict_all_pairs_blend(
        teams, final_df, MODEL_FEATURES, blend_scaler_full, sgd_model_full, final_elos, elo_best["ha"], w_ll
    )
    all_pairs_ll.to_csv(os.path.join(OUT, "ELO_All_Pair_Matchups_LLBlend.csv"), index=False)
    all_pairs_acc = predict_all_pairs_blend(
        teams, final_df, MODEL_FEATURES, blend_scaler_full, lr_model_full, final_elos, elo_best["ha"], w_lr
    )
    all_pairs_acc.to_csv(os.path.join(OUT, "ELO_All_Pair_Matchups_AccBlend.csv"), index=False)
    all_pairs_3way = predict_all_pairs_three_way_blend(
        teams,
        final_df,
        MODEL_FEATURES,
        blend_scaler_full,
        sgd_model_full,
        blend_scaler_full,
        lr_model_full,
        final_elos,
        elo_best["ha"],
        w3_elo,
        w3_sgd,
        w3_lr,
    )
    all_pairs_3way.to_csv(os.path.join(OUT, "ELO_All_Pair_Matchups_3WayBlend.csv"), index=False)
    viz_metrics = perf.set_index("Model").loc[ll_model_name].to_dict()
    make_elo_visualizations(team_tbl, chosen, cal_choice, viz_metrics)
    print(
        "  Saved ELO_All_Pair_Matchups.csv, ELO_All_Pair_Matchups_LLBlend.csv, "
        "ELO_All_Pair_Matchups_AccBlend.csv, ELO_All_Pair_Matchups_3WayBlend.csv, "
        "and ELO_Pipeline_Visualizations.png"
    )

    print("\n[9/9] Completeness report...")
    comp = build_elo_completeness_report()
    ok = int((comp["Status"] == "OK").sum())
    print(f"  ELO_Completeness_Report.csv: {ok}/{len(comp)} OK")
    print("=" * 70)
    print("Elo-only pipeline complete.")


if __name__ == "__main__":
    main()
