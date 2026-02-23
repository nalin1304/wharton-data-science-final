#!/usr/bin/env python3
"""
Create the final submission-facing deliverable pack from finalized model outputs.

This script is intentionally lightweight and deterministic: it only reads finalized
CSV artifacts and writes cleaned presentation files.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))


def _must_read(path):
    p = os.path.join(BASE, path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(p)


def _optional_read(path):
    p = os.path.join(BASE, path)
    return pd.read_csv(p) if os.path.exists(p) else None


def build_final_outputs():
    power = _must_read("ELO_Power_Rankings.csv")
    team = _must_read("ELO_Team_Elo_xG_Power.csv")
    tpred = _must_read("Permutation_Best_Tournament_Predictions.csv")
    msel = _must_read("Permutation_Best_Model_Selection.csv")
    mlead = _must_read("Permutation_Model_Leaderboard.csv")
    dint = _must_read("Permutation_Data_Integrity_Audit.csv")
    leak = _must_read("Permutation_Leakage_Overfit_Report.csv")
    interp = _optional_read("Permutation_Model_Interpretability.csv")
    interp_top = _optional_read("Permutation_Model_Interpretability_Top12.csv")
    weight_explain = _optional_read("Permutation_Model_Weight_Explain.csv")

    # Final power rankings
    final_power = power.copy().sort_values(["Rank", "Power_Score"], ascending=[True, False]).reset_index(drop=True)
    final_power.to_csv(os.path.join(BASE, "Final_Power_Rankings.csv"), index=False)
    final_power.to_csv(os.path.join(BASE, "1A_Power_Rankings.csv"), index=False)

    # Final tournament predictions from selected blend
    tcols = [
        "Game",
        "Home_Team",
        "Away_Team",
        "Win_Prob_SelectedBlend",
        "Predicted_Winner",
        "Confidence",
        "Robust_Win_Prob",
        "Robust_CI_Low_95",
        "Robust_CI_High_95",
        "Fragility_Index",
        "Edge_Stability",
    ]
    for c in ["Top_Edge_Driver", "Top_Edge_Contribution", "Model_Rationale"]:
        if c in tpred.columns:
            tcols.append(c)
    out_t = tpred[tcols].copy()
    out_t = out_t.rename(columns={"Win_Prob_SelectedBlend": "Win_Prob"})
    out_t = out_t.sort_values(["Win_Prob", "Confidence"], ascending=[False, False]).reset_index(drop=True)
    out_t.to_csv(os.path.join(BASE, "Final_Tournament_Predictions.csv"), index=False)
    out_t.to_csv(os.path.join(BASE, "1A_Tournament_Predictions.csv"), index=False)

    # Final line disparity rankings
    line = team[["Team", "Line_Disparity", "Wins", "xGF60", "xGA60", "Power_Rank", "Elo_Rank"]].copy()
    line = line.sort_values(["Line_Disparity", "Wins"], ascending=[False, False]).reset_index(drop=True)
    line.insert(0, "Rank", np.arange(1, len(line) + 1))
    line = line.rename(columns={"Line_Disparity": "Disparity_Ratio"})
    line.to_csv(os.path.join(BASE, "Final_Line_Disparity_Rankings.csv"), index=False)
    line.to_csv(os.path.join(BASE, "1B_Line_Disparity_Rankings.csv"), index=False)
    line.head(10).to_csv(os.path.join(BASE, "Final_Top10_Line_Disparity.csv"), index=False)

    # Full final team ranking list
    final_team = team.sort_values(["Power_Rank", "Elo_Rank"]).reset_index(drop=True)
    final_team.to_csv(os.path.join(BASE, "Final_Team_Rankings_Full.csv"), index=False)

    # Model summary list
    summary_rows = []
    s = msel.iloc[0]
    summary_rows.append(
        {
            "Section": "Selected_Model",
            "Item": s["Selected_Model"],
            "Value": s["Blend_Weights"],
        }
    )
    for k in ["OOF_Log_Loss", "Holdout_Log_Loss", "Holdout_Accuracy_05", "Holdout_AUC_ROC", "Holdout_Brier", "Overfit_Gap", "Overfit_Status", "Permutations_Evaluated"]:
        summary_rows.append({"Section": "Selected_Model_Metric", "Item": k, "Value": s[k]})

    for _, r in mlead.head(6).iterrows():
        summary_rows.append(
            {
                "Section": "Leaderboard_Top6",
                "Item": r["Model"],
                "Value": f"LL={r['Holdout_Log_Loss']:.6f}, Acc={r['Holdout_Accuracy']:.4f}, AUC={r['Holdout_AUC']:.4f}",
            }
        )

    pd.DataFrame(summary_rows).to_csv(os.path.join(BASE, "Final_Model_Summary.csv"), index=False)

    # Integrity and leakage summaries
    dint.to_csv(os.path.join(BASE, "Final_Data_Integrity_Summary.csv"), index=False)
    leak.to_csv(os.path.join(BASE, "Final_Leakage_Overfit_Summary.csv"), index=False)
    if interp is not None:
        interp.to_csv(os.path.join(BASE, "Final_Model_Interpretability.csv"), index=False)
    if interp_top is not None:
        interp_top.to_csv(os.path.join(BASE, "Final_Model_Interpretability_Top12.csv"), index=False)
    if weight_explain is not None:
        weight_explain.to_csv(os.path.join(BASE, "Final_Model_Weight_Explain.csv"), index=False)
    notes_path = os.path.join(BASE, "Permutation_Model_Interpretability_Notes.md")
    if os.path.exists(notes_path):
        with open(notes_path, "r", encoding="utf-8") as src, open(
            os.path.join(BASE, "Final_Model_Interpretability_Notes.md"), "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())

    # Visualization pack
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    top12 = final_power.head(12).iloc[::-1]
    axs[0, 0].barh(top12["Team"], top12["Power_Score"], color="#1f77b4")
    axs[0, 0].set_title("Top 12 Teams by Power Score")
    axs[0, 0].set_xlabel("Power Score")

    axs[0, 1].scatter(team["Line_Disparity"], team["Power_Score"], c=team["Elo_Rating"], cmap="viridis", s=60)
    axs[0, 1].set_title("Line Disparity vs Power Score")
    axs[0, 1].set_xlabel("Line Disparity")
    axs[0, 1].set_ylabel("Power Score")

    m = out_t.sort_values("Win_Prob", ascending=False).reset_index(drop=True)
    labels = [f"{h[:3].upper()}-{a[:3].upper()}" for h, a in zip(m["Home_Team"], m["Away_Team"])]
    axs[1, 0].bar(range(len(m)), m["Win_Prob"], color="#2ca02c")
    axs[1, 0].set_xticks(range(len(m)))
    axs[1, 0].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].set_title("Round-1 Home Win Probabilities (Selected Model)")

    x = np.arange(len(m))
    y = m["Robust_Win_Prob"].values
    lo = m["Robust_CI_Low_95"].values
    hi = m["Robust_CI_High_95"].values
    yerr = np.vstack([y - lo, hi - y])
    axs[1, 1].errorbar(x, y, yerr=yerr, fmt='o', color="#d62728", ecolor="#9467bd", capsize=3)
    axs[1, 1].axhline(0.5, linestyle='--', color='gray', linewidth=1)
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_title("Robust Win Probabilities with 95% CI")

    fig.suptitle("WHL 2026 Finalized Deliverables", fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(BASE, "Final_Visualizations.png"), dpi=220)
    plt.close(fig)


def main():
    build_final_outputs()
    print("Final deliverables created:")
    for f in [
        "Final_Power_Rankings.csv",
        "Final_Tournament_Predictions.csv",
        "Final_Line_Disparity_Rankings.csv",
        "Final_Top10_Line_Disparity.csv",
        "Final_Team_Rankings_Full.csv",
        "Final_Model_Summary.csv",
        "Final_Data_Integrity_Summary.csv",
        "Final_Leakage_Overfit_Summary.csv",
        "Final_Model_Interpretability.csv",
        "Final_Model_Interpretability_Top12.csv",
        "Final_Model_Weight_Explain.csv",
        "Final_Model_Interpretability_Notes.md",
        "Final_Visualizations.png",
        "1A_Power_Rankings.csv",
        "1A_Tournament_Predictions.csv",
        "1B_Line_Disparity_Rankings.csv",
    ]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
