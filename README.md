# WHL 2026 Final Submission Package

This repository contains the WHL 2026 Phase 1 pipelines, model-selection artifacts, and finalized submission deliverables.

## Competition Alignment
- Primary objective: maximize prediction accuracy for Phase 1a matchup probabilities and team-strength ranking quality.
- Selection policy: leakage-safe and holdout-safe (`best OOF log loss`, no holdout tuning).
- Data discipline: pregame-only feature engineering, temporal validation, and explicit leakage/integrity audits.

## Latest Update (v9.8)
- Fixed probability conversion for `decision_function` models to avoid singleton inference distortion.
- Locked permutation model selection to strict OOF objective (`search.iloc[0]`).
- Added post-selection full-data model refit for final tournament/ranking inference while preserving honest holdout evaluation.
- Expanded leakage sanity report with raw-data proxy/leakage diagnostics and row-split contamination checks.
- Added small leakage-safe interaction features (`elo_form_interaction`, `xg_rest_interaction`, `special_teams_net`).
- Added calibration method selection (`raw` vs `platt` vs `isotonic`) with exported selection audits.
- Expanded interpretability exports with top-two matchup edge drivers and summary artifact.
- Added regression test suite: `python3 -m unittest -v test_whl_analysis.py`.

Current selected blend (per permutation optimizer):
- `Elo:0.53`, `SGD:0.47`, `LogReg:0.00`, `XGB:0.00`, `XGB2:0.00`
- Holdout: `LL=0.669612`, `Acc=0.5786`, `AUC=0.5645`, `Brier=0.238420`

## Core Pipelines
- `whl_analysis.py` (main Phase 1 pipeline)
- `whl_elo_pipeline.py` (Elo-centric pipeline + blend variants)
- `whl_permutation_optimizer.py` (large OOF blend search and final model selection)
- `finalize_deliverables.py` (final CSV/PNG deliverable pack)
- `build_final_report_html.py` (submission-ready HTML report)

## Final Deliverables
- `Final_Report.html`
- `Final_Visualizations.png`
- `Final_Power_Rankings.csv`
- `Final_Tournament_Predictions.csv`
- `Final_Line_Disparity_Rankings.csv`
- `Final_Top10_Line_Disparity.csv`
- `Final_Team_Rankings_Full.csv`
- `Final_Model_Summary.csv`
- `Final_Model_Interpretability.csv`
- `Final_Model_Interpretability_Top12.csv`
- `Final_Model_Interpretability_Summary.csv`
- `Final_Model_Weight_Explain.csv`
- `Final_Model_Interpretability_Human_Readout.md`
- `Final_Data_Integrity_Summary.csv`
- `Final_Leakage_Overfit_Summary.csv`
- Submission-formatted:
  - `1A_Power_Rankings.csv`
  - `1A_Tournament_Predictions.csv`
  - `1B_Line_Disparity_Rankings.csv`

## Reproducible Run Order
```bash
python3 whl_analysis.py
python3 whl_elo_pipeline.py
python3 whl_permutation_optimizer.py
python3 finalize_deliverables.py
python3 build_final_report_html.py
python3 -m unittest -v test_whl_analysis.py
```
