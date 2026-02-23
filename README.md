# WHL 2026 Final Submission Package

This repository contains the final, curated outputs and core reproducible scripts for the WHL 2026 analytics submission.

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
- `Final_Model_Weight_Explain.csv`
- `Final_Model_Interpretability_Notes.md`
- `Final_Data_Integrity_Summary.csv`
- `Final_Leakage_Overfit_Summary.csv`
- Submission-formatted outputs:
  - `1A_Power_Rankings.csv`
  - `1A_Tournament_Predictions.csv`
  - `1B_Line_Disparity_Rankings.csv`

## Core Scripts
- `whl_analysis.py`
- `whl_elo_pipeline.py`
- `whl_permutation_optimizer.py`
- `finalize_deliverables.py`
- `build_final_report_html.py`

## Notes
- Model selection policy is leakage-safe: best OOF log loss with no holdout tuning.
- Final selected blend: Elo + SGD.
