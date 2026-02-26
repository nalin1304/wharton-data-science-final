#!/usr/bin/env python3
"""
Regression checks for the WHL pipeline outputs.

These tests are intentionally artifact-based so they run quickly and verify
core competition safety guarantees after each pipeline refresh.
"""

import os
import unittest

import pandas as pd


BASE = os.path.dirname(os.path.abspath(__file__))


def _path(name: str) -> str:
    return os.path.join(BASE, name)


def _must_read_csv(name: str) -> pd.DataFrame:
    path = _path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required artifact: {name}")
    return pd.read_csv(path)


class TestWHLAnalysisRegression(unittest.TestCase):
    def test_selection_policy_is_oof_only(self):
        df = _must_read_csv("Permutation_Best_Model_Selection.csv")
        self.assertGreaterEqual(len(df), 1)

        row = df.iloc[0]
        self.assertEqual(str(row["Selection_Policy"]), "Best_OOF_LogLoss_NoHoldoutTuning")
        self.assertGreaterEqual(int(row["Permutations_Evaluated"]), 1000)

        weights = {}
        for part in str(row["Blend_Weights"]).split(","):
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            weights[k.strip()] = float(v)
        self.assertGreater(len(weights), 0)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_train_test_game_id_overlap_is_zero(self):
        sanity = _must_read_csv("ELO_Leakage_Sanity_Report.csv")
        row = sanity[sanity["Check"] == "train_test_game_id_overlap"]
        self.assertEqual(len(row), 1)
        self.assertEqual(str(row.iloc[0]["Status"]).upper(), "PASS")
        self.assertEqual(int(float(row.iloc[0]["Value"])), 0)

    def test_required_final_artifacts_exist_and_nonempty(self):
        required = [
            "Final_Power_Rankings.csv",
            "Final_Tournament_Predictions.csv",
            "Final_Line_Disparity_Rankings.csv",
            "Final_Model_Summary.csv",
            "Final_Report.html",
            "Final_Visualizations.png",
        ]
        for name in required:
            path = _path(name)
            self.assertTrue(os.path.exists(path), msg=f"Missing artifact: {name}")
            self.assertGreater(os.path.getsize(path), 0, msg=f"Empty artifact: {name}")

    def test_prediction_probability_ranges(self):
        tpred = _must_read_csv("Final_Tournament_Predictions.csv")
        for col in ["Win_Prob", "Confidence", "Predicted_Winner"]:
            self.assertIn(col, tpred.columns)
            self.assertFalse(tpred[col].isna().any(), msg=f"NaN found in {col}")

        self.assertTrue(((tpred["Win_Prob"] >= 0.0) & (tpred["Win_Prob"] <= 1.0)).all())
        self.assertTrue(((tpred["Confidence"] >= 0.5) & (tpred["Confidence"] <= 1.0)).all())

        if "Calibration_Method" in tpred.columns:
            allowed = {"raw", "platt", "isotonic"}
            methods = set(tpred["Calibration_Method"].dropna().astype(str).unique())
            self.assertTrue(methods.issubset(allowed), msg=f"Unexpected calibration methods: {methods - allowed}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
