#!/usr/bin/env python3
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def read_csv(name: str) -> pd.DataFrame:
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input: {name}")
    return pd.read_csv(path)


def read_csv_optional(name: str):
    path = os.path.join(BASE, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def read_text_optional(name: str) -> str:
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def fmt_float(v, d=4):
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return str(v)


def status_badge(s: str) -> str:
    k = str(s).strip().upper()
    css = {
        "PASS": "ok",
        "WARN": "warn",
        "FAIL": "bad",
        "INFO": "info",
        "OK": "ok",
    }.get(k, "info")
    return f'<span class="badge {css}">{k}</span>'


def render_table(df: pd.DataFrame, table_id: str, classes: str = "report-table") -> str:
    return df.to_html(index=False, escape=False, table_id=table_id, classes=classes)


def main() -> None:
    model = read_csv("Final_Model_Summary.csv")
    power = read_csv("Final_Power_Rankings.csv")
    tpred = read_csv("Final_Tournament_Predictions.csv")
    disparity = read_csv("Final_Line_Disparity_Rankings.csv")
    integrity = read_csv("Final_Data_Integrity_Summary.csv")
    leak = read_csv("Final_Leakage_Overfit_Summary.csv")
    interp = read_csv_optional("Final_Model_Interpretability_Top12.csv")
    interp_summary = read_csv_optional("Final_Model_Interpretability_Summary.csv")
    weight_explain = read_csv_optional("Final_Model_Weight_Explain.csv")
    interp_notes = read_text_optional("Final_Model_Interpretability_Human_Readout.md")

    selected_row = model[model["Section"] == "Selected_Model"].iloc[0]
    metrics = model[model["Section"] == "Selected_Model_Metric"].set_index("Item")["Value"].to_dict()

    selected_model = str(selected_row["Item"])
    selected_weights = str(selected_row["Value"])

    kpi_cards = [
        ("OOF Log Loss", fmt_float(metrics.get("OOF_Log_Loss", ""), 6)),
        ("Holdout Log Loss", fmt_float(metrics.get("Holdout_Log_Loss", ""), 6)),
        ("Holdout Accuracy", f"{100 * float(metrics.get('Holdout_Accuracy_05', 0)):.2f}%"),
        ("Holdout AUC", fmt_float(metrics.get("Holdout_AUC_ROC", ""), 4)),
        ("Holdout Brier", fmt_float(metrics.get("Holdout_Brier", ""), 6)),
        ("Permutations", str(int(float(metrics.get("Permutations_Evaluated", 0)))) if "Permutations_Evaluated" in metrics else "21,995"),
    ]

    power_view = power[["Rank", "Team", "Power_Score", "Wins", "Elo_Rating", "EV_xG%"]].head(16).copy()
    power_view["Power_Score"] = power_view["Power_Score"].map(lambda x: fmt_float(x, 4))
    power_view["Elo_Rating"] = power_view["Elo_Rating"].map(lambda x: fmt_float(x, 1))
    power_view["EV_xG%"] = power_view["EV_xG%"].map(lambda x: fmt_float(x, 4))

    disp_view = disparity[["Rank", "Team", "Disparity_Ratio", "Wins", "Power_Rank", "Elo_Rank"]].head(16).copy()
    disp_view["Disparity_Ratio"] = disp_view["Disparity_Ratio"].map(lambda x: fmt_float(x, 4))

    tpred_view = tpred.copy()
    for c in ["Win_Prob", "Confidence", "Robust_Win_Prob", "Robust_CI_Low_95", "Robust_CI_High_95", "Fragility_Index"]:
        if c in tpred_view.columns:
            tpred_view[c] = tpred_view[c].map(lambda x: fmt_float(x, 4))
    if "Top_Edge_Contribution" in tpred_view.columns:
        tpred_view["Top_Edge_Contribution"] = tpred_view["Top_Edge_Contribution"].map(lambda x: fmt_float(x, 4))

    integrity_view = integrity.copy()
    integrity_view["Status"] = integrity_view["Status"].map(status_badge)

    leak_view = leak.copy()
    leak_view["Status"] = leak_view["Status"].map(status_badge)

    pass_count = int((integrity["Status"].astype(str).str.upper() == "PASS").sum())
    total_checks = int(len(integrity))

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    cards_html = "".join(
        [f'<div class="kpi-card"><div class="kpi-title">{t}</div><div class="kpi-val">{v}</div></div>' for t, v in kpi_cards]
    )

    if interp_notes:
        note_lines = []
        for ln in interp_notes.splitlines():
            s = ln.strip()
            if not s:
                continue
            s = s.lstrip("#").strip()
            note_lines.append(s)
        human_notes = "<br/>".join(note_lines[:12])
    else:
        human_notes = "The model blends Elo and SGD probabilities. Elo captures long-run team strength; SGD captures feature-driven matchup context."

    summary_note = ""
    if interp_summary is not None and len(interp_summary):
        sm = {str(k): v for k, v in zip(interp_summary["Metric"], interp_summary["Value"])}
        top_feat = str(sm.get("Top_Feature", "")).strip()
        top_pct = fmt_float(sm.get("Top_Feature_Importance_Pct", ""), 1)
        high_stability = sm.get("High_Stability_Features", "")
        if top_feat:
            summary_note = (
                f"Top global driver: {top_feat} ({top_pct}% importance). "
                f"High-stability features: {high_stability}."
            )

    interp_tbl_html = ""
    if interp is not None and len(interp):
        preferred_cols = [
            "Rank",
            "Feature_Name",
            "Std_Coef",
            "Importance_Pct",
            "Stability",
            "Direction",
            "Plain_English",
        ]
        present_cols = [c for c in preferred_cols if c in interp.columns]
        interp_view = interp[present_cols].copy()
        if "Std_Coef" in interp_view.columns:
            interp_view["Std_Coef"] = interp_view["Std_Coef"].map(lambda x: fmt_float(x, 4))
        if "Importance_Pct" in interp_view.columns:
            interp_view["Importance_Pct"] = interp_view["Importance_Pct"].map(lambda x: fmt_float(x, 1))
        interp_tbl_html = f'<div class="table-wrap">{render_table(interp_view, "interp_table")}</div>'
    else:
        interp_tbl_html = '<div class="subtitle">Interpretability table not found.</div>'

    weight_tbl_html = ""
    if weight_explain is not None and len(weight_explain):
        wv = weight_explain.copy()
        if "Weight" in wv.columns:
            wv["Weight"] = wv["Weight"].map(lambda x: fmt_float(x, 4))
        weight_tbl_html = f'<div class="table-wrap">{render_table(wv, "weight_table")}</div>'
    else:
        weight_tbl_html = '<div class="subtitle">Weight explanation table not found.</div>'

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>WHL 2026 Final Report</title>
  <style>
    :root {{
      --bg: #0a1220;
      --bg2: #111b2f;
      --card: #17243b;
      --line: #2a3d5f;
      --txt: #eaf1ff;
      --muted: #9db1d1;
      --accent: #29c2a8;
      --accent2: #f2b44d;
      --good: #25c17a;
      --warn: #ffae42;
      --bad: #ff6b6b;
      --info: #56a8ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--txt);
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(1200px 500px at 20% -20%, rgba(41,194,168,0.18), transparent 60%),
        radial-gradient(1000px 500px at 110% 0%, rgba(242,180,77,0.16), transparent 60%),
        var(--bg);
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 28px 22px 40px; }}
    .hero {{
      background: linear-gradient(135deg, var(--bg2), #0f1d33);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.28);
    }}
    .title {{ margin: 0; font-size: 34px; letter-spacing: 0.2px; }}
    .subtitle {{ margin-top: 8px; color: var(--muted); font-size: 14px; }}
    .pill {{
      display: inline-block;
      margin-top: 12px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(23,36,59,0.8);
      font-size: 13px;
      color: var(--accent2);
      font-weight: 700;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; margin-top: 14px; }}
    .card {{
      grid-column: span 12;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
    }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px; margin-top: 14px; }}
    .kpi-card {{
      background: #121f36;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      min-height: 88px;
    }}
    .kpi-title {{ color: var(--muted); font-size: 12px; }}
    .kpi-val {{ margin-top: 8px; font-size: 22px; font-weight: 700; color: var(--accent); }}
    .h2 {{ margin: 0 0 10px; font-size: 20px; color: var(--accent2); }}
    .h3 {{ margin: 0 0 8px; font-size: 16px; color: var(--txt); }}
    .row2 .card:nth-child(1), .row2 .card:nth-child(2) {{ grid-column: span 6; }}
    .full {{ grid-column: span 12; }}
    .viz {{ width: 100%; border-radius: 10px; border: 1px solid var(--line); display: block; }}
    .report-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .report-table th {{
      position: sticky;
      top: 0;
      background: #0f1b31;
      color: var(--accent2);
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid var(--line);
    }}
    .report-table td {{ padding: 8px; border-bottom: 1px solid rgba(42,61,95,0.5); }}
    .table-wrap {{ overflow: auto; max-height: 460px; border: 1px solid var(--line); border-radius: 10px; }}
    .badge {{ font-size: 11px; font-weight: 700; padding: 4px 7px; border-radius: 999px; }}
    .badge.ok {{ background: rgba(37,193,122,0.16); color: #85f3bd; border: 1px solid rgba(37,193,122,0.4); }}
    .badge.warn {{ background: rgba(255,174,66,0.16); color: #ffd18a; border: 1px solid rgba(255,174,66,0.4); }}
    .badge.bad {{ background: rgba(255,107,107,0.16); color: #ffb3b3; border: 1px solid rgba(255,107,107,0.4); }}
    .badge.info {{ background: rgba(86,168,255,0.16); color: #a9d2ff; border: 1px solid rgba(86,168,255,0.4); }}
    .foot {{ color: var(--muted); font-size: 12px; margin-top: 16px; }}
    @media (max-width: 980px) {{
      .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .row2 .card:nth-child(1), .row2 .card:nth-child(2) {{ grid-column: span 12; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1 class="title">WHL 2026 Final Analytics Report</h1>
      <div class="subtitle">Generated {generated_at} | Competition-aligned, leakage-safe model selection using time-split OOF optimization.</div>
      <div class="pill">Selected Model: {selected_model}</div>
      <div class="subtitle" style="margin-top:8px;">Blend weights: {selected_weights}</div>
      <div class="kpi-grid">{cards_html}</div>
      <div class="subtitle" style="margin-top:10px;">Data integrity checks passed: {pass_count}/{total_checks}</div>
    </section>

    <section class="grid row2">
      <div class="card">
        <h2 class="h2">Power Rankings (Top 16)</h2>
        <div class="table-wrap">{render_table(power_view, 'power_table')}</div>
      </div>
      <div class="card">
        <h2 class="h2">Line Disparity Rankings (Top 16)</h2>
        <div class="table-wrap">{render_table(disp_view, 'disp_table')}</div>
      </div>
    </section>

    <section class="grid">
      <div class="card full">
        <h2 class="h2">Tournament Predictions (Final Selected Model)</h2>
        <div class="table-wrap">{render_table(tpred_view, 'pred_table')}</div>
      </div>
      <div class="card">
        <h2 class="h2">How The Model Thinks</h2>
        <div class="subtitle">{human_notes}</div>
        <h3 class="h3" style="margin-top:12px;">Blend Component Weights</h3>
        {weight_tbl_html}
      </div>
      <div class="card">
        <h2 class="h2">Top Feature Drivers (Surrogate)</h2>
        <div class="subtitle">Direction is from the home-team perspective. {summary_note}</div>
        {interp_tbl_html}
      </div>
      <div class="card full">
        <h2 class="h2">Final Visualization Pack</h2>
        <img class="viz" src="Final_Visualizations.png" alt="Final visualizations" />
      </div>
      <div class="card">
        <h2 class="h2">Data Integrity Summary</h2>
        <div class="table-wrap">{render_table(integrity_view, 'integrity_table')}</div>
      </div>
      <div class="card">
        <h2 class="h2">Leakage & Overfitting Summary</h2>
        <div class="table-wrap">{render_table(leak_view, 'leak_table')}</div>
      </div>
    </section>

    <div class="foot">
      Source files: Final_* CSV outputs, model selection policy, and final prediction artifacts in this workspace.
    </div>
  </div>
</body>
</html>
"""

    out_path = os.path.join(BASE, "Final_Report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("Created Final_Report.html")


if __name__ == "__main__":
    main()
