"""
Save pipeline results as a Markdown file for easy viewing.
Reads CSV outputs from tables/ and generates a formatted .md summary.
"""

import os
from datetime import datetime
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def df_to_md_table(df):
    """Convert a DataFrame to a markdown table string."""
    lines = []
    headers = [""] + list(df.columns) if df.index.name or not isinstance(df.index, pd.RangeIndex) else list(df.columns)
    has_index = df.index.name or not isinstance(df.index, pd.RangeIndex)

    if has_index:
        header_row = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
        sep_row = "|" + "|".join(["---"] * (len(df.columns) + 1)) + "|"
    else:
        header_row = "| " + " | ".join(str(c) for c in df.columns) + " |"
        sep_row = "|" + "|".join(["---"] * len(df.columns)) + "|"

    lines.append(header_row)
    lines.append(sep_row)

    for idx, row in df.iterrows():
        vals = [str(v) for v in row.values]
        if has_index:
            lines.append("| " + " | ".join([str(idx)] + vals) + " |")
        else:
            lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def read_csv_safe(filename):
    """Read a CSV from tables dir, return None if missing."""
    path = os.path.join(TABLES_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


def save_results_markdown():
    """Generate a comprehensive markdown results file."""
    # Get panel info
    panel_path = os.path.join(PROCESSED_DIR, "analysis_panel.csv")
    panel_info = ""
    if os.path.exists(panel_path):
        df = pd.read_csv(panel_path)
        n_obs = len(df)
        n_firms = df["ticker"].nunique()
        yr_min = df["fiscal_year"].min()
        yr_max = df["fiscal_year"].max()
        panel_info = f"**Sample:** {n_obs:,} firm-year observations, {n_firms} firms, {yr_min}–{yr_max}"

    sections = []
    sections.append(f"# Pipeline Results: Tax Volatility & Credit Ratings\n")
    sections.append(f"**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if panel_info:
        sections.append(panel_info)
    sections.append("")
    sections.append("---\n")

    # Summary statistics
    stats = read_csv_safe("summary_statistics.csv")
    if stats is not None:
        sections.append("## Summary Statistics\n")
        sections.append(df_to_md_table(stats.round(4)))
        sections.append("\n---\n")

    # Summary by rating
    by_rating = read_csv_safe("summary_by_rating.csv")
    if by_rating is not None:
        sections.append("## Summary by Rating\n")
        sections.append(df_to_md_table(by_rating.round(4)))
        sections.append("\n---\n")

    # Model 2: Logit
    logit = read_csv_safe("model_2_logit.csv")
    if logit is not None:
        sections.append("## Model 2: Logit — P(Downgrade Next Year)\n")
        sections.append(df_to_md_table(logit.round(4)))
        sections.append("\n---\n")

    # Model 3: OLS
    ols = read_csv_safe("model_3_ols.csv")
    if ols is not None:
        sections.append("## Model 3: OLS — Rating Level (Robustness)\n")
        sections.append(df_to_md_table(ols.round(4)))
        sections.append("\n---\n")

    # Robustness checks
    robustness_files = [
        ("model_4a_ols.csv", "4a: 5-Year Tax Volatility Window"),
        ("model_4b_ols.csv", "4b: Cash ETR Volatility"),
        ("model_4c_ols.csv", "4c: Tax-to-Assets Volatility"),
        ("model_4d_ols.csv", "4d: Large Firms Only"),
        ("model_4e_ols.csv", "4e: Small Firms Only"),
    ]
    any_robustness = False
    for fname, title in robustness_files:
        rob = read_csv_safe(fname)
        if rob is not None:
            if not any_robustness:
                sections.append("## Model 4: Robustness Checks\n")
                any_robustness = True
            sections.append(f"### {title}\n")
            sections.append(df_to_md_table(rob.round(4)))
            sections.append("")

    if any_robustness:
        sections.append("\n---\n")

    # Figures listing
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    if os.path.exists(figures_dir):
        figs = [f for f in os.listdir(figures_dir) if f.endswith(".png")]
        if figs:
            sections.append("## Figures\n")
            sections.append("Generated in `results/figures/`:\n")
            for f in sorted(figs):
                sections.append(f"- `{f}`")
            sections.append("")

    md_content = "\n".join(sections)
    outpath = os.path.join(RESULTS_DIR, "pipeline_results.md")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(outpath, "w") as f:
        f.write(md_content)

    print(f"Results saved to {outpath}")


if __name__ == "__main__":
    save_results_markdown()
