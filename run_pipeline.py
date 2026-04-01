"""
Main Pipeline Runner
Executes the full research pipeline: data collection → cleaning → features → EDA → models
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_collection import collect_data
from src.data_cleaning import run_cleaning
from src.features import run_feature_engineering
from src.eda import run_eda
from src.models import run_all_models
from src.save_results_md import save_results_markdown


def main():
    print("=" * 70)
    print("RESEARCH PIPELINE: Tax Volatility & Credit Ratings")
    print("=" * 70)

    print("\n\n>>> STEP 1: Data Collection (SEC EDGAR)")
    print("-" * 50)
    collect_data(n_companies=500, min_year=2005)

    print("\n\n>>> STEP 2: Data Cleaning")
    print("-" * 50)
    run_cleaning()

    print("\n\n>>> STEP 3: Feature Engineering")
    print("-" * 50)
    run_feature_engineering()

    print("\n\n>>> STEP 4: Exploratory Data Analysis")
    print("-" * 50)
    run_eda()

    print("\n\n>>> STEP 5: Statistical Models")
    print("-" * 50)
    run_all_models()

    print("\n\n>>> STEP 6: Save Results as Markdown")
    print("-" * 50)
    save_results_markdown()

    print("\n\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nOutputs:")
    print("  data/raw/financial_data.csv        - Raw EDGAR data")
    print("  data/processed/cleaned_panel.csv   - Cleaned panel")
    print("  data/processed/analysis_panel.csv  - Panel with features")
    print("  results/tables/                    - Regression tables (CSV)")
    print("  results/figures/                   - Plots and charts")
    print("  results/pipeline_results.md        - Full results summary")


if __name__ == "__main__":
    main()
