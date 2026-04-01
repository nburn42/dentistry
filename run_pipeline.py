"""
Main Pipeline Runner
Executes the full research pipeline: data collection → cleaning → features → EDA → models
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.synthetic_data import generate_panel
from src.data_cleaning import run_cleaning
from src.features import run_feature_engineering
from src.eda import run_eda
from src.models import run_all_models


def main():
    print("=" * 70)
    print("RESEARCH PIPELINE: Tax Volatility & Credit Ratings")
    print("=" * 70)

    print("\n\n>>> STEP 1: Data Generation (Synthetic Panel)")
    print("-" * 50)
    generate_panel(n_firms=400, years=range(2005, 2025))

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

    print("\n\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nOutputs:")
    print("  data/raw/financial_data.csv        - Raw EDGAR data")
    print("  data/processed/cleaned_panel.csv   - Cleaned panel")
    print("  data/processed/analysis_panel.csv  - Panel with features")
    print("  results/tables/                    - Regression tables")
    print("  results/figures/                   - Plots and charts")


if __name__ == "__main__":
    main()
