"""
Data Cleaning Script
Cleans raw financial data and constructs derived variables.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def load_raw_data():
    path = os.path.join(RAW_DIR, "financial_data.csv")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df['ticker'].nunique()} firms.")
    return df


def clean_data(df):
    """Clean and construct key variables."""

    # Drop rows missing critical fields
    df = df.dropna(subset=["fiscal_year", "ticker"])
    df["fiscal_year"] = df["fiscal_year"].astype(int)

    # Consolidate long-term debt (use primary, fall back to alt)
    if "long_term_debt" in df.columns and "long_term_debt_alt" in df.columns:
        df["long_term_debt"] = df["long_term_debt"].fillna(df["long_term_debt_alt"])
    df = df.drop(columns=["long_term_debt_alt"], errors="ignore")

    # --- Derived variables ---

    # GAAP Effective Tax Rate (compute only if not already present)
    if "gaap_etr" not in df.columns or df["gaap_etr"].isna().all():
        df["gaap_etr"] = np.where(
            (df["pretax_income"].notna()) & (df["pretax_income"] > 0),
            df["tax_expense"] / df["pretax_income"],
            np.nan,
        )

    # Cash Effective Tax Rate
    if "cash_etr" not in df.columns or df["cash_etr"].isna().all():
        df["cash_etr"] = np.where(
            (df["pretax_income"].notna()) & (df["pretax_income"] > 0),
            df["cash_taxes_paid"] / df["pretax_income"],
            np.nan,
        )

    # ROA
    df["roa"] = np.where(
        (df["total_assets"].notna()) & (df["total_assets"] > 0),
        df["net_income"] / df["total_assets"],
        np.nan,
    )

    # Leverage
    df["leverage"] = np.where(
        (df["total_assets"].notna()) & (df["total_assets"] > 0),
        df["total_liabilities"] / df["total_assets"],
        np.nan,
    )

    # Log total assets (firm size)
    df["log_assets"] = np.where(
        (df["total_assets"].notna()) & (df["total_assets"] > 0),
        np.log(df["total_assets"]),
        np.nan,
    )

    # Interest coverage ratio
    df["interest_coverage"] = np.where(
        (df["interest_expense"].notna()) & (df["interest_expense"] > 0),
        df["operating_income"] / df["interest_expense"],
        np.nan,
    )

    # Tax expense to assets ratio
    df["tax_to_assets"] = np.where(
        (df["total_assets"].notna()) & (df["total_assets"] > 0),
        df["tax_expense"] / df["total_assets"],
        np.nan,
    )

    return df


def winsorize(df, cols, lower=0.01, upper=0.99):
    """Winsorize specified columns at given percentiles."""
    for col in cols:
        if col in df.columns:
            lo = df[col].quantile(lower)
            hi = df[col].quantile(upper)
            df[col] = df[col].clip(lo, hi)
    return df


def assign_synthetic_ratings(df):
    """
    Assign synthetic credit ratings based on financial health indicators.
    This approximates actual ratings using known determinants:
    - ROA, leverage, interest coverage, size, earnings stability.

    In a real study you'd use actual S&P/Moody's ratings from WRDS or Kaggle.
    """
    # Compute a credit score (higher = better credit)
    # Based on Altman Z-score inspired weighting
    df = df.copy()

    # Normalize components (per firm-year, cross-sectional)
    def zscore(s):
        return (s - s.mean()) / s.std()

    score = pd.Series(0.0, index=df.index)

    if "roa" in df.columns:
        score += 0.30 * zscore(df["roa"].fillna(0))
    if "leverage" in df.columns:
        score -= 0.25 * zscore(df["leverage"].fillna(0.5))
    if "interest_coverage" in df.columns:
        capped = df["interest_coverage"].clip(-50, 50).fillna(0)
        score += 0.20 * zscore(capped)
    if "log_assets" in df.columns:
        score += 0.15 * zscore(df["log_assets"].fillna(df["log_assets"].median()))
    if "gaap_etr" in df.columns:
        # Moderate ETR is better (very low = aggressive, very high = problems)
        etr_dev = (df["gaap_etr"].fillna(0.25) - 0.25).abs()
        score -= 0.10 * zscore(etr_dev)

    df["credit_score"] = score

    # Map to rating buckets
    bins = [-np.inf, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, np.inf]
    labels = ["CCC", "B", "BB", "BBB", "A", "AA", "AAA", "AAA"]
    # Numerical mapping (higher = worse credit)
    rating_num_map = {"AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6, "CCC": 7}

    df["rating"] = pd.cut(df["credit_score"], bins=bins, labels=labels, ordered=False)
    df["rating"] = df["rating"].astype(str)
    df["rating_num"] = df["rating"].map(rating_num_map)

    # Rating changes (within each firm, year-over-year)
    df = df.sort_values(["ticker", "fiscal_year"])
    df["rating_num_lag"] = df.groupby("ticker")["rating_num"].shift(1)
    df["rating_change"] = df["rating_num"] - df["rating_num_lag"]
    # +1 = downgrade (worse), -1 = upgrade (better), 0 = no change
    df["downgrade"] = (df["rating_change"] > 0).astype(int)
    df["upgrade"] = (df["rating_change"] < 0).astype(int)

    return df


def run_cleaning():
    df = load_raw_data()
    df = clean_data(df)

    # Winsorize key ratios
    ratio_cols = ["gaap_etr", "cash_etr", "roa", "leverage",
                  "interest_coverage", "tax_to_assets"]
    df = winsorize(df, ratio_cols)

    df = assign_synthetic_ratings(df)

    # Drop firm-years with too little data
    key_cols = ["gaap_etr", "roa", "leverage", "log_assets", "rating_num"]
    df = df.dropna(subset=key_cols)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    outpath = os.path.join(PROCESSED_DIR, "cleaned_panel.csv")
    df.to_csv(outpath, index=False)
    print(f"Cleaned panel: {len(df)} obs, {df['ticker'].nunique()} firms.")
    print(f"Years: {df['fiscal_year'].min()}-{df['fiscal_year'].max()}")
    print(f"Saved to {outpath}")
    return df


if __name__ == "__main__":
    run_cleaning()
