"""
Feature Engineering Script
Computes rolling tax volatility measures for the panel dataset.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def compute_rolling_volatility(df, col, windows=(3, 5)):
    """Compute rolling std dev and coefficient of variation for a column."""
    df = df.sort_values(["ticker", "fiscal_year"])
    for w in windows:
        # Rolling standard deviation
        df[f"{col}_vol_{w}yr"] = (
            df.groupby("ticker")[col]
            .transform(lambda x: x.rolling(w, min_periods=w).std())
        )
        # Rolling coefficient of variation
        df[f"{col}_cv_{w}yr"] = (
            df.groupby("ticker")[col]
            .transform(
                lambda x: x.rolling(w, min_periods=w).std()
                / x.rolling(w, min_periods=w).mean().abs().clip(lower=1e-8)
            )
        )
        # Rolling range (max - min)
        df[f"{col}_range_{w}yr"] = (
            df.groupby("ticker")[col]
            .transform(
                lambda x: x.rolling(w, min_periods=w).max()
                - x.rolling(w, min_periods=w).min()
            )
        )
    return df


def compute_earnings_volatility(df, windows=(3, 5)):
    """Compute rolling earnings (ROA) volatility as a control."""
    df = df.sort_values(["ticker", "fiscal_year"])
    for w in windows:
        df[f"roa_vol_{w}yr"] = (
            df.groupby("ticker")["roa"]
            .transform(lambda x: x.rolling(w, min_periods=w).std())
        )
    return df


def compute_tax_level_controls(df, windows=(3, 5)):
    """Compute rolling mean ETR as level control (separate from volatility)."""
    df = df.sort_values(["ticker", "fiscal_year"])
    for w in windows:
        df[f"gaap_etr_mean_{w}yr"] = (
            df.groupby("ticker")["gaap_etr"]
            .transform(lambda x: x.rolling(w, min_periods=w).mean())
        )
        df[f"cash_etr_mean_{w}yr"] = (
            df.groupby("ticker")["cash_etr"]
            .transform(lambda x: x.rolling(w, min_periods=w).mean())
        )
    return df


def add_industry_codes(df):
    """Add 1-digit SIC sector as industry control (placeholder based on ticker hash)."""
    # In a real study, you'd merge SIC codes from Compustat or SEC filings.
    # Here we create a deterministic pseudo-industry based on ticker.
    np.random.seed(42)
    tickers = df["ticker"].unique()
    industry_map = {t: np.random.randint(1, 10) for t in tickers}
    df["industry"] = df["ticker"].map(industry_map)
    return df


def create_lead_variables(df):
    """Create forward-looking (lead) rating variables for prediction models."""
    df = df.sort_values(["ticker", "fiscal_year"])
    df["rating_num_lead"] = df.groupby("ticker")["rating_num"].shift(-1)
    df["downgrade_lead"] = (
        (df["rating_num_lead"] - df["rating_num"]) > 0
    ).astype(float)
    df["downgrade_lead"] = df["downgrade_lead"].where(
        df["rating_num_lead"].notna(), np.nan
    )
    return df


def run_feature_engineering():
    inpath = os.path.join(PROCESSED_DIR, "cleaned_panel.csv")
    df = pd.read_csv(inpath)
    print(f"Loaded cleaned panel: {len(df)} obs.")

    # Tax volatility measures
    df = compute_rolling_volatility(df, "gaap_etr")
    df = compute_rolling_volatility(df, "cash_etr")
    df = compute_rolling_volatility(df, "tax_to_assets")

    # Control volatilities
    df = compute_earnings_volatility(df)
    df = compute_tax_level_controls(df)

    # Industry (only add if not already present)
    if "industry" not in df.columns:
        df = add_industry_codes(df)

    # Lead variables for prediction
    df = create_lead_variables(df)

    outpath = os.path.join(PROCESSED_DIR, "analysis_panel.csv")
    df.to_csv(outpath, index=False)

    # Report coverage
    vol_cols = [c for c in df.columns if "_vol_" in c or "_cv_" in c]
    non_null = df[vol_cols].notna().sum()
    print(f"\nFeature coverage (non-null counts):")
    for col in vol_cols:
        print(f"  {col}: {non_null[col]} / {len(df)}")

    print(f"\nFinal panel: {len(df)} obs, {df['ticker'].nunique()} firms.")
    print(f"Saved to {outpath}")
    return df


if __name__ == "__main__":
    run_feature_engineering()
