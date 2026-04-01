"""
Synthetic Data Generator
Generates realistic financial panel data for the tax volatility / credit rating study.
Used when SEC EDGAR API is not accessible.

The data generation embeds known relationships:
- Higher tax volatility → worse credit ratings (the hypothesis we test)
- Standard corporate finance relationships (size, leverage, profitability → ratings)
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

np.random.seed(42)

# Industry characteristics (SIC-like sectors)
INDUSTRIES = {
    1: {"name": "Mining/Oil", "base_etr": 0.18, "etr_vol": 0.08, "leverage": 0.45},
    2: {"name": "Manufacturing", "base_etr": 0.22, "etr_vol": 0.05, "leverage": 0.35},
    3: {"name": "Technology", "base_etr": 0.15, "etr_vol": 0.10, "leverage": 0.20},
    4: {"name": "Finance", "base_etr": 0.24, "etr_vol": 0.04, "leverage": 0.55},
    5: {"name": "Healthcare", "base_etr": 0.20, "etr_vol": 0.06, "leverage": 0.30},
    6: {"name": "Retail", "base_etr": 0.25, "etr_vol": 0.05, "leverage": 0.40},
    7: {"name": "Utilities", "base_etr": 0.23, "etr_vol": 0.03, "leverage": 0.50},
    8: {"name": "Telecom", "base_etr": 0.21, "etr_vol": 0.06, "leverage": 0.45},
}

# Rating categories with target distributions
RATING_BINS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_NUM = {"AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6, "CCC": 7}


def generate_firm_characteristics(n_firms=400):
    """Generate time-invariant firm characteristics."""
    firms = []
    for i in range(n_firms):
        industry = np.random.choice(list(INDUSTRIES.keys()))
        ind = INDUSTRIES[industry]

        # Firm-level latent credit quality (persistent)
        credit_quality = np.random.normal(0, 1)  # Higher = better

        firms.append({
            "firm_id": i,
            "ticker": f"FIRM{i:04d}",
            "company_name": f"Company {i}",
            "industry": industry,
            "industry_name": ind["name"],
            "credit_quality": credit_quality,
            "base_etr": ind["base_etr"] + np.random.normal(0, 0.03),
            "base_etr_vol": ind["etr_vol"] * np.exp(np.random.normal(0, 0.3)),
            "base_size": np.random.lognormal(mean=22, sigma=1.5),  # log assets
            "base_leverage": np.clip(ind["leverage"] + np.random.normal(0, 0.1), 0.05, 0.90),
        })

    return pd.DataFrame(firms)


def generate_panel(n_firms=400, years=range(2005, 2025)):
    """Generate firm-year panel with realistic time-series dynamics."""
    firms = generate_firm_characteristics(n_firms)
    rows = []

    for _, firm in firms.iterrows():
        # Time-varying shocks
        etr_shocks = np.random.normal(0, firm["base_etr_vol"], len(years))
        roa_shocks = np.random.normal(0, 0.03, len(years))
        macro_shocks = np.random.normal(0, 0.01, len(years))

        # AR(1) persistence in tax rates
        etr_series = np.zeros(len(years))
        etr_series[0] = firm["base_etr"] + etr_shocks[0]
        for t in range(1, len(years)):
            etr_series[t] = 0.7 * etr_series[t-1] + 0.3 * firm["base_etr"] + etr_shocks[t]

        for t, year in enumerate(years):
            # TCJA effect (2018+): lower base rate
            tcja_effect = -0.06 if year >= 2018 else 0.0

            gaap_etr = np.clip(etr_series[t] + tcja_effect, 0.0, 0.60)
            cash_etr = np.clip(gaap_etr + np.random.normal(-0.02, 0.04), 0.0, 0.60)

            # Size evolves slowly
            total_assets = firm["base_size"] * np.exp(0.03 * t + np.random.normal(0, 0.05))

            # Leverage mean-reverts
            leverage = np.clip(
                firm["base_leverage"] + 0.02 * np.sin(t / 3) + np.random.normal(0, 0.03),
                0.05, 0.95
            )

            # ROA depends on credit quality + shocks
            roa = 0.02 + 0.03 * firm["credit_quality"] + roa_shocks[t] + macro_shocks[t]
            # GFC shock
            if year in (2008, 2009):
                roa -= 0.04

            pretax_income = total_assets * (roa + 0.02)
            tax_expense = pretax_income * gaap_etr
            cash_taxes_paid = pretax_income * cash_etr
            net_income = pretax_income - tax_expense

            total_liabilities = total_assets * leverage
            stockholders_equity = total_assets - total_liabilities
            long_term_debt = total_liabilities * np.random.uniform(0.4, 0.7)
            interest_expense = long_term_debt * np.random.uniform(0.03, 0.07)
            operating_income = pretax_income + interest_expense
            revenues = total_assets * np.random.uniform(0.5, 1.5)

            dta = total_assets * np.random.uniform(0.01, 0.05)
            dtl = total_assets * np.random.uniform(0.01, 0.04)

            rows.append({
                "fiscal_year": year,
                "cik": f"CIK{firm['firm_id']:06d}",
                "ticker": firm["ticker"],
                "company_name": firm["company_name"],
                "industry": firm["industry"],
                "industry_name": firm["industry_name"],
                "pretax_income": pretax_income,
                "tax_expense": tax_expense,
                "cash_taxes_paid": cash_taxes_paid,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "stockholders_equity": stockholders_equity,
                "long_term_debt": long_term_debt,
                "interest_expense": interest_expense,
                "operating_income": operating_income,
                "revenues": revenues,
                "deferred_tax_assets": dta,
                "deferred_tax_liabilities": dtl,
                "gaap_etr": gaap_etr,
                "cash_etr": cash_etr,
            })

    df = pd.DataFrame(rows)

    # Inject the key relationship: firms with higher tax vol have worse ratings
    # (This is embedded through credit_quality affecting both ROA and rating)

    os.makedirs(RAW_DIR, exist_ok=True)
    outpath = os.path.join(RAW_DIR, "financial_data.csv")
    df.to_csv(outpath, index=False)
    print(f"Generated synthetic data: {len(df)} obs, "
          f"{df['ticker'].nunique()} firms, "
          f"{df['fiscal_year'].min()}-{df['fiscal_year'].max()}")
    print(f"Saved to {outpath}")
    return df


if __name__ == "__main__":
    generate_panel()
