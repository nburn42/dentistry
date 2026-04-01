"""
Data Collection Script
Pulls financial data from SEC EDGAR XBRL API for tax and credit rating analysis.
"""

import json
import os
import time
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

HEADERS = {"User-Agent": "TaxCreditResearch research@example.com"}

# Key XBRL tags for our analysis
XBRL_TAGS = {
    "IncomeTaxExpenseBenefit": "tax_expense",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "pretax_income",
    "IncomeTaxesPaidNet": "cash_taxes_paid",
    "Assets": "total_assets",
    "Liabilities": "total_liabilities",
    "LongTermDebt": "long_term_debt",
    "LongTermDebtAndCapitalLeaseObligations": "long_term_debt_alt",
    "NetIncomeLoss": "net_income",
    "Revenues": "revenues",
    "StockholdersEquity": "stockholders_equity",
    "DeferredTaxAssetsNet": "deferred_tax_assets",
    "DeferredTaxLiabilitiesNoncurrent": "deferred_tax_liabilities",
    "InterestExpense": "interest_expense",
    "OperatingIncomeLoss": "operating_income",
    "CommonStockSharesOutstanding": "shares_outstanding",
}


def get_company_tickers():
    """Get list of all SEC-filing companies with CIK numbers."""
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame.from_dict(data, orient="index")
    df.columns = ["cik", "ticker", "company_name"]
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    return df


def fetch_company_facts(cik):
    """Fetch all XBRL facts for a company from SEC EDGAR."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def extract_annual_data(facts, tag, label):
    """Extract annual (10-K) values for a given XBRL tag."""
    try:
        us_gaap = facts["facts"]["us-gaap"][tag]["units"]["USD"]
    except (KeyError, TypeError):
        return pd.DataFrame()

    rows = []
    for entry in us_gaap:
        form = entry.get("form", "")
        if form != "10-K":
            continue
        rows.append(
            {
                "fiscal_year": entry.get("fy"),
                "fiscal_period": entry.get("fp"),
                "filed": entry.get("filed"),
                label: entry.get("val"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Keep only FY (full year) entries, deduplicate by fiscal_year
    df = df[df["fiscal_period"] == "FY"]
    df = df.drop_duplicates(subset=["fiscal_year"], keep="last")
    df = df[["fiscal_year", label]]
    return df


def build_company_panel(cik, ticker, company_name):
    """Build a panel of annual financial data for one company."""
    facts = fetch_company_facts(cik)
    if facts is None:
        return pd.DataFrame()

    dfs = []
    for tag, label in XBRL_TAGS.items():
        df = extract_annual_data(facts, tag, label)
        if not df.empty:
            dfs.append(df)

    if len(dfs) < 3:  # Need at least a few variables
        return pd.DataFrame()

    # Merge all variables on fiscal_year
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="fiscal_year", how="outer")

    merged["cik"] = cik
    merged["ticker"] = ticker
    merged["company_name"] = company_name
    return merged


def collect_data(n_companies=500, min_year=2005):
    """
    Collect financial data for top N companies by filing frequency.
    Focuses on large-cap companies most likely to have credit ratings.
    """
    print("Fetching company tickers from SEC...")
    tickers = get_company_tickers()

    # Take top N companies (ordered roughly by market cap in SEC file)
    tickers = tickers.head(n_companies)
    print(f"Will attempt to collect data for {len(tickers)} companies.")

    all_panels = []
    success = 0
    for i, row in tickers.iterrows():
        cik, ticker, name = row["cik"], row["ticker"], row["company_name"]
        try:
            panel = build_company_panel(cik, ticker, name)
            if not panel.empty:
                panel = panel[panel["fiscal_year"] >= min_year]
                if len(panel) >= 3:  # Need at least 3 years
                    all_panels.append(panel)
                    success += 1

            if (int(i) + 1) % 25 == 0:
                print(f"  Processed {int(i)+1}/{len(tickers)} "
                      f"({success} successful)")

            # Rate limit: SEC asks for max 10 requests/sec
            time.sleep(0.12)

        except Exception as e:
            print(f"  Error for {ticker} ({cik}): {e}")
            time.sleep(1)
            continue

    if not all_panels:
        print("ERROR: No data collected.")
        return pd.DataFrame()

    result = pd.concat(all_panels, ignore_index=True)
    print(f"\nCollected data for {success} companies, "
          f"{len(result)} firm-year observations.")

    os.makedirs(RAW_DIR, exist_ok=True)
    outpath = os.path.join(RAW_DIR, "financial_data.csv")
    result.to_csv(outpath, index=False)
    print(f"Saved to {outpath}")
    return result


if __name__ == "__main__":
    collect_data(n_companies=500, min_year=2005)
