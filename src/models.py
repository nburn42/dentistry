"""
Modeling Script
Runs the main empirical models for the tax volatility / credit rating paper.

Model 1: Ordered Logit — Rating level ~ Tax Volatility + Controls
Model 2: Logit — P(Downgrade next year) ~ Tax Volatility + Controls
Model 3: OLS — Rating level ~ Tax Volatility + Controls (robustness)
Model 4: Robustness — Alternative volatility windows and measures
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy import stats
from sklearn.metrics import classification_report, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

BASELINE_CONTROLS = ["roa", "leverage", "log_assets", "interest_coverage", "roa_vol_3yr"]


def load_panel():
    path = os.path.join(PROCESSED_DIR, "analysis_panel.csv")
    df = pd.read_csv(path)
    return df


def prepare_sample(df, y_col, x_tax_vol, controls, dropna=True):
    """Prepare regression sample by dropping NAs and adding constant."""
    cols = [y_col] + [x_tax_vol] + controls
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].copy()
    if dropna:
        sub = sub.dropna()
    if len(sub) < 50:
        return None, None, None
    y = sub[y_col]
    X = sub[[x_tax_vol] + [c for c in controls if c in sub.columns]]
    X = sm.add_constant(X)
    return y, X, sub


def run_ordered_logit(df, tax_vol_col="gaap_etr_vol_3yr", label="Model 1"):
    """Model 1: Ordered logit — rating level explained by tax volatility."""
    print(f"\n{'='*70}")
    print(f"{label}: Ordered Logit — Rating Level ~ Tax Volatility + Controls")
    print(f"{'='*70}")

    y, X, sub = prepare_sample(df, "rating_num", tax_vol_col, BASELINE_CONTROLS)
    if y is None:
        print("  Insufficient data for ordered logit.")
        return None

    try:
        model = OrderedModel(y, X, distr="logit")
        result = model.fit(method="bfgs", disp=False, maxiter=500)
        print(result.summary())

        # Save results
        os.makedirs(TABLES_DIR, exist_ok=True)
        outpath = os.path.join(TABLES_DIR, f"{label.replace(' ', '_').lower()}_ordered_logit.csv")
        params = pd.DataFrame({
            "coefficient": result.params,
            "std_error": result.bse,
            "z_value": result.tvalues,
            "p_value": result.pvalues,
        })
        params.to_csv(outpath)
        print(f"\nResults saved to {outpath}")

        # Key finding
        if tax_vol_col in result.params.index:
            coef = result.params[tax_vol_col]
            pval = result.pvalues[tax_vol_col]
            print(f"\n** Key Result: {tax_vol_col} coefficient = {coef:.4f}, "
                  f"p-value = {pval:.4f} **")
            if pval < 0.05:
                direction = "higher" if coef > 0 else "lower"
                print(f"   => Tax volatility is significantly associated with "
                      f"{direction} rating scores (worse credit).")
            else:
                print(f"   => Tax volatility is NOT statistically significant "
                      f"at the 5% level.")

        return result
    except Exception as e:
        print(f"  Ordered logit failed: {e}")
        return None


def run_ols(df, tax_vol_col="gaap_etr_vol_3yr", label="Model 3"):
    """Model 3: OLS — rating number ~ tax volatility + controls."""
    print(f"\n{'='*70}")
    print(f"{label}: OLS — Rating Level ~ Tax Volatility + Controls")
    print(f"{'='*70}")

    y, X, sub = prepare_sample(df, "rating_num", tax_vol_col, BASELINE_CONTROLS)
    if y is None:
        print("  Insufficient data for OLS.")
        return None

    model = sm.OLS(y, X)
    result = model.fit(cov_type="HC1")  # Heteroskedasticity-robust SEs
    print(result.summary())

    outpath = os.path.join(TABLES_DIR, f"{label.replace(' ', '_').lower()}_ols.csv")
    params = pd.DataFrame({
        "coefficient": result.params,
        "std_error": result.bse,
        "t_value": result.tvalues,
        "p_value": result.pvalues,
    })
    params.to_csv(outpath)
    print(f"Results saved to {outpath}")
    return result


def run_logit_downgrade(df, tax_vol_col="gaap_etr_vol_3yr", label="Model 2"):
    """Model 2: Logit — P(Downgrade t+1) ~ Tax Volatility at t + Controls."""
    print(f"\n{'='*70}")
    print(f"{label}: Logit — P(Downgrade Next Year) ~ Tax Volatility + Controls")
    print(f"{'='*70}")

    y, X, sub = prepare_sample(df, "downgrade_lead", tax_vol_col, BASELINE_CONTROLS)
    if y is None:
        print("  Insufficient data for logit.")
        return None

    y = y.astype(int)
    n_downgrade = y.sum()
    n_total = len(y)
    print(f"  Sample: {n_total} obs, {n_downgrade} downgrades "
          f"({100*n_downgrade/n_total:.1f}%)")

    if n_downgrade < 10:
        print("  Too few downgrades for reliable estimation.")
        return None

    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=False, maxiter=500)
        print(result.summary())

        # Marginal effects
        mfx = result.get_margeff()
        print("\nMarginal Effects (at means):")
        print(mfx.summary())

        # Save
        outpath = os.path.join(TABLES_DIR, f"{label.replace(' ', '_').lower()}_logit.csv")
        params = pd.DataFrame({
            "coefficient": result.params,
            "std_error": result.bse,
            "z_value": result.tvalues,
            "p_value": result.pvalues,
            "odds_ratio": np.exp(result.params),
        })
        params.to_csv(outpath)

        # ROC AUC
        y_pred = result.predict(X)
        try:
            auc = roc_auc_score(y, y_pred)
            print(f"\nROC AUC: {auc:.4f}")
        except ValueError:
            pass

        # Key finding
        if tax_vol_col in result.params.index:
            coef = result.params[tax_vol_col]
            pval = result.pvalues[tax_vol_col]
            odds = np.exp(coef)
            print(f"\n** Key Result: {tax_vol_col} coefficient = {coef:.4f}, "
                  f"odds ratio = {odds:.4f}, p-value = {pval:.4f} **")

        print(f"Results saved to {outpath}")
        return result
    except Exception as e:
        print(f"  Logit failed: {e}")
        return None


def run_robustness(df):
    """Model 4: Robustness checks with alternative measures."""
    print(f"\n{'='*70}")
    print("Model 4: Robustness Checks")
    print(f"{'='*70}")

    results = {}

    # 4a: 5-year window instead of 3-year
    print("\n--- 4a: 5-Year Tax Volatility Window ---")
    r = run_ols(df, tax_vol_col="gaap_etr_vol_5yr", label="Model 4a")
    results["5yr_window"] = r

    # 4b: Cash ETR instead of GAAP ETR
    print("\n--- 4b: Cash ETR Volatility ---")
    r = run_ols(df, tax_vol_col="cash_etr_vol_3yr", label="Model 4b")
    results["cash_etr"] = r

    # 4c: Tax-to-assets volatility
    print("\n--- 4c: Tax-to-Assets Volatility ---")
    r = run_ols(df, tax_vol_col="tax_to_assets_vol_3yr", label="Model 4c")
    results["tax_to_assets"] = r

    # 4d: Subsample — large firms only (above median assets)
    print("\n--- 4d: Large Firms Only (Above Median Assets) ---")
    median_assets = df["log_assets"].median()
    df_large = df[df["log_assets"] >= median_assets]
    r = run_ols(df_large, tax_vol_col="gaap_etr_vol_3yr", label="Model 4d")
    results["large_firms"] = r

    # 4e: Subsample — small firms
    print("\n--- 4e: Small Firms Only (Below Median Assets) ---")
    df_small = df[df["log_assets"] < median_assets]
    r = run_ols(df_small, tax_vol_col="gaap_etr_vol_3yr", label="Model 4e")
    results["small_firms"] = r

    return results


def run_event_study(df):
    """Compare tax volatility before vs. after rating change events."""
    print(f"\n{'='*70}")
    print("Event Study: Tax Volatility Around Rating Changes")
    print(f"{'='*70}")

    if "rating_change" not in df.columns or "gaap_etr_vol_3yr" not in df.columns:
        print("  Missing required columns.")
        return

    # Identify downgrade events
    downgrades = df[df["downgrade"] == 1][["ticker", "fiscal_year"]].copy()
    print(f"  Found {len(downgrades)} downgrade events.")

    if len(downgrades) < 10:
        print("  Too few events for event study.")
        return

    # For each downgrade, get tax vol in year before and year of downgrade
    results = []
    for _, event in downgrades.iterrows():
        t, fy = event["ticker"], event["fiscal_year"]
        pre = df[(df["ticker"] == t) & (df["fiscal_year"] == fy - 1)]
        post = df[(df["ticker"] == t) & (df["fiscal_year"] == fy)]
        if not pre.empty and not post.empty:
            pre_vol = pre["gaap_etr_vol_3yr"].values[0]
            post_vol = post["gaap_etr_vol_3yr"].values[0]
            if not np.isnan(pre_vol) and not np.isnan(post_vol):
                results.append({"pre_vol": pre_vol, "post_vol": post_vol})

    if len(results) < 10:
        print("  Too few matched events.")
        return

    res_df = pd.DataFrame(results)
    pre_mean = res_df["pre_vol"].mean()
    post_mean = res_df["post_vol"].mean()
    t_stat, p_val = stats.ttest_rel(res_df["post_vol"], res_df["pre_vol"])

    print(f"\n  Matched downgrade events: {len(res_df)}")
    print(f"  Mean tax vol BEFORE downgrade: {pre_mean:.4f}")
    print(f"  Mean tax vol AT downgrade:     {post_mean:.4f}")
    print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")

    if p_val < 0.05:
        print("  => Statistically significant difference in tax volatility "
              "around downgrade events.")
    else:
        print("  => No statistically significant difference.")


def run_all_models():
    os.makedirs(TABLES_DIR, exist_ok=True)
    df = load_panel()
    print(f"Loaded panel: {len(df)} obs, {df['ticker'].nunique()} firms.\n")

    # Main models
    run_ordered_logit(df)
    run_logit_downgrade(df)
    run_ols(df)

    # Robustness
    run_robustness(df)

    # Event study
    run_event_study(df)

    print(f"\n{'='*70}")
    print("All models complete. Results saved to results/tables/")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_all_models()
