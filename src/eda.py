"""
Exploratory Data Analysis Script
Generates summary statistics, correlation analysis, and visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def load_panel():
    path = os.path.join(PROCESSED_DIR, "analysis_panel.csv")
    df = pd.read_csv(path)
    print(f"Loaded analysis panel: {len(df)} obs, {df['ticker'].nunique()} firms.")
    return df


def summary_statistics(df):
    """Generate and save summary statistics table."""
    key_vars = [
        "gaap_etr", "cash_etr", "roa", "leverage", "log_assets",
        "interest_coverage", "rating_num",
        "gaap_etr_vol_3yr", "gaap_etr_vol_5yr",
        "cash_etr_vol_3yr", "cash_etr_vol_5yr",
        "roa_vol_3yr", "roa_vol_5yr",
    ]
    key_vars = [v for v in key_vars if v in df.columns]

    stats = df[key_vars].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats = stats.round(4)

    os.makedirs(TABLES_DIR, exist_ok=True)
    outpath = os.path.join(TABLES_DIR, "summary_statistics.csv")
    stats.to_csv(outpath)
    print(f"\nSummary Statistics:")
    print(stats.to_string())
    print(f"\nSaved to {outpath}")
    return stats


def summary_by_rating(df):
    """Summary statistics by credit rating category."""
    if "rating" not in df.columns:
        return

    key_vars = ["gaap_etr", "gaap_etr_vol_3yr", "roa", "leverage", "log_assets"]
    key_vars = [v for v in key_vars if v in df.columns]

    rating_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    df_sub = df[df["rating"].isin(rating_order)]

    grouped = df_sub.groupby("rating")[key_vars].mean()
    grouped = grouped.reindex(rating_order)
    grouped = grouped.round(4)

    outpath = os.path.join(TABLES_DIR, "summary_by_rating.csv")
    grouped.to_csv(outpath)
    print(f"\nMean Values by Rating:")
    print(grouped.to_string())
    print(f"\nSaved to {outpath}")


def correlation_matrix(df):
    """Correlation matrix of key variables."""
    key_vars = [
        "gaap_etr_vol_3yr", "cash_etr_vol_3yr", "roa_vol_3yr",
        "rating_num", "roa", "leverage", "log_assets", "interest_coverage",
    ]
    key_vars = [v for v in key_vars if v in df.columns]

    corr = df[key_vars].corr().round(3)

    outpath = os.path.join(TABLES_DIR, "correlation_matrix.csv")
    corr.to_csv(outpath)

    # Plot
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f",
                square=True, ax=ax)
    ax.set_title("Correlation Matrix: Tax Volatility & Credit Rating Variables")
    plt.tight_layout()
    figpath = os.path.join(FIGURES_DIR, "correlation_matrix.png")
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f"\nCorrelation matrix saved to {figpath}")


def plot_tax_vol_by_rating(df):
    """Box plot of tax volatility by rating category."""
    if "gaap_etr_vol_3yr" not in df.columns:
        return

    rating_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    df_sub = df[df["rating"].isin(rating_order)].copy()
    df_sub["rating"] = pd.Categorical(df_sub["rating"], categories=rating_order,
                                       ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # GAAP ETR volatility
    sns.boxplot(data=df_sub, x="rating", y="gaap_etr_vol_3yr", ax=axes[0],
                order=rating_order)
    axes[0].set_title("GAAP ETR Volatility (3yr) by Credit Rating")
    axes[0].set_ylabel("Std Dev of GAAP ETR")
    axes[0].set_xlabel("Credit Rating")

    # Cash ETR volatility
    if "cash_etr_vol_3yr" in df.columns:
        sns.boxplot(data=df_sub, x="rating", y="cash_etr_vol_3yr", ax=axes[1],
                    order=rating_order)
        axes[1].set_title("Cash ETR Volatility (3yr) by Credit Rating")
        axes[1].set_ylabel("Std Dev of Cash ETR")
        axes[1].set_xlabel("Credit Rating")

    plt.tight_layout()
    figpath = os.path.join(FIGURES_DIR, "tax_vol_by_rating.png")
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f"Box plots saved to {figpath}")


def plot_downgrade_comparison(df):
    """Compare tax volatility for firms that were downgraded vs. not."""
    if "downgrade" not in df.columns or "gaap_etr_vol_3yr" not in df.columns:
        return

    df_sub = df.dropna(subset=["gaap_etr_vol_3yr", "downgrade"])

    fig, ax = plt.subplots(figsize=(8, 6))
    df_sub["Downgraded"] = df_sub["downgrade"].map({1: "Yes", 0: "No"})
    sns.boxplot(data=df_sub, x="Downgraded", y="gaap_etr_vol_3yr", ax=ax)
    ax.set_title("GAAP ETR Volatility: Downgraded vs. Non-Downgraded Firms")
    ax.set_ylabel("3-Year Std Dev of GAAP ETR")

    # Add t-test result
    from scipy import stats
    grp_yes = df_sub[df_sub["downgrade"] == 1]["gaap_etr_vol_3yr"].dropna()
    grp_no = df_sub[df_sub["downgrade"] == 0]["gaap_etr_vol_3yr"].dropna()
    if len(grp_yes) > 5 and len(grp_no) > 5:
        t_stat, p_val = stats.ttest_ind(grp_yes, grp_no, equal_var=False)
        ax.text(0.05, 0.95, f"t = {t_stat:.2f}, p = {p_val:.4f}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    figpath = os.path.join(FIGURES_DIR, "downgrade_comparison.png")
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f"Downgrade comparison saved to {figpath}")


def plot_time_series(df):
    """Time series of average tax volatility."""
    if "gaap_etr_vol_3yr" not in df.columns:
        return

    yearly = df.groupby("fiscal_year").agg(
        mean_gaap_etr_vol=("gaap_etr_vol_3yr", "mean"),
        mean_rating_num=("rating_num", "mean"),
        n_firms=("ticker", "nunique"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(yearly["fiscal_year"], yearly["mean_gaap_etr_vol"],
             "b-o", label="Avg GAAP ETR Vol (3yr)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Avg Tax Volatility", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(yearly["fiscal_year"], yearly["mean_rating_num"],
             "r--s", label="Avg Rating (higher=worse)")
    ax2.set_ylabel("Avg Rating Score", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    ax1.set_title("Tax Volatility and Credit Ratings Over Time")
    plt.tight_layout()
    figpath = os.path.join(FIGURES_DIR, "time_series.png")
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    print(f"Time series plot saved to {figpath}")


def run_eda():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_panel()
    summary_statistics(df)
    summary_by_rating(df)
    correlation_matrix(df)
    plot_tax_vol_by_rating(df)
    plot_downgrade_comparison(df)
    plot_time_series(df)
    print("\nEDA complete.")


if __name__ == "__main__":
    run_eda()
