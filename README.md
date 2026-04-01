# Tax Volatility as a Predictor of Credit Rating Changes

## Research Question

Does higher tax expense volatility predict future credit rating downgrades? This project builds a firm-year panel dataset and tests whether firms with more volatile effective tax rates experience worse credit outcomes.

## Hypotheses

- **H1**: Higher tax volatility is associated with lower (worse) credit ratings in cross-section
- **H2**: Tax volatility at time *t* predicts credit rating downgrades at time *t+1*

## Project Structure

```
├── README.md
├── requirements.txt
├── run_pipeline.py              # Main entry point — runs all steps
├── src/
│   ├── data_collection.py       # SEC EDGAR XBRL API scraper
│   ├── synthetic_data.py        # Synthetic data generator (fallback)
│   ├── data_cleaning.py         # Cleaning, derived variables, synthetic ratings
│   ├── features.py              # Rolling tax volatility measures
│   ├── eda.py                   # Summary stats, correlations, plots
│   └── models.py                # Ordered logit, logit, OLS, robustness, event study
├── data/
│   ├── raw/                     # Raw financial data (generated or downloaded)
│   └── processed/               # Cleaned panel, analysis-ready panel
└── results/
    ├── tables/                  # Regression output CSVs
    └── figures/                 # Plots (PNG)
```

## Setup

```bash
pip install -r requirements.txt
```

### Network Access (for real SEC EDGAR data)

The data collection script pulls from SEC EDGAR's XBRL API. If running on Claude Code on the web:

1. Go to **claude.ai/code** → environment settings
2. Change **Network access** from "Limited" to **"Full"**
3. Save and restart the session

Without network access, the pipeline falls back to a synthetic data generator that embeds realistic financial relationships.

## Running the Pipeline

```bash
python run_pipeline.py
```

This runs all 5 steps in order:

1. **Data Collection** — Pulls 10-K financial data for ~500 firms from SEC EDGAR (or generates synthetic data)
2. **Data Cleaning** — Constructs derived variables (ETR, ROA, leverage), winsorizes outliers, assigns credit ratings
3. **Feature Engineering** — Computes rolling 3-year and 5-year tax volatility measures
4. **Exploratory Data Analysis** — Summary statistics, correlation matrix, box plots, time series charts
5. **Statistical Models** — Runs all regression models and saves results

## Data Sources

| Source | Variables | Access |
|--------|-----------|--------|
| SEC EDGAR XBRL API | Tax expense, pre-tax income, assets, liabilities, debt, cash taxes paid | Free (requires network) |
| Kaggle (alternative) | Corporate credit ratings with S&P ratings | Free download |
| WRDS/Compustat (ideal) | Full Compustat panel + actual S&P/Moody's ratings | University subscription |

### Key Variables Collected

- **Tax**: tax expense, cash taxes paid, deferred tax assets/liabilities
- **Income**: pre-tax income, net income, operating income, revenues
- **Balance sheet**: total assets, total liabilities, stockholders' equity, long-term debt
- **Other**: interest expense, shares outstanding

## Variable Definitions

### Dependent Variables

| Variable | Definition |
|----------|------------|
| `rating_num` | Numerical credit rating (1=AAA, 2=AA, ..., 7=CCC) |
| `downgrade` | Binary: 1 if rating worsened from prior year |
| `downgrade_lead` | Binary: 1 if rating worsens next year (for prediction models) |

### Tax Volatility Measures (Key Independent Variables)

| Variable | Definition |
|----------|------------|
| `gaap_etr_vol_3yr` | Rolling 3-year std dev of GAAP effective tax rate |
| `gaap_etr_vol_5yr` | Rolling 5-year std dev of GAAP effective tax rate |
| `cash_etr_vol_3yr` | Rolling 3-year std dev of cash effective tax rate |
| `cash_etr_vol_5yr` | Rolling 5-year std dev of cash effective tax rate |
| `gaap_etr_cv_3yr` | Rolling 3-year coefficient of variation of GAAP ETR |
| `gaap_etr_range_3yr` | Rolling 3-year range (max - min) of GAAP ETR |
| `tax_to_assets_vol_3yr` | Rolling 3-year std dev of tax expense / total assets |

### Control Variables

| Variable | Definition |
|----------|------------|
| `roa` | Return on assets (net income / total assets) |
| `leverage` | Total liabilities / total assets |
| `log_assets` | Log of total assets (firm size proxy) |
| `interest_coverage` | Operating income / interest expense |
| `roa_vol_3yr` | Rolling 3-year std dev of ROA (earnings volatility) |
| `gaap_etr_mean_3yr` | Rolling 3-year mean GAAP ETR (tax level control) |
| `industry` | Industry sector code |

## Empirical Models

### Model 1: Ordered Logit (Cross-sectional rating levels)

```
Rating_it = β₀ + β₁·TaxVol_it + γ·Controls_it + ε_it
```

Tests whether tax volatility explains the cross-sectional distribution of credit ratings. Ordered logit is appropriate because ratings are ordinal (AAA > AA > A > ...).

### Model 2: Logit (Predictive — downgrade probability)

```
P(Downgrade_{t+1}) = Λ(β₀ + β₁·TaxVol_t + γ·Controls_t)
```

Tests whether tax volatility at time *t* predicts the probability of a rating downgrade at *t+1*. Reports odds ratios, marginal effects, and ROC AUC.

### Model 3: OLS (Robustness)

```
RatingNum_it = β₀ + β₁·TaxVol_it + γ·Controls_it + ε_it
```

Linear model with heteroskedasticity-robust (HC1) standard errors. Serves as a robustness check on the ordered logit.

### Model 4: Robustness Checks

| Variant | Description |
|---------|-------------|
| 4a | 5-year volatility window instead of 3-year |
| 4b | Cash ETR volatility instead of GAAP ETR |
| 4c | Tax-to-assets volatility |
| 4d | Large firms only (above median total assets) |
| 4e | Small firms only (below median total assets) |

### Event Study

Compares mean tax volatility in the year before vs. the year of a rating downgrade event. Uses paired t-tests on matched downgrade observations.

## Expected Outputs

### Tables (results/tables/)

- `summary_statistics.csv` — Descriptive statistics for all key variables
- `summary_by_rating.csv` — Mean values by rating category
- `correlation_matrix.csv` — Pairwise correlations
- `model_1_ordered_logit.csv` — Ordered logit coefficients
- `model_2_logit.csv` — Logit coefficients with odds ratios
- `model_3_ols.csv` — OLS coefficients
- `model_4a-4e_ols.csv` — Robustness check coefficients

### Figures (results/figures/)

- `correlation_matrix.png` — Heatmap of variable correlations
- `tax_vol_by_rating.png` — Box plots of tax volatility by rating bucket
- `downgrade_comparison.png` — Tax volatility for downgraded vs. non-downgraded firms
- `time_series.png` — Average tax volatility and ratings over time

## TODO: Improvements for Publication Quality

### Data
- [ ] Replace synthetic data with real SEC EDGAR data (enable network access)
- [ ] Merge actual S&P/Moody's credit ratings from WRDS or Kaggle
- [ ] Add SIC codes from EDGAR filings for real industry classification
- [ ] Extend panel to include 2024 filings

### Methodology
- [ ] Add firm fixed effects and year fixed effects to panel models
- [ ] Instrument for endogeneity (e.g., state tax rate changes as IV)
- [ ] Cluster standard errors at the firm level
- [ ] Add Fama-MacBeth regressions as alternative specification
- [ ] Include market-to-book ratio as additional control
- [ ] Add VIF multicollinearity diagnostics

### Paper
- [ ] Write introduction with motivation and contribution
- [ ] Literature review: tax avoidance (Hanlon & Heitzman 2010), credit ratings (Kisgen 2006)
- [ ] Formal hypothesis development section
- [ ] Discussion of economic magnitude (not just statistical significance)
- [ ] Address limitations (synthetic ratings, survivorship bias, endogeneity)

## Key References

- Hanlon, M., & Heitzman, S. (2010). A review of tax research. *Journal of Accounting and Economics*, 50(2-3), 127-178.
- Kisgen, D. J. (2006). Credit ratings and capital structure. *Journal of Finance*, 61(3), 1035-1072.
- Hasan, I., Hoi, C. K., Wu, Q., & Zhang, H. (2014). Beauty is in the eye of the beholder: The effect of corporate tax avoidance on the cost of bank loans. *Journal of Financial Economics*, 113(1), 109-130.
- Goh, B. W., Lee, J., Lim, C. Y., & Shevlin, T. (2016). The effect of corporate tax avoidance on the cost of equity. *The Accounting Review*, 91(6), 1647-1670.
- Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance*, 23(4), 589-609.
