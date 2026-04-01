# Project Guidelines

## Run Results
- Always save pipeline run results as Markdown (.md) files so they can be viewed later without re-running the pipeline.
- The `save_results_md.py` module in `src/` generates `results/pipeline_results.md` from CSV outputs.
- When adding new models or tables, update `save_results_md.py` to include them in the markdown output.

## Data
- Data is collected from SEC EDGAR XBRL API (requires network access). There is no synthetic data fallback.
- If network is unavailable, the user will run `python run_pipeline.py` locally.

## Setup
- `pip install -r requirements.txt`
- `python run_pipeline.py` runs the full pipeline
