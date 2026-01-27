# OpenLift

OpenLift is an open-source tool for measuring marketing incrementality using geo-lift methodology with Bayesian counterfactuals. It handles experiment design, data validation, model fitting, and reporting.

## Why OpenLift?

Marketing data is noisy. "Day of Week" seasonality, holidays, and random variance make it hard to know if an ad campaign *actually* worked or if you just got lucky. OpenLift uses **Bayesian Synthetic Control** to create a robust baseline, giving you:
-   **True Incremental Lift**: How many *extra* sales did you get?
-   **Confidence Intervals**: Are we 90% sure or 50% sure?
-   **Seasonality Correction**: Automatically handles weekly patterns.

## Features

- **Bayesian Engine**: Powered by PyMC for robust probabilistic inference.
- **Drag & Drop UI**: Built-in Streamlit app for non-technical users.
- **Strategic Insights**: Auto-calculates Elasticity, Efficiency (CPI), and correlations.
- **Transparent**: See exactly which control markets were used to predict your baseline.

## Installation

### Prerequisites
- Python 3.11+
- Poetry (recommended)

### Setup
```bash
git clone https://github.com/yourusername/openlift
cd openlift
poetry install
```

## Usage

### Option A: The UI (Recommended)
Launch the interactive dashboard to run experiments without writing code.
```bash
poetry run streamlit run app.py
```
1.  Drag & drop your CSV.
2.  Select your Test Geo and Control Geos.
3.  Get a full strategic report.

### Option B: The CLI
For data pipelines and automation.
```bash
# 1. Initialize config
poetry run openlift init

# 2. Run experiment
poetry run openlift run experiment.yaml --out results.json

# 3. View report
poetry run openlift report results.json
```
