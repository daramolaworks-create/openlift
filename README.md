# OpenLift

[![Tests](https://github.com/daramolaworks-create/openlift/actions/workflows/tests.yml/badge.svg)](https://github.com/daramolaworks-create/openlift/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.9--3.12-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-black)](LICENSE)

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
- **Decision Output**: Produces evidence strength, limitations, next action, and Markdown/HTML reports.
- **Next Experiment Planner**: Suggests test geo, controls, MDE, duration, and required input.
- **Incrementality Scorecard**: Tracks cumulative evidence across local experiment runs.

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

## Methodology

OpenLift estimates what would likely have happened without treatment, then
compares that counterfactual with observed post-period outcomes. Read the
methodology guide in [docs/methodology.md](docs/methodology.md).

## Beta User Walkthrough

1. Launch the app with `streamlit run app.py`.
2. Upload `examples/geo_lift_basic/data.csv`.
3. Map `date`, `geo`, and `outcome`.
4. Use Geo Matcher to select controls.
5. Run Measurement and review lift, posterior distribution, economics, and next action.
6. Open Scorecard to see the experiment added to cumulative memory.

## Demo Notebook

A runnable notebook is available at
[examples/notebooks/01_geo_lift_demo.ipynb](examples/notebooks/01_geo_lift_demo.ipynb).
