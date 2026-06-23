# Beta User Onboarding

## Goal

Run one causal growth analysis from CSV upload to recommendation.

## Demo Data

Use `examples/geo_lift_basic/data.csv`.

## Steps

1. Start the app with `streamlit run app.py`.
2. Upload the CSV.
3. Confirm column mapping:
   - Date: `date`
   - Geo: `geo`
   - Outcome: `outcome`
4. Open Geo Matcher and choose a test geo.
5. Review suggested controls and holdout geos.
6. Open Experiment Runner.
7. Select pre-period and post-period dates.
8. Run Measurement.
9. Review:
   - incremental lift
   - posterior distribution
   - evidence strength
   - economics
   - limitations and next action
10. Open Scorecard to confirm the result was recorded.

## What To Send Back

- A screenshot of the results
- Whether the recommendation was understandable
- Whether the selected controls looked plausible
- Any missing fields in your real CSV
