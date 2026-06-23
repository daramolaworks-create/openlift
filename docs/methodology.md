# OpenLift Methodology

OpenLift estimates marketing incrementality by comparing observed outcomes in a
treated market against a counterfactual outcome predicted from matched control
markets.

## Data Shape

The minimum modelling dataset is daily or weekly geo-level data with:

- `date`
- `geo`
- `outcome`

The full decision schema adds:

- `spend`
- `treatment`
- `period`
- `channel`
- `campaign`
- optional creative metadata such as `creative_id`

## Experiment Design

Users define a pre-period and treatment period. The pre-period is used to learn
the relationship between the test geo and control geos. The treatment period is
used only for estimating the gap between observed and predicted outcomes.

Good experiments have:

- enough pre-period history to learn seasonality
- stable outcome measurement
- control markets that move similarly to the test market before treatment
- low risk of spillover or market contamination
- a treatment period long enough to detect the expected effect

## Market Matching

OpenLift ranks candidate control markets using time-series similarity methods:

- dynamic time warping
- Euclidean distance
- correlation distance

The matcher is a design aid, not proof of causal validity. Users should inspect
pre-period trend similarity and exclude contaminated or strategically different
markets.

## Counterfactual Model

The current engine uses a Bayesian regression model with:

- an intercept
- control-market predictors
- day-of-week effects
- optional holiday and weather covariates
- Student-t observation noise for robustness

The model is fit on the pre-period. Posterior samples are then projected into the
treatment period to estimate what would likely have happened without treatment.

## Lift Calculation

For each posterior sample, OpenLift calculates:

- predicted counterfactual outcome
- observed minus predicted lift
- percentage lift
- probability that lift is positive
- 90% highest-density intervals

The reported lift is the posterior mean, not a platform attribution number.

## Evidence Strength

Evidence strength combines:

- posterior probability of positive lift
- credible interval width
- optional match quality
- data quality warnings

Categories are Weak, Directional, Moderate, Strong, and Very Strong. These
categories are decision aids, not guarantees.

## Economics

When spend or another input metric is supplied, OpenLift translates lift into:

- incremental CAC
- incremental ROAS
- incremental profit

These estimates depend on user-provided margin and LTV assumptions.

## Limitations

OpenLift does not provide perfect attribution or guaranteed true ROAS. Results
can be weakened by bad control matches, short test windows, sparse conversions,
volatile revenue, spillover, tracking changes, and overlapping campaigns.

The correct interpretation is evidence-based: OpenLift estimates whether a
campaign likely caused incremental growth and what decision is reasonable under
uncertainty.
