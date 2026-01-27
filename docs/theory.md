# Methodology

OpenLift uses a **Bayesian Counterfactual** approach.

## The Model
We model the outcome $y_t$ in the test geo as a function of control geos $X_t$ and day-of-week effects.

$$ y_t \sim \text{StudentT}(\nu, \mu_t, \sigma) $$
$$ \mu_t = \alpha + X_t \beta + \text{dow}[t] $$

Where:
-   $\alpha$: Intercept
-   $\beta$: Coefficients for control geos (shrinkage priors $\mathcal{N}(0, 1)$)
-   $\text{dow}$: Day-of-week random effects

## Counterfactual
After fitting the model on the **PRE** period, we predict $\hat{y}_t$ for the **POST** period using the observed $X_t$ (control outcomes) in the post period.

The estimated lift at time $t$ is:
$$ \text{Lift}_t = y_t^{\text{obs}} - \hat{y}_t $$

## Uncertainty
By using PyMC, we generate thousands of posterior samples for $\alpha, \beta, \sigma$, allowing us to compute a full posterior distribution for the Lift. We report the 90% High Density Interval (HDI).
