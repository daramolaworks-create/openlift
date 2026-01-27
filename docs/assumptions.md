# Assumptions and Limitations

## Critical Assumptions
1.  **Parallel Trends**: The key assumption of DiD methods. We assume that in the absence of treatment, the test geo would have evolved similarly to the synthetic control (weighted combination of control geos).
2.  **No Spillover**: Marketing in the test geo (Lagos) does not affect outcomes in control geos.
3.  **Stable Relationship**: The relationship between test and control geos observed in the PRE period remains constant in the POST period (barring the treatment effect).

## Limitations of MVP
-   Single test geo only.
-   No automatic outlier detection.
-   Linear relationship assumed (though Bayesian regression handles uncertainty well).
-   No holidays/special events logic beyond day-of-week.
