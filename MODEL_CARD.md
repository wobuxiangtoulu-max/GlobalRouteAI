# GlobalRoute AI delay classifier

## Intended use

This prototype predicts whether a parcel shipment will arrive later than its promised transit time. It compares carriers for the same planned shipment and estimates a decision cost:

`quoted shipping cost + predicted delay probability × user-defined delay penalty`

## Models

- Logistic Regression provides an interpretable baseline.
- Random Forest captures non-linear effects and interactions.
- CatBoost handles category-heavy logistics data and is expected to become the main candidate once real data is available.

The training command evaluates every model on the newest 20% of shipments. The split is chronological rather than random.

## Prediction-time features

The model uses only information available before dispatch: carrier, origin, destination, route, service level, package properties, pickup time, distance, warehouse load, recent carrier performance, forecast weather severity, holiday period, quote, promise, month, and weekday.

Actual delivery date, transit time, delay days, final status, and delay reason are excluded to prevent outcome leakage.

## Current limitations

- The bundled dataset is simulated. Its metrics validate code flow, not commercial accuracy.
- Carrier quotes and promises use simplified deterministic formulas.
- External weather, traffic, tracking, and contract APIs are not connected.
- The recent carrier on-time rate is a trailing historical aggregate and must be computed with strict timestamp controls in production.
- A production release needs customer-specific validation, probability calibration, drift monitoring, access controls, and model-version rollback.

## Promotion rule for a real model

Do not deploy a newly trained model only because its ROC-AUC is higher. Require acceptable PR-AUC, probability calibration, delay recall at the chosen operating threshold, and simulated business benefit on a later untouched time period.

