"""Create like-for-like carrier scenarios for a planned shipment."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .config import CARRIERS, CARRIER_BASE_ON_TIME_RATE
from .modeling import predict_probability
from .simulation import calculate_promised_days, estimate_distance_km, quote_shipping_cost


def recent_carrier_rates(history: pd.DataFrame, as_of_date: date, lookback_days: int = 90) -> dict[str, float]:
    """Calculate trailing rates using outcomes that would have been known by the decision date."""
    if "is_delayed" not in history.columns or "ship_date" not in history.columns:
        return CARRIER_BASE_ON_TIME_RATE.copy()
    dated = history.copy()
    dated["ship_date"] = pd.to_datetime(dated["ship_date"], errors="coerce")
    end = pd.Timestamp(as_of_date)
    start = end - pd.Timedelta(days=lookback_days)
    window = dated[(dated["ship_date"] < end) & (dated["ship_date"] >= start)]
    if window.empty:
        window = dated[dated["ship_date"] < end]
    rates: dict[str, float] = {}
    for carrier in CARRIERS:
        carrier_rows = window[window["carrier"] == carrier]
        if len(carrier_rows) >= 20:
            rates[carrier] = float(1.0 - pd.to_numeric(carrier_rows["is_delayed"]).mean())
        else:
            rates[carrier] = CARRIER_BASE_ON_TIME_RATE[carrier]
    return rates


def build_carrier_scenarios(
    artifact: dict[str, Any],
    history: pd.DataFrame,
    *,
    ship_date: date,
    origin: str,
    destination: str,
    service_level: str,
    package_weight_kg: float,
    package_volume_cm3: float,
    order_value_eur: float,
    fragile: bool,
    pickup_hour: int,
    warehouse_load_pct: float,
    weather_severity: int,
    holiday_period: bool,
    delay_penalty_eur: float,
) -> pd.DataFrame:
    if origin == destination:
        raise ValueError("Origin and destination must be different")
    distance = estimate_distance_km(origin, destination)
    carrier_rates = recent_carrier_rates(history, ship_date)
    rows: list[dict[str, Any]] = []
    for carrier in CARRIERS:
        promised_days = calculate_promised_days(origin, destination, service_level, carrier, distance)
        quote = quote_shipping_cost(
            carrier,
            service_level,
            package_weight_kg,
            package_volume_cm3,
            distance,
            fragile,
        )
        rows.append(
            {
                "ship_date": ship_date.isoformat(),
                "carrier": carrier,
                "origin": origin,
                "destination": destination,
                "route": f"{origin}-{destination}",
                "service_level": service_level,
                "package_weight_kg": package_weight_kg,
                "package_volume_cm3": package_volume_cm3,
                "order_value_eur": order_value_eur,
                "fragile": int(fragile),
                "pickup_hour": pickup_hour,
                "distance_km": distance,
                "warehouse_load_pct": warehouse_load_pct,
                "carrier_recent_on_time_rate": carrier_rates[carrier],
                "weather_severity": weather_severity,
                "holiday_period": int(holiday_period),
                "shipping_cost": quote,
                "promised_days": promised_days,
            }
        )
    scenarios = pd.DataFrame(rows)
    scenarios["delay_probability"] = predict_probability(artifact, scenarios)
    scenarios["expected_decision_cost"] = (
        scenarios["shipping_cost"] + scenarios["delay_probability"] * delay_penalty_eur
    )
    scenarios["risk_level"] = pd.cut(
        scenarios["delay_probability"],
        bins=[-np.inf, 0.20, 0.40, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    scenarios["recommended"] = False
    scenarios.loc[scenarios["expected_decision_cost"].idxmin(), "recommended"] = True
    return scenarios.sort_values("expected_decision_cost", ignore_index=True)

