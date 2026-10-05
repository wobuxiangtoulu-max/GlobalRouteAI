"""Leakage-safe feature preparation shared by training and inference."""

from __future__ import annotations

import pandas as pd


TARGET = "is_delayed"

BASE_FEATURES = [
    "carrier",
    "origin",
    "destination",
    "route",
    "service_level",
    "package_weight_kg",
    "package_volume_cm3",
    "order_value_eur",
    "fragile",
    "pickup_hour",
    "distance_km",
    "warehouse_load_pct",
    "carrier_recent_on_time_rate",
    "weather_severity",
    "holiday_period",
    "shipping_cost",
    "promised_days",
]

CATEGORICAL_FEATURES = [
    "carrier",
    "origin",
    "destination",
    "route",
    "service_level",
    "ship_month",
    "ship_weekday",
]

NUMERIC_FEATURES = [
    "package_weight_kg",
    "package_volume_cm3",
    "order_value_eur",
    "fragile",
    "pickup_hour",
    "distance_km",
    "warehouse_load_pct",
    "carrier_recent_on_time_rate",
    "weather_severity",
    "holiday_period",
    "shipping_cost",
    "promised_days",
    "is_weekend",
    "peak_season",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

LEAKAGE_COLUMNS = {
    "delivery_date",
    "promised_delivery_date",
    "transit_days",
    "delay_days",
    "delay_reason",
    "status",
}


def validate_source_data(df: pd.DataFrame, require_target: bool = True) -> None:
    required = {"ship_date", *BASE_FEATURES}
    if require_target:
        required.add(TARGET)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def prepare_features(df: pd.DataFrame, require_target: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
    validate_source_data(df, require_target=require_target)
    working = df.copy()
    ship_date = pd.to_datetime(working["ship_date"], errors="coerce")
    if ship_date.isna().any():
        raise ValueError("ship_date contains invalid or missing dates")

    working["ship_month"] = ship_date.dt.month.astype(str).str.zfill(2)
    working["ship_weekday"] = ship_date.dt.day_name()
    working["is_weekend"] = ship_date.dt.dayofweek.isin([5, 6]).astype(int)
    working["peak_season"] = ship_date.dt.month.isin([8, 11, 12]).astype(int)

    X = working[MODEL_FEATURES].copy()
    y = None
    if require_target:
        y = pd.to_numeric(working[TARGET], errors="raise").astype(int)
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError("is_delayed must contain only 0 and 1")
    return X, y

