"""ML pipeline for AeroNet Lite using Kaggle Amazon Delivery Dataset.

Place Kaggle CSV here:
    aeronet_lite_interactive/data/raw/amazon_delivery.csv

If the CSV is missing, the code falls back to synthetic demo demand data so the
Streamlit app still runs during viva.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


@dataclass
class DemandModelResult:
    model_name: str
    mae: float
    rmse: float
    model: object
    test_frame: pd.DataFrame
    feature_names: list[str]
    dataset_source: str = "Unknown"


@dataclass
class AnomalyModelResult:
    model_name: str
    accuracy: float
    confusion: np.ndarray
    labels: list[str]
    model: object
    feature_names: list[str]
    sample_frame: pd.DataFrame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

AMAZON_CANDIDATES = [
    RAW_DATA_DIR / "amazon_delivery.csv",
    RAW_DATA_DIR / "Amazon_Delivery.csv",
    RAW_DATA_DIR / "amazon_delivery_dataset.csv",
    RAW_DATA_DIR / "amazon-delivery-dataset" / "amazon_delivery.csv",
]


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _safe_hour(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().mean() > 0.5:
        return dt.dt.hour.fillna(12).astype(int)

    # Handles simple strings like 14:30:00
    extracted = series.astype(str).str.extract(r"(\d{1,2})", expand=False)
    return pd.to_numeric(extracted, errors="coerce").fillna(12).clip(0, 23).astype(int)


def _encode_with_saved_mapping(values: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    le = LabelEncoder()
    text = values.astype(str).fillna("Unknown")
    encoded = le.fit_transform(text)
    mapping = {cls: int(i) for i, cls in enumerate(le.classes_)}
    return pd.Series(encoded, index=values.index), mapping


def make_synthetic_demand_data(n: int = 700, seed: int = 42) -> pd.DataFrame:
    """Fallback data if amazon_delivery.csv is not available."""
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    weekday = rng.integers(0, 7, n)
    traffic_level = rng.choice([1, 2, 3], size=n, p=[0.45, 0.38, 0.17])
    weather_level = rng.choice([1, 2, 3, 4], size=n, p=[0.55, 0.25, 0.15, 0.05])
    area_code = rng.choice([0, 1, 2, 3], size=n)
    category_code = rng.choice([0, 1, 2, 3, 4], size=n)
    zone_density_level = rng.choice([1, 2, 3], size=n, p=[0.25, 0.45, 0.30])

    rush_hour = ((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))
    weekend = weekday >= 5
    demand = (
        10
        + zone_density_level * 15
        + rush_hour.astype(int) * 18
        + weekend.astype(int) * 6
        + traffic_level * 5
        - weather_level * 2
        + rng.normal(0, 5, n)
    ).clip(2, None)

    return pd.DataFrame(
        {
            "hour": hour,
            "weekday": weekday,
            "traffic_level": traffic_level,
            "weather_level": weather_level,
            "area_code": area_code,
            "category_code": category_code,
            "zone_density_level": zone_density_level,
            "demand": np.round(demand, 2),
        }
    )


def load_amazon_delivery_demand(csv_path: Optional[str | Path] = None) -> tuple[pd.DataFrame, list[str], str]:
    """Load and convert Amazon Delivery records into a demand forecasting table.

    The Amazon dataset is record-level delivery data, not a ready-made demand
    time series. To use it for AeroNet Lite demand forecasting, we aggregate
    deliveries into demand units by hour, weekday, area, traffic, weather and
    product category.
    """
    path = Path(csv_path) if csv_path is not None else _first_existing(AMAZON_CANDIDATES)

    if path is None or not path.exists():
        df = make_synthetic_demand_data()
        features = ["hour", "weekday", "traffic_level", "weather_level", "area_code", "category_code", "zone_density_level"]
        return df, features, "Synthetic fallback data - amazon_delivery.csv not found"

    df = _clean_column_names(pd.read_csv(path))

    order_date_col = _find_col(df, ["Order_Date", "order_date", "Date"])
    order_time_col = _find_col(df, ["Order_Time", "order_time", "Pickup_Time", "pickup_time", "Time"])
    weather_col = _find_col(df, ["Weather", "weather"])
    traffic_col = _find_col(df, ["Traffic", "traffic"])
    area_col = _find_col(df, ["Area", "area"])
    category_col = _find_col(df, ["Category", "category"])
    vehicle_col = _find_col(df, ["Vehicle", "vehicle"])
    rating_col = _find_col(df, ["Agent_Rating", "agent_rating"])
    delivery_time_col = _find_col(df, ["Delivery_Time", "delivery_time"])

    if order_time_col is None:
        raise ValueError("Amazon Delivery CSV must contain Order_Time or Pickup_Time column.")

    df["hour"] = _safe_hour(df[order_time_col])

    if order_date_col is not None:
        dt = pd.to_datetime(df[order_date_col], errors="coerce")
        df["weekday"] = dt.dt.weekday.fillna(3).astype(int)
    else:
        # Fallback when date is absent: still keeps the app usable.
        df["weekday"] = 3

    if traffic_col is not None:
        traffic_map = {"low": 1, "medium": 2, "high": 3, "jam": 4, "very high": 4}
        df["traffic_level"] = df[traffic_col].astype(str).str.lower().map(traffic_map)
        df["traffic_level"] = df["traffic_level"].fillna(2).astype(int)
    else:
        df["traffic_level"] = 2

    if weather_col is not None:
        weather_map = {"clear": 1, "sunny": 1, "cloudy": 2, "fog": 2, "rain": 3, "storm": 4, "sandstorms": 4, "windy": 3}
        df["weather_level"] = df[weather_col].astype(str).str.lower().map(weather_map)
        df["weather_level"] = df["weather_level"].fillna(2).astype(int)
    else:
        df["weather_level"] = 2

    if area_col is not None:
        df["area_code"], area_mapping = _encode_with_saved_mapping(df[area_col])
    else:
        df["area_code"] = 0
        area_mapping = {}

    if category_col is not None:
        df["category_code"], category_mapping = _encode_with_saved_mapping(df[category_col])
    else:
        df["category_code"] = 0
        category_mapping = {}

    if vehicle_col is not None:
        df["vehicle_code"], vehicle_mapping = _encode_with_saved_mapping(df[vehicle_col])
    else:
        df["vehicle_code"] = 0
        vehicle_mapping = {}

    if rating_col is not None:
        df["agent_rating"] = pd.to_numeric(df[rating_col], errors="coerce").fillna(pd.to_numeric(df[rating_col], errors="coerce").median())
    else:
        df["agent_rating"] = 4.5

    if delivery_time_col is not None:
        df["avg_delivery_time"] = pd.to_numeric(df[delivery_time_col], errors="coerce")
    else:
        df["avg_delivery_time"] = np.nan

    group_cols = ["hour", "weekday", "traffic_level", "weather_level", "area_code", "category_code", "vehicle_code"]
    demand_df = (
        df.groupby(group_cols, dropna=False)
        .agg(
            demand=(group_cols[0], "size"),
            agent_rating=("agent_rating", "mean"),
            avg_delivery_time=("avg_delivery_time", "mean"),
        )
        .reset_index()
    )

    demand_df["agent_rating"] = demand_df["agent_rating"].fillna(4.5).round(2)
    demand_df["avg_delivery_time"] = demand_df["avg_delivery_time"].fillna(demand_df["avg_delivery_time"].median()).fillna(30).round(2)

    # Higher grouped order counts act like high-density delivery zones.
    # Higher grouped order counts act like high-density delivery zones.
# Fixed version: avoids qcut label/bin mismatch when demand values repeat.
    if len(demand_df) >= 3 and demand_df["demand"].nunique() >= 3:
        ranked_demand = demand_df["demand"].rank(method="first")

        demand_df["zone_density_level"] = (
            pd.qcut(
                ranked_demand,
                q=3,
                labels=False,
                duplicates="drop"
            ).astype(int) + 1
        )
    else:
        demand_df["zone_density_level"] = 2

    features = [
        "hour",
        "weekday",
        "traffic_level",
        "weather_level",
        "area_code",
        "category_code",
        "vehicle_code",
        "agent_rating",
        "avg_delivery_time",
        "zone_density_level",
    ]

    demand_df = demand_df[features + ["demand"]].dropna()
    return demand_df, features, f"Kaggle Amazon Delivery Dataset: {path.name}"


def train_demand_model(model_name: str = "Random Forest", seed: int = 42) -> DemandModelResult:
    df, features, dataset_source = load_amazon_delivery_demand()
    X = df[features]
    y = df["demand"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=120, random_state=seed, max_depth=10)

    model.fit(X_train, y_train)
    model.aeronet_feature_names_ = features
    model.aeronet_dataset_source_ = dataset_source

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    test_frame = X_test.copy()
    test_frame["actual_demand"] = y_test.values
    test_frame["predicted_demand"] = np.round(pred, 2)

    return DemandModelResult(
        model_name=model_name,
        mae=round(float(mae), 3),
        rmse=round(float(rmse), 3),
        model=model,
        test_frame=test_frame,
        feature_names=features,
        dataset_source=dataset_source,
    )


def predict_demand(model: object, hour: int, weekday: int, temperature: float, weather_code: int, zone_density_level: int) -> float:
    """Compatible with the existing Streamlit UI.

    The UI still sends temperature/weather/density. Amazon Delivery does not use
    temperature, so it is ignored. weather_code is mapped to weather_level.
    """
    feature_names = getattr(
        model,
        "aeronet_feature_names_",
        ["hour", "weekday", "traffic_level", "weather_level", "area_code", "category_code", "vehicle_code", "agent_rating", "avg_delivery_time", "zone_density_level"],
    )

    values = {
        "hour": hour,
        "weekday": weekday,
        "traffic_level": 2,
        "weather_level": int(weather_code) + 1 if int(weather_code) in [0, 1, 2, 3] else 2,
        "area_code": max(0, int(zone_density_level) - 1),
        "category_code": 0,
        "vehicle_code": 0,
        "agent_rating": 4.5,
        "avg_delivery_time": 30,
        "zone_density_level": zone_density_level,
    }

    sample = pd.DataFrame([{name: values.get(name, 0) for name in feature_names}])
    return float(model.predict(sample)[0])


# -----------------------------
# Synthetic anomaly detection
# -----------------------------

def make_synthetic_anomaly_data(n: int = 900, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(["Normal", "Battery anomaly", "Route anomaly", "Sensor spike"], size=n, p=[0.58, 0.16, 0.16, 0.10])
    rows = []
    for label in labels:
        if label == "Normal":
            battery_drop = rng.normal(3.0, 0.9)
            speed = rng.normal(8.0, 1.2)
            route_deviation = rng.normal(0.4, 0.25)
            altitude_change = rng.normal(1.2, 0.8)
        elif label == "Battery anomaly":
            battery_drop = rng.normal(13.0, 2.2)
            speed = rng.normal(7.5, 1.4)
            route_deviation = rng.normal(0.7, 0.3)
            altitude_change = rng.normal(1.5, 1.0)
        elif label == "Route anomaly":
            battery_drop = rng.normal(4.5, 1.2)
            speed = rng.normal(8.5, 1.5)
            route_deviation = rng.normal(4.0, 0.9)
            altitude_change = rng.normal(2.0, 1.0)
        else:
            battery_drop = rng.normal(5.0, 1.5)
            speed = rng.normal(16.0, 2.6)
            route_deviation = rng.normal(1.1, 0.5)
            altitude_change = rng.normal(8.0, 2.4)
        rows.append({
            "battery_drop": max(0, battery_drop),
            "speed": max(0, speed),
            "route_deviation": max(0, route_deviation),
            "altitude_change": max(0, altitude_change),
            "label": label,
        })
    return pd.DataFrame(rows).round(3)


def train_anomaly_model(model_name: str = "Random Forest", seed: int = 12) -> AnomalyModelResult:
    df = make_synthetic_anomaly_data(seed=seed)
    features = ["battery_drop", "speed", "route_deviation", "altitude_change"]
    X = df[features]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=seed)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=seed)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = ["Normal", "Battery anomaly", "Route anomaly", "Sensor spike"]
    cm = confusion_matrix(y_test, pred, labels=labels)
    acc = accuracy_score(y_test, pred)
    sample_frame = X_test.copy()
    sample_frame["actual"] = y_test.values
    sample_frame["predicted"] = pred
    return AnomalyModelResult(model_name, round(float(acc), 3), cm, labels, model, features, sample_frame)


def predict_anomaly(model: object, battery_drop: float, speed: float, route_deviation: float, altitude_change: float) -> str:
    sample = pd.DataFrame([{
        "battery_drop": battery_drop,
        "speed": speed,
        "route_deviation": route_deviation,
        "altitude_change": altitude_change,
    }])
    return str(model.predict(sample)[0])
