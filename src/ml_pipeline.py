"""Simple, explainable ML pipeline for demand forecasting and anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


@dataclass
class DemandModelResult:
    model_name: str
    mae: float
    rmse: float
    model: object
    test_frame: pd.DataFrame
    feature_names: list[str]


@dataclass
class AnomalyModelResult:
    model_name: str
    accuracy: float
    confusion: np.ndarray
    labels: list[str]
    model: object
    feature_names: list[str]
    sample_frame: pd.DataFrame


def make_synthetic_demand_data(n: int = 700, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    weekday = rng.integers(0, 7, n)
    temperature = rng.normal(27, 7, n).clip(5, 45)
    weather = rng.choice([0, 1, 2], size=n, p=[0.62, 0.28, 0.10])  # clear, cloudy, rainy
    zone_density = rng.choice([1, 2, 3], size=n, p=[0.25, 0.45, 0.30])

    rush_hour = ((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))
    weekend = weekday >= 5
    demand = (
        12
        + zone_density * 16
        + rush_hour.astype(int) * 20
        + weekend.astype(int) * 6
        + np.maximum(temperature - 25, 0) * 0.9
        - weather * 4
        + rng.normal(0, 6, n)
    ).clip(2, None)

    return pd.DataFrame(
        {
            "hour": hour,
            "weekday": weekday,
            "temperature": np.round(temperature, 2),
            "weather_code": weather,
            "zone_density_level": zone_density,
            "demand": np.round(demand, 2),
        }
    )


def train_demand_model(model_name: str = "Random Forest", seed: int = 42) -> DemandModelResult:
    df = make_synthetic_demand_data(seed=seed)
    features = ["hour", "weekday", "temperature", "weather_code", "zone_density_level"]
    X = df[features]
    y = df["demand"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=90, random_state=seed, max_depth=8)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    test_frame = X_test.copy()
    test_frame["actual_demand"] = y_test.values
    test_frame["predicted_demand"] = np.round(pred, 2)
    return DemandModelResult(model_name, round(mae, 3), round(rmse, 3), model, test_frame, features)


def predict_demand(model: object, hour: int, weekday: int, temperature: float, weather_code: int, zone_density_level: int) -> float:
    sample = pd.DataFrame(
        [
            {
                "hour": hour,
                "weekday": weekday,
                "temperature": temperature,
                "weather_code": weather_code,
                "zone_density_level": zone_density_level,
            }
        ]
    )
    return float(model.predict(sample)[0])


def make_synthetic_anomaly_data(n: int = 900, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(
        ["Normal", "Battery anomaly", "Route anomaly", "Sensor spike"],
        size=n,
        p=[0.58, 0.16, 0.16, 0.10],
    )
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
        rows.append(
            {
                "battery_drop": max(0, battery_drop),
                "speed": max(0, speed),
                "route_deviation": max(0, route_deviation),
                "altitude_change": max(0, altitude_change),
                "label": label,
            }
        )
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
    sample = pd.DataFrame(
        [
            {
                "battery_drop": battery_drop,
                "speed": speed,
                "route_deviation": route_deviation,
                "altitude_change": altitude_change,
            }
        ]
    )
    return str(model.predict(sample)[0])
