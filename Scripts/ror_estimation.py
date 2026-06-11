from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REQUEST_YEARS = (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
RESOLUTION = "d"
LAG_STEPS = range(1, 4)


def ensure_dataframe(data: pd.Series | pd.DataFrame, column_name: str) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        return data.to_frame(name=column_name)

    renamed = data.copy()
    if len(renamed.columns) == 1:
        renamed.columns = [column_name]
    return renamed


def normalize_by_column_max(data: pd.Series | pd.DataFrame) -> tuple[pd.Series | pd.DataFrame, pd.Series]:
    frame = ensure_dataframe(data, data.name or "value") if isinstance(data, pd.Series) else data.copy()
    scale = frame.max(axis=0).replace(0, 1.0)
    normalized = frame.divide(scale, axis=1)

    if isinstance(data, pd.Series):
        return normalized.iloc[:, 0], scale
    return normalized, scale


def normalize_with_scale(data: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    return data.divide(scale.replace(0, 1.0), axis=1)


def combine_years(*year_groups) -> list[int]:
    years = set()
    for year_group in year_groups:
        if year_group is None:
            continue
        if isinstance(year_group, (str, int)):
            year_group = [year_group]
        for year in year_group:
            try:
                years.add(int(year))
            except (TypeError, ValueError):
                continue
    return sorted(years)


def build_lagged_features(data: pd.DataFrame, lags: range) -> pd.DataFrame:
    frames = [data]
    for lag in lags:
        shifted = data.shift(lag)
        shifted.columns = [f"{col}_lag_{lag}" for col in data.columns]
        frames.append(shifted)
    return pd.concat(frames, axis=1).dropna()


def build_random_forest(_feature_count: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=800,
        max_features="sqrt",
        max_depth=9,
        min_samples_leaf=1,
        max_samples=0.8,
        bootstrap=True,
        n_jobs=1,
        random_state=42,
    )


def read_timeseries_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True, sep=None, engine="python")


def load_ror_generation(path: str | Path, resolution: str) -> pd.DataFrame:
    ror = read_timeseries_csv(path).dropna()
    if "Run of River Generation" in ror.columns:
        ror = ror[["Run of River Generation"]]
    elif len(ror.columns) == 1:
        ror.columns = ["Run of River Generation"]

    ror.index = pd.to_datetime(ror.index, utc=True)
    ror = ror.resample("h").mean()

    if resolution != "h":
        return ror.resample(resolution).sum()
    return ror


def load_glofas_discharge(path: str | Path, resolution: str) -> pd.DataFrame:
    glofas = read_timeseries_csv(path)
    glofas.index = pd.to_datetime(glofas.index, utc=True)

    if resolution != "h":
        return glofas.resample(resolution).sum()
    return glofas.resample(resolution).mean().bfill()


def align_inputs(features: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = max(features.index[0], target.index[0])
    end = min(features.index[-1], target.index[-1])
    features = features.loc[start:end].dropna(axis=1).dropna(axis=0)
    target = target.loc[start:end].dropna(axis=0)

    common_index = features.index.intersection(target.index)
    return features.loc[common_index], target.loc[common_index]


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    nrmse = float(rmse / y_true.mean() * 100)
    corr, _ = pearsonr(y_true, y_pred)
    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "nse": float(r2_score(y_true, y_pred)),
        "corr": float(corr),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_and_predict(
    training_features: pd.DataFrame,
    training_target: pd.DataFrame,
    prediction_features: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, float]]:
    if training_features.empty:
        raise ValueError("No GloFAS discharge rows found for training.")
    if prediction_features.empty:
        raise ValueError("No GloFAS discharge rows found for prediction.")

    training_features, training_target = align_inputs(training_features, training_target)
    if training_features.empty or training_target.empty:
        raise ValueError("No overlapping GloFAS/ROR rows found for training.")

    normalized_training_features, feature_scale = normalize_by_column_max(training_features)
    normalized_prediction_features = normalize_with_scale(prediction_features, feature_scale)
    normalized_target, target_scale = normalize_by_column_max(training_target)

    target_column = normalized_target.columns[0]
    target_series = normalized_target[target_column].interpolate(method="linear")
    target_scale_value = float(target_scale[target_column])

    model = build_random_forest(normalized_training_features.shape[1])
    model.fit(normalized_training_features, target_series)

    y_pred = pd.Series(
        model.predict(normalized_prediction_features),
        index=normalized_prediction_features.index,
        name="estimated",
    ).clip(lower=0) * target_scale_value

    train_pred = pd.Series(
        model.predict(normalized_training_features),
        index=target_series.index,
        name="training_estimated",
    ).clip(lower=0) * target_scale_value
    train_true = target_series.rename("training_observed") * target_scale_value
    metrics = evaluate_predictions(train_true, train_pred)
    return y_pred, train_true, train_pred, metrics
