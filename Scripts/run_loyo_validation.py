from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

from Scripts.ror_estimation import (
    LAG_STEPS,
    REQUEST_YEARS,
    RESOLUTION,
    build_lagged_features,
    combine_years,
    evaluate_predictions,
    load_glofas_discharge,
    load_ror_generation,
    train_and_predict,
)


def save_loyo_figure(
    year: int,
    observed: pd.Series,
    estimated: pd.Series,
    metrics: dict[str, float],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(observed.index, observed, label="observed")
    ax.plot(estimated.index, estimated, label="estimated")
    ax.set_title(f"LOYO validation {year}", fontsize=15)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Generation (MWh)", fontsize=12)
    ax.legend(loc="upper right")
    ax.text(
        observed.index[0],
        0.9 * max(estimated.max(), observed.max()),
        f"NSE: {metrics['nse']:.2f} r: {metrics['corr']:.2f}",
        bbox={"boxstyle": "round", "fc": "none", "ec": "k"},
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"loyo_{year}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def run_loyo_validation(config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leave-one-year-out validation over request years with observed ROR data."""
    glofas = load_glofas_discharge(config.glofas_disc_path, RESOLUTION)
    ror = load_ror_generation(config.entsoe_ror_path, RESOLUTION)
    features_all = build_lagged_features(glofas, LAG_STEPS)
    available_years = set(features_all.index.year).intersection(set(ror.index.year))
    validation_years = [year for year in combine_years(REQUEST_YEARS) if year in available_years]
    if not validation_years:
        raise ValueError("No overlapping GloFAS/ROR years found for LOYO validation.")

    prediction_frames = []
    metric_rows = []

    for test_year in validation_years:
        training_years = [year for year in validation_years if year != test_year]
        training_features = features_all.loc[features_all.index.year.isin(training_years)]
        prediction_features = features_all.loc[features_all.index.year == test_year]

        y_pred, _, _, training_metrics = train_and_predict(training_features, ror, prediction_features)
        observed = ror.reindex(y_pred.index).iloc[:, 0].rename("observed").dropna()
        estimated = y_pred.reindex(observed.index).rename("estimated")
        test_metrics = evaluate_predictions(observed, estimated)

        prediction_frames.append(
            pd.concat([observed, estimated], axis=1).assign(validation_year=test_year)
        )
        metric_rows.append(
            {
                "validation_year": test_year,
                "training_nse": training_metrics["nse"],
                "training_corr": training_metrics["corr"],
                "training_rmse": training_metrics["rmse"],
                "test_nse": test_metrics["nse"],
                "test_corr": test_metrics["corr"],
                "test_rmse": test_metrics["rmse"],
                "test_nrmse": test_metrics["nrmse"],
                "test_mae": test_metrics["mae"],
            }
        )
        save_loyo_figure(test_year, observed, estimated, test_metrics, config.loyo_figure_dir)
        print(f"LOYO {test_year}: NSE = {test_metrics['nse']:.2f}, r = {test_metrics['corr']:.2f}")

    predictions = pd.concat(prediction_frames).sort_index()
    metrics = pd.DataFrame(metric_rows)

    predictions_path = Path(config.loyo_output_csv)
    metrics_path = Path(config.loyo_metrics_csv)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path)
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved LOYO predictions to {predictions_path}")
    print(f"Saved LOYO metrics to {metrics_path}")
    return predictions, metrics


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from main import CONFIG

    run_loyo_validation(CONFIG)
