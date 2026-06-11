from pathlib import Path
import sys

import pandas as pd

from Scripts.ror_estimation import (
    LAG_STEPS,
    REQUEST_YEARS,
    RESOLUTION,
    build_lagged_features,
    combine_years,
    load_glofas_discharge,
    load_ror_generation,
    train_and_predict,
)


def run_final_reconstruction(config) -> pd.DataFrame:
    """Train on request years and reconstruct ROR generation for target years."""
    glofas = load_glofas_discharge(config.glofas_disc_path, RESOLUTION)
    ror = load_ror_generation(config.entsoe_ror_path, RESOLUTION)

    training_years = combine_years(REQUEST_YEARS)
    target_years = combine_years(config.target_years)
    features_all = build_lagged_features(glofas, LAG_STEPS)

    training_features = features_all.loc[features_all.index.year.isin(training_years)]
    prediction_features = features_all.loc[features_all.index.year.isin(target_years)]

    y_pred, _, _, metrics = train_and_predict(training_features, ror, prediction_features)
    estimates = y_pred.to_frame()

    observed_target = ror.reindex(y_pred.index)
    if observed_target.notna().any().any():
        estimates = pd.concat(
            [observed_target.rename(columns={observed_target.columns[0]: "observed"}), estimates],
            axis=1,
        )

    output_path = Path(config.reconstruction_output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(output_path)

    print(
        f"Saved target-year ROR reconstruction to {output_path} | "
        f"Training NSE: {metrics['nse']:.2f}, r: {metrics['corr']:.2f}, RMSE: {metrics['rmse']:.2f}"
    )
    return estimates


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from main import CONFIG

    run_final_reconstruction(CONFIG)
