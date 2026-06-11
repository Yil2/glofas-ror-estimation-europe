from dataclasses import dataclass
from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = ROOT_DIR / "test data"
TEST_OUTPUT_DIR = ROOT_DIR / "test outputs"


@dataclass(frozen=True)
class ModelConfig:
    # Shared data paths. These should match the output paths in the fetch scripts.
    glofas_disc_path: Path = Path("your_processed_discharge_path")
    entsoe_ror_path: Path = Path("your_ror_data_path")

    # Outputs.
    reconstruction_output_csv: Path = Path("your_ror_estimates_path")
    loyo_output_csv: Path = Path("your_loyo_predictions_path")
    loyo_metrics_csv: Path = Path("your_loyo_metrics_path")
    loyo_figure_dir: Path = Path("your_loyo_figure_dir")

    # ENTSO-E fetch configuration.
    entsoe_api_key: str = "your_api_key"
    entsoe_area: str = "your_electricity_price_area"
    entsoe_start_time: pd.Timestamp | str = "your_ror_start_time"
    entsoe_end_time: pd.Timestamp | str = "your_ror_end_time"

    # GloFAS fetch configuration.
    glofas_raw_dir: Path = Path("your_glofas_path")
    jrc_database_path: Path = Path("your_jrc_db_path")
    glofas_points_path: Path = Path("your_glofas_points_path")
    country_code: str = "your country code"
    entsoe_zone_code: str = "your entsoe code"
    zone_date: pd.Timestamp = pd.Timestamp("2024-01-01")

    # User-defined reconstruction years.
    target_years: tuple[int | str, ...] = ("your target years for ROR reconstruction",)


CONFIG = ModelConfig()

TEST_CONFIG = ModelConfig(
    glofas_disc_path=TEST_DATA_DIR / "AT_glofas_discharge.csv",
    entsoe_ror_path=TEST_DATA_DIR / "AT_historical_ror_generation.csv",
    reconstruction_output_csv=TEST_OUTPUT_DIR / "AT_reconstruction.csv",
    loyo_output_csv=TEST_OUTPUT_DIR / "AT_loyo_predictions.csv",
    loyo_metrics_csv=TEST_OUTPUT_DIR / "AT_loyo_metrics.csv",
    loyo_figure_dir=TEST_OUTPUT_DIR / "figures",
    target_years=(2024,),
)


def main(
    use_test_data: bool = True,
    run_entsoe_fetch: bool | None = None,
    run_glofas_fetch: bool | None = None,
    run_loyo: bool = True,
    run_reconstruction: bool = True,
) -> None:
    config = TEST_CONFIG if use_test_data else CONFIG

    if run_entsoe_fetch is None:
        run_entsoe_fetch = not use_test_data
    if run_glofas_fetch is None:
        run_glofas_fetch = not use_test_data

    if run_entsoe_fetch:
        from Scripts.entsoe_ror_fetch import fetch_entsoe_ror_generation

        fetch_entsoe_ror_generation(
            api_token=config.entsoe_api_key,
            bidding_zone=config.entsoe_area,
            start=config.entsoe_start_time,
            end=config.entsoe_end_time,
            output_path=config.entsoe_ror_path,
        )

    if run_glofas_fetch:
        from Scripts.glofas_fetch import get_dis_inputs

        get_dis_inputs(
            glofas_raw_dir=config.glofas_raw_dir,
            jrc_database_path=config.jrc_database_path,
            output_disc_path=config.glofas_disc_path,
            output_points_path=config.glofas_points_path,
            country=config.country_code,
            bidding_zone=config.entsoe_zone_code,
            bidding_zone_date=config.zone_date,
            target_years=config.target_years,
        )

    if run_loyo:
        from Scripts.run_loyo_validation import run_loyo_validation
        run_loyo_validation(config)

    if run_reconstruction:
        from Scripts.run_final_reconstruction import run_final_reconstruction
        run_final_reconstruction(config)


if __name__ == "__main__":
    main()
