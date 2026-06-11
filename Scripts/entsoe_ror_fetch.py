from pathlib import Path
import sys

import pandas as pd


def fetch_entsoe_ror_generation(
    api_token: str,
    bidding_zone: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    output_path: str | Path,
    psr_type: str = "B11",
) -> pd.DataFrame:
    """Download ENTSO-E run-of-river generation and save it as a CSV file.
    """
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=api_token)
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    output_path = Path(output_path)

    ror_requested = client.query_generation(
        bidding_zone,
        start=start,
        end=end,
        psr_type=psr_type,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ror_requested.to_csv(output_path)
    print(f"Retrieve run-of-river generation data: {bidding_zone} ---> Finished")
    return ror_requested


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from main import CONFIG

    fetch_entsoe_ror_generation(
        api_token=CONFIG.entsoe_api_key,
        bidding_zone=CONFIG.entsoe_area,
        start=CONFIG.entsoe_start_time,
        end=CONFIG.entsoe_end_time,
        output_path=CONFIG.entsoe_ror_path,
    )
