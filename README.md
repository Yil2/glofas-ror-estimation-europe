# eu_ror_model
This is the repository of run-of-river daily generation estimation with referenced data from entose transparency platform, by GloFAS river discharge data.



## How to run

### 1. Test with included data

The repository includes a small `test data` folder for functionality testing. No API key, download, or path configuration is needed.

From the `eu_ror_model` folder, run:

```bash
python main.py
```

By default, `main.py` uses `TEST_CONFIG`, skips both data-fetching scripts, runs leave-one-year-out validation, and runs final reconstruction. Outputs are written to:

```text
test outputs/
```

### 2. Run with your own data
#### Before running
API tokens are necessary for running this code: [GloFAS](https://ewds.climate.copernicus.eu/how-to-api), [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide_prod_backup_06_11_2024.html)

Hydropower database should be downloaded: [JRC Hydro-power database](https://data.europa.eu/data/datasets/52b00441-d3e0-44e0-8281-fda86a63546d?locale=en)
Edit `CONFIG` in `main.py` with your own paths, API key, country code in JRC database, ENTSO-E code, and target years. Then call:

```python
main(use_test_data=False)
```

In this mode, the code runs the ENTSO-E and GloFAS fetching/processing steps by default, then runs validation and reconstruction.

### 3. Skip fetching if needed

If the fetched/processed files already exist and you only want to rerun the model, call:

```python
main(
    use_test_data=False,
    run_entsoe_fetch=False,
    run_glofas_fetch=False,
)
```
