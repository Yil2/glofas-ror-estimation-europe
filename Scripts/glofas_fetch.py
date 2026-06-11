
import pandas as pd
import numpy as np
import xarray as xr
import os
import sys
from pathlib import Path
from glob import glob


REQUEST_YEAR=[2015, 2016, 2017, 2018, 2019, 2020,
                2021, 2022, 2023, 2024, 2025]
REQUEST_MONTH=["01","02","03","04","05","06","07","08","09","10","11","12"]


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


def glofas_request(
    glofas_raw_dir: str | Path,
    years: list[int] = REQUEST_YEAR,
    target_years: list[int] | tuple[int | str, ...] | None = None,
    months: list[str] = REQUEST_MONTH,
) -> None:
    """Download monthly GloFAS historical discharge NetCDF files.
    """
    datapath = Path(glofas_raw_dir)
    years = combine_years(years, target_years)
    os.makedirs(datapath, exist_ok=True)
    for year in years:
        for month in months:
            import cdsapi

            dataset = "cems-glofas-historical"
            request = {
                "system_version": ["version_4_0"],
                "hydrological_model": ["lisflood"],
                "product_type": ["consolidated"],
                "variable": ["river_discharge_in_the_last_24_hours"],
                "hyear": str(year),
                "hmonth": month,
                "hday": [
                    "01", "02", "03","04", "05", "06", "07", "08", "09", "10", "11", "12",
                    "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
                    "25","26","27","28","29","30","31"
                ],
                "data_format": "netcdf4",
                "download_format": "unarchived",
                "area": [72, -25, 34, 45]
            }

            client = cdsapi.Client(url="https://ewds.climate.copernicus.eu/api")
            
            client.retrieve(dataset, request, os.path.join(datapath,f"{year}_{month}_00utc.nc"))
            print(f"Retrieve {year}_{month} successfully") 

#===========================================read Glofas disc data============================================
def sjoin_gdf(gdf1, gdf2):
    gdf=gdf1.sjoin(gdf2)
    if 'index_right' in gdf.columns:
        gdf=gdf.drop(columns='index_right')

    if 'index_left' in gdf.columns:
        gdf=gdf.drop(columns='index_left')

    return gdf

def extract_values_glofas( ds, variable_name, plant_loc):

    lats = xr.DataArray(plant_loc["Latitude"].to_numpy(), dims="site")
    lons = xr.DataArray(plant_loc["Longitude"].to_numpy(), dims="site")

    da = ds[variable_name].sel(latitude=lats, longitude=lons, method="nearest")

    return da

def reshape_values( da, time_dim="valid_time"):
    df = da.to_pandas()  # index=valid_time, columns=site
    df.columns = [f"ROR_{i}" for i in range(df.shape[1])]
    df.index = pd.to_datetime(df.index - pd.Timedelta(days=1), format="%Y%m%d%H")
    # the glofas data is shown as 00:00:00 as the next day's data, so we shift it back to the previous day
    return df

def correct_glofas_points(cds_disc_path, disc_points_path, plant_loc, var):
    
    if os.path.exists(disc_points_path):
        plant_loc_updated = pd.read_csv(disc_points_path)
    else:
        data_2024 = read_glofas_netcdfs(os.path.join(cds_disc_path, "*.nc"), "valid_time", years=[2024], determine_local_points=True)
        plant_loc_updated = update_plant_locations(
                                plant_loc=plant_loc,
                                ds=data_2024,
                                variable_name=var,
                                half_size=0.025
                        )
        Path(disc_points_path).parent.mkdir(parents=True, exist_ok=True)
        plant_loc_updated.to_csv(disc_points_path, index=False)
    
    return plant_loc_updated


def read_glofas_netcdfs( files, dim, years, determine_local_points):
    paths_all = sorted(glob(files))
    if determine_local_points:
        paths =[p for p in paths_all if int(os.path.basename(p).split("_", 1)[0]) == 2024]  
        #determine accuracte local points by 2024 data
    else:
        years = {int(year) for year in years}
        paths =[p for p in paths_all if int(os.path.basename(p).split("_", 1)[0]) in years]
    #only read necessary years to save memory
    if not paths:
        raise FileNotFoundError(f"No GloFAS NetCDF files found for years {years} matching {files}")

    ds = xr.open_mfdataset(
        paths,
        combine="nested",        
        concat_dim=dim,
        parallel=False,
        engine="netcdf4",
        chunks={dim: 200},        
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    return ds

def update_plant_locations(plant_loc, ds, variable_name, half_size=0.025):
    updated = plant_loc.copy()
    latitudes = updated["Latitude"].to_numpy(dtype=float)
    longitudes = updated["Longitude"].to_numpy(dtype=float)

    candidate_lats = np.column_stack(
        [
            latitudes - half_size,
            latitudes - half_size,
            latitudes + half_size,
            latitudes + half_size,
            latitudes,
        ]
    )
    candidate_lons = np.column_stack(
        [
            longitudes - half_size,
            longitudes + half_size,
            longitudes - half_size,
            longitudes + half_size,
            longitudes,
        ]
    )
    da = ds[variable_name].sel(
        latitude=xr.DataArray(candidate_lats, dims=("site", "candidate")),
        longitude=xr.DataArray(candidate_lons, dims=("site", "candidate")),
        method="nearest",
    )
    reduce_dims = [dim for dim in da.dims if dim not in {"site", "candidate"}]
    scores = da.mean(dim=reduce_dims, skipna=True) if reduce_dims else da
    best_idx = scores.argmax(dim="candidate").to_numpy().astype(int)
    site_idx = np.arange(len(updated))

    updated['Latitude'] = candidate_lats[site_idx, best_idx]
    updated['Longitude'] = candidate_lons[site_idx, best_idx]
    print(
        f"update_plant_locations: {len(updated)} sites, "
    )
    return updated


def get_dis_inputs(
    glofas_raw_dir: str | Path,
    jrc_database_path: str | Path,
    output_disc_path: str | Path,
    output_points_path: str | Path,
    country: str,
    bidding_zone: str,
    bidding_zone_date: pd.Timestamp,
    years: list[int] = REQUEST_YEAR,
    target_years: list[int] | tuple[int | str, ...] | None = None,
    variable_name: str = "dis24",
) -> pd.DataFrame:
    """Create local HROR GloFAS discharge inputs for one country/zone.
    """
    glofas_raw_dir = Path(glofas_raw_dir)
    output_disc_path = Path(output_disc_path)
    output_points_path = Path(output_points_path)
    years = combine_years(years, target_years)

    import geopandas as gpd
    from entsoe.geo.utils import load_zones
    from shapely.geometry import Point

    loc_csv = pd.read_csv(jrc_database_path)
    location=loc_csv[loc_csv['type'] == "HROR"].copy()
    location['geometry'] = location.apply(lambda row: Point(row['lon'], row['lat']),axis=1)
    zone = load_zones([bidding_zone], bidding_zone_date)
    loc_geo=gpd.GeoDataFrame(location, geometry='geometry',crs=zone.crs)
    zone_hror = loc_geo[loc_geo['country_code']==country]
    hror_point=zone_hror[['lat', 'lon']].rename(columns={'lat': 'Latitude', 'lon': 'Longitude'})

    os.makedirs(glofas_raw_dir, exist_ok=True)
    if len(os.listdir(glofas_raw_dir))==0:
        print(f"Downloading data from GloFAS to {glofas_raw_dir}")
        glofas_request(glofas_raw_dir, years=years, target_years=None)

    plant_loc_updated = correct_glofas_points(glofas_raw_dir, output_points_path, hror_point, variable_name)
    data = read_glofas_netcdfs(os.path.join(glofas_raw_dir, "*.nc"), "valid_time", years=years, determine_local_points=False)
    da = extract_values_glofas(data, variable_name, plant_loc_updated)
    da = da.chunk({"valid_time": -1})   
    da = da.compute()                  # one compute
    data_local = reshape_values(da)
    output_disc_path.parent.mkdir(parents=True, exist_ok=True)
    data_local.to_csv(output_disc_path)
    return data_local


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from main import CONFIG

    get_dis_inputs(
        glofas_raw_dir=CONFIG.glofas_raw_dir,
        jrc_database_path=CONFIG.jrc_database_path,
        output_disc_path=CONFIG.glofas_disc_path,
        output_points_path=CONFIG.glofas_points_path,
        country=CONFIG.country_code,
        bidding_zone=CONFIG.entsoe_zone_code,
        bidding_zone_date=CONFIG.zone_date,
        target_years=CONFIG.target_years,
    )
