
import cdsapi
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import os
from glob import glob
from shapely.geometry import Point
from entsoe.geo.utils import load_zones


datapath = "your_glofas_path"
jrc_database ="your_jrc_db_path"
disc_path = "your_processed_discharge_path"
disc_points_path = "your_glofas_points_path"
country_code = "your country code"
entsoe_code = "your entsoe code"
zone_date = pd.Timestamp("2024-01-01")  # default is the 2024 definition
#===============================================================================================================


#===========================================fetch and read Glofas original data============================================
REQUEST_YEAR=[2015, 2016, 2017, 2018, 2019, 2020,
                2021, 2022, 2023, 2024]
REQUEST_MONTH=["01","02","03","04","05","06","07","08","09","10","11","12"]
def glofas_request(datapath):
    os.makedirs(datapath, exist_ok=True)
    for year in REQUEST_YEAR:
        for month in REQUEST_MONTH:
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
    df.columns = [f"RES_{i}" for i in range(df.shape[1])]
    df.index = pd.to_datetime(df.index - pd.Timedelta(days=1), format="%Y%m%d%H")
    # the glofas data is shown as 00:00:00 as the next day's data, so we shift it back to the previous day
    return df

def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

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
        ensure_parent_dir(disc_points_path)
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




loc_csv = pd.read_csv(jrc_database)
location=loc_csv[loc_csv['type'] == "HROR"].copy()
location['geometry'] = location.apply(lambda row: Point(row['lon'], row['lat']),axis=1)
zone = load_zones([entsoe_code], zone_date)
loc_geo=gpd.GeoDataFrame(location, geometry='geometry',crs=zone.crs)
zone_hror = loc_geo[loc_geo['country_code']==country_code]
hror_point=zone_hror[['lat', 'lon']].rename(columns={'lat': 'Latitude', 'lon': 'Longitude'})
var = 'dis24'

os.makedirs(datapath, exist_ok=True)
if len(os.listdir(datapath))==0:
    print(f"Downloading data from GloFAS to {datapath}")
    glofas_request(datapath)

plant_loc_updated = correct_glofas_points(datapath, disc_points_path, hror_point, var)
data = read_glofas_netcdfs(os.path.join(datapath, "*.nc"), "valid_time", years=REQUEST_YEAR, determine_local_points=False)
da = extract_values_glofas(data, 'dis24', plant_loc_updated)
da = da.chunk({"valid_time": -1})   
da = da.compute()                  # one compute
data_local = reshape_values(da)
ensure_parent_dir(disc_path)
data_local.to_csv(disc_path)
