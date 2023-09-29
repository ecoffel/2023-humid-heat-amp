import rasterio as rio
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

from matplotlib.colors import Normalize
import numpy as np
import numpy.matlib
from scipy import interpolate
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
import scipy
import os, sys, pickle, gzip
import datetime
import geopy.distance
import xarray as xr
import pandas as pd
import rasterio
import geopandas as gpd
import shapely.geometry
import shapely.ops
import xesmf as xe
import cartopy
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.util import add_cyclic_point
import itertools
import random
import metpy
from metpy.plots import USCOUNTIES

import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress

import warnings
warnings.filterwarnings('ignore')

decile_var = 'tx'

ds_var = 'tp'
file_var = 'tp'
year = int(sys.argv[1])

dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'

land_sea_mask = xr.open_dataset('%s/land-sea-mask.nc'%dirEra5)
land_sea_mask.load()
land_sea_mask = land_sea_mask.lsm.mean(dim='time')
land_sea_mask_binary = land_sea_mask > 0.1

# Load the DataArray containing the months of annual maximum temperature
if decile_var == 'tx':
    annual_max_months_da = xr.open_dataarray("txx_months_1981_2021.nc")
elif decile_var == 'tw':
    annual_max_months_da = xr.open_dataarray("tw_months_1981_2021.nc")


# Load the soil moisture dataset for the specified year
era5_var_file_path = '%s/daily/%s_%d.nc'%(dirEra5, file_var, year)
ds_era5_var = xr.open_dataset(era5_var_file_path)
ds_era5_var = ds_era5_var.sel(latitude=slice(60,-60))
ds_era5_var = ds_era5_var.where(land_sea_mask_binary)
ds_era5_var.load()


# First create a boolean mask
mask = xr.full_like(ds_era5_var.time, False, dtype=bool)

# Iterate over the years
for y in annual_max_months_da.year:
    # Find the month of max temperature in this year
    month_of_max = annual_max_months_da.sel(year=y)
    
    # Set True in the mask for all days of this month in this year
    mask = mask | (ds_era5_var.time.dt.month == month_of_max)

# Apply the mask to select temperature data for the months of interest
# if decile_var == 'tx':
#     temperature_months_of_interest = ds_temperature['mx2t'].where(mask, drop=True)
# elif decile_var == 'tw':
#     temperature_months_of_interest = ds_temperature['tw'].where(mask, drop=True)
    
era5_var_months_of_interest = ds_era5_var[ds_var].where(mask, drop=True)



# Step 1: Create a mask where True indicates a dry day (precipitation < 0.02)
dry_mask = era5_var_months_of_interest < 0.005

# Initialize an empty DataArray to store median lengths
median_lengths = xr.DataArray(
    np.nan,
    coords={'latitude': era5_var_months_of_interest.latitude, 'longitude': era5_var_months_of_interest.longitude},
    dims=['latitude', 'longitude']
)

# Step 2: Loop through each grid cell
for lat in range(era5_var_months_of_interest.latitude.size):
    print(lat)
    for lon in range(era5_var_months_of_interest.longitude.size):
        
        if np.where(dry_mask[:,lat,lon].values)[0].size < 5:
            continue
        
        dry_sequences = []
        count = 0
        for t in range(dry_mask.sizes['time']):
            if dry_mask[t, lat, lon].values:
                count += 1
            else:
                if count > 0:
                    dry_sequences.append(count)
                count = 0
        
        # Don't forget the last sequence if it ends with a dry day
        if count > 0:
            dry_sequences.append(count)
        
        # Step 3: Compute the median length of dry sequences
        if len(dry_sequences) > 0:
            median_length = np.median(dry_sequences)
        else:
            median_length = np.nan
        
        median_lengths.loc[{'latitude': era5_var_months_of_interest.latitude.values[lat], 'longitude': era5_var_months_of_interest.longitude.values[lon]}] = median_length


# Save the results to a netcdf file
if decile_var == 'tx':
    output_file = f"output/cdd_on_tx/cdd_on_tx_warm_season_{year}.nc"
elif decile_var == 'tw':
    output_file = f"output/cdd_on_tw/cdd_on_tw_warm_season_{year}.nc"
median_lengths.to_netcdf(output_file)
