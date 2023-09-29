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

year = int(sys.argv[1])

dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'

# Load the DataArray containing the months of annual maximum temperature
if decile_var == 'tx':
    annual_max_months_da = xr.open_dataarray("txx_months_1981_2021.nc")
elif decile_var == 'tw':
    annual_max_months_da = xr.open_dataarray("tw_months_1981_2021.nc")

# Load the temperature dataset for the specified year
if decile_var == 'tx':
    file_path = '%s/daily/tasmax_%d.nc'%(dirEra5, year)
elif decile_var == 'tw':
    file_path = '%s/daily/tw_max_%d.nc'%(dirEra5, year)
ds_temperature = xr.open_dataset(file_path)

# CALC HUMIDITY ADVECTION

print('loading data...')
era5_u = xr.open_dataset('%s/daily/u_1000mb_%d.nc'%(dirEra5, year))
era5_v = xr.open_dataset('%s/daily/v_1000mb_%d.nc'%(dirEra5, year))
era5_q = xr.open_dataset('%s/daily/huss_%d.nc'%(dirEra5, year))

# Compute gradients along latitude and longitude
gradient_lat = era5_q['q'].differentiate('latitude')
gradient_lon = era5_q['q'].differentiate('longitude')

# compute advection
era5_q_adv = era5_u.u*gradient_lat + era5_v.v*gradient_lon

# Check if the dimensions differ
if (ds_temperature.longitude.size != era5_q_adv.longitude.size) or (ds_temperature.latitude.size != era5_q_adv.latitude.size):

    ds_temperature = ds_temperature.rename({'latitude':'lat', 'longitude':'lon'})
    era5_q_adv = era5_q_adv.rename({'latitude':'lat', 'longitude':'lon'})
    
    # Define regridder
    regridder = xe.Regridder(ds_era5_var, ds_temperature, 'bilinear', reuse_weights=True)

    # Apply regridding
    era5_q_adv = regridder(era5_q_adv)
    
    ds_temperature = ds_temperature.rename({'lat':'latitude', 'lon':'longitude'})
    era5_q_adv = era5_q_adv.rename({'lat':'latitude', 'lon':'longitude'})


print('building mask...')
# First create a boolean mask
mask = xr.full_like(ds_temperature.time, False, dtype=bool)

# Iterate over the years
for y in annual_max_months_da.year:
    # Find the month of max temperature in this year
    month_of_max = annual_max_months_da.sel(year=y)
    
    # Set True in the mask for all days of this month in this year
    mask = mask | (ds_temperature.time.dt.month == month_of_max)

# Apply the mask to select temperature data for the months of interest
if decile_var == 'tx':
    temperature_months_of_interest = ds_temperature['mx2t'].where(mask, drop=True)
elif decile_var == 'tw':
    temperature_months_of_interest = ds_temperature['tw'].where(mask, drop=True)
    
era5_q_adv_months_of_interest = era5_q_adv.where(mask, drop=True)


print('computing bins...')
# Calculate the temperature bins for each grid cell
temperature_bins = temperature_months_of_interest.quantile(
    np.linspace(0, 1, 21), dim="time"
)

# Calculate the average soil moisture value on days within warm season temperature bins
era5_var_bin_means = []
for i in range(20):
    lower_bound = temperature_bins.isel(quantile=i)
    upper_bound = temperature_bins.isel(quantile=i + 1)
    era5_var_bin = era5_q_adv_months_of_interest.where(
        (temperature_months_of_interest >= lower_bound) &
        (temperature_months_of_interest < upper_bound)
    )
    era5_var_bin_mean = era5_var_bin.mean(dim="time", skipna=True)
    era5_var_bin_means.append(era5_var_bin_mean)

era5_var_bin_means_da = xr.concat(
    era5_var_bin_means, dim="quantile"
)


# Add the bins coordinate to the DataArray
era5_var_bin_means_da = era5_var_bin_means_da.assign_coords(
    quantile=("quantile", np.arange(0, 1, .05))
)

# Save the results to a netcdf file
if decile_var == 'tx':
    output_file = f"output/q_adv_on_tx/q_adv_on_warm_season_tx_deciles_{year}.nc"
elif decile_var == 'tw':
    output_file = f"output/q_adv_on_tw/q_adv_on_warm_season_tw_deciles_{year}.nc"
era5_var_bin_means_da.to_netcdf(output_file)
