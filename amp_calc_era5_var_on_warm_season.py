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

decile_var = 'tw'

ds_var = 'tp'
file_var = 'tp'
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


# Load the soil moisture dataset for the specified year
era5_var_file_path = '%s/daily/%s_%d.nc'%(dirEra5, file_var, year)
ds_era5_var = xr.open_dataset(era5_var_file_path)


# Check if the dimensions differ
if (ds_temperature.longitude.size != ds_era5_var.longitude.size) or (ds_temperature.latitude.size != ds_era5_var.latitude.size):

    ds_temperature = ds_temperature.rename({'latitude':'lat', 'longitude':'lon'})
    ds_era5_var = ds_era5_var.rename({'latitude':'lat', 'longitude':'lon'})
    
    # Define regridder
    regridder = xe.Regridder(ds_era5_var, ds_temperature, 'bilinear', reuse_weights=True)

    # Apply regridding
    ds_era5_var = regridder(ds_era5_var)
    
    ds_temperature = ds_temperature.rename({'lat':'latitude', 'lon':'longitude'})
    ds_era5_var = ds_era5_var.rename({'lat':'latitude', 'lon':'longitude'})


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
    
era5_var_months_of_interest = ds_era5_var[ds_var].where(mask, drop=True).sum(dim='time')


# Save the results to a netcdf file
if decile_var == 'tx':
    output_file = f"output/{file_var}_on_tx/{file_var}_on_tx_warm_season_{year}.nc"
elif decile_var == 'tw':
    output_file = f"output/{file_var}_on_tw/{file_var}_on_tw_warm_season_{year}.nc"
era5_var_months_of_interest.to_netcdf(output_file)
