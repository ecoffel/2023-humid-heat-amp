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

low_v_var = 'lai_lv'
high_v_var = 'lai_hv'
year = int(sys.argv[1])

dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'

# Load the DataArray containing the months of annual maximum temperature
annual_max_months_da_tx = xr.open_dataarray("txx_months_1981_2021.nc")
annual_max_months_da_tw = xr.open_dataarray("tw_months_1981_2021.nc")

# Load the temperature dataset for the specified year
file_path = '%s/daily/tasmax_%d.nc'%(dirEra5, year)
ds_temperature = xr.open_dataset(file_path)

# Load the soil moisture dataset for the specified year
era5_lai_low_file_path = '%s/monthly/lai_low_%d.nc'%(dirEra5Land, year)
era5_lai_high_file_path = '%s/monthly/lai_high_%d.nc'%(dirEra5Land, year)
ds_era5_lai_low = xr.open_dataset(era5_lai_low_file_path)
ds_era5_lai_high = xr.open_dataset(era5_lai_high_file_path)


# Check if the dimensions differ
if (ds_temperature.longitude.size != ds_era5_lai_low.longitude.size) or (ds_temperature.latitude.size != ds_era5_lai_low.latitude.size):

    ds_temperature = ds_temperature.rename({'latitude':'lat', 'longitude':'lon'})
    ds_era5_lai_low = ds_era5_lai_low.rename({'latitude':'lat', 'longitude':'lon'})
    ds_era5_lai_high = ds_era5_lai_high.rename({'latitude':'lat', 'longitude':'lon'})
    
    # Define regridder
    regridder = xe.Regridder(ds_era5_lai_low, ds_temperature, 'bilinear', reuse_weights=True)

    # Apply regridding
    ds_era5_lai_low = regridder(ds_era5_lai_low)
    ds_era5_lai_high = regridder(ds_era5_lai_high)
    
    ds_temperature = ds_temperature.rename({'lat':'latitude', 'lon':'longitude'})
    ds_era5_lai_low = ds_era5_lai_low.rename({'lat':'latitude', 'lon':'longitude'})
    ds_era5_lai_high = ds_era5_lai_high.rename({'lat':'latitude', 'lon':'longitude'})


# First create a boolean mask
mask = xr.full_like(ds_temperature.time, False, dtype=bool)

# Iterate over the years
for y in annual_max_months_da_tx.year:
    # Find the month of max temperature in this year
    month_of_max_tx = annual_max_months_da_tx.sel(year=y)
    month_of_max_tw = annual_max_months_da_tw.sel(year=y)
    
    # Set True in the mask for all days of this month in this year
    mask = mask | (ds_era5_lai_low.time.dt.month == month_of_max_tx) | (ds_era5_lai_low.time.dt.month == month_of_max_tw)

# Apply the mask to select temperature data for the months of interest
ds_era5_lai_low = ds_era5_lai_low['lai_lv'].where(mask, drop=True).mean(dim='time')
ds_era5_lai_high = ds_era5_lai_high['lai_hv'].where(mask, drop=True).mean(dim='time')

ds_era5_lai = ds_era5_lai_low+ds_era5_lai_high


# Save the results to a netcdf file
output_file = f"output/lai_on_tx_season/lai_on_tx_warm_season_{year}.nc"
ds_era5_lai.to_netcdf(output_file)
