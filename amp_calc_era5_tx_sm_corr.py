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
annual_max_months_da_tx = xr.open_dataarray("txx_months_1981_2021.nc")
annual_max_months_da_tw = xr.open_dataarray("tw_months_1981_2021.nc")

# Load the temperature dataset for the specified year
file_path = '%s/daily/tasmax_%d.nc'%(dirEra5, year)
ds_tx_var = xr.open_dataset(file_path)


# Load the soil moisture dataset for the specified year
huss_file_path = '%s/daily/sm_%d.nc'%(dirEra5Land, year)
ds_huss_var = xr.open_dataset(huss_file_path)


# Check if the dimensions differ
if (ds_tx_var.longitude.size != ds_huss_var.longitude.size) or (ds_tx_var.latitude.size != ds_huss_var.latitude.size):

    ds_tx_var = ds_tx_var.rename({'latitude':'lat', 'longitude':'lon'})
    ds_huss_var = ds_huss_var.rename({'latitude':'lat', 'longitude':'lon'})
    
    # Define regridder
    regridder = xe.Regridder(ds_huss_var, ds_tx_var, 'bilinear', reuse_weights=True)

    # Apply regridding
    ds_huss_var = regridder(ds_huss_var)
    
    ds_tx_var = ds_tx_var.rename({'lat':'latitude', 'lon':'longitude'})
    ds_huss_var = ds_huss_var.rename({'lat':'latitude', 'lon':'longitude'})


# First create a boolean mask
mask = xr.full_like(ds_tx_var.time, False, dtype=bool)

# Iterate over the years
for y in annual_max_months_da_tx.year:
    # Find the month of max temperature in this year
    month_of_max_tx = annual_max_months_da_tx.sel(year=y)
    month_of_max_tw = annual_max_months_da_tw.sel(year=y)
    
    # Set True in the mask for all days of this month in this year
    mask = mask | (ds_tx_var.time.dt.month == month_of_max_tx) | (ds_tx_var.time.dt.month == month_of_max_tw)

# Apply the mask to select temperature data for the months of interest
tx_months_of_interest = ds_tx_var['mx2t'].where(mask, drop=True)
huss_months_of_interest = ds_huss_var['swvl1'].where(mask, drop=True)
    
corr = xr.corr(tx_months_of_interest, huss_months_of_interest, dim='time')

# Save the results to a netcdf file
output_file = f"output/tx_sm_corr/tx_sm_corr_warm_season_{year}.nc"
corr.to_netcdf(output_file)
