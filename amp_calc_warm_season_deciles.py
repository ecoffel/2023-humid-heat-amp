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

file_var = 'tasmax'
ds_var = 'mx2t'

year = int(sys.argv[1])

dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'
dirCMIP6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/CMIP6'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'
dirAg6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/research/2020-ag-cmip6'


# Load the DataArray containing the months of annual maximum temperature
annual_max_months_da = xr.open_dataarray("tw_months_1981_2021.nc")

# Load the temperature dataset for the specified year
file_path = '%s/daily/%s_%d.nc'%(dirEra5, file_var, year)
ds = xr.open_dataset(file_path)
if ds_var == 'mx2t':
    ds['mx2t'] -= 273.15

# # Find the months when the annual max temperature has historically occurred for each grid cell
# months_of_interest = annual_max_months_da.sel(year=year)

# # Select the daily temperature data for those months
# ds_temperature_months_of_interest = ds[ds_var].where(
#     ds.time.dt.month.isin(months_of_interest), drop=True
# )

# First create a boolean mask
mask = xr.full_like(ds.time, False, dtype=bool)

# Iterate over the years
for y in annual_max_months_da.year:
    # Find the month of max temperature in this year
    month_of_max = annual_max_months_da.sel(year=y)
    
    # Set True in the mask for all days of this month in this year
    mask = mask | (ds.time.dt.month == month_of_max)

# Apply the mask to select temperature data for the months of interest
if ds_var == 'mx2t':
    temperature_months_of_interest = ds['mx2t'].where(mask, drop=True)
elif ds_var == 'tw':
    temperature_months_of_interest = ds['tw'].where(mask, drop=True)
    
# Calculate the percentiles for each grid cell
daily_temperature_percentiles = temperature_months_of_interest.quantile(
    np.linspace(0, 1, 101), dim="time"
)


# Save the results to a netcdf file
output_file = f"deciles/tw/era5_{file_var}_deciles_warm_season_{year}_new.nc"
daily_temperature_percentiles.to_netcdf(output_file)
