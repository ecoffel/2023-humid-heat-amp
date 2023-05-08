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

ds_var = 'swvl1'
file_var = 'sm'
year = int(sys.argv[1])

dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'

# Load the DataArray containing the months of annual maximum temperature
annual_max_months_da = xr.open_dataarray("txx_months_1981_2021.nc")

# Load the temperature dataset for the specified year
file_path = '%s/daily/tasmax_%d.nc'%(dirEra5, year)
ds_temperature = xr.open_dataset(file_path)

# Load the soil moisture dataset for the specified year
era5_var_file_path = '%s/daily/sm_%d.nc'%(dirEra5Land, year)
ds_era5_var = xr.open_dataset(era5_var_file_path)

# Find the months when the annual max temperature has historically occurred for each grid cell
months_of_interest = annual_max_months_da.sel(year=year)

# Select the daily temperature and soil moisture data for those months
temperature_months_of_interest = ds_temperature.mx2t.where(
    ds_temperature.time.dt.month.isin(months_of_interest), drop=True
)
era5_var_months_of_interest = ds_era5_var[ds_var].where(
    ds_era5_var.time.dt.month.isin(months_of_interest), drop=True
)

# Calculate the temperature bins for each grid cell
temperature_bins = temperature_months_of_interest.quantile(
    np.linspace(0, 1, 21), dim="time"
)

# Calculate the average soil moisture value on days within warm season temperature bins
era5_var_bin_means = []
for i in range(20):
    lower_bound = temperature_bins.isel(quantile=i)
    upper_bound = temperature_bins.isel(quantile=i + 1)
    era5_var_bin = era5_var_months_of_interest.where(
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
output_file = f"output/sm_on_txx/{file_var}_on_warm_season_tx_deciles_{year}.nc"
era5_var_bin_means_da.to_netcdf(output_file)
