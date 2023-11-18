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

member = int(sys.argv[1])


dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'
dirCMIP6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/CMIP6'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'
dirAg6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/research/2020-ag-cmip6'
dirUtil = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/research/util'
dataDirLens = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Data-edcoffel-F20/LENS/daily/atm'

regridMesh_global = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1.5)),
                                'lon': (['lon'], np.arange(0, 360, 1.5)),})

land_sea_mask = xr.open_dataset('%s/land-sea-mask.nc'%dirEra5)
land_sea_mask.load()
land_sea_mask = land_sea_mask.lsm.mean(dim='time')
land_sea_mask_binary = land_sea_mask > 0.1

# Load the DataArray containing the months of annual maximum temperature
annual_max_months_da_tx = xr.open_dataarray("txx_months_1981_2021.nc")
annual_max_months_da_tx.load();
annual_max_months_da_tw = xr.open_dataarray("tw_months_1981_2021.nc")
annual_max_months_da_tw.load();
annual_max_months_da_tx = annual_max_months_da_tx.rename({'latitude':'lat', 'longitude':'lon'})


# REGRID ALL THE LENS MODELS AND CALC TX_TW_CORR

print('opening lens member %d'%member)
lens_tw = xr.open_mfdataset(f'{dirUtil}/tw_lens_{member}_*.nc')
lens_tw = lens_tw.rename({'__xarray_dataarray_variable__': 'tw'})

lens_tx = xr.open_mfdataset(f'{dataDirLens}/TASMAX/HIST-RCP85/tasmax_day_CESM1-CAM5_historical_rcp85_r{member}i1p1_*.nc')

lens_tw = lens_tw.sel(time=slice('1981', '2021')).tw
lens_tx = lens_tx.sel(time=slice('1981', '2021')).tasmax

time_dim = pd.date_range("1981-01-01", "2021-12-31", freq="D")
time_dim_no_leap = time_dim[~((time_dim.month == 2) & (time_dim.day == 29))]

lens_tx['time'] = time_dim_no_leap

# lens_tx = lens_tx.reindex(lat=lens_tx.lat[::-1])
lens_tx.load();

# lens_tw = lens_tw.reindex(lat=lens_tw.lat[::-1])
lens_tw.load();


print('regridding...')

regridder = xe.Regridder(land_sea_mask_binary.rename({'latitude':'lat', 'longitude':'lon'}), regridMesh_global, 'bilinear', \
                         reuse_weights=True)
land_sea_mask_binary_regrid = regridder(land_sea_mask_binary)

regridder = xe.Regridder(lens_tx, regridMesh_global, 'bilinear', reuse_weights=False)
lens_tx_regrid = regridder(lens_tx)
lens_tw_regrid = regridder(lens_tw)

lens_tx_regrid = lens_tx_regrid.where(land_sea_mask_binary_regrid)
lens_tw_regrid = lens_tw_regrid.where(land_sea_mask_binary_regrid)

lens_tx_regrid = lens_tx_regrid.sel(lat=slice(-60,60))
lens_tw_regrid = lens_tw_regrid.sel(lat=slice(-60,60))

regridder = xe.Regridder(annual_max_months_da_tx, regridMesh_global, 'bilinear', reuse_weights=True)
annual_max_months_da_tx_regrid = regridder(annual_max_months_da_tx)
annual_max_months_da_tw_regrid = regridder(annual_max_months_da_tw)

print('creating warm season mask')
# First create a boolean mask
mask = xr.full_like(lens_tx_regrid.time, False, dtype=bool)


# Iterate over the years
for y in annual_max_months_da_tx_regrid.year:
    # Find the month of max temperature in this year
    month_of_max_tx = annual_max_months_da_tx_regrid.sel(year=y)
    month_of_max_tw = annual_max_months_da_tw_regrid .sel(year=y)

    # Set True in the mask for all days of this month in this year
    mask = mask | (lens_tx_regrid.time.dt.month == month_of_max_tx) | (lens_tx_regrid.time.dt.month == month_of_max_tw)


# Apply the mask to select temperature data for the months of interest
lens_tx_regrid = lens_tx_regrid.where(mask, drop=True)
lens_tw_regrid = lens_tw_regrid.where(mask, drop=True)

print('calculating correlation...')
correlation_per_year = []

# Iterate over the years
for y in range(1981, 2021 + 1):
    # Select data for this year
    lens_tx_year = lens_tx_regrid.sel(time=lens_tx_regrid.time.dt.year == y)
    lens_tw_year = lens_tw_regrid.sel(time=lens_tw_regrid.time.dt.year == y)

#         print(y)
    # Calculate the correlation for this year and store it in the list
    correlation = xr.corr(lens_tx_year, lens_tw_year, dim='time')
    correlation['year'] = y  # Add a coordinate for the year
    correlation_per_year.append(correlation)

# Combine all the DataArrays along the 'year' dimension
correlation_per_year_da = xr.concat(correlation_per_year, dim='year')


print('saving file...')
correlation_per_year_da.to_netcdf('tx_tw_corr_lens_member_%d.nc'%member)

