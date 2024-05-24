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

model = sys.argv[1]

on_tx = False


dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirCMIP6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/CMIP6'

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


cmip6_lh_hist = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/hfls/hfls_day_*.nc'%(dirCMIP6, model))
cmip6_lh_hist = cmip6_lh_hist.sel(time=slice('1981', '2015'))

cmip6_lh_fut = xr.open_mfdataset('%s/%s/r1i1p1f1/ssp245/hfls/hfls_day*.nc'%(dirCMIP6, model))
cmip6_lh_fut = cmip6_lh_fut.sel(time=slice('2015', '2100'))

cmip6_lh = xr.concat([cmip6_lh_hist, cmip6_lh_fut], dim='time')

cmip6_lh = cmip6_lh['hfls']
cmip6_lh = cmip6_lh.reindex(lat=cmip6_lh.lat[::-1])
cmip6_lh.load();

    
print('regridding...')
regridder = xe.Regridder(land_sea_mask_binary.rename({'latitude':'lat', 'longitude':'lon'}), regridMesh_global, 'bilinear', reuse_weights=False)
land_sea_mask_binary_regrid = regridder(land_sea_mask_binary)

regridder = xe.Regridder(cmip6_lh, regridMesh_global, 'bilinear', reuse_weights=False)
cmip6_lh_regrid = regridder(cmip6_lh)

del cmip6_lh

cmip6_lh_regrid = cmip6_lh_regrid.where(land_sea_mask_binary_regrid)

cmip6_lh_regrid = cmip6_lh_regrid.sel(lat=slice(-60,60))

regridder = xe.Regridder(annual_max_months_da_tx, regridMesh_global, 'bilinear', reuse_weights=False)
annual_max_months_da_tx_regrid = regridder(annual_max_months_da_tx)
annual_max_months_da_tw_regrid = regridder(annual_max_months_da_tw)

print('creating warm season mask')
# First create a boolean mask
mask = xr.full_like(cmip6_lh_regrid.time, False, dtype=bool)

# Iterate over the years
for y in annual_max_months_da_tx_regrid.year:
    # Find the month of max temperature in this year
    if on_tx:
        month_of_max = annual_max_months_da_tx_regrid.sel(year=y)
    else:
        month_of_max = annual_max_months_da_tw_regrid .sel(year=y)

    # Set True in the mask for all days of this month in this year
    mask = mask | (cmip6_lh_regrid.time.dt.month == month_of_max)

# Apply the mask to select temperature data for the months of interest
cmip6_lh_regrid = cmip6_lh_regrid.where(mask, drop=True).resample(time='1Y').mean()

# Save the results to a netcdf file
# if decile_var == 'tx':
if not on_tx:
    output_file = f"output/cmip6/lh_on_tw_warm_season_1981_2100_ssp245_{model}.nc"
else:
    output_file = f"output/cmip6/lh_on_tx_warm_season_1981_2100_ssp245_{model}.nc"

cmip6_lh_regrid.to_netcdf(output_file)
