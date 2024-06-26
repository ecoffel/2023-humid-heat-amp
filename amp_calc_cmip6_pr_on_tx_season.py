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

tw_on_tw = False

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


cmip6_pr_hist = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/pr/pr_day_*.nc'%(dirCMIP6, model))
cmip6_pr_hist = cmip6_pr_hist.sel(time=slice('1981', '2015'))

cmip6_pr_fut = xr.open_mfdataset('%s/%s/r1i1p1f1/ssp245/pr/pr_day*.nc'%(dirCMIP6, model))
cmip6_pr_fut = cmip6_pr_fut.sel(time=slice('2015', '2100'))

cmip6_pr = xr.concat([cmip6_pr_hist, cmip6_pr_fut], dim='time')

cmip6_pr = cmip6_pr['pr']
cmip6_pr = cmip6_pr.reindex(lat=cmip6_pr.lat[::-1])
cmip6_pr.load();

    
print('regridding...')
regridder = xe.Regridder(land_sea_mask_binary.rename({'latitude':'lat', 'longitude':'lon'}), regridMesh_global, 'bilinear', reuse_weights=True)
land_sea_mask_binary_regrid = regridder(land_sea_mask_binary)

regridder = xe.Regridder(cmip6_pr, regridMesh_global, 'bilinear', reuse_weights=False)
cmip6_pr_regrid = regridder(cmip6_pr)

del cmip6_pr

cmip6_pr_regrid = cmip6_pr_regrid.where(land_sea_mask_binary_regrid)

cmip6_pr_regrid = cmip6_pr_regrid.sel(lat=slice(-60,60))

regridder = xe.Regridder(annual_max_months_da_tx, regridMesh_global, 'bilinear', reuse_weights=True)
annual_max_months_da_tx_regrid = regridder(annual_max_months_da_tx)
annual_max_months_da_tw_regrid = regridder(annual_max_months_da_tw)

print('creating warm season mask')
# First create a boolean mask
mask = xr.full_like(cmip6_pr_regrid.time, False, dtype=bool)

# Iterate over the years
for y in annual_max_months_da_tx_regrid.year:
    # Find the month of max temperature in this year
    month_of_max_tx = annual_max_months_da_tx_regrid.sel(year=y)
#     month_of_max_tw = annual_max_months_da_tw_regrid .sel(year=y)

    # Set True in the mask for all days of this month in this year
    mask = mask | (cmip6_pr_regrid.time.dt.month == month_of_max_tx)

# Apply the mask to select temperature data for the months of interest
cmip6_pr_regrid = cmip6_pr_regrid.where(mask, drop=True).resample(time='1Y').mean()

# Save the results to a netcdf file
# if decile_var == 'tx':
if tw_on_tw:
    output_file = f"output/cmip6/pr_on_tw_1981_2100_ssp245_{model}.nc"
else:
    output_file = f"output/cmip6/pr_on_tx_1981_2100_ssp245_{model}.nc"

cmip6_pr_regrid.to_netcdf(output_file)
