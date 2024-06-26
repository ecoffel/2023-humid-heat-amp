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

cmip6_models = ['access-cm2', 'awi-esm-1-1-lr', 'bcc-csm2-mr',
                'bcc-esm1', 'canesm5', 'cmcc-esm2',
                'fgoals-f3-l', 'fgoals-g3', 'inm-cm4-8',
                'inm-cm5-0', 'ipsl-cm6a-lr', 'kace-1-0-g',
                'miroc6', 'mpi-esm1-2-hr', 'mpi-esm1-2-lr',
                'mri-esm2-0', 'noresm2-lm', 'taiesm1']

model = sys.argv[1]

on_tx = True

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

print('opening %s'%model)

if on_tx:
    cmip6_tx_hist = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/tasmax/*day*.nc'%(dirCMIP6, model))
    cmip6_tx_hist = cmip6_tx_hist.sel(time=slice('1981', '2015'))
    cmip6_tx_fut = xr.open_mfdataset('%s/%s/r1i1p1f1/ssp245/tasmax/*day*.nc'%(dirCMIP6, model))
    cmip6_tx_fut = cmip6_tx_fut.sel(time=slice('2015', '2100'))
    cmip6_tx = xr.concat([cmip6_tx_hist, cmip6_tx_fut], dim='time')
    cmip6_tx = cmip6_tx.sel(time=slice('1981', '2100'))
    
    cmip6_tx = cmip6_tx['tasmax']-273.15
    cmip6_tx = cmip6_tx.reindex(lat=cmip6_tx.lat[::-1])
    cmip6_tx.load();
else:
    cmip6_tw_hist = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/tw/*.nc'%(dirCMIP6, model))
    cmip6_tw_fut = xr.open_mfdataset('%s/%s/r1i1p1f1/ssp245/tw/*.nc'%(dirCMIP6, model))
    cmip6_tw = xr.concat([cmip6_tw_hist, cmip6_tw_fut], dim='time')
    cmip6_tw = cmip6_tw.sel(time=slice('1981', '2100'))
    
    cmip6_tw = cmip6_tw['tw']
    cmip6_tw = cmip6_tw.reindex(lat=cmip6_tw.lat[::-1])
    cmip6_tw.load();



print('regridding...')
regridder = xe.Regridder(land_sea_mask_binary.rename({'latitude':'lat', 'longitude':'lon'}), regridMesh_global, 'bilinear', reuse_weights=False)
land_sea_mask_binary_regrid = regridder(land_sea_mask_binary)

if on_tx:
    regridder = xe.Regridder(cmip6_tx, regridMesh_global, 'bilinear', reuse_weights=False)
    cmip6_tx_regrid = regridder(cmip6_tx)

    del cmip6_tx
    
    cmip6_tx_regrid = cmip6_tx_regrid.where(land_sea_mask_binary_regrid)
    cmip6_tx_regrid = cmip6_tx_regrid.sel(lat=slice(-60,60))
else:
    regridder = xe.Regridder(cmip6_tw, regridMesh_global, 'bilinear', reuse_weights=False)
    cmip6_tw_regrid = regridder(cmip6_tw)

    del cmip6_tw

    cmip6_tw_regrid = cmip6_tw_regrid.where(land_sea_mask_binary_regrid)

    cmip6_tw_regrid = cmip6_tw_regrid.sel(lat=slice(-60,60))


cmip6_huss_hist = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/huss/huss_day_*.nc'%(dirCMIP6, model))
cmip6_huss_hist = cmip6_huss_hist.sel(time=slice('1981', '2015'))

cmip6_huss_fut = xr.open_mfdataset('%s/%s/r1i1p1f1/ssp245/huss/huss_day*.nc'%(dirCMIP6, model))
cmip6_huss_fut = cmip6_huss_fut.sel(time=slice('2015', '2100'))

cmip6_huss = xr.concat([cmip6_huss_hist, cmip6_huss_fut], dim='time')

cmip6_huss = cmip6_huss['huss']
cmip6_huss = cmip6_huss.reindex(lat=cmip6_huss.lat[::-1])
cmip6_huss.load();

    
print('regridding...')
regridder = xe.Regridder(land_sea_mask_binary.rename({'latitude':'lat', 'longitude':'lon'}), regridMesh_global, 'bilinear', reuse_weights=False)
land_sea_mask_binary_regrid = regridder(land_sea_mask_binary)

regridder = xe.Regridder(cmip6_huss, regridMesh_global, 'bilinear', reuse_weights=False)
cmip6_huss_regrid = regridder(cmip6_huss)

del cmip6_huss

cmip6_huss_regrid = cmip6_huss_regrid.where(land_sea_mask_binary_regrid)

cmip6_huss_regrid = cmip6_huss_regrid.sel(lat=slice(-60,60))


regridder = xe.Regridder(annual_max_months_da_tx, regridMesh_global, 'bilinear', reuse_weights=False)
annual_max_months_da_tx_regrid = regridder(annual_max_months_da_tx)

regridder = xe.Regridder(annual_max_months_da_tw, regridMesh_global, 'bilinear', reuse_weights=False)
annual_max_months_da_tw_regrid = regridder(annual_max_months_da_tw)

print('creating warm season mask')
# First create a boolean mask
mask = xr.full_like(cmip6_huss_regrid.time, False, dtype=bool)

# Iterate over the years
for y in annual_max_months_da_tx_regrid.year:
    # Find the month of max temperature in this year
    if on_tx:
        month_of_max_tx = annual_max_months_da_tx_regrid.sel(year=y)
        mask = mask | (cmip6_tx_regrid.time.dt.month == month_of_max_tx)# | (cmip6_tx_regrid.time.dt.month == month_of_max_tw)
    else:
        month_of_max_tw = annual_max_months_da_tw_regrid .sel(year=y)
        mask = mask | (cmip6_tw_regrid.time.dt.month == month_of_max_tw)

    # Set True in the mask for all days of this month in this year
    

# Apply the mask to select temperature data for the months of interest
if on_tx:
    cmip6_temp_regrid = cmip6_tx_regrid.where(mask, drop=True)
else:
    cmip6_temp_regrid = cmip6_tw_regrid.where(mask, drop=True)

# Initialize an empty list to store the tw values corresponding to the highest tx days for each year
val_on_warm_season = []


# For each year, find the day with the highest tx and then get the corresponding tw value
for year in np.unique(cmip6_temp_regrid.time.dt.year.values):
    print(year)
    temp_yearly = cmip6_temp_regrid.sel(time=str(year))


    # Convert the numpy array to a DataArray
    cur_val_on_warm_season = xr.DataArray(
        temp_yearly.mean(dim='time'),
        coords=[temp_yearly.lat, temp_yearly.lon],
        dims=["lat", "lon"]
    )
    
    val_on_warm_season.append(cur_val_on_warm_season)

# Combine the tw values into a single DataArray
val_on_warm_season_da = xr.concat(val_on_warm_season, dim="time")

# Save the results to a netcdf file
# if decile_var == 'tx':
if on_tx:
    output_file = f"output/cmip6/huss_on_tx_warm_season_1981_2100_ssp245_{model}.nc"
else:
    output_file = f"output/cmip6/huss_on_tw_warm_season_1981_2100_ssp245_{model}.nc"

val_on_warm_season_da.to_netcdf(output_file)
