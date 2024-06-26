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

regridder = xe.Regridder(cmip6_tx, regridMesh_global, 'bilinear', reuse_weights=False)
if on_tx:
    cmip6_temp_regrid = regridder(cmip6_tx)
    del cmip6_tx
else:
    cmip6_temp_regrid = regridder(cmip6_tw)
    del cmip6_tw

cmip6_temp_regrid = cmip6_temp_regrid.where(land_sea_mask_binary_regrid)
cmip6_temp_regrid = cmip6_temp_regrid.sel(lat=slice(-60,60))

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
mask = xr.full_like(cmip6_temp_regrid.time, False, dtype=bool)

# Iterate over the years
for y in annual_max_months_da_tx_regrid.year:
    # Find the month of max temperature in this year
    if on_tx:
        month_of_max_tx = annual_max_months_da_tx_regrid.sel(year=y)
        mask = mask | (cmip6_temp_regrid.time.dt.month == month_of_max_tx)
    else:
        month_of_max_tw = annual_max_months_da_tw_regrid .sel(year=y)
        mask = mask | (cmip6_temp_regrid.time.dt.month == month_of_max_tw)
    

# Apply the mask to select temperature data for the months of interest
cmip6_temp_regrid = cmip6_temp_regrid.where(mask, drop=True)

# Initialize an empty list to store the tw values corresponding to the highest tx days for each year
val_on_max_days = []


# For each year, find the day with the highest tx and then get the corresponding tw value
for year in np.unique(cmip6_temp_regrid.time.dt.year.values):
    print(year)
    temp_yearly = cmip6_temp_regrid.sel(time=str(year))

    # Get the time index of the day with the maximum tx value
    day_of_max = temp_yearly.idxmax(dim="time")
    
    # Get the tw value on that day
    val_on_max_day = np.full([day_of_max.lat.size, day_of_max.lon.size], np.nan)
    
    for x in range(day_of_max.lat.size):
        for y in range(day_of_max.lon.size):
            if ~pd.isnull(day_of_max[x,y]):
                val_on_max_day[x,y] = cmip6_lh_regrid[:,x,y].sel(time=day_of_max[x,y])
                
    
    # Convert the numpy array to a DataArray
    val_on_max_day = xr.DataArray(
        val_on_max_day,
        coords=[temp_yearly.lat, temp_yearly.lon],
        dims=["lat", "lon"]
    )
    
    val_on_max_days.append(val_on_max_day)

# Combine the tw values into a single DataArray
val_on_max_days_da = xr.concat(val_on_max_days, dim="time")


# Save the results to a netcdf file
# if decile_var == 'tx':
if on_tx:
    output_file = f"output/cmip6/lh_on_txx_1981_2100_ssp245_{model}.nc"
else:
    output_file = f"output/cmip6/lh_on_tww_1981_2100_ssp245_{model}.nc"

val_on_max_days_da.to_netcdf(output_file)
