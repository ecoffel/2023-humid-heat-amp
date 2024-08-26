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

tw_on_tw = True

dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirCMIP6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/CMIP6'

regridMesh_global = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1.5)),
                                'lon': (['lon'], np.arange(0, 360, 1.5)),})

land_sea_mask = xr.open_dataset('%s/land-sea-mask.nc'%dirEra5)
land_sea_mask.load()
land_sea_mask = land_sea_mask.lsm.mean(dim='time')
land_sea_mask_binary = land_sea_mask > 0.1

percentile = .99

print('opening %s'%model)

if tw_on_tw:
    cmip6_temp_hist = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/tw/*.nc'%(dirCMIP6, model))
    cmip6_temp_fut = xr.open_mfdataset('%s/%s/r1i1p1f1/ssp245/tw/*.nc'%(dirCMIP6, model))
    
    cmip6_temp = xr.concat([cmip6_temp_hist, cmip6_temp_fut], dim='time')
    cmip6_temp = cmip6_temp.sel(time=slice('1981', '2100'))
    
    cmip6_temp = cmip6_temp['tw']
    cmip6_temp = cmip6_temp.reindex(lat=cmip6_temp.lat[::-1])
    cmip6_temp.load();
    
else:
    cmip6_temp_hist = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/tasmax/*day*.nc'%(dirCMIP6, model))
    cmip6_temp_hist = cmip6_temp_hist.sel(time=slice('1981', '2015'))

    cmip6_temp_fut = xr.open_mfdataset('%s/%s/r1i1p1f1/ssp245/tasmax/*day*.nc'%(dirCMIP6, model))
    cmip6_temp_fut = cmip6_temp_fut.sel(time=slice('2015', '2100'))

    cmip6_temp = xr.concat([cmip6_temp_hist, cmip6_temp_fut], dim='time')

    cmip6_temp = cmip6_temp['tasmax']-273.15
    cmip6_temp = cmip6_temp.reindex(lat=cmip6_temp.lat[::-1])
    cmip6_temp.load();
    

print('regridding...')
regridder = xe.Regridder(land_sea_mask_binary.rename({'latitude':'lat', 'longitude':'lon'}), regridMesh_global, 'bilinear', reuse_weights=False)
land_sea_mask_binary_regrid = regridder(land_sea_mask_binary)

regridder = xe.Regridder(cmip6_temp, regridMesh_global, 'bilinear', reuse_weights=False)
cmip6_temp_regrid = regridder(cmip6_temp)

del cmip6_temp

cmip6_temp_regrid = cmip6_temp_regrid.where(land_sea_mask_binary_regrid)
cmip6_temp_regrid = cmip6_temp_regrid.sel(lat=slice(-60,60))

if tw_on_tw:
    thresh_file_path = f'cmip6_tw_{percentile}_thresholds_1981_2021_{model}.nc'
else:
    thresh_file_path = f'cmip6_tx_{percentile}_thresholds_1981_2021_{model}.nc'
if not os.path.isfile(thresh_file_path):
    print('creating percentiles file')
    hist_temp_data = cmip6_temp_regrid.sel(time=slice('1981','2021'))
    hist_temp_data = hist_temp_data.chunk(dict(time=-1))
    temp_thresh = hist_temp_data.quantile(percentile, dim='time')
    temp_thresh.to_netcdf(thresh_file_path)
else:
    print('loading percentiles file')
    temp_thresh = xr.open_dataset(thresh_file_path)

# Count the number of days above the threshold
# Convert time to year
cmip6_temp_regrid['year'] = cmip6_temp_regrid['time'].dt.year

# Calculate the sum of days above the threshold for each year
days_above_threshold_by_year = (cmip6_temp_regrid > temp_thresh).groupby('year').sum(dim='time')

days_above_threshold_by_year = days_above_threshold_by_year.drop(['quantile', 'height'])

# Save the results to a netcdf file
if tw_on_tw:
    days_above_threshold_by_year.rename({'__xarray_dataarray_variable__':'tw'})
    output_file = f"output/cmip6/days_above_tw{percentile}_{model}.nc"
else:
    days_above_threshold_by_year.rename({'__xarray_dataarray_variable__':'tx'})
    output_file = f"output/cmip6/days_above_tx{percentile}_{model}.nc"
    
days_above_threshold_by_year.to_netcdf(output_file)
