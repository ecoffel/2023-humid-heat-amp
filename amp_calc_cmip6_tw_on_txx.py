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

tw_on_tx = False
output_ts = True

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
cmip6_tw = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/tw/*.nc'%(dirCMIP6, model))
cmip6_tx = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/tasmax/*day*.nc'%(dirCMIP6, model))

cmip6_tx = cmip6_tx.sel(time=slice('1981', '2014'))
cmip6_tw = cmip6_tw.sel(time=slice('1981', '2014'))

cmip6_tx = cmip6_tx['tasmax']-273.15
cmip6_tx = cmip6_tx.reindex(lat=cmip6_tx.lat[::-1])
cmip6_tx.load();


cmip6_tw = cmip6_tw['tw']
cmip6_tw = cmip6_tw.reindex(lat=cmip6_tw.lat[::-1])
cmip6_tw.load();

print('regridding...')
regridder = xe.Regridder(land_sea_mask_binary.rename({'latitude':'lat', 'longitude':'lon'}), regridMesh_global, 'bilinear', reuse_weights=True)
land_sea_mask_binary_regrid = regridder(land_sea_mask_binary)

regridder = xe.Regridder(cmip6_tx, regridMesh_global, 'bilinear', reuse_weights=False)
cmip6_tx_regrid = regridder(cmip6_tx)
cmip6_tw_regrid = regridder(cmip6_tw)

cmip6_tx_regrid = cmip6_tx_regrid.where(land_sea_mask_binary_regrid)
cmip6_tw_regrid = cmip6_tw_regrid.where(land_sea_mask_binary_regrid)

cmip6_tx_regrid = cmip6_tx_regrid.sel(lat=slice(-60,60))
cmip6_tw_regrid = cmip6_tw_regrid.sel(lat=slice(-60,60))

regridder = xe.Regridder(annual_max_months_da_tx, regridMesh_global, 'bilinear', reuse_weights=True)
annual_max_months_da_tx_regrid = regridder(annual_max_months_da_tx)
annual_max_months_da_tw_regrid = regridder(annual_max_months_da_tw)

print('creating warm season mask')
# First create a boolean mask
mask = xr.full_like(cmip6_tx.time, False, dtype=bool)

# Iterate over the years
for y in annual_max_months_da_tx_regrid.year:
    # Find the month of max temperature in this year
    month_of_max_tx = annual_max_months_da_tx_regrid.sel(year=y)
    month_of_max_tw = annual_max_months_da_tw_regrid .sel(year=y)

    # Set True in the mask for all days of this month in this year
    mask = mask | (cmip6_tx.time.dt.month == month_of_max_tx) | (cmip6_tx.time.dt.month == month_of_max_tw)


# Apply the mask to select temperature data for the months of interest
cmip6_tx_regrid = cmip6_tx_regrid.where(mask, drop=True)
cmip6_tw_regrid = cmip6_tw_regrid.where(mask, drop=True)

# Calculate the temperature bins for each grid cell
if tw_on_tx:
    temperature_bins = cmip6_tx_regrid.quantile(
        np.linspace(0, 1, 21), dim="time"
    )
else:
    temperature_bins = cmip6_tw_regrid.quantile(
        np.linspace(0, 1, 21), dim="time"
    )

# Calculate the average soil moisture value on days within warm season temperature bins
temp_bin_means = []
for i in range(20):
    
    lower_bound = temperature_bins.isel(quantile=i)
    upper_bound = temperature_bins.isel(quantile=i + 1)
    
    if tw_on_tx:
        temp_bin = cmip6_tw_regrid.where(
            (cmip6_tx_regrid >= lower_bound) &
            (cmip6_tx_regrid < upper_bound)
        )
    else:
        temp_bin = cmip6_tx_regrid.where(
            (cmip6_tw_regrid >= lower_bound) &
            (cmip6_tw_regrid < upper_bound)
        )
        
    temp_bin_mean = temp_bin.mean(dim="time", skipna=True)
    temp_bin_means.append(temp_bin_mean)

temp_bin_means_da = xr.concat(
    temp_bin_means, dim="quantile"
)

# Add the bins coordinate to the DataArray
temp_bin_means_da = temp_bin_means_da.assign_coords(
    quantile=("quantile", np.arange(0, 1, .05))
)

# Save the results to a netcdf file
# if decile_var == 'tx':
if tw_on_tx:
    output_file = f"output/cmip6/tw_on_tx_{model}.nc"
else:
    output_file = f"output/cmip6/tx_on_tw_{model}.nc"
# elif decile_var == 'tw':
#     output_file = f"output/cmip6/tw_on_tx_{model}.nc"
temp_bin_means_da.to_netcdf(output_file)
