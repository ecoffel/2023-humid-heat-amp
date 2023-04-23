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

year = int(sys.argv[1])

dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'
dirCMIP6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/CMIP6'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'
dirAg6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/research/2020-ag-cmip6'


# Open and concatenate datasets using Dask
era5_tasmax = xr.open_dataset('%s/daily/tasmax_%d.nc'%(dirEra5, year))
era5_tasmax.load()
era5_tasmax['mx2t'] -= 273.15
era5_tasmax = era5_tasmax.rename_dims({'latitude':'lat', 'longitude':'lon'})

era5_sm = xr.open_dataset('%s/daily/sm_%d.nc'%(dirEra5Land, year))
era5_sm.load()
era5_sm = era5_sm.rename_dims({'latitude':'lat', 'longitude':'lon'})

target_grid = xr.Dataset({
    'lat': era5_tasmax['lat'],
    'lon': era5_tasmax['lon']
})

# Create a regridder object using the source and target grids
regridder = xe.Regridder(era5_sm, target_grid, 'bilinear', reuse_weights=True)

# Regrid the era5_soil_moisture DataArray
era5_sm_regrid = regridder(era5_sm)

# Clean up the regridder to delete the intermediate weight file
regridder.clean_weight_file()


# Find the hottest day for the current year
hottest_day = era5_tasmax['mx2t'].idxmax(dim="time")

# Extract the soil moisture data corresponding to the hottest day
soil_moisture_on_hottest_day = era5_sm_regrid.sel(time=hottest_day)

print('computing %d'%year)
# Compute the result and convert it to a Dataset
soil_moisture_on_hottest_day = era5_sm_regrid['swvl1'].compute().to_dataset(name='sm_on_txx')

# Write the output to a netCDF file
mode = "w"# if year == 1981 else "a"
soil_moisture_on_hottest_day.to_netcdf('sm_on_txx_%d.nc'%year, mode=mode)