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

dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'

years = range(1981, 2022)

file_var = 'tw_max'
xarray_var = 'tw'

# Initialize an empty list to store DataArrays with months of annual max temperature
annual_max_months_list = []

for year in years:
    print(year)
    file_path = '%s/daily/%s_%d.nc'%(dirEra5, file_var, year)
    ds = xr.open_dataset(file_path)

    # Calculate annual maximum temperature and its index
    annual_max_temperature = ds[xarray_var].max(dim="time")
    annual_max_temperature_idx = ds[xarray_var].argmax(dim="time")

    # Convert the index to timestamps and extract the month
    annual_max_temperature_time = ds.time[annual_max_temperature_idx]
    annual_max_months = annual_max_temperature_time.dt.month

    annual_max_months_list.append(annual_max_months)

# Concatenate the list of DataArrays along a new dimension 'year'
annual_max_months_da = xr.concat(annual_max_months_list, dim=pd.Index(years, name='year'))

# Save the results to a netcdf file
annual_max_months_da.to_netcdf("%s_months_1981_2021.nc"%xarray_var)
