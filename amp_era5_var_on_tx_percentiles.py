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
era5_tasmax = era5_tasmax.rename({'latitude':'lat', 'longitude':'lon'})

file_name = 'huss'
var_name = 'q'

era5_var = xr.open_dataset('%s/daily/%s_%d.nc'%(dirEra5, file_name, year))
era5_var.load()
era5_var = era5_var.rename({'latitude':'lat', 'longitude':'lon'})

era5_tx_deciles = xr.open_dataset('%s/era5_tasmax_deciles.nc'%dirHeatData)
era5_tx_deciles.load()
era5_tx_deciles['mx2t'] -= 273.15
era5_tx_deciles_values = era5_tx_deciles.mx2t.values.copy()


# target_grid = xr.Dataset({
#     'lat': era5_tasmax['lat'],
#     'lon': era5_tasmax['lon']
# })

# # Create a regridder object using the source and target grids
# regridder = xe.Regridder(era5_var[var_name], target_grid, 'bilinear', reuse_weights=False)

# # Regrid the era5_soil_moisture DataArray
# era5_var_regrid_data = regridder(era5_var[var_name])

# Clean up the regridder to delete the intermediate weight file
# regridder.clean_weight_file()
era5_var_regrid_data = era5_var[var_name]

# Calculate the bins for the groupby_bins operation
bin_edges = [i*5 for i in range(21)]  # Assuming deciles are in fractions (0, 0.1, 0.2, ..., 0.9, 1)

# Create a new DataArray to store the results
era5_var_decile_means = np.full([len(bin_edges)-1, era5_tasmax.lat.size, era5_tasmax.lon.size], fill_value=np.nan)
era5_var_decile_means = xr.DataArray(
    era5_var_decile_means,
    dims=["decile", "latitude", "longitude"],
    coords={
        "decile": range(0, 100, 5),
        "latitude": era5_tasmax.lat.values,
        "longitude": era5_tasmax.lon.values
    }
)

# Iterate through each decile and calculate the average soil moisture
for i in range(20):
    print('processing bin', i)
    lower_bound = era5_tx_deciles_values[bin_edges[i], :, :]
    upper_bound = era5_tx_deciles_values[bin_edges[i+1], :, :]
    
    # Create a mask for the current decile
    decile_mask = ((era5_tasmax.mx2t >= lower_bound) & (era5_tasmax.mx2t < upper_bound))
    
    # Select soil moisture data corresponding to the current decile using the mask
    era5_var_decile = era5_var_regrid_data.where(decile_mask)
    
    # Calculate the mean soil moisture for the current decile
    era5_var_decile_mean = era5_var_decile.mean(dim='time', skipna=True)
    
    # Assign the results to the corresponding decile in the output DataArray
    era5_var_decile_means[i,:,:] = era5_var_decile_mean
    era5_var_decile_means.loc[{"decile": bin_edges[i]}] = era5_var_decile_mean


# Save the result to a netCDF file
era5_var_decile_means.to_netcdf('output/q_on_txx/%s_on_tx_deciles_%d.nc'%(file_name, year))











