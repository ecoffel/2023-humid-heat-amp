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

from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress

import warnings
warnings.filterwarnings('ignore')

def linregress_ufunc(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return slope, p_value

from datetime import datetime
def add_time_dim(xda):
    xda = xda.expand_dims(time = [datetime.now()])
    return xda


dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'

land_sea_mask = xr.open_dataset('%s/land-sea-mask.nc'%dirEra5)
land_sea_mask.load()
land_sea_mask = land_sea_mask.lsm.mean(dim='time')
land_sea_mask_binary = land_sea_mask > 0.1




time_dim = pd.date_range("1981-01-01", "2021-12-31", freq="AS")
era5_var_deciles = xr.open_mfdataset('deciles/tx/era5_tasmax_deciles_warm_season_*.nc', preprocess = add_time_dim, concat_dim='time')
era5_var_deciles['time'] = time_dim
era5_var_deciles = era5_var_deciles.chunk(chunks={"time": -1, "latitude": 50, "longitude": 50})
era5_var_deciles = era5_var_deciles.chunk({"time": -1})
era5_var_deciles = era5_var_deciles.chunk({"quantile": -1})



# time_dim = pd.date_range("1981-01-01", "2020-12-31", freq="AS")
# era5_var_deciles = xr.open_mfdataset('output/sm_on_tw/sm_on_warm_season_tw_deciles_*.nc', preprocess = add_time_dim, concat_dim='time')
# # era5_var_deciles = era5_var_deciles.rename({'bin':'quantile'})
# era5_var_deciles['time'] = time_dim
# era5_var_deciles = era5_var_deciles.chunk(chunks={"time": -1, "latitude": 50, "longitude": 50})
# era5_var_deciles = era5_var_deciles.chunk({"time": -1})
# era5_var_deciles = era5_var_deciles.chunk({"quantile": -1})


land_sea_mask_binary = land_sea_mask_binary.broadcast_like(era5_var_deciles)
era5_var_deciles_masked = era5_var_deciles.where(land_sea_mask_binary)

era5_var_deciles_masked = era5_var_deciles_masked.sel(quantile=[0.5, 0.99], method='nearest')
era5_var_deciles_masked = era5_var_deciles_masked['mx2t']



n_bootstrap=30
time_dim='time'
x_dim='longitude'
y_dim='latitude'
bin_dim='quantile'
    
unique_years = np.unique(era5_var_deciles_masked[time_dim].dt.year.values)

# Initialize output array for storing bootstrapped trends and p-values
bootstrap_trend = xr.DataArray(
    np.empty((n_bootstrap, len(era5_var_deciles_masked[bin_dim]), len(era5_var_deciles_masked[y_dim]), len(era5_var_deciles_masked[x_dim]))),
    dims=['bootstrap', bin_dim, y_dim, x_dim],
    coords={
        'bootstrap': range(n_bootstrap),
        bin_dim: era5_var_deciles_masked[bin_dim],
        y_dim: era5_var_deciles_masked[y_dim],
        x_dim: era5_var_deciles_masked[x_dim],
    }
)

bootstrap_pvalue = bootstrap_trend.copy()



for i in range(n_bootstrap):
    bootstrap_years = np.random.choice(unique_years, size=len(unique_years))

    # Initialize an empty list to store DataArrays for each bootstrap year
    dataarrays = []

    # Loop over each year in bootstrap_years
    for year in bootstrap_years:
        # Select data for the current year and append it to the list
        ds_year = era5_var_deciles_masked.sel(time=str(year))
        dataarrays.append(ds_year)

    # Concatenate all DataArrays along the time dimension
    era5_var_deciles_masked_bootstrap = xr.concat(dataarrays, dim='time')

    sys.exit()

    
    # Update time coordinate to the bootstrapped years
    era5_var_deciles_masked_bootstrap[time_dim] = pd.to_datetime(bootstrap_years, format='%Y')

    # Broadcast unique_years along the time dimension
    unique_years_broadcasted = xr.DataArray(bootstrap_years, dims=[time_dim], coords={time_dim: era5_var_deciles_masked_bootstrap[time_dim]})

    trend, p_value = xr.apply_ufunc(
        linregress_ufunc,
        unique_years_broadcasted,
        era5_var_deciles_masked_bootstrap,
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]
    )

    bootstrap_trend[i, :, :, :] = trend
    bootstrap_pvalue[i, :, :, :] = p_value

output_ds = xr.Dataset({
    "evaporation_on_warm_season_tw_trend": bootstrap_trend,
    "evaporation_on_warm_season_tw_p_value": bootstrap_pvalue
})


