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

import xgrid_utils

import numpy as np
import statsmodels.api as sm
from scipy import signal

import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress

import warnings
warnings.filterwarnings('ignore')

decile_var = 'tx'

decile = int(sys.argv[1])

dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'

land_sea_mask = xr.open_dataset('%s/land-sea-mask.nc'%dirEra5)
land_sea_mask.load()
land_sea_mask = land_sea_mask.lsm.mean(dim='time')
land_sea_mask_binary = land_sea_mask > 0.1


from datetime import datetime
def add_time_dim(xda):
    xda = xda.expand_dims(time = [datetime.now()])
    return xda

time_dim = pd.date_range("1981-01-01", "2021-12-31", freq="AS")
tx_tw_corr_full = xr.open_mfdataset('output/tx_tw_corr/tx_tw_corr_warm_season_*.nc', preprocess = add_time_dim, concat_dim='time')
tx_tw_corr_full['time'] = time_dim
tx_tw_corr_full = tx_tw_corr_full.rename({'__xarray_dataarray_variable__':'tx_tw_corr'})
tx_tw_corr_full = tx_tw_corr_full.where(land_sea_mask_binary)

tx_tw_corr_full = tx_tw_corr_full.sel(latitude=slice(60,-60))
tx_tw_corr_full.load()

tw_on_tw_full = xr.open_dataset('intermediate/tw_on_tw_full.nc')
tx_on_tx_full = xr.open_dataset('intermediate/tx_on_tx_full.nc')
tw_on_tx_full = xr.open_dataset('intermediate/tw_on_tx_full.nc')
tx_on_tw_full = xr.open_dataset('intermediate/tx_on_tw_full.nc')

tx_era5_annual = xr.open_dataset('era5_global_mean_tx_annual.nc')

# Assuming the necessary data and libraries are already loaded
# Modify the shape of the slope arrays to store bootstrapped trends
num_bootstraps = 100

lat_start = (decile*20)
lat_end = min(lat_start+20, tx_tw_corr_full.latitude.size)

r_tx_tw_vs_tw_on_tx_slope = np.full([lat_end-lat_start, tx_tw_corr_full.longitude.size, num_bootstraps], np.nan)
r_tx_tw_vs_tw_on_tx_int = np.full([lat_end-lat_start, tx_tw_corr_full.longitude.size, num_bootstraps], np.nan)

r_tx_tw_vs_tx_on_tw_slope = np.full([lat_end-lat_start, tx_tw_corr_full.longitude.size, num_bootstraps], np.nan)
r_tx_tw_vs_tx_on_tw_int = np.full([lat_end-lat_start, tx_tw_corr_full.longitude.size, num_bootstraps], np.nan)

r_tx_tw_slope = np.full([lat_end-lat_start, tx_tw_corr_full.longitude.size, num_bootstraps], np.nan)
r_tx_tw_int = np.full([lat_end-lat_start, tx_tw_corr_full.longitude.size, num_bootstraps], np.nan)

r_tx_tw_slope_per_deg = np.full([lat_end-lat_start, tx_tw_corr_full.longitude.size, num_bootstraps], np.nan)
r_tx_tw_int_per_deg = np.full([lat_end-lat_start, tx_tw_corr_full.longitude.size, num_bootstraps], np.nan)

for xlat in range(lat_start, lat_end):
    print(xlat)
    for ylon in range(tx_tw_corr_full.longitude.size):
        if ~np.isnan(tx_tw_corr_full.tx_tw_corr[0, xlat, ylon]):
            for bootstrap in range(num_bootstraps):
                
                # Bootstrap sampling with replacement
                indices = np.random.choice(np.arange(tx_tw_corr_full.tx_tw_corr.shape[0]), size=tx_tw_corr_full.tx_tw_corr.shape[0], replace=True)
                annual_tx_ts = tx_era5_annual.mx2t[indices, xlat, ylon]
                tx_tw_corr_ts = tx_tw_corr_full.tx_tw_corr[indices, xlat, ylon]
                tw_on_tx_ts = (tw_on_tx_full.tw[indices, -1, xlat, ylon].values - tw_on_tw_full.tw[indices, -1, xlat, ylon].values)
                tx_on_tw_ts = (tx_on_tw_full.mx2t[indices, -1, xlat, ylon].values - tx_on_tx_full.mx2t[indices, -1, xlat, ylon].values)

                nn_ts = np.where((~np.isnan(tx_tw_corr_ts)) & (~np.isnan(tw_on_tx_ts)) & (~np.isnan(tx_on_tw_ts)) & (tx_on_tw_ts>-100) & (tw_on_tx_ts>-100))[0]
                
                if nn_ts.size < 30: break
                
                tx_tw_corr_ts_detrend = signal.detrend(tx_tw_corr_ts[nn_ts])
                tw_on_tx_ts_detrend = signal.detrend(tw_on_tx_ts[nn_ts])
                tx_on_tw_ts_detrend = signal.detrend(tx_on_tw_ts[nn_ts])
                

                X = sm.add_constant(tx_tw_corr_ts_detrend)
                mdl = sm.OLS(tw_on_tx_ts_detrend, X).fit()
                r_tx_tw_vs_tw_on_tx_int[xlat-lat_start, ylon, bootstrap] = mdl.params[0]
                r_tx_tw_vs_tw_on_tx_slope[xlat-lat_start, ylon, bootstrap] = mdl.params[1]
                
                X = sm.add_constant(tx_tw_corr_ts_detrend)
                mdl = sm.OLS(tx_on_tw_ts_detrend, X).fit()
                r_tx_tw_vs_tx_on_tw_int[xlat-lat_start, ylon, bootstrap] = mdl.params[0]
                r_tx_tw_vs_tx_on_tw_slope[xlat-lat_start, ylon, bootstrap] = mdl.params[1]

                X = sm.add_constant(indices)
                mdl = sm.OLS(tx_tw_corr_ts.values, X).fit()
                r_tx_tw_int_per_deg[xlat-lat_start, ylon, bootstrap] = mdl.params[0]
                r_tx_tw_slope_per_deg[xlat-lat_start, ylon, bootstrap] = mdl.params[1]

                X = sm.add_constant(indices)
                mdl = sm.OLS(tx_tw_corr_ts.values, X).fit()
                r_tx_tw_int[xlat-lat_start, ylon, bootstrap] = mdl.params[0]
                r_tx_tw_slope[xlat-lat_start, ylon, bootstrap] = mdl.params[1]
                
                

with open(f'era5_r_tx_tw_slope_bootstrap_100_decile{decile}_2_27_5pm.dat', 'wb') as f:
    pickle.dump({'r_tx_tw_int':r_tx_tw_int, 'r_tx_tw_slope':r_tx_tw_slope,
                 'r_tx_tw_int_per_deg':r_tx_tw_int, 'r_tx_tw_slope_per_deg':r_tx_tw_slope,
                 'r_tx_tw_vs_tx_on_tw_int':r_tx_tw_vs_tx_on_tw_int, 'r_tx_tw_vs_tx_on_tw_slope':r_tx_tw_vs_tx_on_tw_slope,
                 'r_tx_tw_vs_tw_on_tx_int':r_tx_tw_vs_tw_on_tx_int, 'r_tx_tw_vs_tw_on_tx_slope':r_tx_tw_vs_tw_on_tx_slope}, f)