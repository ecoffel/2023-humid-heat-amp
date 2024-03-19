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
from statsmodels.tsa.stattools import ccf

import random
from scipy import signal


import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress

import warnings
warnings.filterwarnings('ignore')

lag = int(sys.argv[1])

dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'

from datetime import datetime
def add_time_dim(xda):
    xda = xda.expand_dims(time = [datetime.now()])
    return xda

land_sea_mask = xr.open_dataset('%s/land-sea-mask.nc'%dirEra5)
land_sea_mask.load()
land_sea_mask = land_sea_mask.lsm.mean(dim='time')
land_sea_mask_binary = land_sea_mask > 0.1

annual_max_months_da_tx = xr.open_dataarray("txx_months_1981_2021.nc").sel(latitude=slice(60,-60))
annual_max_months_da_tw = xr.open_dataarray("tw_months_1981_2021.nc").sel(latitude=slice(60,-60))

annual_max_month_mean_tx = annual_max_months_da_tx.mean(dim='year')
annual_max_month_mean_tx.load()

time_dim = pd.date_range("1981-01-01", "2021-12-31", freq="AS")
tx_tw_corr_full = xr.open_mfdataset('output/tx_tw_corr/tx_tw_corr_warm_season_*.nc', preprocess = add_time_dim, concat_dim='time')
tx_tw_corr_full['time'] = time_dim
tx_tw_corr_full = tx_tw_corr_full.rename({'__xarray_dataarray_variable__':'tx_tw_corr'})
tx_tw_corr_full = tx_tw_corr_full.where(land_sea_mask_binary)

tx_tw_corr_full = tx_tw_corr_full.sel(latitude=slice(60,-60))
tx_tw_corr_full.load()


era5_monthly_tp = []
for y in range(1981, 2019+1):
    cur_ds_era5_tp = xr.open_mfdataset(f'era5_tp_monthly_{y}.nc')
    era5_monthly_tp.append(cur_ds_era5_tp)

ds_era5_monthly_tp = xr.concat(era5_monthly_tp, dim='time')
ds_era5_monthly_tp.load()


print('LAG ------------ ', lag)

monthly_tp_slope = np.full([tx_tw_corr_full.latitude.size, tx_tw_corr_full.longitude.size], np.nan)
monthly_tp_corr_r2 = np.full([tx_tw_corr_full.latitude.size, tx_tw_corr_full.longitude.size], np.nan)
monthly_tp_corr = np.full([6, tx_tw_corr_full.latitude.size, tx_tw_corr_full.longitude.size], np.nan)

for xlat in range(tx_tw_corr_full.latitude.size):
    if xlat%50 == 0: print(xlat)
    for ylon in range(tx_tw_corr_full.longitude.size):
        
        cur_max_month = int(np.round(annual_max_month_mean_tx.values[xlat, ylon]))
                
        if cur_max_month-lag <= 0:
            cur_tp = ds_era5_monthly_tp.tp[:, xlat, ylon].sel(time=ds_era5_monthly_tp['time'].dt.month == cur_max_month-lag+12)
            cur_corr_ts = tx_tw_corr_full.tx_tw_corr[0:39, xlat, ylon]
            cur_tp = cur_tp.roll(time=1).values
            cur_corr_ts = cur_corr_ts.roll(time=1).values
        else:
            cur_tp = ds_era5_monthly_tp.tp[:, xlat, ylon].sel(time=ds_era5_monthly_tp['time'].dt.month == cur_max_month-lag).values
            cur_corr_ts = tx_tw_corr_full.tx_tw_corr[0:39, xlat, ylon].values
        
        nn = np.where((~np.isnan(cur_tp)) & ~np.isnan(cur_corr_ts))[0]
        
        if nn.size > 10:
            
            cur_tp_ts_detrend = signal.detrend(cur_tp[nn])
            cur_corr_ts_detrend = signal.detrend(cur_corr_ts[nn])
        
            X = sm.add_constant(cur_tp_ts_detrend)
            mdl = sm.OLS(cur_corr_ts_detrend, X).fit()

            monthly_tp_slope[xlat, ylon] = mdl.params[1]
            monthly_tp_corr_r2[xlat, ylon] = mdl.rsquared
            
            lags = ccf(cur_tp, cur_corr_ts, unbiased=True)
            monthly_tp_corr[:, xlat, ylon] = lags[0:6]

with open(f'era5_monthly_tp_r_tx_tw_lag{lag}.dat', 'wb') as f:
    pickle.dump({'era5_monthly_tp_r_tx_tw_slope':monthly_tp_slope, 'era5_monthly_tp_r_tx_tw_r2':monthly_tp_corr_r2, 'monthly_tp_corr':monthly_tp_corr}, f)