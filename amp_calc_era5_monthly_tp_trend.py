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
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'

m = int(sys.argv[1])

ds_era5_tp = []
for y in range(1981, 2019+1):
    cur_ds_era5_tp = xr.open_mfdataset(f'era5_tp_monthly_{y}.nc')
    ds_era5_tp.append(cur_ds_era5_tp)

ds_era5_tp = xr.concat(ds_era5_tp, dim='time')
ds_era5_tp.load()

era5_tp_trend = np.full([ds_era5_tp.latitude.size, ds_era5_tp.longitude.size], np.nan)

print('MONTH ------------ ', m)
cur_month_ts = ds_era5_tp.sel(time=ds_era5_tp['time'].dt.month == m+1)

for xlat in range(ds_era5_tp.latitude.size):
    if xlat%50==0:print(xlat)
    for ylon in range(ds_era5_tp.longitude.size):

        cur_tp_ts = cur_month_ts.tp.values[:, xlat, ylon]
        nn = np.where(~np.isnan(cur_tp_ts))[0]

        X = sm.add_constant(np.arange(cur_tp_ts.size))
        mdl = sm.OLS(cur_tp_ts[nn], X).fit()

        era5_tp_trend[xlat, ylon] = mdl.params[1]
with open(f'era5_tp_monthly_trend_{m}.dat', 'wb') as f:
    pickle.dump(era5_tp_trend, f)