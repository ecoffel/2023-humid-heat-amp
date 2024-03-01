# import rasterio as rio
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize
import numpy as np
import numpy.matlib
from scipy import interpolate
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
from scipy import signal
import scipy
import os, sys, pickle, gzip
import datetime
# import geopy.distance
import xarray as xr
import pandas as pd
# import geopandas as gpd
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
# import metpy
# from metpy.plots import USCOUNTIES
import xgrid_utils
import pickle

from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress

import warnings
warnings.filterwarnings('ignore')

# we have tw historical for awi-cm-1-1-lr and ssp245 for awi-cm-1-1-mr

cmip6_models = ['access-cm2',
                'bcc-csm2-mr', 'canesm5', 'cmcc-esm2',
                'fgoals-g3', 'inm-cm4-8',
                'inm-cm5-0', 'kace-1-0-g',
                'mpi-esm1-2-hr', 'mpi-esm1-2-lr',
                'mri-esm2-0', 'noresm2-lm', 'taiesm1']
# cmip6_models = ['taiesm1']

cmip6_lat = np.arange(-90, 90, 1.5)
cmip6_lon = np.arange(0, 360, 1.5)

regridMesh_global = xr.Dataset({'lat': (['lat'], np.arange(-90, 90, 1.5)),
                                'lon': (['lon'], np.arange(0, 360, 1.5)),})

# should be the number of the model
m = int(sys.argv[1])

dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'
dirCMIP6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/CMIP6'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'
dirAg6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/research/2020-ag-cmip6'
dirUtil = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/research/util'
dataDirLens = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Data-edcoffel-F20/LENS/daily/atm'

land_sea_mask = xr.open_dataset('%s/land-sea-mask.nc'%dirEra5)
land_sea_mask.load()
land_sea_mask = land_sea_mask.lsm.mean(dim='time')
land_sea_mask_binary = land_sea_mask > 0.1

with open('cmip6_tx_tw_hist_fut.pkl', 'rb') as file:
    tw_hist, tw_fut, tw_chg, tx_hist, tx_fut, tx_chg = pickle.load(file)
    

tx_chg = []
tw_chg = []
tw_max_chg = []
tx_max_chg = []

global_mean_tx_ts_hist = []
global_mean_tx_ts_fut = []
global_mean_tw_ts_hist = []
global_mean_tw_ts_fut = []

tx_ann_max_chg_spatial = []
tw_ann_max_chg_spatial = []

print(f'processing {cmip6_models[m]}')
tx_hist_time_mean = tx_hist[m].sel(time=slice('1981', '2010')).mean(dim='time')
tx_fut_time_mean = tx_fut[m].sel(time=slice('2070', '2100')).mean(dim='time')

tx_hist_time_mean = tx_hist_time_mean.sel(lat=slice(-60,60))
tx_fut_time_mean = tx_fut_time_mean.sel(lat=slice(-60,60))

tx_hist_spatial_mean = xgrid_utils.calc_spatial_mean(tx_hist_time_mean.tasmax.rename({'lat':'latitude', 'lon':'longitude'}))
tx_hist_max = tx_hist_time_mean.tasmax.max(dim=['lat', 'lon'])
tx_fut_spatial_mean = xgrid_utils.calc_spatial_mean(tx_fut_time_mean.tasmax.rename({'lat':'latitude', 'lon':'longitude'}))
tx_fut_max = tx_fut_time_mean.tasmax.max(dim=['lat', 'lon'])

tx_hist_ann_max = tx_hist[m].sel(time=slice('1981', '2010')).tasmax.resample(time='Y').max().mean(dim='time')
tx_fut_ann_max = tx_fut[m].sel(time=slice('2070', '2100')).tasmax.resample(time='Y').max().mean(dim='time')

cur_global_mean_tx_ts_hist = xgrid_utils.calc_spatial_mean(tx_hist[m].sel(time=slice('1981', '2020')).tasmax.resample(time='Y').max().rename({'lat':'latitude', 'lon':'longitude'}))
cur_global_mean_tx_ts_fut = xgrid_utils.calc_spatial_mean(tx_fut[m].sel(time=slice('2070', '2100')).tasmax.resample(time='Y').max().rename({'lat':'latitude', 'lon':'longitude'}))

tx_chg = ((tx_fut_spatial_mean.load() - tx_hist_spatial_mean.load()).values.tolist())
tx_max_chg = ((tx_fut_max.load() - tx_hist_max.load()).values.tolist())
tx_ann_max_chg_spatial = ((tx_fut_ann_max.load() - tx_hist_ann_max.load()))

global_mean_tx_ts_hist = (cur_global_mean_tx_ts_hist.load())
global_mean_tx_ts_fut = (cur_global_mean_tx_ts_fut.load())


tw_hist_time_mean = tw_hist[m].sel(time=slice('1981', '2010')).mean(dim='time')
tw_fut_time_mean = tw_fut[m].sel(time=slice('2070', '2100')).mean(dim='time')

tw_hist_time_mean = tw_hist_time_mean.sel(lat=slice(-60,60))
tw_fut_time_mean = tw_fut_time_mean.sel(lat=slice(-60,60))

tw_hist_spatial_mean = xgrid_utils.calc_spatial_mean(tw_hist_time_mean.tw.rename({'lat':'latitude', 'lon':'longitude'}))
tw_hist_max = tw_hist_time_mean.tw.max(dim=['lat', 'lon'])
tw_fut_spatial_mean = xgrid_utils.calc_spatial_mean(tw_fut_time_mean.tw.rename({'lat':'latitude', 'lon':'longitude'}))
tw_fut_max = tw_fut_time_mean.tw.max(dim=['lat', 'lon'])

tw_hist_ann_max = tw_hist[m].sel(time=slice('1981', '2010')).tw.resample(time='Y').max().mean(dim='time')
tw_fut_ann_max = tw_fut[m].sel(time=slice('2070', '2100')).tw.resample(time='Y').max().mean(dim='time')

cur_global_mean_tw_ts_hist = xgrid_utils.calc_spatial_mean(tw_hist[m].sel(time=slice('1981', '2020')).tw.resample(time='Y').max().rename({'lat':'latitude', 'lon':'longitude'}))
cur_global_mean_tw_ts_fut = xgrid_utils.calc_spatial_mean(tw_fut[m].sel(time=slice('2070', '2100')).tw.resample(time='Y').max().rename({'lat':'latitude', 'lon':'longitude'}))

tw_chg = ((tw_fut_spatial_mean.load() - tw_hist_spatial_mean.load()).values.tolist())
tw_max_chg = ((tw_fut_max.load() - tw_hist_max.load()).values.tolist())
tw_ann_max_chg_spatial = ((tw_fut_ann_max.load() - tw_hist_ann_max.load()))

global_mean_tw_ts_hist = (cur_global_mean_tw_ts_hist.load())
global_mean_tw_ts_fut = (cur_global_mean_tw_ts_fut.load())


with open(f'cmip6_tx_tw_spatial_mean_max_{m}.pkl', 'wb') as file:
    pickle.dump((tx_chg, tw_chg, tw_max_chg, tx_max_chg, tx_ann_max_chg_spatial, tw_ann_max_chg_spatial,\
                 global_mean_tx_ts_hist, global_mean_tx_ts_fut, global_mean_tw_ts_hist, global_mean_tw_ts_fut), file)
