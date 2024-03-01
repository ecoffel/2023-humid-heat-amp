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

tw_hist = []
tw_fut = []
tw_chg = []

tx_hist = []
tx_fut = []
tx_chg = []


for m in cmip6_models:
    print(f'opening {m}')
    cur_tw_hist = xr.open_mfdataset(f"{dirCMIP6}/{m}/r1i1p1f1/historical/tw/tw_daily*.nc", concat_dim='time')
    cur_tw_fut = xr.open_mfdataset(f"{dirCMIP6}/{m}/r1i1p1f1/ssp245/tw/tw_daily*.nc", concat_dim='time')

    cur_tx_hist = xr.open_mfdataset(f"{dirCMIP6}/{m}/r1i1p1f1/historical/tasmax/tasmax_day_*.nc", concat_dim='time')
    cur_tx_fut = xr.open_mfdataset(f"{dirCMIP6}/{m}/r1i1p1f1/ssp245/tasmax/tasmax_day_*.nc", concat_dim='time')
    cur_tx_hist['tasmax'] -= 273.15
    cur_tx_fut['tasmax'] -= 273.15

    tw_hist.append(cur_tw_hist)
    tw_fut.append(cur_tw_fut)

    tx_hist.append(cur_tx_hist)
    tx_fut.append(cur_tx_fut)

with open(f'cmip6_tx_tw_hist_fut.pkl', 'wb') as file:
    pickle.dump((tw_hist, tw_fut, tw_chg, tx_hist, tx_fut, tx_chg), file)
