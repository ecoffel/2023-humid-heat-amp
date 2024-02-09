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
import pickle

import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress

import warnings
warnings.filterwarnings('ignore')

dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'
dirCMIP6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/CMIP6'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'
dirAg6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/research/2020-ag-cmip6'
dirUtil = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/research/util'
dataDirLens = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Data-edcoffel-F20/LENS/daily/atm'

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


# REGRID ALL THE LENS MODELS AND CALC TX_TW_CORR
lens_tx_spatial_means = []
for member in range(1,41):
    print('opening lens member %d'%member)
    lens_tx = xr.open_mfdataset(f'{dataDirLens}/TASMAX/HIST-RCP85/tasmax_day_CESM1-CAM5_historical_rcp85_r{member}i1p1_*.nc')

    lens_tx = lens_tx.sel(time=slice('1981', '2021')).tasmax-273.15

    time_dim = pd.date_range("1981-01-01", "2021-12-31", freq="D")
    time_dim_no_leap = time_dim[~((time_dim.month == 2) & (time_dim.day == 29))]

    lens_tx['time'] = time_dim_no_leap

    # lens_tx = lens_tx.reindex(lat=lens_tx.lat[::-1])
    lens_tx.load();

    lens_tx_spatial_mean = xgrid_utils.calc_spatial_mean(lens_tx.sel(time=slice('1981', '2021')).resample(time='Y').max().rename({'lat':'latitude', 'lon':'longitude'}))
    lens_tx_spatial_means.append(lens_tx_spatial_mean)
    
with open(f'lens_global_mean_tx.pkl', 'wb') as file:
    pickle.dump((lens_tx_spatial_means), file)

