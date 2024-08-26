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

era5_tx_on_tx_full = xr.open_dataset('intermediate/tx_on_tx_full.nc')
era5_tw_on_tx_full = xr.open_dataset('intermediate/tw_on_tx_full.nc')
# era5_tw_on_tx_full['tw'][-1,:,:,:] += 273.15

era5_tw_on_tw_full = xr.open_dataset('intermediate/tw_on_tw_full.nc')
era5_tx_on_tw_full = xr.open_dataset('intermediate/tx_on_tw_full.nc')
era5_huss_on_txx_full = xr.open_dataset('intermediate/huss_on_txx.nc')
era5_huss_on_tww_full = xr.open_dataset('intermediate/huss_on_tww.nc')

era5_vpd_daily_mean_on_tww_full = xr.open_dataset('intermediate/vpd_daily_mean_on_tww.nc')
era5_vpd_daily_mean_on_txx_full = xr.open_dataset('intermediate/vpd_daily_mean_on_txx.nc')

era5_n_days_above_tx90_full = xr.open_dataset('intermediate/n_days_above_tx90.nc')
era5_n_days_above_tx99_full = xr.open_dataset('intermediate/n_days_above_tx99.nc')

era5_n_days_above_tw90_full = xr.open_dataset('intermediate/n_days_above_tw90.nc')
era5_n_days_above_tw99_full = xr.open_dataset('intermediate/n_days_above_tw99.nc')

era5_tx_tw_corr_full = xr.open_dataset('intermediate/tx_tw_corr.nc')

era5_huss_on_txx_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_huss_on_txx_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_huss_on_tww_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_huss_on_tww_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

era5_vpd_daily_mean_on_txx_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_vpd_daily_mean_on_txx_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

era5_vpd_daily_mean_on_tww_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_vpd_daily_mean_on_tww_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

era5_n_days_above_tx_90_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_n_days_above_tx_90_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

era5_n_days_above_tx_99_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_n_days_above_tx_99_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

era5_n_days_above_tw_90_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_n_days_above_tw_90_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

era5_n_days_above_tw_99_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_n_days_above_tw_99_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

era5_tw_on_txx_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_tw_on_txx_per_rtet_int = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_tw_on_txx_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

era5_tw_on_txx_minus_tw_on_tww_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_tw_on_txx_minus_tw_on_tww_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

era5_tw_on_tww_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_tw_on_tww_per_rtet_int = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_tw_on_tww_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

era5_tx_on_tww_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_tx_on_tww_per_rtet_int = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_tx_on_tww_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

era5_tx_on_tww_minus_tx_on_txx_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_tx_on_tww_minus_tx_on_txx_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

era5_tx_on_txx_per_rtet = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_tx_on_txx_per_rtet_int = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)
era5_tx_on_txx_per_rtet_r2 = np.full([era5_huss_on_txx_full.latitude.size, era5_huss_on_txx_full.longitude.size], np.nan)

for xlat in range(era5_huss_on_txx_full.latitude.size):
    if xlat%25==0 and xlat > 10:
        print(xlat)
        
    for ylon in range(era5_huss_on_txx_full.longitude.size):
        

        v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
        v2 = era5_n_days_above_tw99_full.tw[:, xlat, ylon]
        
        nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
        if nn.size > 20:
            v1 = signal.detrend(v1[nn])
            v2 = signal.detrend(v2[nn])

            X = sm.add_constant(v1[nn])
            mdl = sm.OLS(v2[nn], X).fit()

            era5_n_days_above_tw_99_per_rtet[xlat, ylon] = mdl.params[1]
            era5_n_days_above_tw_99_per_rtet_r2[xlat, ylon] = mdl.rsquared
        
        v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
        v2 = era5_n_days_above_tw90_full.tw[:, xlat, ylon]
        
        nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
        if nn.size > 20:
            v1 = signal.detrend(v1[nn])
            v2 = signal.detrend(v2[nn])

            X = sm.add_constant(v1[nn])
            mdl = sm.OLS(v2[nn], X).fit()

            era5_n_days_above_tw_90_per_rtet[xlat, ylon] = mdl.params[1]
            era5_n_days_above_tw_90_per_rtet_r2[xlat, ylon] = mdl.rsquared
            
        
#         v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
#         v2 = era5_n_days_above_tx99_full.mx2t[:, xlat, ylon]
        
#         nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
#         if nn.size > 20:
#             v1 = signal.detrend(v1[nn])
#             v2 = signal.detrend(v2[nn])

#             X = sm.add_constant(v1[nn])
#             mdl = sm.OLS(v2[nn], X).fit()

#             era5_n_days_above_tx_99_per_rtet[xlat, ylon] = mdl.params[1]
#             era5_n_days_above_tx_99_per_rtet_r2[xlat, ylon] = mdl.rsquared
        
#         v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
#         v2 = era5_n_days_above_tx90_full.mx2t[:, xlat, ylon]
        
#         nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
#         if nn.size > 20:
#             v1 = signal.detrend(v1[nn])
#             v2 = signal.detrend(v2[nn])

#             X = sm.add_constant(v1[nn])
#             mdl = sm.OLS(v2[nn], X).fit()

#             era5_n_days_above_tx_90_per_rtet[xlat, ylon] = mdl.params[1]
#             era5_n_days_above_tx_90_per_rtet_r2[xlat, ylon] = mdl.rsquared
            
            
        
#         v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
#         v2 = era5_vpd_daily_mean_on_tww_full.vpd_daily_mean_on_tww[:, xlat, ylon]
        
#         nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
#         if nn.size > 20:
#             v1 = signal.detrend(v1[nn])
#             v2 = signal.detrend(v2[nn])

#             X = sm.add_constant(v1[nn])
#             mdl = sm.OLS(v2[nn], X).fit()

#             era5_vpd_daily_mean_on_tww_per_rtet[xlat, ylon] = mdl.params[1]
#             era5_vpd_daily_mean_on_tww_per_rtet_r2[xlat, ylon] = mdl.rsquared
        
#         v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
#         v2 = era5_vpd_daily_mean_on_txx_full.vpd_daily_mean_on_txx[:, xlat, ylon]
        
#         nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
#         if nn.size > 20:
#             v1 = signal.detrend(v1[nn])
#             v2 = signal.detrend(v2[nn])

#             X = sm.add_constant(v1[nn])
#             mdl = sm.OLS(v2[nn], X).fit()

#             era5_vpd_daily_mean_on_txx_per_rtet[xlat, ylon] = mdl.params[1]
#             era5_vpd_daily_mean_on_txx_per_rtet_r2[xlat, ylon] = mdl.rsquared
        
        
        
        
#         v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
#         v2 = era5_huss_on_txx_full.huss_on_txx[:, xlat, ylon]
        
#         nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
#         if nn.size > 20:
#             v1 = signal.detrend(v1[nn])
#             v2 = signal.detrend(v2[nn])

#             X = sm.add_constant(v1[nn])
#             mdl = sm.OLS(v2[nn], X).fit()

#             era5_huss_on_txx_per_rtet[xlat, ylon] = mdl.params[1]
#             era5_huss_on_txx_per_rtet_r2[xlat, ylon] = mdl.rsquared
            
#         v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
#         v2 = era5_huss_on_tww_full.huss_on_tww[:, xlat, ylon]
        
#         nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
#         if nn.size > 20:
#             v1 = signal.detrend(v1[nn])
#             v2 = signal.detrend(v2[nn])

#             X = sm.add_constant(v1[nn])
#             mdl = sm.OLS(v2[nn], X).fit()

#             era5_huss_on_tww_per_rtet[xlat, ylon] = mdl.params[1]
#             era5_huss_on_tww_per_rtet_r2[xlat, ylon] = mdl.rsquared


#         v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
#         v2 = era5_tx_on_tx_full.tx_on_txx[:, xlat, ylon]
        
#         nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
#         if nn.size > 20:
#             v1 = signal.detrend(v1[nn])
#             v2 = signal.detrend(v2[nn])

#             X = sm.add_constant(v1[nn])
#             mdl = sm.OLS(v2[nn], X).fit()

#             era5_tx_on_txx_per_rtet_int[xlat, ylon] = mdl.params[0]
#             era5_tx_on_txx_per_rtet[xlat, ylon] = mdl.params[1]
#             era5_tx_on_txx_per_rtet_r2[xlat, ylon] = mdl.rsquared


        
#         v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
#         v2 = era5_tw_on_tx_full.tw_on_txx[:, xlat, ylon]
        
#         nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
#         if nn.size > 20:
#             v1 = signal.detrend(v1[nn])
#             v2 = signal.detrend(v2[nn])

#             X = sm.add_constant(v1[nn])
#             mdl = sm.OLS(v2[nn], X).fit()

#             era5_tw_on_txx_per_rtet_int[xlat, ylon] = mdl.params[0]
#             era5_tw_on_txx_per_rtet[xlat, ylon] = mdl.params[1]
#             era5_tw_on_txx_per_rtet_r2[xlat, ylon] = mdl.rsquared
            
            
            
            
#         v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
#         v2 = era5_tw_on_tx_full.tw_on_txx[:, xlat, ylon] - era5_tw_on_tw_full.tw_on_tww[:, xlat, ylon]
        
#         nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
#         if nn.size > 20:
#             v1 = signal.detrend(v1[nn])
#             v2 = signal.detrend(v2[nn])

#             X = sm.add_constant(v1[nn])
#             mdl = sm.OLS(v2[nn], X).fit()

#             era5_tw_on_txx_minus_tw_on_tww_per_rtet[xlat, ylon] = mdl.params[1]
#             era5_tw_on_txx_minus_tw_on_tww_per_rtet_r2[xlat, ylon] = mdl.rsquared
            
            
            
            
        


#         v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
#         v2 = era5_tw_on_tw_full.tw_on_tww[:, xlat, ylon]
        
#         nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
#         if nn.size > 20:
#             v1 = signal.detrend(v1[nn])
#             v2 = signal.detrend(v2[nn])

#             X = sm.add_constant(v1[nn])
#             mdl = sm.OLS(v2[nn], X).fit()

#             era5_tw_on_tww_per_rtet_int[xlat, ylon] = mdl.params[0]
#             era5_tw_on_tww_per_rtet[xlat, ylon] = mdl.params[1]
#             era5_tw_on_tww_per_rtet_r2[xlat, ylon] = mdl.rsquared


#         v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
#         v2 = era5_tx_on_tw_full.tx_on_tww[:, xlat, ylon]
        
#         nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
#         if nn.size > 20:
#             v1 = signal.detrend(v1[nn])
#             v2 = signal.detrend(v2[nn])

#             X = sm.add_constant(v1[nn])
#             mdl = sm.OLS(v2[nn], X).fit()

#             era5_tx_on_tww_per_rtet_int[xlat, ylon] = mdl.params[0]
#             era5_tx_on_tww_per_rtet[xlat, ylon] = mdl.params[1]
#             era5_tx_on_tww_per_rtet_r2[xlat, ylon] = mdl.rsquared
            
            
            
#         v1 = era5_tx_tw_corr_full.tx_tw_corr[:,xlat,ylon]
#         v2 = era5_tx_on_tw_full.mx2t[:, 0, xlat, ylon] - era5_tx_on_tx_full.mx2t[:, 0, xlat, ylon]
        
#         nn = np.where((~np.isnan(v1)) & (~np.isnan(v2)))[0]
        
#         if nn.size > 20:
#             v1 = signal.detrend(v1[nn])
#             v2 = signal.detrend(v2[nn])

#             X = sm.add_constant(v1[nn])
#             mdl = sm.OLS(v2[nn], X).fit()

#             era5_tx_on_tww_minus_tx_on_txx_per_rtet[xlat, ylon] = mdl.params[1]
#             era5_tx_on_tww_minus_tx_on_txx_per_rtet_r2[xlat, ylon] = mdl.rsquared



with open('era5_n_days_above_tw_99_per_rtet.dat', 'wb') as f:
    pickle.dump(era5_n_days_above_tw_99_per_rtet, f)
with open('era5_n_days_above_tw_99_per_rtet_r2.dat', 'wb') as f:
    pickle.dump(era5_n_days_above_tw_99_per_rtet_r2, f)  

with open('era5_n_days_above_tw_90_per_rtet.dat', 'wb') as f:
    pickle.dump(era5_n_days_above_tw_90_per_rtet, f)
with open('era5_n_days_above_tw_90_per_rtet_r2.dat', 'wb') as f:
    pickle.dump(era5_n_days_above_tw_90_per_rtet_r2, f)    


# with open('era5_n_days_above_tx_99_per_rtet.dat', 'wb') as f:
#     pickle.dump(era5_n_days_above_tx_99_per_rtet, f)
# with open('era5_n_days_above_tx_99_per_rtet_r2.dat', 'wb') as f:
#     pickle.dump(era5_n_days_above_tx_99_per_rtet_r2, f)  

# with open('era5_n_days_above_tx_90_per_rtet.dat', 'wb') as f:
#     pickle.dump(era5_n_days_above_tx_90_per_rtet, f)
# with open('era5_n_days_above_tx_90_per_rtet_r2.dat', 'wb') as f:
#     pickle.dump(era5_n_days_above_tx_90_per_rtet_r2, f)    

# with open('era5_vpd_daily_mean_on_tww_per_rtet.dat', 'wb') as f:
#     pickle.dump(era5_vpd_daily_mean_on_tww_per_rtet, f)
# with open('era5_vpd_daily_mean_on_tww_per_rtet_r2.dat', 'wb') as f:
#     pickle.dump(era5_vpd_daily_mean_on_tww_per_rtet_r2, f)    

# with open('era5_vpd_daily_mean_on_txx_per_rtet.dat', 'wb') as f:
#     pickle.dump(era5_vpd_daily_mean_on_txx_per_rtet, f)
# with open('era5_vpd_daily_mean_on_txx_per_rtet_r2.dat', 'wb') as f:
#     pickle.dump(era5_vpd_daily_mean_on_txx_per_rtet_r2, f)    


# with open('era5_huss_on_txx_per_rtet.dat', 'wb') as f:
#     pickle.dump(era5_huss_on_txx_per_rtet, f)
# with open('era5_huss_on_txx_per_rtet_r2.dat', 'wb') as f:
#     pickle.dump(era5_huss_on_txx_per_rtet_r2, f)    
# with open('era5_huss_on_tww_per_rtet.dat', 'wb') as f:
#     pickle.dump(era5_huss_on_tww_per_rtet, f)
# with open('era5_huss_on_tww_per_rtet_r2.dat', 'wb') as f:
#     pickle.dump(era5_huss_on_tww_per_rtet_r2, f)    

# with open('era5_tx_on_txx_per_rtet_int.dat', 'wb') as f:
#     pickle.dump(era5_tx_on_txx_per_rtet_int, f)
# with open('era5_tx_on_txx_per_rtet.dat', 'wb') as f:
#     pickle.dump(era5_tx_on_txx_per_rtet, f)
# with open('era5_tx_on_txx_per_rtet_r2.dat', 'wb') as f:
#     pickle.dump(era5_tx_on_txx_per_rtet_r2, f)
    
# with open('era5_tw_on_txx_per_rtet_int.dat', 'wb') as f:
#     pickle.dump(era5_tw_on_txx_per_rtet_int, f)
# with open('era5_tw_on_txx_per_rtet.dat', 'wb') as f:
#     pickle.dump(era5_tw_on_txx_per_rtet, f)
# with open('era5_tw_on_txx_per_rtet_r2.dat', 'wb') as f:
#     pickle.dump(era5_tw_on_txx_per_rtet_r2, f)    
    
# with open('era5_tx_on_tww_minus_tx_on_txx_per_rtet.dat', 'wb') as f:
#     pickle.dump(era5_tx_on_tww_minus_tx_on_txx_per_rtet, f)
# with open('era5_tx_on_tww_minus_tx_on_txx_per_rtet_r2.dat', 'wb') as f:
#     pickle.dump(era5_tx_on_tww_minus_tx_on_txx_per_rtet_r2, f)    

# with open('era5_tw_on_tww_per_rtet_int.dat', 'wb') as f:
#     pickle.dump(era5_tw_on_tww_per_rtet_int, f)
# with open('era5_tw_on_tww_per_rtet.dat', 'wb') as f:
#     pickle.dump(era5_tw_on_tww_per_rtet, f)
# with open('era5_tw_on_tww_per_rtet_r2.dat', 'wb') as f:
#     pickle.dump(era5_tw_on_tww_per_rtet_r2, f)    

# with open('era5_tx_on_tww_per_rtet_int.dat', 'wb') as f:
#     pickle.dump(era5_tx_on_tww_per_rtet_int, f)
# with open('era5_tx_on_tww_per_rtet.dat', 'wb') as f:
#     pickle.dump(era5_tx_on_tww_per_rtet, f)
# with open('era5_tx_on_tww_per_rtet_r2.dat', 'wb') as f:
#     pickle.dump(era5_tx_on_tww_per_rtet_r2, f)    

# with open('era5_tw_on_txx_minus_tw_on_tww_per_rtet.dat', 'wb') as f:
#     pickle.dump(era5_tw_on_txx_minus_tw_on_tww_per_rtet, f)
# with open('era5_tw_on_txx_minus_tw_on_tww_per_rtet_r2.dat', 'wb') as f:
#     pickle.dump(era5_tw_on_txx_minus_tw_on_tww_per_rtet_r2, f)    