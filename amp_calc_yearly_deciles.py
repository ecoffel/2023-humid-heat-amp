import xarray as xr
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
import cartopy
import cartopy.crs as ccrs
import glob
import sys
import datetime

dirERA5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'

file_var = 'huss'
orig_var = 'q'

year = int(sys.argv[1])

latRange = [-90, 90]
lonRange = [0, 360]

land_sea_mask = xr.open_dataset('%s/land-sea-mask.nc'%dirERA5)
land_sea_mask.load()
N_gridcells = np.where((land_sea_mask.lsm[0,:,:].values.reshape([land_sea_mask.latitude.size*land_sea_mask.longitude.size]))>0)[0].size
# land_sea_mask = land_sea_mask.rename({'latitude':'lat', 'longitude':'lon'})


print('opening era5 %d...'%year)
era5_var = xr.open_dataset('%s/daily/%s_%d.nc'%(dirERA5, file_var, year))
era5_var.load()
era5_var = era5_var[orig_var]
# era5_var = era5_var.rename({'latitude':'lat', 'longitude':'lon'})
# era5_var = era5_var[orig_var] - 273.15

# target_grid = xr.Dataset({
#     'lat': land_sea_mask['lat'],
#     'lon': land_sea_mask['lon']
# })

# # Create a regridder object using the source and target grids
# regridder = xe.Regridder(era5_var.swvl1, target_grid, 'bilinear', reuse_weights=True)

# # Regrid the era5_soil_moisture DataArray
# era5_var = regridder(era5_var.swvl1)

# era5_var = era5_var.rename({'lat':'latitude', 'lon':'longitude'})

# decile bins
bins = np.arange(0, 101, 1)
            
# this will hold the decile cutoffs for every grid cell for the current year
yearly_era5_deciles = np.full([era5_var.latitude.size, era5_var.longitude.size, bins.size], np.nan)

n = 0
# loop over all latitudes
for xlat in range(era5_var.latitude.size):

    # loop over all longitudes
    for ylon in range(era5_var.longitude.size):

        # skip water grid cells
        if land_sea_mask.lsm[0, xlat, ylon] < 0.1: continue
        
        # print out progress through loop
        if n % 25000 == 0:
            print('%.2f%%'%(n/N_gridcells*100))

        curTw = era5_var[:, xlat, ylon]
        if len(curTw) > 0:
            yearly_era5_deciles[xlat, ylon, :] = np.nanpercentile(curTw, bins)
        n += 1

print('renaming dims...')

da_grow_era5_var_deciles = xr.DataArray(data   = yearly_era5_deciles, 
                      dims   = ['latitude', 'longitude', 'bin'],
                      coords = {'bin':bins, 'latitude':era5_var.latitude, 'longitude':era5_var.longitude},
                      attrs  = {'units'     : 'C'
                        })
ds_grow_era5_var_deciles = xr.Dataset()
ds_grow_era5_var_deciles['%s_deciles'%orig_var] = da_grow_era5_var_deciles


print('saving netcdf...')
ds_grow_era5_var_deciles.to_netcdf('deciles/q/era5_%s_deciles_%d.nc'%(file_var, year))
