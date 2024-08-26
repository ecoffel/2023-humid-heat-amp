import sys
import xarray as xr
import numpy as np
import pandas as pd
import os

decile_var = 'tw'
percentile = .99

dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'


# Load the temperature dataset for the specified year
if decile_var == 'tx':
    thresh_file_path = f'tx_{percentile}_thresholds_1981_2021.nc'
    if not os.path.isfile(thresh_file_path):
        print('creating percentiles file')
        hist_temp_data = xr.open_mfdataset(f'{dirEra5}/daily/tasmax_*.nc')
        hist_temp_data = hist_temp_data.sel(time=slice('1981','2021'))
        hist_temp_data = hist_temp_data.chunk(dict(time=-1))
        temp_thresh = hist_temp_data.mx2t.quantile(percentile, dim='time')
        temp_thresh.to_netcdf(thresh_file_path)
    else:
        print('loading percentiles file')
        temp_thresh = xr.open_dataset(thresh_file_path)
        
elif decile_var == 'tw':
    thresh_file_path = f'tw_{percentile}_thresholds_1981_2021.nc'
    if not os.path.isfile(thresh_file_path):
        print('creating percentiles file')
        hist_temp_data = xr.open_mfdataset(f'{dirEra5}/daily/tw_max_*.nc')
        hist_temp_data = hist_temp_data.sel(time=slice('1981','2021'))
        hist_temp_data = hist_temp_data.chunk(dict(time=-1))
        temp_thresh = hist_temp_data.tw.quantile(percentile, dim='time')
        temp_thresh.to_netcdf(thresh_file_path)
    else:
        print('loading percentiles file')
        temp_thresh = xr.open_dataset(thresh_file_path)

for year in range(1981,2022):
    if decile_var == 'tx':
        file_path = f'{dirEra5}/daily/tasmax_{year}.nc'
    elif decile_var == 'tw':
        file_path = f'{dirEra5}/daily/tw_max_{year}.nc'

    ds_temperature = xr.open_dataset(file_path)
    
    print(f'calc percentiles for {year}')
    # Calculate the temperature threshold for the specified percentile
    if decile_var == 'tx':
        temperature_data = ds_temperature['mx2t']
    elif decile_var == 'tw':
        temperature_data = ds_temperature['tw']
    
    
    # Count the number of days above the threshold
    days_above_threshold = (temperature_data > temp_thresh).sum(dim='time')

    # Save the results to a netcdf file
    if decile_var == 'tx':
        output_file = f"output/days_above_tx_pct/days_above_tx{percentile}_{year}.nc"
    elif decile_var == 'tw':
        output_file = f"output/days_above_tw_pct/days_above_tw{percentile}_{year}.nc"
    days_above_threshold.to_netcdf(output_file)
