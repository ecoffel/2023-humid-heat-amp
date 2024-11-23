import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Directory containing the data files
dirHadisd = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Data-edcoffel-F20/hadisd'
data_dir = f'{dirHadisd}/stations/'

# Output NetCDF file to save progress
output_netcdf = f'{dirHadisd}/station_correlations_with_latlon.nc'

# Error log file
error_log_file = f'{dirHadisd}/error_log.txt'

# Ensure the error log file exists
open(error_log_file, 'a').close()

# Find all .nc files in the directory
nc_files = glob.glob(os.path.join(data_dir, '*.nc'))

# Initialize arrays to store correlation trends and p-values
cor_trends = []
cor_p = []

# Load existing NetCDF file if it exists, otherwise create a new one
if os.path.exists(output_netcdf):
    ds_out = xr.open_dataset(output_netcdf)
    existing_stations = ds_out.station_id.values.tolist()
    ds_out.close()
else:
    existing_stations = []

n = 0

for file in nc_files:
    try:
        # Open the dataset
        data_hum = xr.open_dataset(file)

        # Extract station metadata
        station_id = data_hum.attrs.get('station_id', os.path.basename(file).split('_')[0])

        station_lon = data_hum.attrs.get('longitude', None)
        station_lat = data_hum.attrs.get('latitude', None)

        # If not in attrs, check for longitude and latitude as variables or coordinates
        if station_lon is None and 'longitude' in data_hum:
            station_lon = float(data_hum['longitude'].values[0])

        if station_lat is None and 'latitude' in data_hum:
            station_lat = float(data_hum['latitude'].values[0])

        if station_id in existing_stations:
            n += 1
            print(f"Station {station_id} already processed, skipping.")
            continue

        if n % 10 == 0:
            print(f'{n} of {len(nc_files)}')
        n += 1

        data_hum.load()

        # Calculate daily maximum temperature
        daily_max_temp = data_hum['temperatures'].resample(time='1D').max()
        daily_max_tw = data_hum['wet_bulb_temperature'].resample(time='1D').max()

        # Combine the results into a new Dataset
        daily_data = xr.Dataset({
            'daily_max_temperature': daily_max_temp,
            'daily_max_tw': daily_max_tw
        })

        hottest_times = []
        hottest_months = set()

        for y in np.unique(daily_data.time.dt.year):
            cur_year = daily_data['daily_max_temperature'].sel(time=daily_data.time.dt.year.isin([y]))
            cur_year_txx = cur_year.max(dim="time")

            if np.isnan(cur_year_txx.values) or np.where(~np.isnan(cur_year))[0].size < 300:
                continue

            cur_year_txx_idx = cur_year.argmax(dim="time")
            cur_year_txx_time = cur_year.time[cur_year_txx_idx]
            cur_year_txx_month = cur_year_txx_time.dt.month

            hottest_times.append(cur_year_txx_time)
            hottest_months.add(int(cur_year_txx_month.values))

        # Convert the set of months to a sorted list
        hottest_months = sorted(hottest_months)

        # Select data for only the hottest months
        selected_data = daily_data.sel(time=daily_data.time.dt.month.isin(hottest_months))

        # Initialize a dictionary to store correlations by year
        correlations = {}

        # Loop through each year
        for year, group in selected_data.groupby("time.year"):
            # Convert the group to a pandas DataFrame for correlation calculation
            df = group.to_dataframe()
            # Calculate the correlation between temperature and specific humidity for this year
            correlation = df['daily_max_temperature'].corr(df['daily_max_tw'])
            # Store the result
            correlations[year] = correlation

        # Convert to a pandas Series for easier viewing and manipulation
        correlations_series = pd.Series(correlations)

        # Filter out NaN values from the correlation series
        correlations_series_clean = correlations_series.dropna()

        # Add the cleaned correlation series and metadata to the NetCDF file
        new_data = xr.Dataset(
            {
                'correlations': ('year', correlations_series_clean.values),
                'station_longitude': (('station',), [station_lon]),
                'station_latitude': (('station',), [station_lat]),
                'station_id': (('station',), [station_id]),
            },
            coords={
                'year': correlations_series_clean.index,
                'station': [station_id]
            }
        )
        
        
        # Append the new station data to the NetCDF file
        if os.path.exists(output_netcdf):
            # Open the existing NetCDF file and combine it with the new data
            with xr.open_dataset(output_netcdf) as ds_existing:
                ds_combined = xr.concat([ds_existing, new_data], dim='station')
            # Save the combined dataset to the output file
            ds_combined.to_netcdf(output_netcdf, mode='w')
        else:
            new_data.to_netcdf(output_netcdf, mode='w')

    except Exception as e:
        # Write the error to the log file
        with open(error_log_file, 'a') as error_log:
            error_log.write(f"Error processing {file}: {e}\n")
        continue

print("Processing complete. Results saved.")
