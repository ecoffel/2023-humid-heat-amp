
import sys, os, time

for y, year in enumerate(range(1981, 2020+1)):
    print('running %d'%year)
    os.system('ipython amp_calc_era5_huss_on_txx.py %d'%(year))

    