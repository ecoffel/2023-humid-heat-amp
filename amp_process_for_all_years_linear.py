
import sys, os, time

for y, year in enumerate(range(1988,2020+1)):
    
    print('running %d'%year)
    os.system('ipython amp_calc_era5_huss_sm_corr.py %d'%(year))

    