
import sys, os, time

for y, year in enumerate(range(2001,2019+1)):
    
    print('running %d'%year)
    os.system('ipython amp_calc_era5_var_on_warm_season_deciles.py %d'%(year))

    