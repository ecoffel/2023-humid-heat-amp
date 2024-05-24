
import sys, os, time

for y, year in enumerate(range(1981,2001+1)):
    
    print('running %d'%year)
    os.system('screen -d -m ipython amp_calc_era5_evap_on_txx.py %d'%(year))
    
    

    