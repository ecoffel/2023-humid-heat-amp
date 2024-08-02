
import sys, os, time

for y, year in enumerate(range(1986,2022+1)):
    
    print('running %d'%year)
    os.system('screen -d -m ipython amp_calc_era5_vpd_on_txx.py %d'%(year))
    
    

    