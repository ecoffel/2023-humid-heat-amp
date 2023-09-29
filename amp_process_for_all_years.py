
import sys, os, time

for y, year in enumerate(range(1988,2021+1)):
    
    print('running %d'%year)
    
    time.sleep()
    os.system('screen -d -m ipython amp_calc_era5_cdd_on_warm_season.py %d'%(year))
    
    

    