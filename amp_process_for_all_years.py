
import sys, os, time

for y, year in enumerate(range(1981,2021+1)):
    
    print('running %d'%year)
    
    os.system('screen -d -m ipython amp_calc_era5_et_on_warm_season.py %d'%(year))
    time.sleep(5)
    

    