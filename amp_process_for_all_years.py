
import sys, os, time

for y, year in enumerate(range(2013,2021+1)):
    
    print('running %d'%year)
    
    os.system('screen -d -m ipython amp_calc_warm_season_deciles.py %d'%(year))
    time.sleep(120)
    

    