
import sys, os, time

for y, year in enumerate(range(1982,2021+1)):
    
    print('running %d'%year)
    
    os.system('screen -d -m ipython amp_calc_era5_tx_tw_std.py %d'%(year))
    time.sleep(15)
    

    