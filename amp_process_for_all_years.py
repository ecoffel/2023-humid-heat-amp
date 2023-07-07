
import sys, os, time

for y, year in enumerate(range(1982,2021+1)):
    
    print('running %d'%year)
    
    os.system('screen -d -m ipython amp_calc_era5_huss_sm_corr.py %d'%(year))
    time.sleep(120)
    

    