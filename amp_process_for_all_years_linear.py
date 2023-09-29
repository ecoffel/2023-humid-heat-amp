
import sys, os, time

for y, year in enumerate(range(1981,1999+1)):
    
    print('running %d'%year)
    os.system('ipython amp_calc_q_adv_on_warm_season_deciles.py %d'%(year))

    