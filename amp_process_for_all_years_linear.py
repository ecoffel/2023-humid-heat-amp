
import sys, os, time

for y, year in enumerate(range(26,30+1)):
    
    print('running %d'%year)
    os.system('ipython amp_calc_lens_tx_tw_corr.py %d'%(year))

    