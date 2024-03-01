
import sys, os, time

for b in range(25):
    
    print(f'running {b}')
    os.system(f'screen -d -m ipython amp_calc_era5_tx_tw_corr-bootstrapped-trend.py {b}')

    