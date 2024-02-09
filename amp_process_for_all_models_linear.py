
import sys, os, time

cmip6_models = ['access-cm2',
                'bcc-csm2-mr', 'canesm5', 'cmcc-esm2',
                'fgoals-g3', 'inm-cm4-8',
                'inm-cm5-0', 'kace-1-0-g',
                'mpi-esm1-2-hr', 'mpi-esm1-2-lr',
                'mri-esm2-0', 'noresm2-lm', 'taiesm1']

for model in range(len(cmip6_models)):
    
    print(f'running {model}')
    os.system(f'ipython amp_calc_cmip6_tx_tw_corr.py {model}')

    