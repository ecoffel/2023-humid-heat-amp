
import sys, os, time

cmip6_models = ['access-cm2',
                'bcc-csm2-mr', 'canesm5', 'cmcc-esm2',
                'fgoals-g3', 'inm-cm4-8',
                'inm-cm5-0', 'kace-1-0-g',
                'mpi-esm1-2-hr', 'mpi-esm1-2-lr',
                'mri-esm2-0', 'noresm2-lm', 'taiesm1']

for model in range(0, len(cmip6_models)):
    print(f'running {cmip6_models[model]}')
    os.system(f'screen -d -m ipython amp_calc_cmip6_n_days_above_temp_pct.py {cmip6_models[model]}')

    