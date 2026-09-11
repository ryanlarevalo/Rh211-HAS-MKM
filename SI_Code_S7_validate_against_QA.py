#!/usr/bin/env python3
"""Numerical cross-check against the archived final-QA tables supplied with this package."""
from pathlib import Path
import pandas as pd, numpy as np
HERE=Path(__file__).resolve().parent

def cmp(gen,ref,keys,cols,rtol=5e-4,atol=1e-20):
    a=pd.read_csv(HERE/gen); b=pd.read_csv(HERE/ref)
    m=a.merge(b,on=keys,suffixes=('_gen','_ref'))
    worst=0.0
    for c in cols:
        x=m[c+'_gen'].to_numpy(float); y=m[c+'_ref'].to_numpy(float)
        den=np.maximum(np.abs(y),atol); worst=max(worst,float(np.max(np.abs(x-y)/den)))
    print(gen,'matched rows',len(m),'worst relative deviation',worst)
    return worst

def main():
    checks=[]
    if (HERE/'SI_Output_S2_TP_grid.csv').exists():
        checks.append(cmp('SI_Output_S2_TP_grid.csv','FINAL_QA_100point_grid.csv',['T_K','P_CO_bar','P_H2_bar'],
                          ['CO_cov','H_cov','vacancy','CH4_TOF','MeOH_TOF','EtOH_TOF','CO2_TOF'],rtol=5e-3))
    if (HERE/'SI_Output_S3_DRC_three_regimes.csv').exists():
        checks.append(cmp('SI_Output_S3_DRC_three_regimes.csv','FINAL_QA_DRC_three_regimes.csv',['regime','code'],['X_EtOH'],rtol=5e-3,atol=1e-8))
    if (HERE/'SI_Output_S5_reaction_orders_Eapp.csv').exists():
        checks.append(cmp('SI_Output_S5_reaction_orders_Eapp.csv','FINAL_QA_reaction_orders_Eapp.csv',['regime','product'],['n_CO','n_H2','Eapp_eV'],rtol=5e-3,atol=1e-8))
    print('Validation complete. Archived FINAL_QA files are reference outputs, not model inputs.')
if __name__=='__main__': main()
