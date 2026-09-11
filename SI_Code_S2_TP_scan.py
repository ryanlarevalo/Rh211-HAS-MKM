#!/usr/bin/env python3
"""Supplementary Code S2 — audited 100-point T/P survey."""
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
import SI_Code_S1_core_mkm as core

HERE=Path(__file__).resolve().parent
T_LIST=[523.0,573.0,623.0,673.0]
PCO_LIST=[0.1,0.3,1.0,3.0,10.0]
PH2_LIST=[0.5,1.0,2.0,5.0,10.0]


def center_solution(T):
    c=core.build_context(573.0,1.0,2.0)
    r=core.solve_steady_state(c,core.load_baseline_guess(),max_nfev=5000,floor_rel=1e-6)
    y=r['y']
    if T==573.0: return c,r
    step=10.0 if T>573 else -10.0
    temps=list(np.arange(573.0+step,T,step))+[T]
    for tt in temps:
        c=core.build_context(float(tt),1.0,2.0)
        r=core.solve_with_staging(c,y,max_nfev=6000); y=r['y']
    return c,r


def solve_target(T,pco,ph2,center_y):
    # Continuation first in CO then H2. Retry with a finer path if required.
    for nsteps in (8,16,28):
        try:
            c0=core.build_context(T,1.0,2.0)
            r=core.solve_with_staging(c0,center_y,max_nfev=6000); y=r['y']
            if pco!=1.0:
                for lp in np.linspace(0.0,math.log(pco),nsteps+1)[1:]:
                    c=core.build_context(T,math.exp(lp),2.0)
                    r=core.solve_with_staging(c,y,max_nfev=6000); y=r['y']
            if ph2!=2.0:
                for lp in np.linspace(math.log(2.0),math.log(ph2),nsteps+1)[1:]:
                    c=core.build_context(T,pco,math.exp(lp))
                    r=core.solve_with_staging(c,y,max_nfev=6000); y=r['y']
            c=core.build_context(T,pco,ph2)
            r=core.solve_with_staging(c,y,max_nfev=8000)
            if r['max_rel_residual']<1e-8: return c,r
        except Exception:
            pass
    raise RuntimeError(f'Failed to converge physical branch at T={T}, PCO={pco}, PH2={ph2}')


def row_from_result(T,pco,ph2,r):
    s=core.organic_selectivity(r['product_TOF'])
    return dict(T_K=T,P_CO_bar=pco,P_H2_bar=ph2,
        CO_cov=r['coverages']['CO'],H_cov=r['coverages']['H'],vacancy=r['coverages']['*'],
        CH4_TOF=r['product_TOF']['CH4'],MeOH_TOF=r['product_TOF']['CH3OH'],
        EtOH_TOF=r['product_TOF']['C2H5OH'],PrOH_TOF=r['product_TOF']['C3H7OH'],
        CO2_TOF=r['product_TOF']['CO2'],S_CH4=s['CH4'],S_MeOH=s['CH3OH'],
        S_EtOH=s['C2H5OH'],S_PrOH=s['C3H7OH'],
        max_rel_residual=r['max_rel_residual'],max_abs_balance=r['max_abs_balance'],success=r['max_rel_residual']<1e-8)


def run_scan():
    rows=[]
    for T in T_LIST:
        _,center=center_solution(T); cy=center['y']
        for pco in PCO_LIST:
            for ph2 in PH2_LIST:
                _,r=solve_target(T,pco,ph2,cy)
                rows.append(row_from_result(T,pco,ph2,r))
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',default=str(HERE/'SI_Output_S2_TP_grid.csv'))
    args=ap.parse_args(); df=run_scan(); df.to_csv(args.csv,index=False)
    print('Saved',args.csv); print('points',len(df),'max scaled residual',df.max_rel_residual.max())

if __name__=='__main__': main()
