#!/usr/bin/env python3
"""Supplementary Code S3 — ethanol degree of rate control for 44 kinetic reactions."""
import math
from pathlib import Path
import pandas as pd
import SI_Code_S1_core_mkm as core
import SI_Code_S2_TP_scan as scan
HERE=Path(__file__).resolve().parent
REGIMES=[('low_activity',573.0,10.0,5.0),('intermediate_activity',623.0,10.0,10.0),('high_activity',673.0,10.0,10.0)]
DELTA=0.001

def base_solution(T,pco,ph2):
    _,cen=scan.center_solution(T); return scan.solve_target(T,pco,ph2,cen['y'])

def run():
    rows=[]
    for name,T,pco,ph2 in REGIMES:
        ctx,base=base_solution(T,pco,ph2)
        for code in core.KINETIC_CODES:
            rr=[]
            for delta in (+DELTA,-DELTA):
                pc=core.perturb_transition_state(ctx,code,delta)
                r=core.solve_with_staging(pc,base['y'],max_nfev=12000)
                if r['max_rel_residual']>=1e-7:
                    r=core.solve_steady_state(pc,r['y'],max_nfev=20000,floor_rel=1e-6)
                rr.append(r)
            low,high=rr
            X=(math.log(high['product_TOF']['C2H5OH'])-math.log(low['product_TOF']['C2H5OH']))/(2*DELTA/(core.KB_EV*T))
            rows.append(dict(regime=name,code=code,type='surface_TS',X_EtOH=X,
                             residual_low=low['max_rel_residual'],residual_high=high['max_rel_residual']))
    return pd.DataFrame(rows)

def main():
    df=run(); path=HERE/'SI_Output_S3_DRC_three_regimes.csv'; df.to_csv(path,index=False)
    print('Saved',path)
    for reg,g in df.groupby('regime'): print(reg,'sum X =',g.X_EtOH.sum())
if __name__=='__main__': main()
