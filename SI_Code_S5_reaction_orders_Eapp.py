#!/usr/bin/env python3
"""Supplementary Code S5 — local reaction orders and apparent activation energies."""
import math
from pathlib import Path
import pandas as pd
import SI_Code_S1_core_mkm as core
import SI_Code_S2_TP_scan as scan
HERE=Path(__file__).resolve().parent
REGIMES=[('low_activity',573.0,10.0,5.0),('intermediate_activity',623.0,10.0,10.0),('high_activity',673.0,10.0,10.0)]
PRODUCTS=['CH4','CH3OH','C2H5OH']
DP=0.005; DT=1.0

def solve_near(T,pco,ph2,y):
    c=core.build_context(T,pco,ph2); r=core.solve_with_staging(c,y,max_nfev=10000)
    if r['max_rel_residual']>=1e-7: r=core.solve_steady_state(c,r['y'],max_nfev=18000,floor_rel=1e-6)
    return c,r

def run():
    rows=[]
    for name,T,pco,ph2 in REGIMES:
        _,cen=scan.center_solution(T); _,base=scan.solve_target(T,pco,ph2,cen['y']); y=base['y']
        cp,cm=solve_near(T,pco*math.exp(DP),ph2,y)[1],solve_near(T,pco*math.exp(-DP),ph2,y)[1]
        hp,hm=solve_near(T,pco,ph2*math.exp(DP),y)[1],solve_near(T,pco,ph2*math.exp(-DP),y)[1]
        tp,tm=solve_near(T+DT,pco,ph2,y)[1],solve_near(T-DT,pco,ph2,y)[1]
        maxres=max(x['max_rel_residual'] for x in (cp,cm,hp,hm,tp,tm))
        for prod in PRODUCTS:
            nco=(math.log(cp['product_TOF'][prod])-math.log(cm['product_TOF'][prod]))/(2*DP)
            nh2=(math.log(hp['product_TOF'][prod])-math.log(hm['product_TOF'][prod]))/(2*DP)
            slope=(math.log(tp['product_TOF'][prod])-math.log(tm['product_TOF'][prod]))/(1/(T+DT)-1/(T-DT))
            Eapp=-core.KB_EV*slope
            rows.append(dict(regime=name,T_K=T,P_CO_bar=pco,P_H2_bar=ph2,product=prod,
                **{'TOF_site-1_s-1':base['product_TOF'][prod]},n_CO=nco,n_H2=nh2,Eapp_eV=Eapp,
                Eapp_kJ_mol=Eapp*core.EV_TO_KJMOL,max_perturbed_residual=maxres))
    return pd.DataFrame(rows)

def main():
    df=run(); p=HERE/'SI_Output_S5_reaction_orders_Eapp.csv'; df.to_csv(p,index=False); print('Saved',p)
if __name__=='__main__': main()
