#!/usr/bin/env python3
"""Supplementary Code S4 — net reaction and product fluxes at representative states."""
from pathlib import Path
import pandas as pd
import SI_Code_S1_core_mkm as core
import SI_Code_S2_TP_scan as scan
HERE=Path(__file__).resolve().parent
REGIMES=[('low_activity',573.0,10.0,5.0),('intermediate_activity',623.0,10.0,10.0),('high_activity',673.0,10.0,10.0)]

def run():
    rows=[]
    for name,T,pco,ph2 in REGIMES:
        _,cen=scan.center_solution(T); ctx,r=scan.solve_target(T,pco,ph2,cen['y'])
        for code in core.KINETIC_CODES:
            rows.append(dict(regime=name,process=code,type='surface',net_flux_site_m1_s_m1=r['net_flux'][code]))
        for gas,_ in core.PRODUCT_MAP:
            rows.append(dict(regime=name,process='D_'+gas,type='gas_surface_boundary',net_flux_site_m1_s_m1=r['product_TOF'][gas]))
    return pd.DataFrame(rows)

def main():
    df=run(); p=HERE/'SI_Output_S4_net_fluxes_three_regimes.csv'; df.to_csv(p,index=False); print('Saved',p)
if __name__=='__main__': main()
