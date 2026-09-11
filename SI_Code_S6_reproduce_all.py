#!/usr/bin/env python3
"""Supplementary Code S6 — master reproduction workflow."""
import subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
for script in ['SI_Code_S2_TP_scan.py','SI_Code_S3_DRC.py','SI_Code_S4_flux_analysis.py','SI_Code_S5_reaction_orders_Eapp.py','SI_Code_S7_validate_against_QA.py']:
    print('\n===',script,'===')
    subprocess.run([sys.executable,str(HERE/script)],check=True,cwd=HERE)
