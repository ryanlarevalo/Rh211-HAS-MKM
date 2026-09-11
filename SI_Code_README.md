# Final supplementary Python code — Rh(211) higher-alcohol microkinetic model

This directory contains the **audited final implementation** corresponding to the manuscript *Carbonyl Recycling Limits Productive Ethanol Formation during Syngas Conversion on Rh(211): A First-Principles Microkinetic Analysis*.

## Final model convention

- **45-reaction DFT inventory:** S1–S45.
- **Thermodynamic reconciliation:** performed on all 45 reactions. The inventory has rank 28 and 17 independent cycles.
- **Operational steady-state kinetic network:** 44 reactions, S1–S37 and S39–S45.
- **S38** (`CO* + OH* <=> HCOO* + *`) remains in the DFT/vibrational/thermodynamic inventory but is omitted from the steady-state kinetic equations because the isolated formate branch is kinetically frozen over the studied conditions and the DFT database contains no independent downstream or gas-surface boundary process for HCOO*.
- **H* vibrational free energy:** least-squares decomposition using all 45 DFT reaction-state datasets.
- **Surface representation:** one effective Rh(211) site population.
- **Coverage variables:** site-balance-preserving log ratios, `y_i = ln(theta_i/theta_*)`; no arbitrary individual-coverage ceiling is imposed.
- **Feed adsorption A1/A2:** nonactivated quasi-equilibrated CO adsorption and dissociative H2 adsorption.
- **Product boundaries D1–D6:** reversible gas-surface boundary closure.
- **Product pressures:** zero for the reported differential/inlet-limit calculations.

## Files

- `SI_Code_S1_core_mkm.py` — thermochemistry, 45-reaction reconciliation, A/D boundaries, 44-reaction kinetic model, site balance, and nonlinear steady-state solver.
- `SI_Code_S2_TP_scan.py` — 100-point T/P survey: T = 523, 573, 623, 673 K; PCO = 0.1, 0.3, 1, 3, 10 bar; PH2 = 0.5, 1, 2, 5, 10 bar.
- `SI_Code_S3_DRC.py` — ethanol degree of rate control using symmetric ±0.001 eV transition-state perturbations while keeping reaction thermodynamics fixed.
- `SI_Code_S4_flux_analysis.py` — net surface-reaction and product-boundary fluxes at the low-, intermediate-, and high-activity representative states.
- `SI_Code_S5_reaction_orders_Eapp.py` — local CO/H2 reaction orders using symmetric logarithmic ±0.5% pressure perturbations and apparent activation energies using ±1 K.
- `SI_Code_S6_reproduce_all.py` — master workflow.
- `SI_Code_S7_validate_against_QA.py` — optional comparison of regenerated outputs with the archived final-QA tables.

Companion inputs:

- `SI_Data_S1_surface_frequencies.json`
- `SI_Data_S2_reaction_parameters.json`
- `SI_Data_S3_baseline_initial_guess.json`
- `SI_Data_S4_center_guesses.json` (archived continuation seeds; HCOO is ignored by the final 44-reaction solver)

Archived reference outputs beginning with `FINAL_QA_` are included **only for independent validation**. They are not read by the kinetic solver or used to generate rates.

## Representative regimes

- Low activity: 573 K, PCO = 10 bar, PH2 = 5 bar
- Intermediate activity: 623 K, PCO = 10 bar, PH2 = 10 bar
- High activity: 673 K, PCO = 10 bar, PH2 = 10 bar

Activity classification is based on calculated ethanol TOF.

## Requirements

Python 3.10+ with:

```text
numpy
scipy
pandas
```

## Reproduction

From this directory:

```bash
python SI_Code_S6_reproduce_all.py
```

The full 100-point survey and sensitivity calculations are numerically stiff and may require substantial CPU time. Individual analyses can be run separately.

## Numerical notes

The solver normalizes each dynamic-species balance by its gross formation/consumption rate plus a small relative floor tied to the maximum gross rate. This prevents kinetically dormant species from dominating the nonlinear least-squares problem while retaining the physical steady-state branch. Low-temperature states are obtained by logarithmic continuation and retry with progressively finer continuation paths.
