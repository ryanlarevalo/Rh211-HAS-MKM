#!/usr/bin/env python3
"""Supplementary Code S1 — final audited Rh(211) microkinetic model.

Model convention
----------------
* DFT/thermodynamic inventory: S1-S45 (45 reactions)
* Steady-state kinetic network: S1-S37 and S39-S45 (44 reactions)
* S38 (CO* + OH* <=> HCOO* + *) is retained in the DFT inventory,
  thermodynamic reconciliation, and vibrational bookkeeping, but excluded
  from steady-state kinetics because the isolated formate branch is
  kinetically frozen and has no downstream/boundary process in the DFT set.
* H* vibrational free energy is inferred from all 45 DFT reaction-state data.
* Coverages are represented with site-balance-preserving log-ratio variables.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
KB_EV = 8.617333262145e-5
H_EV_S = 4.135667696e-15
HC_EV_CM = 1.2398419843320026e-4
KB_SI = 1.380649e-23
H_SI = 6.62607015e-34
C_SI = 299792458.0
AMU = 1.66053906660e-27
P0_PA = 1.0e5
R_KJMOLK = 8.31446261815324e-3
EV_TO_KJMOL = 96.48533212331002

INVENTORY_CODES = [f"S{i}" for i in range(1, 46)]
KINETIC_CODES = [c for c in INVENTORY_CODES if c != "S38"]

GAS_DATA = {
    "CO": {"mass":28.0101,"linear":True,"B":[1.92253],"sigma":1,"freqs":[2143.0]},
    "H2": {"mass":2.01588,"linear":True,"B":[60.853],"sigma":2,"freqs":[4161.0]},
    "CH4": {"mass":16.04246,"linear":False,"B":[5.24120,5.24120,5.24120],"sigma":12,
            "freqs":[2917,1534,1534,3019,3019,3019,1306,1306,1306]},
    "CH3OH": {"mass":32.04186,"linear":False,"B":[4.25730,0.82338,0.79273],"sigma":1,
              "freqs":[3681,3000,2844,1477,1455,1345,1060,1033,2960,1477,1165,200]},
    "C2H5OH": {"mass":46.06844,"linear":False,"B":[1.16386,0.31190,0.27136],"sigma":1,
               "freqs":[3653,2984,2939,2900,1490,1464,1412,1371,1256,1091,1028,888,417,
                        2991,2910,1446,1275,1161,812,251,200]},
    "C3H7OH": {"mass":60.09502,"linear":False,"B":[0.47869,0.17055,0.14393],"sigma":1,
               "freqs":[3683,3021,2995,2967,2931,2927,2918,2884,1473,1462,1449,1433,1393,
                        1369,1332,1272,1223,1189,1119,1070,1029,946,893,835,746,458,314,239,216,136]},
    "H2O": {"mass":18.01528,"linear":False,"B":[27.877,14.512,9.285],"sigma":2,
            "freqs":[3657,1595,3756]},
    "CO2": {"mass":44.0095,"linear":True,"B":[0.39021],"sigma":2,
            "freqs":[1333,667,667,2349]},
}

ADS_DE = {
    "CO": -2.270, "H2": -1.298, "CH4": -0.3474, "CH3OH": -0.750,
    "C2H5OH": -1.020, "C3H7OH": -1.144, "H2O": -0.750, "CO2": -0.650,
}
SINGLE_ADS_STATE = {
    "CO": ("S1","IS"), "CH4": ("S25","IS"), "CH3OH": ("S15","IS"),
    "C2H5OH": ("S16","IS"), "C3H7OH": ("S45","FS"),
    "H2O": ("S34","FS"), "CO2": ("S35","FS"),
}
PRODUCT_MAP = [
    ("CH4","CH4"), ("CH3OH","CH3OH"), ("C2H5OH","CH3CH2OH"),
    ("C3H7OH","CH3CH2CH2OH"), ("H2O","H2O"), ("CO2","CO2")
]


def load_inputs():
    freq = json.load(open(HERE / "SI_Data_S1_surface_frequencies.json"))
    raw = json.load(open(HERE / "SI_Data_S2_reaction_parameters.json"))
    params = {r["code"]: r for r in raw}
    assert set(INVENTORY_CODES) == set(freq) == set(params)
    return freq, params


def positive_freqs(vals):
    out = []
    for v in vals:
        if isinstance(v, str) and v.strip().lower().startswith("i"):
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if x > 0:
            out.append(x)
    return out


def fvib(freqs, T):
    vals = np.asarray(positive_freqs(freqs), float)
    if vals.size == 0:
        return 0.0
    eps = HC_EV_CM * vals
    x = eps / (KB_EV * T)
    return float(np.sum(0.5 * eps + KB_EV * T * np.log1p(-np.exp(-x))))


def gas_gcorr(spec, T):
    d = GAS_DATA[spec]
    m = d["mass"] * AMU
    qtrans = (2*np.pi*m*KB_SI*T/H_SI**2)**1.5 * (KB_SI*T/P0_PA)
    if d["linear"]:
        theta = H_SI*C_SI*(d["B"][0]*100.0)/KB_SI
        qrot = T/(d["sigma"]*theta)
    else:
        thetas = [H_SI*C_SI*(B*100.0)/KB_SI for B in d["B"]]
        qrot = np.sqrt(np.pi)*T**1.5/(d["sigma"]*np.sqrt(np.prod(thetas)))
    return fvib(d["freqs"],T) - KB_EV*T*np.log(qtrans) - KB_EV*T*np.log(qrot)


def inventory_thermochemistry(T):
    """Return 45-reaction raw/TC free energies and forward barriers."""
    freq, params = load_inputs()
    species = sorted({s for code in INVENTORY_CODES
                      for side in (params[code]["reactants"], params[code]["products"])
                      for s in side if s != "*"})
    idx = {s:i for i,s in enumerate(species)}
    N = np.zeros((len(species), len(INVENTORY_CODES)))
    raw_dg = np.zeros(len(INVENTORY_CODES))
    dga = np.zeros(len(INVENTORY_CODES))
    for j,code in enumerate(INVENTORY_CODES):
        r = params[code]
        for s,n in r["products"].items():
            if s != "*": N[idx[s],j] += n
        for s,n in r["reactants"].items():
            if s != "*": N[idx[s],j] -= n
        Fis, Fts, Ffs = (fvib(freq[code][q],T) for q in ("IS","TS","FS"))
        dga[j] = r["Ea_eV"] + Fts - Fis
        raw_dg[j] = r["dE_eV"] + Ffs - Fis
    g_opt, *_ = np.linalg.lstsq(N.T, raw_dg, rcond=None)
    tc_dg = N.T @ g_opt
    return {
        "species":species, "idx":idx, "N":N, "raw_dG":raw_dg,
        "tc_dG":tc_dg, "dGact":dga, "g_opt":g_opt,
        "rank":int(np.linalg.matrix_rank(N)),
        "cycle_dim":int(len(INVENTORY_CODES)-np.linalg.matrix_rank(N)),
    }


def fit_species_fvib_all45(T, thermo=None):
    """Least-squares state decomposition using all 45 DFT reaction-state datasets."""
    freq, params = load_inputs()
    if thermo is None:
        thermo = inventory_thermochemistry(T)
    species, idx = thermo["species"], thermo["idx"]
    rows, ys = [], []
    for code in INVENTORY_CODES:
        for state, side in (("IS",params[code]["reactants"]),("FS",params[code]["products"])):
            row = np.zeros(len(species))
            for s,n in side.items():
                if s != "*": row[idx[s]] += n
            rows.append(row)
            ys.append(fvib(freq[code][state],T))
    A = np.asarray(rows); y = np.asarray(ys)
    coef, *_ = np.linalg.lstsq(A,y,rcond=None)
    rms = float(np.sqrt(np.mean((A@coef-y)**2)))
    return {s:float(coef[idx[s]]) for s in species}, rms


def build_context(T, pco, ph2, product_pressures=None):
    freq, params = load_inputs()
    thermo45 = inventory_thermochemistry(T)
    spF, hfit_rms = fit_species_fvib_all45(T, thermo45)
    k0 = KB_EV*T/H_EV_S

    def adsF(spec):
        if spec == "H": return spF["H"]
        code,state = SINGLE_ADS_STATE[spec]
        return fvib(freq[code][state],T)

    boundary = {}
    for gas in ADS_DE:
        Fads = 2*adsF("H") if gas == "H2" else adsF(gas)
        dGads = ADS_DE[gas] + Fads - gas_gcorr(gas,T)
        Kads = float(np.exp(-dGads/(KB_EV*T)))
        boundary[gas] = {"dGads_eV":dGads,"Kads":Kads,"kdes":k0/Kads}

    alphaCO = boundary["CO"]["Kads"] * pco
    alphaH = math.sqrt(boundary["H2"]["Kads"] * ph2)

    species = sorted({s for code in KINETIC_CODES
                      for side in (params[code]["reactants"], params[code]["products"])
                      for s in side if s != "*"})
    idx = {s:i for i,s in enumerate(species)}
    M, NS = len(KINETIC_CODES), len(species)
    reac = np.zeros((NS,M)); prod = np.zeros((NS,M)); sr = np.zeros(M); sp = np.zeros(M)
    kf = np.zeros(M); kr = np.zeros(M)
    for j,code in enumerate(KINETIC_CODES):
        r = params[code]
        for s,n in r["reactants"].items():
            if s == "*": sr[j] += n
            else: reac[idx[s],j] += n
        for s,n in r["products"].items():
            if s == "*": sp[j] += n
            else: prod[idx[s],j] += n
        ii = INVENTORY_CODES.index(code)
        kf[j] = k0*np.exp(-thermo45["dGact"][ii]/(KB_EV*T))
        kr[j] = kf[j]*np.exp(thermo45["tc_dG"][ii]/(KB_EV*T))
    nu = prod-reac
    dynamic = [s for s in species if s not in ("CO","H")]
    didx = np.array([idx[s] for s in dynamic],int)
    if product_pressures is None:
        product_pressures = {g:0.0 for g,_ in PRODUCT_MAP}
    return locals()


def _state_from_y(ctx, y):
    """Convert y_i=ln(theta_i/theta*) to coverages satisfying the site balance exactly."""
    z = np.exp(np.clip(np.asarray(y,float), -745.0, 100.0))
    den = 1.0 + ctx["alphaCO"] + ctx["alphaH"] + float(z.sum())
    star = 1.0/den
    th = np.zeros(len(ctx["species"]))
    th[ctx["idx"]["CO"]] = ctx["alphaCO"]*star
    th[ctx["idx"]["H"]] = ctx["alphaH"]*star
    th[ctx["didx"]] = z*star
    return th, star


def evaluate(ctx, y, with_jac=False, floor_rel=1e-6):
    th, star = _state_from_y(ctx,y)
    lt = np.log(np.maximum(th,1e-300)); ls = math.log(star)
    mf = np.exp(lt@ctx["reac"] + ctx["sr"]*ls)
    mr = np.exp(lt@ctx["prod"] + ctx["sp"]*ls)
    rf = ctx["kf"]*mf; rr = ctx["kr"]*mr; net = rf-rr
    d = ctx["nu"]@net
    gross = np.abs(ctx["nu"])@(rf+rr)
    pflux = {}

    if with_jac:
        theta_dyn = th[ctx["didx"]]
        ER = ctx["reac"][ctx["didx"],:].T
        EP = ctx["prod"][ctx["didx"],:].T
        nrtot = ctx["sr"] + np.sum(ctx["reac"],axis=0)
        nptot = ctx["sp"] + np.sum(ctx["prod"],axis=0)
        Ar = ER - nrtot[:,None]*theta_dyn[None,:]
        Ap = EP - nptot[:,None]*theta_dyn[None,:]
        drdy = rf[:,None]*Ar - rr[:,None]*Ap
        Jfull = ctx["nu"]@drdy

    for gas,surf in PRODUCT_MAP:
        a = float(ctx["product_pressures"].get(gas,0.0))
        f = ctx["k0"]*a*star
        rev = ctx["boundary"][gas]["kdes"]*th[ctx["idx"][surf]]
        rb = f-rev
        d[ctx["idx"][surf]] += rb
        gross[ctx["idx"][surf]] += f+rev
        pflux[gas] = -rb
        if with_jac:
            theta_dyn = th[ctx["didx"]]
            row = -f*theta_dyn
            if surf in ctx["dynamic"]:
                k = ctx["dynamic"].index(surf)
                row -= rev*(-theta_dyn)
                row[k] -= rev
            else:
                row += rev*theta_dyn
            Jfull[ctx["idx"][surf],:] += row

    gmax = max(float(np.max(gross[ctx["didx"]])), 1e-300)
    scale = gross[ctx["didx"]] + floor_rel*gmax
    res = d[ctx["didx"]]/scale
    if with_jac:
        # Derivative of the residual scale is intentionally neglected; this is
        # a stable quasi-Newton Jacobian for the normalized balances.
        return res, Jfull[ctx["didx"],:]/scale[:,None], (th,star,net,rf,rr,d,gross,pflux)
    return res, (th,star,net,rf,rr,d,gross,pflux)


def guess_to_y(ctx, guess):
    if isinstance(guess, dict):
        vals = np.array([max(float(guess.get(s,1e-300)),1e-300) for s in ctx["dynamic"]])
        # Preserve the relative dynamic coverages supplied by the seed. The
        # vacant fraction consistent with A1/A2 and the site balance follows.
        total = min(float(vals.sum()), 0.999999999999)
        star = max((1-total)/(1+ctx["alphaCO"]+ctx["alphaH"]),1e-300)
        return np.log(vals/star)
    arr = np.asarray(guess,float)
    if arr.size != len(ctx["dynamic"]):
        raise ValueError("Initial vector has wrong length")
    return arr.copy()


def solve_steady_state(ctx, guess, max_nfev=4000, floor_rel=1e-6):
    """Solve the 44-reaction steady state in log-ratio variables."""
    y0 = guess_to_y(ctx,guess)
    def fun(y): return evaluate(ctx,y,False,floor_rel)[0]
    # Bounds are on log(theta_i/theta*), not on the coverages themselves;
    # the site balance is exact and there is no arbitrary individual-coverage ceiling.
    # Finite-difference Jacobians are used because they are more robust for the very
    # stiff low-temperature branch and dormant C3 subnetwork.
    lo = np.full(len(y0),-120.0); hi = np.full(len(y0),35.0)
    sol = least_squares(fun,np.clip(y0,lo+1e-12,hi-1e-12),jac="2-point",bounds=(lo,hi),
                        max_nfev=max_nfev,xtol=1e-13,ftol=1e-13,gtol=1e-13,
                        x_scale="jac",diff_step=1e-5)
    res,_,data = evaluate(ctx,sol.x,True,floor_rel)
    th,star,net,rf,rr,d,gross,pflux = data
    return {
        "y":sol.x,
        "coverages":{s:float(th[ctx["idx"][s]]) for s in ctx["species"]}|{"*":float(star)},
        "net_flux":{code:float(net[j]) for j,code in enumerate(KINETIC_CODES)},
        "forward_flux":{code:float(rf[j]) for j,code in enumerate(KINETIC_CODES)},
        "reverse_flux":{code:float(rr[j]) for j,code in enumerate(KINETIC_CODES)},
        "product_TOF":{k:float(v) for k,v in pflux.items()},
        "max_rel_residual":float(np.max(np.abs(res))),
        "max_abs_balance":float(np.max(np.abs(d[ctx["didx"]]))),
        "nfev":int(sol.nfev), "success":bool(sol.success), "message":sol.message,
    }


def solve_with_staging(ctx, guess, max_nfev=4000):
    """Robust two-stage solve used for the audited manuscript branch."""
    r1 = solve_steady_state(ctx,guess,max_nfev=max_nfev,floor_rel=1e-5)
    r2 = solve_steady_state(ctx,r1["y"],max_nfev=max_nfev,floor_rel=1e-6)
    return r2


def continuation(T,pco,ph2,seed,max_nfev=4000,nsteps=8):
    """Logarithmic continuation from (1 bar CO, 2 bar H2) to a target state."""
    c0 = build_context(T,1.0,2.0)
    r = solve_with_staging(c0,seed,max_nfev=max_nfev)
    y = r["y"]
    # First CO, then H2, matching the manuscript implementation.
    if pco != 1.0:
        for lp in np.linspace(0.0,math.log(pco),nsteps+1)[1:]:
            c = build_context(T,math.exp(lp),2.0)
            r = solve_with_staging(c,y,max_nfev=max_nfev); y=r["y"]
    if ph2 != 2.0:
        for lp in np.linspace(math.log(2.0),math.log(ph2),nsteps+1)[1:]:
            c = build_context(T,pco,math.exp(lp))
            r = solve_with_staging(c,y,max_nfev=max_nfev); y=r["y"]
    c = build_context(T,pco,ph2)
    r = solve_with_staging(c,y,max_nfev=max_nfev)
    return c,r


def organic_selectivity(tof):
    carbon = {"CH4":1,"CH3OH":1,"C2H5OH":2,"C3H7OH":3}
    denom = sum(carbon[k]*max(float(tof[k]),0.0) for k in carbon)
    if denom <= 0: return {k:0.0 for k in carbon}
    return {k:carbon[k]*max(float(tof[k]),0.0)/denom for k in carbon}


def load_baseline_guess():
    d = json.load(open(HERE/"SI_Data_S3_baseline_initial_guess.json"))
    d.pop("HCOO",None)
    return d


def load_center_guess(T):
    d = json.load(open(HERE/"SI_Data_S4_center_guesses.json"))[str(int(T))]
    d.pop("HCOO",None)
    return d


def perturb_transition_state(ctx, code, delta_eV):
    """Return shallow copy with forward/reverse rates of one step changed equally."""
    out = dict(ctx)
    out["kf"] = ctx["kf"].copy(); out["kr"] = ctx["kr"].copy()
    j = KINETIC_CODES.index(code)
    fac = math.exp(-delta_eV/(KB_EV*ctx["T"]))
    out["kf"][j] *= fac; out["kr"][j] *= fac
    return out


def thermo_reconciliation_stats(T):
    th = inventory_thermochemistry(T)
    corr = th["tc_dG"]-th["raw_dG"]
    # Null-space cycle residual norm from projection residual.
    U,s,Vt = np.linalg.svd(th["N"],full_matrices=True)
    # For the reaction-space nullspace use SVD of N.
    _,sN,VtN = np.linalg.svd(th["N"],full_matrices=True)
    rank = th["rank"]; C = VtN[rank:,:]
    raw_cycle = C@th["raw_dG"] if C.size else np.zeros(0)
    tc_cycle = C@th["tc_dG"] if C.size else np.zeros(0)
    rev_bar = th["dGact"]-th["tc_dG"]
    return {
        "T_K":T,"rank":rank,"cycle_dim":th["cycle_dim"],
        "RMS_correction_eV":float(np.sqrt(np.mean(corr*corr))),
        "max_abs_correction_eV":float(np.max(np.abs(corr))),
        "raw_cycle_residual_norm_eV":float(np.linalg.norm(raw_cycle)),
        "max_cycle_residual_after_eV":float(np.max(np.abs(tc_cycle))) if tc_cycle.size else 0.0,
        "min_reverse_barrier_eV":float(np.min(rev_bar)),
    }

if __name__ == "__main__":
    for T,pco,ph2 in [(573.0,10.0,5.0),(623.0,10.0,10.0),(673.0,10.0,10.0)]:
        ctx,res = continuation(T,pco,ph2,load_center_guess(T))
        sel = organic_selectivity(res["product_TOF"])
        print(T,pco,ph2,"CO",res["coverages"]["CO"],"EtOH",res["product_TOF"]["C2H5OH"],
              "S_EtOH",sel["C2H5OH"],"res",res["max_rel_residual"])
