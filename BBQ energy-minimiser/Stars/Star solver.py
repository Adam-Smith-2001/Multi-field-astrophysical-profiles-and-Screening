"""Author: Adam Smith"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import numpy as np

# -------------------------
# Matplotlib functions and flags
# -------------------------
HEADLESS_FLAG = ("--headless" in sys.argv)
import matplotlib
if HEADLESS_FLAG:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.optimize import minimize_scalar

# numpy trapz compatibility (the ususal stupid shit)
TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# -------------------------
# Numerical safety
# -------------------------
EXP_CLIP = 700.0
def exp_safe(x):
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))

W2_FLOOR_FACTOR = 10.0

class BBQNoMinimum(RuntimeError): pass
class BBQSingular(RuntimeError): pass

@dataclass
class Env:
    Rs: float = 1.0
    rmin: float = 1e-4
    N: int = 2000

    # density: smoothed truncated polytrope
    rho_c: float = 1e-5
    rho_env: float = 1e-12
    ell_rho: float = 1e-3
    n_poly: int = 3

    # model
    beta: float = 0.2
    V0: float = 5e-10
    kappa: float = None
    Mpl: float = 1.0

    # axion kinetic coupling
    W0: float = 1.0
    zeta: float = -2.0e10
    W2_floor: float = 1e-300

    # axion BC + window
    a_minus: float = 0.0
    a_plus: float = 5.0e-9
    ell_a: float = 0.4
    window_centered: bool = False

    # geometry factors
    ell_S: float = None #smooths the axion gradient turn-on so dilaton doesn't recive instantaneous kick
    Lov: float = None

    # fixed-point loop
    max_iter: int = 60
    damp: float = 0.6
    tol_rel_J: float = 1e-3
    tol_abs_chi: float = 1e-12

    # BVP settings
    bvp_tol: float = 5e-4
    bvp_max_nodes: int = None

    # chi_s bracket + refine
    chis_bounds_halfwidth: float = 2e-9
    chis_xatol: float = 1e-12
    chis_maxiter: int = 80
    chis_min: float = 0.0

    # scan + candidate selection
    dense_n: int = 201
    topk_scan: int = 8
    refine_halfwidth_factor: float = 0.25

    # progress control
    scan_print_every: int = 1
    refine_print_every: int = 1

    # FAST scan-only env
    use_fast_scan_refine: bool = True
    N_fast: int = 400
    max_iter_fast: int = 25
    bvp_tol_fast: float = 3e-5
    bvp_max_nodes_factor_fast: int = 50

    # selection behaviour
    include_chi_guess_as_candidate: bool = False
    scan_warm_start: bool = True

    # seed behaviour
    use_analytic_seed: bool = False
    fallback_chi_guess: float = 0.0

    # flatness detection
    flat_rel_tol: float = 1e-10
    flat_trim_frac: float = 0.10
    flat_best_band: float = 5e-12

    # sanity cut for "OK but garbage" points
    L_sanity_max: float = 1e-6
    J_sanity_max: float = 1e-6

    # robust fallback continuation in flux strength (only if needed)
    use_flux_continuation_on_fail: bool = True
    flux_scales: tuple = (0.25, 0.5, 0.8, 1.0)

    def __post_init__(self):
        if self.kappa is None:
            self.kappa = 4*self.beta - 1.0
        if self.Lov is None:
            self.Lov = self.ell_a
        if self.ell_S is None:
            dr = (self.Rs - self.rmin) / (self.N - 1)
            self.ell_S = 1.0*dr
        if self.bvp_max_nodes is None:
            self.bvp_max_nodes = 50 * self.N

def _recompute_ellS(env: Env):
    dr = (env.Rs - env.rmin) / (env.N - 1)
    env.ell_S = 1.0 * dr

def make_fast_env(env_full: Env) -> Env:
    envf = Env(**vars(env_full))
    envf.N = int(env_full.N_fast)
    envf.max_iter = int(env_full.max_iter_fast)
    envf.bvp_tol = float(env_full.bvp_tol_fast)
    envf.bvp_max_nodes = int(env_full.bvp_max_nodes_factor_fast * envf.N)
    envf.tol_abs_chi = max(env_full.tol_abs_chi, 10.0*env_full.tol_abs_chi)
    envf.tol_rel_J   = max(env_full.tol_rel_J,   10.0*env_full.tol_rel_J)
    _recompute_ellS(envf)
    return envf

# ---------------- density ----------------
#star density
def rho_poly_truncated(r, env: Env):
    r = np.asarray(r, dtype=float)
    rho = np.empty_like(r)
    inside = r <= env.Rs
    x2 = (r[inside] / env.Rs)**2
    rho[inside] = env.rho_c * np.maximum(1.0 - x2, 0.0)**env.n_poly
    rho[~inside] = env.rho_env
    return rho

#total density
def rho_of_r(r, env: Env):
    r = np.asarray(r, dtype=float)
    if env.ell_rho <= 0:
        return rho_poly_truncated(r, env)
    t = (r - env.Rs) / env.ell_rho
    s = 0.5 * (1.0 - np.tanh(t))
    rho_in = rho_poly_truncated(r, env)
    return env.rho_env + (rho_in - env.rho_env) * s

#computes surface newtonian potential (not used for computations)
def Phi_surface_from_density(env: Env, use_reduced_planck=True):
    r = np.linspace(0.0, env.Rs, 200000)
    rho = rho_poly_truncated(r, env)
    M = 4*np.pi * TRAPZ(rho * r**2, r)
    G = (1.0/(8*np.pi)) if use_reduced_planck else 1.0
    return float(G * M / env.Rs)

# ---------------- model ----------------
def Vchi(chi, env): return env.kappa * env.V0 * exp_safe(env.kappa*chi)
def V_of_chi(chi, env): return env.V0 * exp_safe(env.kappa*chi)

def Achi_rho(chi, r, env):
    return env.beta * rho_of_r(r, env) * exp_safe(env.beta*chi)

def A_of_chi(chi, env): return exp_safe(env.beta*chi)

def logW2(chi, env): return np.log(env.W0**2) + 2.0*env.zeta*chi/env.Mpl
def W2_raw(chi, env): return (env.W0**2) * exp_safe(2.0*env.zeta*chi/env.Mpl)
def W2_eff(chi, env): return np.maximum(W2_raw(chi, env), env.W2_floor)

# ---------------- window ----------------
#smotting windo to make sure the axion gradient isn't singular
def window_edges(env):
    if env.window_centered:
        r1 = env.Rs - 0.3*env.ell_a
        r2 = env.Rs + 0.3*env.ell_a
    else:
        r1 = env.Rs - env.ell_a
        r2 = env.Rs
    return r1, r2

def S_window_smooth(r, env):
    r = np.asarray(r, dtype=float)
    r1, r2 = window_edges(env)
    ellS = env.ell_S
    return 0.5*(np.tanh((r - r1)/ellS) - np.tanh((r - r2)/ellS))

# ---------------- flux + a' ----------------
def compute_flux_J(r, chi, env):
    S = S_window_smooth(r, env)
    rr = np.maximum(r, env.rmin)
    denom = TRAPZ(S / (rr*rr*W2_eff(chi, env)), r)
    Da = env.a_plus - env.a_minus
    if denom <= 0 or (not np.isfinite(denom)):
        raise RuntimeError(f"Bad flux denominator: denom={denom}")
    return Da / denom

def aprime_from_J(r, chi, env, J):
    S = S_window_smooth(r, env)
    rr = np.maximum(r, env.rmin)
    return J * S / (rr*rr*W2_eff(chi, env))

# ---------------- checks (FIXED) ----------------
def _min_allowed_W2(env: Env):
    # The only *hard* criterion we enforce is "not floor dominated"
    return W2_FLOOR_FACTOR * float(env.W2_floor)

def check_surface_W2(env, chi_s):
    W2_s = float(W2_raw(chi_s, env))
    minW2 = _min_allowed_W2(env)
    if (not np.isfinite(W2_s)) or (W2_s < minW2):
        raise BBQSingular(
            f"[STOP] Surface W2 floor-dominated: W2={W2_s:.3e}, min_allowed={minW2:.3e} "
            f"(chi_s={chi_s:.3e}, zeta={env.zeta:.3e})"
        )
    return float(logW2(chi_s, env)), W2_s

def check_ramp_W2(env, r, chi):
    S = S_window_smooth(r, env)
    mask = S > 1e-6
    if not np.any(mask):
        return
    W2m = float(np.min(W2_raw(chi[mask], env)))
    minW2 = _min_allowed_W2(env)
    if (not np.isfinite(W2m)) or (W2m < minW2):
        raise BBQSingular(
            f"[STOP] Ramp W2 floor-dominated: min W2 in ramp={W2m:.3e}, min_allowed={minW2:.3e}"
        )

# ---------------- backreaction ----------------
#axion term in dilaton EoM
def axion_backreaction(rr, chi, env, J):
    S = S_window_smooth(rr, env)
    rr_eff = np.maximum(rr, env.rmin)
    return (env.zeta/env.Mpl) * (J*J) * (S*S) / (rr_eff**4 * W2_eff(chi, env))

# ---------------- BVP ----------------
def solve_chi_bvp_given_J_and_chis(r, J, env, chi_guess, chi_s):
    check_surface_W2(env, chi_s)
    chi0 = np.array(chi_guess, copy=True)
    chip0 = np.gradient(chi0, r, edge_order=2)
    u0 = (r*r) * chip0
    u0[0] = 0.0
    y0 = np.vstack((chi0, u0))

    def ode(rr, y):
        chi = y[0]
        u   = y[1]
        rr_eff = np.maximum(rr, env.rmin)
        dchi = u / (rr_eff*rr_eff)
        du   = (rr_eff*rr_eff) * (Vchi(chi, env) + Achi_rho(chi, rr_eff, env)
                                  + axion_backreaction(rr_eff, chi, env, J))
        return np.vstack((dchi, du))

    def bc(ya, yb):
        return np.array([ya[1], yb[0] - chi_s])

    sol = solve_bvp(
        ode, bc, r, y0,
        tol=env.bvp_tol,
        max_nodes=env.bvp_max_nodes,
        verbose=0
    )
    if not sol.success:
        raise RuntimeError("chi BVP failed: " + sol.message)

    chi  = sol.sol(r)[0]
    u    = sol.sol(r)[1]
    rr_eff = np.maximum(r, env.rmin)
    chip = u / (rr_eff*rr_eff)
    return chi, chip, u

# ---------------- fixed-point (base) ----------------
def solve_flux_given_chis(env, chi_s, chi_init=None, J_init=None, do_print=False):
    r = np.linspace(env.rmin, env.Rs, env.N)

    if chi_init is None:
        chi_guess = np.full_like(r, chi_s)
    else:
        chi_guess = np.array(chi_init, copy=True)
        chi_guess += (chi_s - chi_guess[-1])

    if J_init is None:
        J = compute_flux_J(r, chi_guess, env)
    else:
        J = float(J_init)

    chi = chi_guess
    converged = False
    dJ_rel = np.inf
    dchi_abs = np.inf

    for it in range(env.max_iter):
        J_old   = J
        chi_old = chi

        chi_new, chip_new, u_new = solve_chi_bvp_given_J_and_chis(
            r, J, env, chi_guess=chi, chi_s=chi_s
        )
        J_new = compute_flux_J(r, chi_new, env)

        J   = (1-env.damp)*J_old   + env.damp*J_new
        chi = (1-env.damp)*chi_old + env.damp*chi_new

        dJ_rel   = abs(J_new - J_old) / max(abs(J_new), abs(J_old), 1e-300)
        dchi_abs = float(np.max(np.abs(chi_new - chi_old)))

        if do_print:
            print(f"it={it:02d}  dJ_rel={dJ_rel:.3e}  dchi={dchi_abs:.3e}", flush=True)

        if (dJ_rel < env.tol_rel_J) and (dchi_abs < env.tol_abs_chi):
            converged = True
            chi, chip, u = chi_new, chip_new, u_new
            J = J_new
            break

    if not converged:
        raise RuntimeError(
            f"Fixed-point did NOT converge (max_iter={env.max_iter}). "
            f"Last dJ_rel={dJ_rel:.3e}, dchi={dchi_abs:.3e}"
        )

    check_ramp_W2(env, r, chi)

    ap = aprime_from_J(r, chi, env, J)
    a  = cumulative_trapezoid(ap, r, initial=0.0)
    a += env.a_minus - a[0]

    L = float(u[-1])
    return r, chi, chip, a, ap, J, L

# ---------------- robust wrapper ----------------
def solve_flux_given_chis_robust(env, chi_s, chi_init=None, J_init=None, do_print=False):
    try:
        return solve_flux_given_chis(env, chi_s, chi_init=chi_init, J_init=J_init, do_print=do_print)
    except Exception as first_err:
        if not env.use_flux_continuation_on_fail:
            raise
        # continuation in Delta a, warm-start each stage
        Da_full = float(env.a_plus - env.a_minus)
        aplus_full = float(env.a_plus)
        chi_state = chi_init
        J_state = J_init
        last_err = first_err
        for s in env.flux_scales:
            try:
                env.a_plus = env.a_minus + s*Da_full
                out = solve_flux_given_chis(env, chi_s, chi_init=chi_state, J_init=J_state, do_print=False)
                # warm starts for next stage
                r, chi, chip, a, ap, J, L = out
                chi_state, J_state = chi, J
                last_out = out
                last_err = None
            except Exception as e:
                last_err = e
                break
        env.a_plus = aplus_full
        if last_err is not None:
            raise RuntimeError(f"robust solve failed (after continuation): {last_err}") from last_err
        # final stage already used full scale=1.0 and restored a_plus; return last_out
        return last_out

# ---------------- energy ----------------
def total_energy_from_profiles(r, chi, chip, ap, rho, env, L):
    W2 = W2_raw(chi, env)
    S = S_window_smooth(r, env)
    mask = S > 1e-6
    if np.any(mask):
        if np.min(W2[mask]) < _min_allowed_W2(env):
            raise BBQSingular(
                f"[STOP] Energy eval floor-dominated in ramp: min W2={np.min(W2[mask]):.3e}, "
                f"min_allowed={_min_allowed_W2(env):.3e}"
            )

    integrand = 0.5*chip**2 + 0.5*W2*ap**2 + V_of_chi(chi, env) + A_of_chi(chi, env)*rho
    Ein = 4*np.pi * TRAPZ(r*r*integrand, r)
    Eout_grad = 2*np.pi * (L*L) / env.Rs
    return float(Ein + Eout_grad)

# ---------------- seed ----------------
def chi_surface_from_eq_analytic(env):
    PhiN = Phi_surface_from_density(env, use_reduced_planck=True)
    Da = env.a_plus - env.a_minus
    pref = (env.ell_a**2) / (env.Rs * env.Lov) * (env.Mpl**2 / (Da**2 + 1e-300))
    denom = (2.0 * env.zeta**2) * (4.0*env.zeta*env.beta*PhiN + 1.0) * (env.W0**2)
    arg = - pref / denom
    if (not np.isfinite(arg)) or (arg <= 0.0):
        raise BBQNoMinimum(f"eq_analytic: no real chi_s (arg={arg:.3e}).")
    chi_s = (env.Mpl/(2.0*env.zeta)) * np.log(arg)
    return float(chi_s)

def get_initial_chi_guess(env: Env):
    chi_eq_analytic = None
    if env.use_analytic_seed:
        try:
            chi_eq_analytic = chi_surface_from_eq_analytic(env)
        except Exception:
            chi_eq_analytic = None
    if (chi_eq_analytic is not None) and np.isfinite(chi_eq_analytic):
        return float(chi_eq_analytic), float(chi_eq_analytic)
    return float(env.fallback_chi_guess), chi_eq_analytic

# ---------------- scan machinery ----------------
def _center_out_order(n, i0):
    order = [i0]
    for k in range(1, n):
        j1 = i0 + k
        j2 = i0 - k
        if j1 < n: order.append(j1)
        if j2 >= 0: order.append(j2)
        if len(order) >= n: break
    return order[:n]

def scan_energy_over_bounds(env, bounds, n_scan=201, center=None, do_print=True):
    chi_grid = np.linspace(bounds[0], bounds[1], int(n_scan))
    E = np.full_like(chi_grid, np.nan, dtype=float)
    ok = np.zeros_like(chi_grid, dtype=bool)

    if center is None:
        center = 0.5*(bounds[0] + bounds[1])
    i0 = int(np.argmin(np.abs(chi_grid - center)))
    order = _center_out_order(len(chi_grid), i0)

    state_chi = None
    state_J = None
    t0 = time.time()

    for count, i in enumerate(order, start=1):
        chi_s = float(chi_grid[i])
        if do_print and (count % max(1, env.scan_print_every) == 0):
            dt = time.time() - t0
            print(f"[scan {count:03d}/{len(order)}] chi_s={chi_s:.6e}  (elapsed {dt:.1f}s)", flush=True)

        try:
            chi_init = state_chi if env.scan_warm_start else None
            J_init   = state_J   if env.scan_warm_start else None

            r, chi, chip, a, ap, J, L = solve_flux_given_chis_robust(
                env, chi_s, chi_init=chi_init, J_init=J_init, do_print=False
            )
            rho = rho_of_r(r, env)
            Eraw = total_energy_from_profiles(r, chi, chip, ap, rho, env, L)

            E[i] = float(Eraw)
            ok[i] = True

            if env.scan_warm_start:
                state_chi, state_J = chi, J

            if do_print and (count % max(1, env.scan_print_every) == 0):
                print(f"    -> OK  E={Eraw:.6e}  J={J:.3e}  L={L:.3e}", flush=True)

        except Exception as e:
            if do_print and (count % max(1, env.scan_print_every) == 0):
                print(f"    -> FAIL: {e}", flush=True)

    return chi_grid, E, ok

def build_candidates_from_full_grid(chi_grid, E_grid, ok, topk=8, include_center_idx=None):
    valid = np.where(ok)[0]
    if valid.size == 0:
        return []
    imin = valid[np.argmin(E_grid[valid])]
    cand = {int(imin)}

    valid_set = set(map(int, valid))
    for i in valid:
        i = int(i)
        if (i-1 in valid_set) and (i+1 in valid_set):
            if np.isfinite(E_grid[i-1]) and np.isfinite(E_grid[i]) and np.isfinite(E_grid[i+1]):
                if (E_grid[i] < E_grid[i-1]) and (E_grid[i] < E_grid[i+1]):
                    cand.add(i)

    k = int(min(topk, valid.size))
    top_idx = valid[np.argsort(E_grid[valid])[:k]]
    for i in top_idx:
        cand.add(int(i))

    if include_center_idx is not None and int(include_center_idx) in valid_set:
        cand.add(int(include_center_idx))

    cand = sorted(list(cand), key=lambda j: E_grid[j])
    return cand

def _trimmed_minmax(arr, trim_frac):
    a = np.array(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan, a
    a.sort()
    m = a.size
    k = int(np.floor(trim_frac * m))
    if (m > 2*k) and (k > 0):
        a = a[k:-k]
    return float(np.min(a)), float(np.max(a)), a

def _is_sane_solution(J, L, env: Env):
    if not np.isfinite(J) or not np.isfinite(L): return False
    if abs(L) > env.L_sanity_max: return False
    if abs(J) > env.J_sanity_max: return False
    return True

# ---------------- refine ----------------
#refining around best found chi_s from output
def refine_around_candidate_FULL(env_full, bounds0, chi0, scan_dx=None, do_print=True):
    half_ref_global = env_full.refine_halfwidth_factor * env_full.chis_bounds_halfwidth
    if (scan_dx is not None) and np.isfinite(scan_dx) and (scan_dx > 0):
        half_ref_local = 8.0 * float(scan_dx)
        half_ref_local = max(half_ref_local, 10.0 * env_full.chis_xatol)
        half_ref = min(half_ref_global, half_ref_local)
    else:
        half_ref = half_ref_global

    lo = max(bounds0[0], chi0 - half_ref)
    hi = min(bounds0[1], chi0 + half_ref)
    if hi <= lo:
        raise RuntimeError(f"Refine bounds collapsed: lo={lo}, hi={hi}, chi0={chi0}, half_ref={half_ref}")

    if do_print:
        print("\n=== Refine candidate (FULL objective) ===", flush=True)
        print(f"chi0          = {chi0:.6e}", flush=True)
        print(f"refine bounds = [{lo:.6e}, {hi:.6e}]", flush=True)

    eval_log = []
    eval_counter = 0
    t0 = time.time()

    def obj(x):
        nonlocal eval_counter
        eval_counter += 1
        x = float(x)
        if do_print and (eval_counter % max(1, env_full.refine_print_every) == 0):
            dt = time.time() - t0
            print(f"  [refine {eval_counter:02d}/{env_full.chis_maxiter}] chi_s={x:.12e}  (dt={dt:.1f}s)", flush=True)
        try:
            r, chi, chip, a, ap, J, L = solve_flux_given_chis_robust(env_full, x, chi_init=None, J_init=None, do_print=False)
            rho = rho_of_r(r, env_full)
            E = total_energy_from_profiles(r, chi, chip, ap, rho, env_full, L)
            sane = _is_sane_solution(J, L, env_full)
            eval_log.append({"chi_s": x, "E": float(E), "ok": True, "sane": sane, "J": float(J), "L": float(L), "err": ""})
            return float(E) if sane else float(E) * (1.0 + 1e-8)
        except Exception as e:
            eval_log.append({"chi_s": x, "E": np.nan, "ok": False, "sane": False, "J": np.nan, "L": np.nan, "err": str(e)})
            return 1e300

    res = minimize_scalar(
        obj,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": env_full.chis_xatol, "maxiter": env_full.chis_maxiter}
    )
    if not res.success:
        raise RuntimeError("bounded refine failed: " + str(res.message))

    ok_pts = [d for d in eval_log if d["ok"] and np.isfinite(d["E"])]
    if not ok_pts:
        raise RuntimeError("No successful FULL objective evaluations in refine bracket.")

    sane_pts = [d for d in ok_pts if d["sane"]]
    pool = sane_pts if sane_pts else ok_pts
    best_d = min(pool, key=lambda d: d["E"])
    chi_star = float(best_d["chi_s"])

    r, chi, chip, a, ap, J, L = solve_flux_given_chis_robust(env_full, chi_star, chi_init=None, J_init=None, do_print=False)
    rho = rho_of_r(r, env_full)
    E_full = total_energy_from_profiles(r, chi, chip, ap, rho, env_full, L)

    sol = (r, chi, chip, a, ap, J, L, float(E_full))
    return chi_star, sol, eval_log

# ---------------- global selection ----------------
"""energy calculation is unstable for certain chi_s, 
need to find minimum then do lots of checks to make sure its not a garbage point"""
def find_best_chis(env_full, chi_guess, do_print=True):
    lo = chi_guess - env_full.chis_bounds_halfwidth
    hi = chi_guess + env_full.chis_bounds_halfwidth
    if env_full.chis_min is not None:
        lo = max(lo, float(env_full.chis_min))
        hi = max(hi, float(env_full.chis_min) + 1e-30)
    bounds0 = (lo, hi)
    if bounds0[1] <= bounds0[0]:
        raise RuntimeError(f"Bad chi_s bounds: {bounds0}")

    env_scan = make_fast_env(env_full) if env_full.use_fast_scan_refine else env_full

    if do_print:
        print("\n=== Global search (scan -> FULL refine -> pick min FULL) ===", flush=True)
        print(f"chi_guess = {chi_guess:.6e}", flush=True)
        print(f"bounds    = [{bounds0[0]:.6e}, {bounds0[1]:.6e}]", flush=True)
        print(f"dense_n   = {env_full.dense_n}", flush=True)

    chi_grid, E_grid, ok = scan_energy_over_bounds(
        env_scan, bounds0, n_scan=env_full.dense_n, center=chi_guess, do_print=do_print
    )
    scan_dx = float(chi_grid[1] - chi_grid[0]) if len(chi_grid) > 1 else None

    if not np.any(ok):
        raise RuntimeError("Dense scan: no successful points in bounds.")

    valid = np.where(ok)[0]
    Emin_t, Emax_t, _ = _trimmed_minmax(E_grid[valid], env_full.flat_trim_frac)
    if not np.isfinite(Emin_t):
        raise RuntimeError("Flatness check: no finite scan energies among ok points.")
    relspan_trim = (Emax_t - Emin_t) / max(abs(Emin_t), 1e-300)

    imin = int(valid[np.argmin(E_grid[valid])])
    i0 = int(np.argmin(np.abs(chi_grid - chi_guess)))
    include_center = i0 if env_full.include_chi_guess_as_candidate else None

    cand_idx = build_candidates_from_full_grid(
        chi_grid, E_grid, ok,
        topk=env_full.topk_scan,
        include_center_idx=include_center
    )
    candidates = [float(chi_grid[i]) for i in cand_idx]

    best_scan_chi = float(chi_grid[imin])
    if best_scan_chi not in candidates:
        candidates = [best_scan_chi] + candidates
    else:
        candidates = [best_scan_chi] + [c for c in candidates if c != best_scan_chi]

    best = None
    all_eval_logs = []

    for ic, c0 in enumerate(candidates, start=1):
        if do_print:
            print(f"\n[CAND {ic}/{len(candidates)}] chi0={c0:.6e}", flush=True)
        chi_star, sol, eval_log = refine_around_candidate_FULL(env_full, bounds0, c0, scan_dx=scan_dx, do_print=do_print)
        all_eval_logs.append(eval_log)
        E_full = float(sol[-1])
        if best is None or E_full < best[0]:
            best = (E_full, chi_star, sol)

    if best is None:
        raise RuntimeError("All candidate refines failed.")

    return best[1], best[2], bounds0, all_eval_logs, chi_grid, E_grid, ok, relspan_trim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ell-a", type=float, default=None, help="override ell_a")
    ap.add_argument("--outdir", type=str, default="bbq_outputs", help="output directory")
    ap.add_argument("--quick", action="store_true", help="reduce N and scan points for a fast sanity run")
    ap.add_argument("--no-plots", action="store_true", help="skip writing plots")
    ap.add_argument("--no-show", action="store_true", help="do not open plot windows (still saves)")
    ap.add_argument("--headless", action="store_true", help="force non-interactive backend (no windows)")
    args = ap.parse_args()

    env = Env()
    if args.ell_a is not None:
        env.ell_a = float(args.ell_a)
        env.Lov = env.ell_a  # keep your convention

    if args.quick:
        env.N = 300
        env.bvp_max_nodes = 50 * env.N
        env.N_fast = 150
        env.dense_n = 41
        env.max_iter_fast = 25
        env.bvp_tol_fast = 3e-4
        env.bvp_max_nodes_factor_fast = 30
        env.bvp_tol = 3e-3
        env.max_iter = 40
        _recompute_ellS(env)

    print(f"Using fixed zeta = {env.zeta:.6e}")
    print(f"ell_a={env.ell_a:.3e}  Da={env.a_plus-env.a_minus:.3e}")
    print(f"BVP: tol={env.bvp_tol:.1e}, max_nodes={env.bvp_max_nodes}  N={env.N}")
    print(f"W2_floor={env.W2_floor:.1e}  min_allowed={_min_allowed_W2(env):.1e}")

    Phi_s = Phi_surface_from_density(env, use_reduced_planck=True)
    print(f"Estimated Phi_s = {Phi_s:.6e}")

    chi_guess, chi_eq_analytic = get_initial_chi_guess(env)
    print(f"chi_guess = {chi_guess:.6e} (analytic seed {'ON' if env.use_analytic_seed else 'OFF'})")

    chi_s_star, sol, bounds, all_eval_logs, chi_grid, E_grid, ok_scan, relspan_trim = find_best_chis(env, chi_guess, do_print=True)
    r, chi, chip, a, ap_prof, J, L, E_full = sol
    print(f"\nFINAL: chi_s*={chi_s_star:.12e}, J={J:.12e}, L={L:.12e}, E_full={E_full:.12e}")
    print(f"bounds: [{bounds[0]:.3e}, {bounds[1]:.3e}]")
    print(f"scan trimmed relspan={relspan_trim:.3e}")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    rho_vals = rho_of_r(r, env)
    W2_vals  = W2_raw(chi, env)
    W_vals   = np.sqrt(np.maximum(W2_vals, 0.0))
    data = np.column_stack([r, chi, chip, a, ap_prof, rho_vals, W_vals])

    profiles_file = os.path.join(outdir, "profiles_best.txt")
    np.savetxt(profiles_file, data, fmt="%.12e", header="r  chi  chi_prime  a  a_prime  rho  W")
    print("Saved:", os.path.abspath(profiles_file))

    # save dense scan
    dense_file = os.path.join(outdir, "dense_scan.txt")
    with open(dense_file, "w") as f:
        f.write("# chi_s   E_scan   ok(1/0)\n")
        for x, e, okv in zip(chi_grid, E_grid, ok_scan):
            ok_i = 1 if okv and np.isfinite(e) else 0
            if np.isfinite(e) and okv:
                f.write(f"{x:.12e} {e:.12e} {ok_i:d}\n")
            else:
                f.write(f"{x:.12e} NaN {ok_i:d}\n")
    print("Saved:", os.path.abspath(dense_file))

    # summary
    summ_file = os.path.join(outdir, "summary.txt")
    with open(summ_file, "w") as f:
        f.write(f"ell_a = {env.ell_a:.12e}\n")
        f.write(f"Da = {env.a_plus-env.a_minus:.12e}\n")
        f.write(f"chi_guess = {chi_guess:.12e}\n")
        f.write(f"chi_eq_analytic = {chi_eq_analytic if chi_eq_analytic is not None else np.nan:.12e}\n")
        f.write(f"chi_s_star = {chi_s_star:.12e}\n")
        f.write(f"E_full = {E_full:.12e}\n")
        f.write(f"J = {J:.12e}\n")
        f.write(f"L = {L:.12e}\n")
        f.write(f"W2_floor = {env.W2_floor:.12e}\n")
        f.write(f"W2_min_allowed = {_min_allowed_W2(env):.12e}\n")
    print("Saved:", os.path.abspath(summ_file))

    # Objective-eval log
    obj_file = os.path.join(outdir, "objective_evals_full.txt")
    with open(obj_file, "w") as f:
        f.write("# chi_s   E_full   ok(1/0)   sane(1/0)   J   L   err\n")
        for log in all_eval_logs:
            for d in log:
                ok_i = 1 if d.get("ok", False) else 0
                sane_i = 1 if d.get("sane", False) else 0
                x = d.get("chi_s", np.nan)
                E = d.get("E", np.nan)
                Jd = d.get("J", np.nan)
                Ld = d.get("L", np.nan)
                err = d.get("err", "")
                if np.isfinite(E):
                    f.write(f"{x:.12e} {E:.12e} {ok_i:d} {sane_i:d} {Jd:.12e} {Ld:.12e} {err}\n")
                else:
                    f.write(f"{x:.12e} NaN {ok_i:d} {sane_i:d} NaN NaN {err}\n")
    print("Saved:", os.path.abspath(obj_file))

    # ---- plotting: save PDFs AND (optionally) show windows ----
    do_plots = (not args.no_plots)
    do_show  = do_plots and (not args.no_show) and (not args.headless)

    if do_plots:
        # Dense scan diagnostic plot
        scan_file = os.path.join(outdir, "scan_energy.pdf")
        fig_scan, ax_scan = plt.subplots(1, 1, figsize=(8, 4))
        msk = np.asarray(ok_scan, dtype=bool) & np.isfinite(E_grid)
        ax_scan.plot(chi_grid[msk], E_grid[msk], marker="o", ms=3, lw=1)
        ax_scan.axvline(chi_guess, ls="--", alpha=0.6, label="chi_guess")
        ax_scan.axvline(chi_s_star, ls="-.", alpha=0.8, label="chi_s*")
        ax_scan.set_xlabel(r"$\chi_s$")
        ax_scan.set_ylabel(r"$E_{\rm scan}$")
        ax_scan.grid(alpha=0.3)
        ax_scan.legend(loc="best", frameon=False)
        fig_scan.tight_layout()
        fig_scan.savefig(scan_file, bbox_inches="tight")
        print("Saved:", os.path.abspath(scan_file))
        if not do_show:
            plt.close(fig_scan)

        # Profiles plot
        prof_file = os.path.join(outdir, "profiles.pdf")
        fig_prof, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

        ax[0].plot(r/env.Rs, rho_vals/max(env.rho_c, 1e-300))
        ax[0].set_ylabel(r"$\rho/\rho_c$")
        ax[0].grid(alpha=0.3)

        ax[1].plot(r/env.Rs, chi)
        ax[1].axhline(chi_s_star, ls="--", alpha=0.6)
        ax[1].set_ylabel(r"$\chi$")
        ax[1].grid(alpha=0.3)

        ax[2].plot(r/env.Rs, a)
        ax[2].set_ylabel(r"$a$")
        ax[2].grid(alpha=0.3)

        ax[3].plot(r/env.Rs, chip*r*r)
        ax[3].set_ylabel(r"$L=r^2\chi'$")
        ax[3].set_xlabel(r"$r/R_s$")
        ax[3].grid(alpha=0.3)

        fig_prof.tight_layout()
        fig_prof.savefig(prof_file, bbox_inches="tight")
        print("Saved:", os.path.abspath(prof_file))
        if not do_show:
            plt.close(fig_prof)

        if do_show:
            plt.show()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())