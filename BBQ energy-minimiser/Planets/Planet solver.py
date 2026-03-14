#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BBQ thin-shell + axion-flux solver with chi_s chosen by numerical energy minimisation.

This version:
  - verifies the BVP solution directly from the solver interpolant on the solver mesh,
  - saves ALL outputs into a single folder created next to this script,
  - reads ALL later plotting inputs from that same single folder.

Checks on the solver mesh sol.x:
    (1)  chi'(r) ?= L(r) / r^2
    (2)  L'(r)   ?= r^2 [ V_,chi + beta rho e^{beta chi} + axion ]
    (3)  BCs: L(rmin)=0, chi(Rs)=chi_s
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator
from scipy.integrate import cumulative_trapezoid, solve_bvp
from scipy.optimize import minimize_scalar

# -------------------------------------------------
# NumPy trapz compatibility
# -------------------------------------------------

TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# -------------------------
# Numerical safety
# -------------------------

EXP_CLIP = 700.0


def exp_safe(x):
    """Exponent with clipping to avoid overflow/underflow."""
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))


# -------------------------
# Single run directory
# -------------------------

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd().resolve()

RUN_DIR = SCRIPT_DIR / "bbq_outputs"
RUN_DIR.mkdir(parents=True, exist_ok=True)


def run_path(filename: str) -> str:
    """Return absolute path inside the single run directory."""
    return str(RUN_DIR / filename)


def ensure_directory(path):
    """Ensure a directory exists and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_plot_directory(path=None):
    """Ensure the plot output directory exists."""
    if path is None:
        return str(ensure_directory(RUN_DIR))
    return str(ensure_directory(path))


# -------------------------
# Physical / formal stop thresholds
# -------------------------

W2_SURFACE_THRESH = 1e-260
W2_RAMP_MIN_THRESH = 1e-260
W2_FLOOR_FACTOR = 10.0


class BBQNoMinimum(RuntimeError):
    """Raised when the analytic chi_s estimate yields no valid real solution."""
    pass


class BBQSingular(RuntimeError):
    """Raised when W^2 hits thresholds or floors indicating a singular regime."""
    pass


# ============================================================
# Environment
# ============================================================

@dataclass
class Env:
    """Bundle of physical and numerical parameters for one BBQ run."""
    Rs: float = 1.0
    rmin: float = 1e-4
    N: int = 2000

    # Density profile
    rho_c: float = 1e-9
    rho_env: float = 1e-10
    ell_rho: float = 1e-3

    # Scalar model
    beta: float = 0.2
    V0: float = 5e-10
    kappa: float | None = None
    Mpl: float = 1.0

    # Axion kinetic coupling
    W0: float = 1.0
    zeta: float = -2.0e10
    W2_floor: float = 1e-300

    # Axion boundary data and window
    a_minus: float = 0.0
    a_plus: float = 0.04
    ell_a: float = 0.4
    window_centered: bool = False

    ell_S: float | None = None
    Lov: float | None = None

    # Fixed-point loop
    max_iter: int = 60
    damp: float = 0.6
    tol_rel_J: float = 1e-3
    tol_abs_chi: float = 1e-12

    # BVP settings
    bvp_tol: float = 1e-4
    bvp_max_nodes: int | None = None

    # chi_s bracket and local refine
    chis_bounds_halfwidth: float = 2e-10
    chis_xatol: float = 1e-12
    chis_maxiter: int = 80

    # Scan and candidate selection
    dense_n: int = 201
    topk_scan: int = 8
    refine_halfwidth_factor: float = 0.25

    # Residual reporting
    report_bvp_residuals: bool = True
    report_bvp_every_iter: bool = False

    def __post_init__(self):
        """Fill derived defaults."""
        if self.kappa is None:
            self.kappa = 4.0 * self.beta - 1.0
        if self.Lov is None:
            self.Lov = self.ell_a
        if self.ell_S is None:
            dr = (self.Rs - self.rmin) / (self.N - 1)
            self.ell_S = dr
        if self.bvp_max_nodes is None:
            self.bvp_max_nodes = 50 * self.N


env = Env()

# ============================================================
# Density profile
# ============================================================


def rho_of_r(r, env):
    """Smooth step density profile."""
    t = np.tanh((r - env.Rs) / env.ell_rho)
    return 0.5 * env.rho_c * (1 - t) + 0.5 * env.rho_env * (1 + t)


# ============================================================
# Model functions
# ============================================================


def Vchi(chi, env):
    """Potential derivative V_,chi for V = V0 exp(kappa chi)."""
    return env.kappa * env.V0 * exp_safe(env.kappa * chi)


def V_of_chi(chi, env):
    """Potential V(chi) = V0 exp(kappa chi)."""
    return env.V0 * exp_safe(env.kappa * chi)


def Achi_rho(chi, r, env):
    """Matter source term beta rho(r) exp(beta chi)."""
    return env.beta * rho_of_r(r, env) * exp_safe(env.beta * chi)


def A_of_chi(chi, env):
    """Conformal factor A(chi) = exp(beta chi)."""
    return exp_safe(env.beta * chi)


def logW2(chi, env):
    """log(W^2) for diagnostics."""
    return np.log(env.W0**2) + 2.0 * env.zeta * chi / env.Mpl


def W2_raw(chi, env):
    """Raw W^2 without flooring."""
    return (env.W0**2) * exp_safe(2.0 * env.zeta * chi / env.Mpl)


def W2_eff(chi, env):
    """Effective W^2 with a hard floor for numerical stability."""
    return np.maximum(W2_raw(chi, env), env.W2_floor)


# ============================================================
# Smooth axion window
# ============================================================


def window_edges(env):
    """Return the inner and outer edges of the axion window."""
    if env.window_centered:
        r1 = env.Rs - 0.3 * env.ell_a
        r2 = env.Rs + 0.3 * env.ell_a
    else:
        r1 = env.Rs - env.ell_a
        r2 = env.Rs
    return r1, r2


def S_window_smooth(r, env):
    """Smooth top-hat window S(r) with tanh edges of width ell_S."""
    r = np.asarray(r, dtype=float)
    r1, r2 = window_edges(env)
    return 0.5 * (
        np.tanh((r - r1) / env.ell_S) - np.tanh((r - r2) / env.ell_S)
    )


# ============================================================
# Flux J and a'(r)
# ============================================================


def compute_flux_J(r, chi, env):
    """Compute flux J enforcing the total axion jump across the window."""
    S = S_window_smooth(r, env)
    denom = TRAPZ(S / (r * r * W2_eff(chi, env)), r)
    delta_a = env.a_plus - env.a_minus
    if denom <= 0 or (not np.isfinite(denom)):
        raise RuntimeError(f"Bad flux denominator: denom={denom}")
    return delta_a / denom


def aprime_from_J(r, chi, env, J):
    """Compute a'(r) from the flux J and the current chi(r)."""
    S = S_window_smooth(r, env)
    return J * S / (r * r * W2_eff(chi, env))


# ============================================================
# Singularity / floor checks
# ============================================================


def check_surface_W2(env, chi_s):
    """Abort if W^2 at the surface is below threshold or too close to the floor."""
    logW2_s = float(logW2(chi_s, env))
    W2_s = float(np.exp(np.clip(logW2_s, -EXP_CLIP, EXP_CLIP)))

    if logW2_s < np.log(W2_SURFACE_THRESH):
        raise BBQSingular(
            f"[STOP] Surface singular: W2(chi_s) < {W2_SURFACE_THRESH:.1e} "
            f"(logW2_s={logW2_s:.3e}, chi_s={chi_s:.3e}, zeta={env.zeta:.3e})"
        )
    if W2_s < W2_FLOOR_FACTOR * env.W2_floor:
        raise BBQSingular(
            f"[STOP] Surface W2 near floor: W2={W2_s:.3e}, floor={env.W2_floor:.3e}"
        )

    return logW2_s, W2_s


def check_ramp_W2(env, r, chi):
    """Abort if W^2 becomes too small inside the active axion window."""
    S = S_window_smooth(r, env)
    mask = S > 1e-6
    if not np.any(mask):
        return

    logW2_min = float(np.min(logW2(chi[mask], env)))
    if logW2_min < np.log(W2_RAMP_MIN_THRESH):
        raise BBQSingular(
            f"[STOP] Ramp singular: min W2 in ramp < {W2_RAMP_MIN_THRESH:.1e} "
            f"(min logW2={logW2_min:.3e}, zeta={env.zeta:.3e})"
        )

    W2min_clip = float(np.exp(np.clip(logW2_min, -EXP_CLIP, EXP_CLIP)))
    if W2min_clip < W2_FLOOR_FACTOR * env.W2_floor:
        raise BBQSingular(
            f"[STOP] Ramp W2 near floor: min(W2)~{W2min_clip:.3e}, floor={env.W2_floor:.3e}"
        )


# ============================================================
# Axion backreaction term
# ============================================================


def axion_backreaction(rr, chi, env, J):
    """Axion contribution to the chi equation in flux form."""
    S = S_window_smooth(rr, env)
    rr_eff = np.maximum(rr, env.rmin)
    return (env.zeta / env.Mpl) * (J * J) * (S * S) / (
        rr_eff**4 * W2_eff(chi, env)
    )


# ============================================================
# BVP residuals on solver mesh
# ============================================================


def bvp_residuals_on_mesh(sol, env, J, chi_s_target):
    """
    Compute ODE residuals on the solver mesh using sol.sol'(r).

    Returns:
        res1 = chi'(r) - L(r)/r^2
        res2 = L'(r) - r^2 [Vchi + Achi_rho + axion]
    together with boundary-condition errors.
    """
    rr = sol.x
    yy = sol.y
    dy = sol.sol(rr, 1)

    chi_mesh = yy[0]
    L_mesh = yy[1]
    dchi_mesh = dy[0]
    dL_mesh = dy[1]

    rr_eff = np.maximum(rr, env.rmin)

    res1 = dchi_mesh - L_mesh / (rr_eff * rr_eff)
    rhs = (
        Vchi(chi_mesh, env)
        + Achi_rho(chi_mesh, rr_eff, env)
        + axion_backreaction(rr_eff, chi_mesh, env, J)
    )
    res2 = dL_mesh - (rr_eff * rr_eff) * rhs

    bc_L0 = float(L_mesh[0])
    bc_chi = float(chi_mesh[-1] - chi_s_target)

    return rr, res1, res2, bc_L0, bc_chi


# ============================================================
# BVP solve in (chi, L = r^2 chi')
# ============================================================


def solve_chi_bvp_given_J_and_chi_surface(
    r, J, env, chi_guess, chi_surface, want_diag=False
):
    """
    Solve the chi BVP at fixed (J, chi_surface) in variables (chi, L = r^2 chi').

    Equations:
        chi' = L/r^2
        L'   = r^2 [Vchi + Achi_rho + axion]

    Boundary conditions:
        L(rmin) = 0
        chi(Rs) = chi_surface
    """
    check_surface_W2(env, chi_surface)

    chi0 = np.array(chi_guess, copy=True)
    chip0 = np.gradient(chi0, r, edge_order=2)
    L0 = (r * r) * chip0
    L0[0] = 0.0
    y0 = np.vstack((chi0, L0))

    def ode(rr, y):
        chi = y[0]
        L = y[1]
        rr_eff = np.maximum(rr, env.rmin)
        dchi = L / (rr_eff * rr_eff)
        dL = (rr_eff * rr_eff) * (
            Vchi(chi, env)
            + Achi_rho(chi, rr_eff, env)
            + axion_backreaction(rr_eff, chi, env, J)
        )
        return np.vstack((dchi, dL))

    def bc(ya, yb):
        return np.array([ya[1], yb[0] - chi_surface])

    sol = solve_bvp(
        ode,
        bc,
        r,
        y0,
        tol=env.bvp_tol,
        max_nodes=env.bvp_max_nodes,
        verbose=0,
    )
    if not sol.success:
        raise RuntimeError("chi BVP failed: " + sol.message)

    chi = sol.sol(r)[0]
    L = sol.sol(r)[1]
    rr_eff = np.maximum(r, env.rmin)
    chip = L / (rr_eff * rr_eff)

    if not want_diag:
        return chi, chip, L

    rr_mesh, res1, res2, bc_L0, bc_chi = bvp_residuals_on_mesh(
        sol, env, J, chi_s_target=chi_surface
    )
    diag = {
        "sol": sol,
        "mesh_r": rr_mesh,
        "res_chi_prime_minus_L_over_r2": res1,
        "res_L_prime_minus_r2_rhs": res2,
        "bc_L0": bc_L0,
        "bc_chiR": bc_chi,
        "max_res1": float(np.max(np.abs(res1))),
        "max_res2": float(np.max(np.abs(res2))),
    }
    return chi, chip, L, diag


# ============================================================
# Fixed-point loop for a given chi_s
# ============================================================


def solve_flux_given_chi_surface(
    env, chi_surface, chi_init=None, J_init=None, do_print=False
):
    """
    Solve the coupled problem at fixed chi_surface.

    Iteration:
        solve chi BVP at fixed J
        update J from the flux constraint
        apply damping
        stop when both J and chi have converged
    """
    r = np.linspace(env.rmin, env.Rs, env.N)

    if chi_init is None:
        chi_guess = np.full_like(r, chi_surface)
    else:
        chi_guess = np.array(chi_init, copy=True)
        chi_guess += (chi_surface - chi_guess[-1])

    J = compute_flux_J(r, chi_guess, env) if J_init is None else float(J_init)
    chi = chi_guess

    if do_print:
        logW2_s, _ = check_surface_W2(env, chi_surface)
        print("Inner solve (given chi_s):", flush=True)
        print(f"  zeta = {env.zeta:.6e}", flush=True)
        print(f"  chi_s(trial) = {chi_surface:.6e}", flush=True)
        print(f"  logW2(chi_s) = {logW2_s:.6e}", flush=True)
        print(f"  initial J = {J:.6e}\n", flush=True)

    converged = False
    dJ_rel = np.inf
    dchi_abs = np.inf
    last_diag = None

    for it in range(env.max_iter):
        J_old = J
        chi_old = chi

        want_diag = bool(
            env.report_bvp_residuals and (do_print or env.report_bvp_every_iter)
        )

        if want_diag:
            chi_new, chip_new, L_new, diag = solve_chi_bvp_given_J_and_chi_surface(
                r, J, env, chi_guess=chi, chi_surface=chi_surface, want_diag=True
            )
            last_diag = diag
        else:
            chi_new, chip_new, L_new = solve_chi_bvp_given_J_and_chi_surface(
                r, J, env, chi_guess=chi, chi_surface=chi_surface, want_diag=False
            )

        J_new = compute_flux_J(r, chi_new, env)

        J = (1 - env.damp) * J_old + env.damp * J_new
        chi = (1 - env.damp) * chi_old + env.damp * chi_new

        dJ_rel = abs(J_new - J_old) / max(abs(J_new), abs(J_old), 1e-300)
        dchi_abs = float(np.max(np.abs(chi_new - chi_old)))

        if do_print:
            print(
                f"it={it:02d}  J_new={J_new:.6e}  J_old={J_old:.6e}  "
                f"dJ_rel={dJ_rel:.3e}  dchi={dchi_abs:.3e}",
                flush=True,
            )
            if want_diag:
                print(
                    f"    BVP residuals (mesh): max|chi' - L/r^2|={diag['max_res1']:.3e}  "
                    f"max|L' - r^2 RHS|={diag['max_res2']:.3e}  "
                    f"BC L(rmin)={diag['bc_L0']:.3e}  BC chi(R)-chi_s={diag['bc_chiR']:.3e}",
                    flush=True,
                )

        if (dJ_rel < env.tol_rel_J) and (dchi_abs < env.tol_abs_chi):
            converged = True
            chi, chip, L = chi_new, chip_new, L_new
            J = J_new

            if env.report_bvp_residuals:
                _, _, _, diag = solve_chi_bvp_given_J_and_chi_surface(
                    r, J, env, chi_guess=chi, chi_surface=chi_surface, want_diag=True
                )
                last_diag = diag

            if do_print:
                print("-> converged", flush=True)
            break

    if not converged:
        raise RuntimeError(
            f"Fixed-point did NOT converge (max_iter={env.max_iter}). "
            f"Last dJ_rel={dJ_rel:.3e}, dchi={dchi_abs:.3e}"
        )

    check_ramp_W2(env, r, chi)

    ap = aprime_from_J(r, chi, env, J)
    a = cumulative_trapezoid(ap, r, initial=0.0)
    a += env.a_minus - a[0]

    Lsurf = float(L[-1])
    return r, chi, chip, a, ap, J, Lsurf, last_diag


# ============================================================
# Total energy
# ============================================================


def total_energy_from_profiles(r, chi, chip, ap, rho, env, L):
    """
    Total energy for profiles (chi, a), including the exterior gradient piece.
    Uses W2_raw to match the original energy definition.
    """
    W2 = W2_raw(chi, env)

    S = S_window_smooth(r, env)
    mask = S > 1e-6
    if np.any(mask):
        if np.min(W2[mask]) < W2_FLOOR_FACTOR * env.W2_floor:
            raise BBQSingular(
                f"[STOP] Energy eval floor-dominated in ramp: "
                f"min W2={np.min(W2[mask]):.3e}, floor={env.W2_floor:.3e}"
            )

    integrand = (
        0.5 * chip**2
        + 0.5 * W2 * ap**2
        + V_of_chi(chi, env)
        + A_of_chi(chi, env) * rho
    )
    Ein = 4 * np.pi * TRAPZ(r * r * integrand, r)
    Eout_grad = 2 * np.pi * (L * L) / env.Rs
    return float(Ein + Eout_grad)


# ============================================================
# Analytic chi_s seed
# ============================================================


def phiN_surface(env):
    """Newtonian potential proxy at the surface."""
    return (env.rho_c * env.Rs**2) / (6.0 * env.Mpl**2)


def chi_surface_from_analytic_estimate(env):
    """Analytic estimate for chi_s."""
    PhiN = phiN_surface(env)
    delta_a = env.a_plus - env.a_minus
    pref = (env.ell_a**2) / (env.Rs * env.Lov) * (env.Mpl**2 / (delta_a**2 + 1e-300))
    denom = (2.0 * env.zeta**2) * (4.0 * env.zeta * env.beta * PhiN + 1.0) * (env.W0**2)
    arg = -pref / denom

    if (not np.isfinite(arg)) or (arg <= 0.0):
        raise BBQNoMinimum(f"Analytic estimate gives no real chi_s (arg={arg:.3e}).")

    chi_surface = (env.Mpl / (2.0 * env.zeta)) * np.log(arg)
    return float(chi_surface)


# ============================================================
# Dense scan helper
# ============================================================


def scan_energy_over_bounds(env, bounds, n_scan=201, do_print=False):
    """
    Coarse scan of chi_s over bounds.
    Returns (chi_grid, E, ok) where ok flags successful points.
    """
    chi_grid = np.linspace(bounds[0], bounds[1], int(n_scan))
    E = np.full_like(chi_grid, np.nan, dtype=float)
    ok = np.zeros_like(chi_grid, dtype=bool)

    state_chi = None
    state_J = None

    for i, chi_surface in enumerate(chi_grid):
        try:
            r, chi, chip, a, ap, J, L, _diag = solve_flux_given_chi_surface(
                env, float(chi_surface), chi_init=state_chi, J_init=state_J, do_print=False
            )
            rho = rho_of_r(r, env)
            Eraw = total_energy_from_profiles(r, chi, chip, ap, rho, env, L)

            E[i] = float(Eraw)
            ok[i] = True

            state_chi, state_J = chi, J

            if do_print:
                print(
                    f"[scan {i+1:03d}/{len(chi_grid)}] chi_s={chi_surface:.6e}  E={Eraw:.6e}",
                    flush=True,
                )

        except Exception as e:
            if do_print:
                print(
                    f"[scan {i+1:03d}/{len(chi_grid)}] chi_s={chi_surface:.6e}  FAIL: {e}",
                    flush=True,
                )

    return chi_grid, E, ok


# ============================================================
# Candidate detection from scan
# ============================================================


def pick_candidates_from_scan(chi_ok, E_ok, topk=8):
    """Pick candidate minima from successful scan points."""
    n = len(E_ok)
    if n == 0:
        return []

    candidate_indices = set()

    for i in range(1, n - 1):
        if np.isfinite(E_ok[i - 1]) and np.isfinite(E_ok[i]) and np.isfinite(E_ok[i + 1]):
            if (E_ok[i] < E_ok[i - 1]) and (E_ok[i] < E_ok[i + 1]):
                candidate_indices.add(i)

    candidate_indices.add(int(np.argmin(E_ok)))

    k = int(min(topk, n))
    for i in np.argsort(E_ok)[:k]:
        candidate_indices.add(int(i))

    candidate_indices = sorted(list(candidate_indices), key=lambda j: E_ok[j])
    return [float(chi_ok[j]) for j in candidate_indices]


# ============================================================
# Local refine around candidate
# ============================================================


def refine_around_candidate(env, bounds0, chi0, do_print=True):
    """
    Local bounded 1D minimisation around chi0 using the full solver in the objective.
    Returns (chi_star, sol) where sol includes a verified final solve.
    """
    half_ref = env.refine_halfwidth_factor * env.chis_bounds_halfwidth
    bounds = (max(bounds0[0], chi0 - half_ref), min(bounds0[1], chi0 + half_ref))

    if do_print:
        print("\n=== Refine candidate ===", flush=True)
        print(f"candidate chi0 = {chi0:.6e}", flush=True)
        print(f"refine bounds  = [{bounds[0]:.6e}, {bounds[1]:.6e}]", flush=True)

    def objective(x):
        try:
            r, chi, chip, a, ap, J, L, _diag = solve_flux_given_chi_surface(
                env, float(x), do_print=False
            )
            rho = rho_of_r(r, env)
            return total_energy_from_profiles(r, chi, chip, ap, rho, env, L)
        except Exception:
            return 1e300

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": env.chis_xatol, "maxiter": env.chis_maxiter},
    )
    if not result.success:
        raise RuntimeError("bounded refine failed: " + str(result.message))

    chi_star = float(result.x)

    r, chi, chip, a, ap, J, L, diag = solve_flux_given_chi_surface(
        env, chi_star, do_print=do_print
    )
    rho = rho_of_r(r, env)
    E_full = total_energy_from_profiles(r, chi, chip, ap, rho, env, L)

    if do_print:
        print(f"-> refined chi_s = {chi_star:.6e}", flush=True)
        print(f"-> E_full       = {E_full:.6e}", flush=True)
        print(f"-> J            = {J:.6e}", flush=True)
        print(f"-> L            = {L:.6e}", flush=True)
        if diag is not None:
            print(
                f"-> FINAL BVP residuals (mesh): max|chi' - L/r^2|={diag['max_res1']:.3e}  "
                f"max|L' - r^2 RHS|={diag['max_res2']:.3e}  "
                f"BC L(rmin)={diag['bc_L0']:.3e}  BC chi(R)-chi_s={diag['bc_chiR']:.3e}",
                flush=True,
            )

    solution = (r, chi, chip, a, ap, J, L, E_full, diag)
    return chi_star, solution


# ============================================================
# Global selection
# ============================================================


def find_best_chi_surface(env, chi_guess, do_print=True):
    """
    Global chi_s minimisation:
      scan across bounds -> pick candidates -> refine each -> choose lowest-energy solution.
    """
    bounds0 = (chi_guess - env.chis_bounds_halfwidth, chi_guess + env.chis_bounds_halfwidth)

    if do_print:
        print("\n=== Global search (scan -> multi-refine) ===", flush=True)
        print(f"chi_guess = {chi_guess:.6e}", flush=True)
        print(f"bounds    = [{bounds0[0]:.6e}, {bounds0[1]:.6e}]", flush=True)
        print(f"dense_n   = {env.dense_n}", flush=True)

    chi_grid, E_grid, ok = scan_energy_over_bounds(
        env, bounds0, n_scan=env.dense_n, do_print=False
    )
    if not np.any(ok):
        raise RuntimeError(
            "Dense scan: no successful points in bounds. "
            "Widen bracket or relax inner solver."
        )

    chi_ok = chi_grid[ok]
    E_ok = E_grid[ok]

    candidates = pick_candidates_from_scan(chi_ok, E_ok, topk=env.topk_scan)
    candidates = [float(chi_guess)] + candidates

    candidates_sorted = []
    grid_dx = 0.5 * (bounds0[1] - bounds0[0]) / max(env.dense_n - 1, 1)
    for c in candidates:
        if all(abs(c - cc) > grid_dx for cc in candidates_sorted):
            candidates_sorted.append(c)

    if do_print:
        print(f"Candidates to refine: {len(candidates_sorted)}", flush=True)
        for i, c in enumerate(candidates_sorted[:12]):
            print(f"  [{i+1:02d}] {c:.6e}", flush=True)
        if len(candidates_sorted) > 12:
            print("  ...", flush=True)

    best = None

    for i, c0 in enumerate(candidates_sorted):
        try:
            if do_print:
                print(
                    f"\n[CAND {i+1}/{len(candidates_sorted)}] starting at chi0={c0:.6e}",
                    flush=True,
                )
            chi_star, solution = refine_around_candidate(env, bounds0, c0, do_print=do_print)
            E_full = float(solution[7])

            if best is None or E_full < best[0]:
                best = (E_full, chi_star, solution)
                if do_print:
                    print(f"[BEST UPDATE] chi_s={chi_star:.6e}  E={E_full:.6e}", flush=True)

        except Exception as e:
            if do_print:
                print(f"[CAND FAIL] chi0={c0:.6e}: {e}", flush=True)

    if best is None:
        raise RuntimeError(
            "All candidate refines failed. Inner solve unstable across all candidates."
        )

    chi_grid2, E_grid2, ok2 = scan_energy_over_bounds(
        env, bounds0, n_scan=env.dense_n, do_print=False
    )
    return best[1], best[2], (chi_grid2, E_grid2, ok2), bounds0


# ============================================================
# Term diagnostics
# ============================================================


def kg_terms(r, chi, env, J):
    """Return the separate contributions entering L'(r) = r^2(...)."""
    r_eff = np.maximum(r, env.rmin)
    V_term = Vchi(chi, env)
    M_term = Achi_rho(chi, r_eff, env)
    Ax_term = axion_backreaction(r_eff, chi, env, J)
    Sum_term = V_term + M_term + Ax_term
    return V_term, M_term, Ax_term, Sum_term


# ============================================================
# File I/O helpers
# ============================================================


def read_key_value_summary(path):
    """Parse a summary file with lines of the form 'key = value'."""
    result = {}
    with open(path, "r") as f:
        for line in f:
            text = line.strip()
            if (not text) or text.startswith("#") or ("=" not in text):
                continue
            key, value = [x.strip() for x in text.split("=", 1)]
            try:
                result[key] = float(value)
            except Exception:
                result[key] = value
    return result


def write_key_value_summary(path, summary_dict):
    """Write a summary file with lines of the form 'key = value'."""
    with open(path, "w") as f:
        for key, value in summary_dict.items():
            f.write(f"{key} = {value}\n")


# ============================================================
# Plotting helpers
# ============================================================


def format_radial_tick(x, pos):
    """Format x ticks, showing R_s at x = 1."""
    if abs(x - 1.0) < 1e-12:
        return r"$R_s$"
    return rf"${x:.1f}$"


def format_linear_mathtext_tick(value, pos):
    """Compact mathtext formatter for linear axes."""
    value = float(value)
    abs_value = abs(value)

    if abs_value == 0.0:
        return r"$0$"

    if (abs_value < 1e-3) or (abs_value >= 1e3):
        exponent = int(np.floor(np.log10(abs_value)))
        mantissa = value / (10.0 ** exponent)
        if abs(mantissa - round(mantissa)) < 1e-12:
            mantissa_str = f"{int(round(mantissa))}"
        else:
            mantissa_str = f"{mantissa:.2g}"
        return rf"${mantissa_str}\times 10^{{{exponent}}}$"

    return rf"${value:.2f}$"


def apply_cm_tick_format(ax, do_x=True, do_y=True):
    """Apply CM-style mathtext formatting to a linear-axis plot."""
    if do_x:
        ax.xaxis.set_major_formatter(FuncFormatter(format_radial_tick))
    if do_y:
        ax.yaxis.set_major_formatter(FuncFormatter(format_linear_mathtext_tick))


def plot_objective_from_files(
    objective_file,
    summary_file,
    output_label="bbq",
    output_prefix="bbq_energy",
    axis_labelsize=18,
    tick_labelsize=16,
    legend_fontsize=16,
    y_floor=1e-18,
):
    """
    Plot objective_scan.txt with columns:
        chi_s, E, dx, y

    where dx is the shifted/rescaled chi_s axis and y is typically
    (E - E_min) / |E_min|.
    """
    if not os.path.exists(objective_file):
        raise FileNotFoundError(f"Missing: {objective_file}")
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Missing: {summary_file}")

    summary = read_key_value_summary(summary_file)
    data = np.loadtxt(objective_file, comments="#")

    dx = data[:, 2]
    y = np.maximum(data[:, 3], y_floor)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
    ax.plot(dx, y, marker="o", ls="-", ms=3, label=r"$\mathrm{scan\ points}$")

    ax.axvline(0.0, ls="--", lw=1.5, label=r"$\chi_{s,\min}^{\rm num}$")

    dx_analytic = summary.get("dx_analytic_guess/1e-12", np.nan)
    if np.isfinite(dx_analytic):
        ax.axvline(dx_analytic, ls=":", lw=1.5, label=r"$\chi_{s,\min}^{\rm analytic}$")

    ax.set_xlabel(r"$(\chi_s-\chi_{s,\min})10^{12}/M_{\rm pl}$", fontsize=axis_labelsize)
    ax.set_ylabel(r"$(E(\chi_s)-E_{\min})/|E_{\min}|$", fontsize=axis_labelsize)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=tick_labelsize)
    ax.legend(fontsize=legend_fontsize, loc="best")

    output_path = run_path(f"{output_prefix}_objective_{output_label}.pdf".replace(" ", "_"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    return output_path


# ============================================================
# Main
# ============================================================

print(f"Using fixed zeta = {env.zeta:.6e}", flush=True)
print(f"Single run directory: {RUN_DIR}", flush=True)

chi_analytic_guess = None
try:
    chi_analytic_guess = chi_surface_from_analytic_estimate(env)
    print(f"Seed chi_s(analytic) = {chi_analytic_guess:.6e}", flush=True)
    print(f"logW2(seed)          = {logW2(chi_analytic_guess, env):.6e}", flush=True)
    chi_guess = chi_analytic_guess
except Exception as e:
    chi_guess = 0.0
    print("Analytic seed failed; using chi_guess = 0.0", flush=True)
    print("Reason:", str(e), flush=True)

chi_surface_best, solution, scan_pack, bounds = find_best_chi_surface(
    env, chi_guess, do_print=True
)
r, chi, chip, a, ap, J, L, E_full, diag = solution
chi_grid, E_grid, ok_grid = scan_pack

print(
    f"\nFINAL: chi_s*={chi_surface_best:.6e}, J={J:.6e}, L={L:.6e}, E={E_full:.6e}",
    flush=True,
)

S = S_window_smooth(r, env)
mask = S > 1e-6
if np.any(mask):
    ratio = W2_raw(chi, env)[mask] / W2_eff(chi, env)[mask]
    print(
        f"min(W2_raw/W2_eff) in window = {np.min(ratio):.6e}  (1 means floor NOT active)",
        flush=True,
    )
    print(
        f"logW2 range in window = "
        f"[{np.min(logW2(chi[mask], env)):.3e}, {np.max(logW2(chi[mask], env)):.3e}]",
        flush=True,
    )

# ------------------------------------------------------------
# Save core profiles
# ------------------------------------------------------------

rho_vals = rho_of_r(r, env)
W2_vals = W2_raw(chi, env)
W_vals = np.sqrt(np.maximum(W2_vals, 0.0))
L_profile = (r * r) * chip

profiles_array = np.column_stack([r, chi, chip, a, ap, rho_vals, W_vals, L_profile])
profiles_output_file = run_path("profiles_best.txt")
np.savetxt(
    profiles_output_file,
    profiles_array,
    fmt="%.12e",
    header="Columns: r, chi, chi_prime, a, a_prime, rho, W, L=r^2 chi_prime",
)
print(f"Saved: {profiles_output_file}", flush=True)

# ------------------------------------------------------------
# Save scan and summary
# ------------------------------------------------------------

scan_success = ok_grid & np.isfinite(E_grid)
if np.any(scan_success):
    E_min_scan = float(np.min(E_grid[scan_success]))
else:
    E_min_scan = float(E_full)

dx_scan = (chi_grid - chi_surface_best) * 1.0e12 / env.Mpl
y_scan = np.full_like(chi_grid, np.nan, dtype=float)

denom = max(abs(E_min_scan), 1e-300)
finite_mask = np.isfinite(E_grid)
y_scan[finite_mask] = (E_grid[finite_mask] - E_min_scan) / denom

objective_scan_file = run_path("objective_scan.txt")
objective_scan_array = np.column_stack([chi_grid, E_grid, dx_scan, y_scan, ok_grid.astype(float)])
np.savetxt(
    objective_scan_file,
    objective_scan_array,
    fmt="%.12e",
    header="Columns: chi_s, E, dx=(chi_s-chi_s_min)*1e12/Mpl, y=(E-Emin)/|Emin|, ok_flag",
)
print(f"Saved: {objective_scan_file}", flush=True)

if chi_analytic_guess is not None:
    dx_analytic = (chi_analytic_guess - chi_surface_best) * 1.0e12 / env.Mpl
else:
    dx_analytic = np.nan

summary_file = run_path("summary.txt")
summary_dict = {
    "run_directory": str(RUN_DIR),
    "chi_s_best": chi_surface_best,
    "chi_s_analytic_guess": chi_analytic_guess if chi_analytic_guess is not None else np.nan,
    "dx_analytic_guess/1e-12": dx_analytic,
    "J_best": J,
    "L_best": L,
    "E_best": E_full,
    "zeta": env.zeta,
    "beta": env.beta,
    "V0": env.V0,
    "kappa": env.kappa,
    "Rs": env.Rs,
    "rmin": env.rmin,
    "N": env.N,
    "rho_c": env.rho_c,
    "rho_env": env.rho_env,
    "ell_rho": env.ell_rho,
    "a_minus": env.a_minus,
    "a_plus": env.a_plus,
    "ell_a": env.ell_a,
    "W0": env.W0,
    "W2_floor": env.W2_floor,
    "scan_bounds_lo": bounds[0],
    "scan_bounds_hi": bounds[1],
}
write_key_value_summary(summary_file, summary_dict)
print(f"Saved: {summary_file}", flush=True)

# ------------------------------------------------------------
# Save residual diagnostics
# ------------------------------------------------------------

if diag is not None:
    rr_mesh = diag["mesh_r"]
    res1 = diag["res_chi_prime_minus_L_over_r2"]
    res2 = diag["res_L_prime_minus_r2_rhs"]

    residuals_output_file = run_path("bvp_residuals_mesh.txt")
    residual_array = np.column_stack([rr_mesh / env.Rs, res1, res2])
    np.savetxt(
        residuals_output_file,
        residual_array,
        fmt="%.12e",
        header="x=r/Rs  res1=chi' - L/r^2  res2=L' - r^2*(Vchi + Achi_rho + axion)",
    )
    print(f"Saved: {residuals_output_file}", flush=True)

    print("\n=== FINAL BVP RESIDUAL SUMMARY (solver mesh) ===", flush=True)
    print(f"max|chi' - L/r^2|       = {diag['max_res1']:.6e}", flush=True)
    print(f"max|L' - r^2 RHS|       = {diag['max_res2']:.6e}", flush=True)
    print(f"BC L(rmin)              = {diag['bc_L0']:.6e}", flush=True)
    print(f"BC chi(R)-chi_s         = {diag['bc_chiR']:.6e}", flush=True)

# ------------------------------------------------------------
# Diagnostic terms
# ------------------------------------------------------------

V_term, M_term, Ax_term, Sum_term = kg_terms(r, chi, env, J)
rr_eff = np.maximum(r, env.rmin)

uprime_V = (rr_eff * rr_eff) * V_term
uprime_M = (rr_eff * rr_eff) * M_term
uprime_Ax = (rr_eff * rr_eff) * Ax_term
uprime_sum = uprime_V + uprime_M + uprime_Ax

x = r / env.Rs
RMAX_FRAC = 0.999
m = x <= RMAX_FRAC

S = S_window_smooth(r, env)
mw = S > 1e-6
uprime_Ax_plot = np.where(mw, uprime_Ax, np.nan)

# ------------------------------------------------------------
# Plot 1 — Profiles
# ------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax1.plot(x[m], chi[m], label=r"$\chi$", lw=2.0)
ax1.axvline(1.0, ls="--", color="k", alpha=0.4)
ax1.set_ylabel(r"$\chi$")
ax1.grid(alpha=0.3)
ax1.legend(loc="best")

ax2.plot(x[m], a[m], label=r"$a$", lw=2.0)
ax2.axvline(1.0, ls="--", color="k", alpha=0.4)
ax2.set_xlabel(r"$r/R_s$")
ax2.set_ylabel(r"$a$")
ax2.grid(alpha=0.3)
ax2.legend(loc="best")

plt.tight_layout()
profiles_plot_file = run_path("profiles_plot.pdf")
plt.savefig(profiles_plot_file, dpi=300)
plt.show()
print(f"Saved: {profiles_plot_file}", flush=True)

# ------------------------------------------------------------
# Plot 2 — L(r)
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x[m], L_profile[m], label=r"$L=r^2\chi'$", lw=2.2)
ax.axvline(1.0, ls="--", color="k", alpha=0.4)
ax.axhline(0.0, color="k", alpha=0.3)

ax.set_xlabel(r"$r/R_s$")
ax.set_ylabel(r"$L(r)$")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
L_plot_file = run_path("L_profile_plot.pdf")
plt.savefig(L_plot_file, dpi=300)
plt.show()
print(f"Saved: {L_plot_file}", flush=True)

# ------------------------------------------------------------
# Plot 3 — L'(r) contributions
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x[m], uprime_V[m], label=r"$r^2 V_{,\chi}$")
ax.plot(x[m], uprime_M[m], label=r"$r^2(\beta\rho e^{\beta\chi})$")
ax.plot(x[m], uprime_Ax_plot[m], label=r"$r^2(\mathrm{axion})$")
ax.plot(x[m], uprime_sum[m], label=r"$L'(\mathrm{total})$", lw=2.2)

ax.axvline(1.0, ls="--", color="k", alpha=0.4)
ax.axhline(0.0, color="k", alpha=0.3)

ax.set_xlabel(r"$r/R_s$")
ax.set_ylabel(r"$L'(r)$")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
Lprime_plot_file = run_path("Lprime_contributions_plot.pdf")
plt.savefig(Lprime_plot_file, dpi=300)
plt.show()
print(f"Saved: {Lprime_plot_file}", flush=True)
#%%
# ============================================================
# Configuration for plotting from the single run directory
# ============================================================

run_directory = str(RUN_DIR)

profiles_file = run_path("profiles_best.txt")
objective_file = run_path("objective_scan.txt")
summary_file = run_path("summary.txt")

profile_x_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
axion_x_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
density_x_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]

chi_y_ticks = None
axion_y_ticks = None
density_y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

profile_files = [profiles_file]
profile_labels = [r"$\mathrm{current\ run}$"]

objective_plot_path = plot_objective_from_files(
    objective_file=objective_file,
    summary_file=summary_file,
    output_label="bbq",
    output_prefix="bbq_energy",
    axis_labelsize=18,
    tick_labelsize=16,
    legend_fontsize=16,
)
print("Saved objective plot:", objective_plot_path, flush=True)

# Example calls if these helper functions are enabled elsewhere in your script:
# output_profiles = plot_profiles_multi(
#     profile_files,
#     profile_labels,
#     output_label="bbq",
#     output_prefix="bbq_profiles",
#     chi_x_ticks=profile_x_ticks,
#     axion_x_ticks=axion_x_ticks,
#     density_x_ticks=density_x_ticks,
# )
# print("Saved profile overlay:", output_profiles)
#
# output_L = plot_L_abs_multi(
#     profile_files,
#     profile_labels,
#     output_label="bbq",
#     output_prefix="bbq",
#     x_ticks=profile_x_ticks,
#     rmin=0.0,
#     rmax=1.000001,
# )
# print("Saved L plot:", output_L)
