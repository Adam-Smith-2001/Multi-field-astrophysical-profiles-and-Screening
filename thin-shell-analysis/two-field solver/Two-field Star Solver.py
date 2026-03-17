#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coupled axion–dilaton star solver (Jordan density variant) — Picard-stable + continuation.

This version is designed to fix the "works at a_plus=10 but blows up at 11" behaviour by:
  1) doing continuation in a_plus (homotopy): solve 10 -> 11 in small steps,
  2) integrating the chi IVP in three segments with a small max_step only near the surface,
  3) guarding against NaNs/Infs from a bad intermediate iterate (shrinks omega_chi).

No flux-form rewrite. Axion BVP equation is unchanged.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp, solve_bvp
from scipy.interpolate import interp1d, PchipInterpolator

plt.rcParams["mathtext.fontset"] = "cm"


# Numerical safety
EXP_CLIP = 700.0
def exp_safe(x):
    x = np.asarray(x, dtype=float)
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))


# Environment / parameters
@dataclass
class Env:
    Rc: float = 1.0

    # density profile
    rho_c: float = 1.0e-6
    rho_env: float = 1e-14
    n_poly: int = 3
    ell_rho: float = 1e-3

    # dilaton sector
    beta: float = 0.2
    V0: float = 5.0e-7
    kappa: float = None  # if None -> 4*beta - 1

    # W(chi)=W0 exp(zeta chi)
    W0: float = 1.0e0
    zeta: float = np.sqrt(2)

    # axion BCs
    a_minus: float = 0.0
    a_plus: float = 10.0

    # axion microphysics
    m0: float = 1.0
    Lambda_a: float = 1.0e-3

    # domain
    rmin: float = 1e-4
    rmax: float = 6.0

    # main grid (dense near Rc)
    n_in: int = 450
    n_shell: int = 1800
    n_out: int = 450
    w_shell: float = None  # if None -> max(10*ell_rho, 0.03)

    # Picard
    max_iter: int = 80
    tol_chi: float = 2e-6
    tol_a: float = 2e-6
    damp_chi: float = 0.1
    max_dchi_step: float = 0.5  # absolute cap on chi update per iteration

    # axion update: Gauss–Seidel
    use_gauss_seidel_a: bool = True

    # chi IVP
    chi_method: str = "Radau"  
    chi_rtol: float = 2e-6
    chi_atol: float = 1e-9
    max_step_outer: float = 0.05      # outside shell
    max_step_shell: float = None      # if None -> ell_rho/5

    # initial chi guess 
    chi_env_init: float = 0.0
    chi_init_width: float = None

    # continuation
    a_plus_target: float = 9.0e0
    a_plus_start: float = 5.0e0
    a_plus_step: float = 1.0e0  # smaller step = more robust

    def __post_init__(self):
        if self.kappa is None:
            self.kappa = 4.0 * self.beta - 1.0
        if self.max_step_shell is None:
            self.max_step_shell = max(1e-4, (self.ell_rho if self.ell_rho > 0 else 5e-3) / 5.0)

# Density profile rho(r)
def rho_star_poly(r, env: Env):
    r = np.asarray(r, dtype=float)
    inside = r <= env.Rc
    rho = np.empty_like(r)
    x2 = (r[inside] / env.Rc) ** 2
    rho[inside] = env.rho_c * np.maximum(1.0 - x2, 0.0) ** env.n_poly
    rho[~inside] = env.rho_env
    return rho

def rho_of_r(r, env: Env):
    r = np.asarray(r, dtype=float)
    if env.ell_rho <= 0:
        return rho_star_poly(r, env)
    t = (r - env.Rc) / env.ell_rho
    s = 0.5 * (1.0 - np.tanh(t))  # 1 inside -> 0 outside
    rho_in = rho_star_poly(r, env)
    return env.rho_env + (rho_in - env.rho_env) * s

# ============================================================
# Model functions
# ============================================================
def A(chi, env: Env):
    return exp_safe(env.beta * chi)

def Vchi(chi, env: Env):
    # dV/dchi for V = V0 exp(kappa chi)
    return env.kappa * env.V0 * exp_safe(env.kappa * chi)

def W2(chi, env: Env):
    return (env.W0 ** 2) * exp_safe(2.0 * env.zeta * chi)

def dW2_dchi(chi, env: Env):
    return 2.0 * env.zeta * (env.W0 ** 2) * exp_safe(2.0 * env.zeta * chi)

def U(a, env: Env):
    return 0.5 * ((a - env.a_minus) / env.Lambda_a) ** 2

def dU_da(a, env: Env):
    return (a - env.a_minus) / (env.Lambda_a ** 2)

def dV_da(a, env: Env):
    return (env.m0 ** 2) * (a - env.a_plus)


# chi_min(rho) for diagnostics / IC (U ignored)
def solve_chi_min(rho_val, env: Env):
    rho_val = float(rho_val)
    chi = 0.0
    for _ in range(200):
        f = Vchi(chi, env) + env.beta * rho_val * exp_safe(env.beta * chi)
        if abs(f) < 1e-14:
            break
        fp = (env.kappa ** 2) * env.V0 * exp_safe(env.kappa * chi) + (env.beta ** 2) * rho_val * exp_safe(env.beta * chi)
        chi -= f / max(fp, 1e-30)
    return chi


# Nonuniform main grid (dense near Rc)
def build_r_grid(env: Env):
    w = env.w_shell
    if w is None:
        w = max(10.0 * env.ell_rho if env.ell_rho > 0 else 0.0, 0.03)
    rL = max(env.rmin, env.Rc - w)
    rR = min(env.rmax, env.Rc + w)

    parts = []
    if rL > env.rmin:
        parts.append(np.linspace(env.rmin, rL, env.n_in, endpoint=False))
    # include Rc exactly
    parts.append(np.linspace(rL, env.Rc, env.n_shell // 2, endpoint=False))
    parts.append(np.linspace(env.Rc, rR, env.n_shell - env.n_shell // 2, endpoint=False))
    parts.append(np.linspace(rR, env.rmax, env.n_out))

    r = np.unique(np.concatenate(parts))
    if not np.any(np.isclose(r, env.Rc)):
        r = np.sort(np.append(r, env.Rc))
    return r


# Axion BVP given chi, chi'
def solve_axion_bvp(r_full, chi, chip, env: Env, a_init=None):
    r_bvp = r_full
    chi_fun  = interp1d(r_full, chi,  kind="linear", bounds_error=True)
    chip_fun = interp1d(r_full, chip, kind="linear", bounds_error=True)

    def rhs(r, y):
        a, ap = y
        ch  = np.clip(chi_fun(r),  -1e8, 1e8)
        chp = np.clip(chip_fun(r), -1e8, 1e8)
        source = dV_da(a, env) + A(ch, env) * rho_of_r(r, env) * dU_da(a, env)
        app = -(2.0 / r) * ap - 2.0 * env.zeta * chp * ap + source / W2(ch, env)
        return np.vstack((ap, app))

    def bc(y0, yR):
        return np.array([y0[1], yR[0] - env.a_plus])

    # initial guess: smooth step + anchor outer BC
    w = max(env.ell_rho if env.ell_rho > 0 else 0.0, 5e-3)
    s = 0.5 * (1.0 + np.tanh((r_bvp - env.Rc) / w))
    if a_init is None:
        a_guess = env.a_minus + (env.a_plus - env.a_minus) * s
    else:
        a_guess = np.interp(r_bvp, r_full, a_init)
        a_guess = a_guess + (env.a_plus - a_guess[-1]) * s
    ap_guess = np.gradient(a_guess, r_bvp, edge_order=2)

    sol = solve_bvp(rhs, bc, r_bvp, np.vstack((a_guess, ap_guess)),
                    tol=1e-4, max_nodes=300000, verbose=0)
    if not sol.success:
        raise RuntimeError("Axion BVP failed: " + sol.message)
    return sol.sol(r_full)


# Chi IVP (piecewise max_step: outer / shell / outer)
def solve_chi_ivp_piecewise(r_grid, a, ap, env: Env):
    r0 = float(r_grid[0])
    r1 = float(r_grid[-1])

    ap_spline = PchipInterpolator(r_grid, np.array(ap, copy=True), extrapolate=False)
    a_spline  = PchipInterpolator(r_grid, np.array(a,  copy=True), extrapolate=False)

    def rhs(r, y):
        chi, chip = y
        rc = r0 if r < r0 else (r1 if r > r1 else r)
        apr = float(ap_spline(rc))
        ar  = float(a_spline(rc))

        fric = -2.0 * chip / r
        Vt   = Vchi(chi, env)

        rh   = float(rho_of_r(rc, env))
        At   = env.beta * rh * exp_safe(env.beta * chi) * (1.0 + U(ar, env))

        axt  = 0.5 * dW2_dchi(chi, env) * (apr ** 2)
        return np.array([chip, fric + Vt + At + axt], dtype=float)

    # shell band
    w = env.w_shell
    if w is None:
        w = max(10.0 * env.ell_rho if env.ell_rho > 0 else 0.0, 0.03)
    rL = max(env.rmin, env.Rc - w)
    rR = min(env.rmax, env.Rc + w)

    def run_seg(t0, t1, y0, t_eval, max_step):
        if t_eval.size == 0:
            return y0, np.empty((2, 0))
        sol = solve_ivp(rhs, (t0, t1), y0, method=env.chi_method,
                        t_eval=t_eval, rtol=env.chi_rtol, atol=env.chi_atol,
                        max_step=max_step)
        if not sol.success:
            raise RuntimeError("Chi IVP failed: " + sol.message)
        if not np.all(np.isfinite(sol.y)):
            raise RuntimeError("Chi IVP produced non-finite values.")
        return sol.y[:, -1], sol.y

    chi_c = solve_chi_min(float(rho_of_r(env.rmin, env)), env)
    y = np.array([chi_c, 0.0], dtype=float)

    # segment 1
    m1 = (r_grid >= env.rmin) & (r_grid <= rL)
    r1_eval = r_grid[m1]
    y, Y1 = run_seg(env.rmin, rL, y, r1_eval, env.max_step_outer)

    # segment 2 (shell)
    m2 = (r_grid > rL) & (r_grid <= rR)
    r2_eval = r_grid[m2]
    y, Y2 = run_seg(rL, rR, y, r2_eval, env.max_step_shell)

    # segment 3
    m3 = (r_grid > rR) & (r_grid <= env.rmax)
    r3_eval = r_grid[m3]
    y, Y3 = run_seg(rR, env.rmax, y, r3_eval, env.max_step_outer)

    # stitch
    chi = np.empty_like(r_grid)
    chip = np.empty_like(r_grid)
    chi[m1] = Y1[0]; chip[m1] = Y1[1]
    chi[m2] = Y2[0]; chip[m2] = Y2[1]
    chi[m3] = Y3[0]; chip[m3] = Y3[1]
    return chi, chip

# ============================================================
# Coupled Picard for fixed a_plus
# ============================================================
def solve_coupled_fixed_aplus(env: Env, r, rho_vals, chi, chip, a, ap):
    for it in range(env.max_iter):
        chi_old = chi.copy()
        chip_old = chip.copy()
        a_old = a.copy()

        a_new, ap_new = solve_axion_bvp(r, chi, chip, env, a_init=a)
        a, ap = a_new, ap_new  # Gauss–Seidel

        chi_ivp, chip_ivp = solve_chi_ivp_piecewise(r, a, ap, env)

        dchi_vec = chi_ivp - chi
        max_dchi = float(np.max(np.abs(dchi_vec)))
        omega = env.damp_chi
        if (env.max_dchi_step is not None) and (env.max_dchi_step > 0.0) and (max_dchi > 0.0):
            omega = min(omega, env.max_dchi_step / (max_dchi + 1e-30))

        if not np.all(np.isfinite(chi_ivp)) or not np.all(np.isfinite(chip_ivp)):
            omega *= 0.1

        # consistent damping for BOTH chi and chi'
        chi  = chi  + omega * (chi_ivp  - chi)
        chip = chip + omega * (chip_ivp - chip)

        dchi_upd  = float(np.max(np.abs(chi - chi_old)))
        dchip_upd = float(np.max(np.abs(chip - chip_old)))
        da_upd    = float(np.max(np.abs(a - a_old)))

        print(f"  it={it:02d}  |Δchi|={dchi_upd:.3e}  |Δchip|={dchip_upd:.3e}  |Δa|={da_upd:.3e}  omega={omega:.3e}  a(rmax)={a[-1]:.3e}")

        if (dchi_upd < env.tol_chi) and (da_upd < env.tol_a):
            print("  -> converged")
            break

    # final projection: return IVP-consistent profiles
    chi, chip = solve_chi_ivp_piecewise(r, a, ap, env)
    return chi, chip, a, ap


# ============================================================
# Continuation driver
# ============================================================
def solve_with_continuation(env: Env):
    r = build_r_grid(env)
    rho_vals = rho_of_r(r, env)

    # initial chi guess: smooth interior min -> modest outer guess
    chi_c = solve_chi_min(env.rho_c, env)
    chi_e = env.chi_env_init
    w = env.chi_init_width
    if w is None:
        w = max(env.ell_rho if env.ell_rho > 0 else 0.0, 5e-3)
    s = 0.5 * (1.0 + np.tanh((r - env.Rc) / w))
    chi = chi_c + (chi_e - chi_c) * s
    chip = np.gradient(chi, r, edge_order=2)

    # initial a for a_plus_start
    env.a_plus = env.a_plus_start
    a = env.a_minus + (env.a_plus - env.a_minus) * s
    ap = np.gradient(a, r, edge_order=2)

    print("=== effective potential minima (analytic; U ignored) ===")
    print("center chi_min:", solve_chi_min(env.rho_c, env))
    print("env chi_min:   ", solve_chi_min(env.rho_env, env))
    print("=======================================================")

    # continuation steps
    a_steps = [env.a_plus_start]
    x = env.a_plus_start
    while x + 1e-12 < env.a_plus_target:
        x = min(env.a_plus_target, x + env.a_plus_step)
        a_steps.append(x)

    for ap_target in a_steps:
        env.a_plus = float(ap_target)
        print(f"\n=== Solving for a_plus = {env.a_plus:.6g} ===")
        chi, chip, a, ap = solve_coupled_fixed_aplus(env, r, rho_vals, chi, chip, a, ap)

    return r, rho_vals, chi, chip, a, ap

# ============================================================
# Main
# ============================================================
def main():
    env = Env()
    r, rho_vals, chi, chip, a, ap = solve_with_continuation(env)
    chi, chip = solve_chi_ivp_piecewise(r, a, ap, env)


    # Save
    Wvals = np.sqrt(W2(chi, env))
    data2 = np.column_stack([r, rho_of_r(r, env), chi, a, ap, Wvals, a * Wvals])
    header2 = "r   rho   chi(r)   a(r)   a_prime(r)   W(r)   a(r)*W(r)"
    np.savetxt("axion_dilaton_full_profile.txt", data2, header=header2, fmt="%.10e")
    print("\nSaved: axion_dilaton_full_profile.txt")

    # Plots
    x = r / env.Rc
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax1.plot(x, rho_vals / env.rho_c, lw=2)
    ax1.set_ylabel(r"$\rho(r)/\rho_c$")
    ax1.axvline(1.0, color="k", ls="--", alpha=0.6)
    ax1.grid(alpha=0.3)

    ax2.plot(x, chi, lw=2)
    ax2.set_ylabel(r"$\chi(r)$")
    ax2.axvline(1.0, color="k", ls="--", alpha=0.6)
    ax2.grid(alpha=0.3)

    ax3.plot(x, a, lw=2)
    ax3.set_ylabel(r"$a(r)$")
    ax3.set_xlabel(r"$r/R_\star$")
    ax3.axvline(1.0, color="k", ls="--", alpha=0.6)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # zoom around Rc
    plt.figure(figsize=(8, 3))
    m = (x > 0.96) & (x < 1.04)
    plt.plot(x[m], chi[m], marker="o", ms=3)
    plt.axvline(1.0, color="k", ls="--", alpha=0.6)
    plt.grid(alpha=0.3)
    plt.xlabel(r"$r/R_\star$")
    plt.ylabel(r"$\chi$")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
