#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Axion-Dilaton Planetary Solver (Flux Approximation)
This script solves the coupled axio-dilaton field equations for a planetary density profile.
It fully integrates the dilaton field as an IVP, while treating the axion gradient using a 
conserved flux approximation. A fixed-point iteration is used to find the self-consistent 
backreaction between the two fields. 
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp, cumulative_trapezoid

plt.rcParams["mathtext.fontset"] = "cm"

@dataclass
class Env:
    Rc: float = 1.0
    rho_c: float = 1e-9
    rho_env: float = 1e-10
    ell_rho: float = 1e-3

    beta: float = 0.2
    V0: float = 5.e-10
    kappa: float = 4*beta - 1   # = -0.2

    W0: float = 1.0e0
    zeta: float = np.sqrt(2)

    a_minus: float = 0.0
    a_plus:  float = 8.2

    # alpha controls the ramp width
    alpha: float = 0.70
    ell_a: float = None

    rin: float = 1e-4
    rout: float = 10.0
    N: int = 1000000

    stiff_method: str = "Radau"

    fp_max_iter: int = 30
    fp_tol: float = 5e-6
    fp_damp: float = 0.6

    def __post_init__(self):
        if self.ell_a is None:
            self.ell_a = (1.0 - self.alpha) * self.Rc

# profile definitions
def rho_of_r(r, env):
    t = np.tanh((r - env.Rc)/env.ell_rho)
    return 0.5 * env.rho_c * (1 - t) + 0.5 * env.rho_env * (1 + t)

def S_window(r, env):
    # symmetric axion window centered on the surface Rc
    # makes sure we always have an axion gradient there
    r1 = env.Rc - 0.5 * env.ell_a
    r2 = env.Rc + 0.5 * env.ell_a
    return np.where((r > r1) & (r < r2), 1.0, 0.0)

def Vchi(chi, env):
    return env.kappa * env.V0 * np.exp(env.kappa * chi)

def Achi_rho(chi, r, env):
    return env.beta * np.exp(env.beta * chi) * rho_of_r(r, env)

def W2(chi, env):
    return env.W0**2 * np.exp(2 * env.zeta * chi)

def WWchi(chi, env):
    return env.zeta * (env.W0**2) * np.exp(2 * env.zeta * chi)

# dilaton interior minimum 
def solve_chi_min(rho_val, env):
    def f(chi):
        return Vchi(chi,env) + env.beta * rho_val * np.exp(env.beta * chi)
    chi = 0.0
    for _ in range(200):
        val = f(chi)
        if abs(val) < 1e-14:
            break
        fp = env.kappa**2 * env.V0 * np.exp(env.kappa * chi) \
             + env.beta**2 * rho_val * np.exp(env.beta * chi)
        chi -= val/max(fp, 1e-30)
    return chi

# flux reconstruction
def compute_J(r, chi, env):
    S = S_window(r,env)
    Wsq = W2(chi,env)
    denom = np.trapz(S/(r*r*Wsq), r)
    return (env.a_plus - env.a_minus) / denom

def apr_from_J(r, chi, env, J):
    return J * S_window(r,env) / (r*r*W2(chi,env))

# ode for the dilaton
def rhs_chi(r, y, env, a_prime_fun):
    chi, chip = y
    ap = a_prime_fun(r)

    chi_pp = -2*chip/r \
             + Vchi(chi,env) \
             + Achi_rho(chi,r,env) \
             + WWchi(chi,env)*(ap**2)

    return np.array([chip, chi_pp])

def plot_chi_source_terms(r, chi, ap, env):
    # grab the components driving chi''
    chip = np.gradient(chi, r, edge_order=2)

    V_term   = Vchi(chi, env)
    M_term   = Achi_rho(chi, r, env)
    Ax_term  = WWchi(chi, env) * (ap**2)
    Fric     = -2.0 * chip / r

    total = Fric + V_term + M_term + Ax_term
    x = r / env.Rc

    eps = 1e-300
    plt.figure(figsize=(9,5))
    plt.semilogy(x, np.abs(M_term)  + eps, lw=2, label=r"$|\beta\rho e^{\beta\chi}|$")
    plt.semilogy(x, np.abs(Ax_term) + eps, lw=2, label=r"$|W W_{,\chi}\,(a')^2|$")
    plt.semilogy(x, np.abs(V_term)  + eps, lw=2, label=r"$|V_{,\chi}|$")
    plt.axvline(1.0, color="k", ls="--", alpha=0.6)
    plt.xlabel(r"$r/R_\star$")
    plt.ylabel("term magnitude")
    plt.title("Dilaton equation term magnitudes")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(9,5))
    plt.plot(x, M_term,  lw=2, label=r"$\beta\rho e^{\beta\chi}$")
    plt.plot(x, Ax_term, lw=2, label=r"$W W_{,\chi}\,(a')^2$")
    plt.plot(x, V_term,  lw=2, label=r"$V_{,\chi}$")
    plt.axhline(0.0, color="k", lw=1, alpha=0.4)
    plt.axvline(1.0, color="k", ls="--", alpha=0.6)
    plt.xlabel(r"$r/R_\star$")
    plt.ylabel("signed contribution")
    plt.title("Dilaton equation signed contributions")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()

# run the integration for a given axion gradient
def integrate_chi(env, a_prime_array, r_grid):
    a_prime_fun = lambda rr: np.interp(rr, r_grid, a_prime_array)
    chi_c = solve_chi_min(env.rho_c, env)

    sol = solve_ivp(lambda rr,yy: rhs_chi(rr,yy,env,a_prime_fun),
                    (env.rin, env.rout),
                    [chi_c, 0.0], 
                    method=env.stiff_method,
                    t_eval=r_grid,
                    rtol=1e-7, atol=1e-9)

    if not sol.success:
        raise RuntimeError("χ IVP failed: " + sol.message)

    return sol.y[0] 

# main iterative solver using flux approx
def solve_flux_ivp(env):
    r_grid = np.linspace(env.rin, env.rout, env.N)

    # initial guess assuming W = W0
    S = S_window(r_grid, env)
    Wsq0 = env.W0**2
    denom0 = np.trapz(S/(r_grid*r_grid*Wsq0), r_grid)
    J0 = (env.a_plus - env.a_minus) / denom0
    a_prime = J0 * S / (r_grid*r_grid*Wsq0)

    print("\n--- Fixed-point iteration diagnostics ---")
    print("alpha =", env.alpha, " ell_a =", env.ell_a)
    print("grid dr =", r_grid[1] - r_grid[0],
          "  window points =", np.count_nonzero(S))
    print("----------------------------------------")

    for it in range(env.fp_max_iter):
        
        # integrate chi based on current axion gradient
        chi = integrate_chi(env, a_prime, r_grid)

        S = S_window(r_grid, env)
        Wsq = W2(chi, env)

        denom = np.trapz(S/(r_grid*r_grid*Wsq), r_grid)
        J_new = (env.a_plus - env.a_minus) / denom
        a_prime_flux = J_new * S / (r_grid*r_grid*Wsq)

        print(
            f"it={it:02d}  "
            f"nwin={np.count_nonzero(S):7d}  "
            f"denom={denom:.6e}  "
            f"J={J_new:.6e}  "
            f"max|ap|={np.max(np.abs(a_prime_flux)):.6e}  "
            f"nan_ap={np.any(~np.isfinite(a_prime_flux))}"
        )

        # damp the update to prevent blowing up
        a_prime = env.fp_damp * a_prime_flux + (1 - env.fp_damp) * a_prime

        if np.max(np.abs(a_prime_flux - a_prime)) < env.fp_tol:
            print("  -> converged")
            break

    # reconstruct the full axion field from the gradient
    a = cumulative_trapezoid(a_prime, r_grid, initial=0.0)
    a += env.a_minus - a[0]

    return r_grid, chi, a, a_prime


def main():
    env = Env()

    # show minima
    chi_c = solve_chi_min(env.rho_c, env)
    chi_inf = solve_chi_min(env.rho_env, env)
    print("χ_c   (inside min)      =", chi_c)
    print("χ_inf (outside min)     =", chi_inf)
    print("Δχ =", chi_inf - chi_c)
    print()

    # get the profiles
    r, chi, a, ap = solve_flux_ivp(env)
    
    # uncomment below to debug dilaton sourcing terms
    # plot_chi_source_terms(r, chi, ap, env)

    plt.figure(figsize=(9,4))
    plt.plot(r/env.Rc, chi, lw=2)
    plt.axvline(1.0, color='k', ls='--')
    plt.ylabel(r'$\chi(r)$')
    plt.grid()

    plt.figure(figsize=(9,4))
    plt.plot(r/env.Rc, a, lw=2)
    plt.axvline(1.0, color='k', ls='--')
    plt.ylabel(r'$a(r)$')
    plt.grid()

    W = np.sqrt(W2(chi, env))
    plt.figure(figsize=(9,4))
    plt.plot(r/env.Rc, a*W, lw=2)
    plt.axvline(1.0, color='k', ls='--')
    plt.ylabel(r"$a*W$")
    plt.grid()
    
    # write out to file
    data2 = np.column_stack([r, rho_of_r(r, env), chi, a, ap, W, a*W])
    header2 = "r   rho   chi(r)   a(r)   a_prime(r)   W(r)   a(r)*W(r)"
    
    np.savetxt("axion_dilaton_full_profile.txt", data2,
               header=header2,
               fmt="%.10e")
    print("Saved: axion_dilaton_full_profile.txt")


if __name__ == "__main__":
    main()