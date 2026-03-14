#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Author: Adam Smith"""
"""
Plot stacked profile comparisons and scan diagnostics from BBQ output files.

This script can:
  1. Overlay multiple profile files in a four-panel figure.
  2. Plot a normalized scan objective from a scan file.
  3. Plot the raw scan energy as a function of chi_s.

Input files
-----------
Profile files are expected to contain at least 7 columns:
    r, chi, chi_prime, a, a_prime, rho, W

Scan files are expected to contain three columns:
    chi_s   E_scan   ok(1/0)

Blank lines and comment lines beginning with '#' are ignored.

Output
------
Figures are written to the configured output directory.

Usage
-----
Edit the configuration section at the bottom, then run:

    python bbq_star_plotter.py

For non-interactive use:

    python bbq_star_plotter.py --headless
"""



import os
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

HEADLESS = ("--headless" in sys.argv) or ("--no-show" in sys.argv)

import matplotlib
if HEADLESS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator


# -------------------------------------------------------------------
# Plot style
# -------------------------------------------------------------------

plt.rcParams["mathtext.fontset"] = "cm"

AXIS_LABELSIZE = 20
TICK_LABELSIZE = 18
LEGEND_FONTSIZE = 18


# -------------------------------------------------------------------
# Data containers
# -------------------------------------------------------------------

@dataclass
class ProfileData:
    """Container for one radial profile file."""
    r: np.ndarray
    chi: np.ndarray
    chi_prime: np.ndarray
    a: np.ndarray
    a_prime: np.ndarray
    rho: np.ndarray
    W: np.ndarray

    @property
    def surface_radius(self) -> float:
        rs = float(np.max(self.r))
        return rs if rs > 0.0 else 1.0

    @property
    def chi_surface(self) -> float:
        return float(self.chi[-1])

    @property
    def chi_center(self) -> float:
        return float(self.chi[0])

    @property
    def rho_center(self) -> float:
        if np.isfinite(self.rho[0]):
            return max(float(self.rho[0]), 1e-300)
        return max(float(np.nanmax(self.rho)), 1e-300)


@dataclass
class ScanData:
    """Container for a scan file with columns chi_s, E_scan, ok."""
    chi_surface: np.ndarray
    total_energy: np.ndarray
    is_valid: np.ndarray


@dataclass
class ProfileSpec:
    """Description of one profile to include in the overlay plot."""
    path: str
    label: str


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_figure(fig: plt.Figure, path: str, show: bool) -> None:
    """Save a figure and close it when plots are not shown interactively."""
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(path)}")
    if not show:
        plt.close(fig)


# -------------------------------------------------------------------
# File loading
# -------------------------------------------------------------------

def load_profile_file(path: str) -> ProfileData:
    """
    Load a radial profile file.

    The file must contain at least 7 columns:
        r, chi, chi_prime, a, a_prime, rho, W
    """
    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"Expected at least 7 columns in profile file: {path}")

    arr = arr[:, :7]
    r, chi, chi_prime, a, a_prime, rho, W = arr.T

    return ProfileData(
        r=r,
        chi=chi,
        chi_prime=chi_prime,
        a=a,
        a_prime=a_prime,
        rho=rho,
        W=W,
    )


def load_scan_file(path: str) -> Optional[ScanData]:
    """
    Load a scan file with columns:
        chi_s   E_scan   ok(1/0)
    """
    chi_surface_values = []
    total_energy_values = []
    is_valid_values = []

    with open(path, "r") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue

            parts = text.split()
            if len(parts) < 3:
                continue

            try:
                chi_surface = float(parts[0])
                total_energy = float(parts[1]) if parts[1].lower() != "nan" else np.nan
                is_valid = int(parts[2])
            except Exception:
                continue

            chi_surface_values.append(chi_surface)
            total_energy_values.append(total_energy)
            is_valid_values.append(is_valid)

    if not chi_surface_values:
        return None

    return ScanData(
        chi_surface=np.asarray(chi_surface_values, dtype=float),
        total_energy=np.asarray(total_energy_values, dtype=float),
        is_valid=np.asarray(is_valid_values, dtype=int),
    )


# -------------------------------------------------------------------
# Tick formatting
# -------------------------------------------------------------------

def format_sci_mathtext(value: float) -> str:
    """Format a number using CM-style mathtext."""
    value = float(value)
    abs_value = abs(value)

    if abs_value == 0.0:
        return r"$0$"

    if abs_value < 1e-3 or abs_value >= 1e3:
        exponent = int(np.floor(np.log10(abs_value)))
        mantissa = value / (10.0 ** exponent)

        if abs(mantissa - round(mantissa)) < 1e-12:
            mantissa_str = f"{int(round(mantissa))}"
        else:
            mantissa_str = f"{mantissa:.2g}"

        return rf"${mantissa_str}\times 10^{{{exponent}}}$"

    if abs(value - round(value)) < 1e-12:
        return rf"${int(round(value))}$"

    return rf"${value:.2f}$"


def generic_tick_formatter(x, pos):
    return format_sci_mathtext(x)


def radial_tick_formatter(x, pos):
    if abs(x - 1.0) < 1e-12:
        return r"$R_s$"
    if abs(x - round(x)) < 1e-12:
        return rf"${int(round(x))}$"
    return rf"${x:.1f}$"


def set_fixed_ticks(ax, xticks=None, yticks=None) -> None:
    """Apply fixed major tick locations when provided."""
    if xticks is not None:
        ax.xaxis.set_major_locator(FixedLocator(xticks))
    if yticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(yticks))


def apply_radial_tick_format(ax, format_y: bool = True, xticks=None, yticks=None) -> None:
    """Apply r/R_s formatting to the x-axis and optional fixed tick locations."""
    set_fixed_ticks(ax, xticks=xticks, yticks=yticks)
    ax.xaxis.set_major_formatter(FuncFormatter(radial_tick_formatter))
    if format_y:
        ax.yaxis.set_major_formatter(FuncFormatter(generic_tick_formatter))


def apply_generic_tick_format(
    ax,
    format_x: bool = True,
    format_y: bool = True,
    xticks=None,
    yticks=None,
) -> None:
    """Apply generic mathtext formatting and optional fixed tick locations."""
    set_fixed_ticks(ax, xticks=xticks, yticks=yticks)
    if format_x:
        ax.xaxis.set_major_formatter(FuncFormatter(generic_tick_formatter))
    if format_y:
        ax.yaxis.set_major_formatter(FuncFormatter(generic_tick_formatter))


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

def plot_profiles(
    profile_specs: Sequence[ProfileSpec],
    output_dir: str,
    show: bool,
    show_surface_chi_lines: bool = False,
    filename: str = "profiles.pdf",
    ticks: Optional[dict] = None,
) -> None:
    """
    Plot multiple radial profiles in a four-panel stacked figure.

    Optional tick dictionary format:
        {
            "x":   [...],
            "chi": [...],
            "a":   [...],
            "rho": [...],
            "L":   [...],
        }
    """
    if ticks is None:
        ticks = {}

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    ax_chi, ax_a, ax_rho, ax_L = axes

    loaded_any = False

    for spec in profile_specs:
        if not os.path.exists(spec.path):
            print(f"[WARN] Missing profile file: {spec.path} (skipping)")
            continue

        data = load_profile_file(spec.path)
        loaded_any = True

        x = data.r / data.surface_radius
        chi_center = data.chi_center
        rho_center = data.rho_center
        chi_surface = data.chi_surface

        mask_chi = x < 6.0
        mask_inner = x < 2.0

        line_chi, = ax_chi.plot(
            x[mask_chi],
            data.chi[mask_chi] - chi_center,
            label=spec.label,
        )

        if show_surface_chi_lines:
            ax_chi.axhline(
                chi_surface - chi_center,
                ls="--",
                alpha=0.35,
                color=line_chi.get_color(),
            )

        ax_a.plot(x[mask_inner], data.a[mask_inner], label=spec.label)
        ax_rho.plot(x[mask_inner], data.rho[mask_inner] / rho_center, label=spec.label)
        ax_L.plot(
            x[mask_inner],
            data.chi_prime[mask_inner] * data.r[mask_inner] ** 2,
            label=spec.label,
        )

    if not loaded_any:
        raise RuntimeError("No valid profile files were loaded.")

    apply_radial_tick_format(ax_chi, format_y=True, xticks=ticks.get("x"), yticks=ticks.get("chi"))
    apply_radial_tick_format(ax_a,   format_y=True, xticks=ticks.get("x"), yticks=ticks.get("a"))
    apply_radial_tick_format(ax_rho, format_y=True, xticks=ticks.get("x"), yticks=ticks.get("rho"))
    apply_radial_tick_format(ax_L,   format_y=True, xticks=ticks.get("x"), yticks=ticks.get("L"))

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.axvline(1.0, ls="--", color="k", alpha=0.5)
        ax.tick_params(labelsize=TICK_LABELSIZE)

    ax_chi.set_ylabel(r"$(\chi-\chi_c)/M_{\rm pl}$", fontsize=AXIS_LABELSIZE)
    ax_a.set_ylabel(r"$\mathfrak{a}/M_{\rm pl}$", fontsize=AXIS_LABELSIZE)
    ax_rho.set_ylabel(r"$\rho/\rho_c$", fontsize=AXIS_LABELSIZE)
    ax_L.set_ylabel(r"$L=r^2\chi'$", fontsize=AXIS_LABELSIZE)
    ax_L.set_xlabel(r"$r/R_s$", fontsize=AXIS_LABELSIZE)

    ax_a.legend(loc="best", frameon=False, fontsize=LEGEND_FONTSIZE)

    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, filename), show)


def plot_normalized_objective(
    scan_file: str,
    output_dir: str,
    chi_surface_reference: float,
    show: bool,
    chi_initial_guess: Optional[float] = None,
    chi_analytic_guess: Optional[float] = None,
    y_floor: float = 1e-18,
    filename: str = "objective_from_scan.pdf",
    ticks: Optional[dict] = None,
) -> None:
    """
    Plot the normalized scan objective:
        (E(chi_s) - E_min) / |E_min|

    Optional tick dictionary format:
        {
            "x": [...],
            "y": [...],
        }

    For the log y-axis, supplied y ticks should be values such as
    [1e-18, 1e-15, 1e-12, 1e-9].
    """
    if ticks is None:
        ticks = {}

    scan = load_scan_file(scan_file)
    if scan is None:
        print("[WARN] No valid rows found in scan file; skipping normalized objective plot.")
        return

    valid = (scan.is_valid == 1) & np.isfinite(scan.total_energy)
    if not np.any(valid):
        print("[WARN] No valid scan points with finite energy; skipping.")
        return

    chi_surface = scan.chi_surface[valid]
    total_energy = scan.total_energy[valid]

    energy_min = float(np.min(total_energy))
    denominator = max(abs(energy_min), 1e-300)

    dx = (chi_surface - chi_surface_reference) * 1e12
    y = (total_energy - energy_min) / denominator
    y = np.maximum(y, y_floor)

    order = np.argsort(dx)
    dx_sorted = dx[order]
    y_sorted = y[order]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.4))
    ax.set_yscale("log")

    apply_generic_tick_format(
        ax,
        format_x=True,
        format_y=True,
        xticks=ticks.get("x"),
        yticks=ticks.get("y"),
    )

    ax.plot(
        dx_sorted,
        y_sorted,
        marker="o",
        ms=3,
        lw=1,
        ls="-",
        label=r"$\text{valid scan points}$",
    )

    ax.axvline(0.0, ls="--", alpha=0.8, label=r"$\chi_{s,\min}\;\text{(numerical)}$")

    if chi_initial_guess is not None and np.isfinite(chi_initial_guess):
        ax.axvline(
            (chi_initial_guess - chi_surface_reference) * 1e12,
            ls=":",
            alpha=0.8,
            label=r"$\chi_{s,\rm initial}$",
        )

    if chi_analytic_guess is not None and np.isfinite(chi_analytic_guess):
        ax.axvline(
            (chi_analytic_guess - chi_surface_reference) * 1e12,
            ls="-.",
            alpha=0.8,
            label=r"$\chi_{s,\rm analytic}$",
        )

    ax.set_xlabel(r"$(\chi_s-\chi_{s,\min})\,10^{12}/M_{\rm pl}$", fontsize=AXIS_LABELSIZE)
    ax.set_ylabel(r"$(E(\chi_s)-E_{\min})/|E_{\min}|$", fontsize=AXIS_LABELSIZE)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=TICK_LABELSIZE)
    
    ax.legend(loc="best", frameon=False, fontsize=LEGEND_FONTSIZE)

    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, filename), show)


def plot_scan_energy(
    scan_file: str,
    output_dir: str,
    chi_surface_reference: float,
    show: bool,
    chi_initial_guess: Optional[float] = None,
    chi_analytic_guess: Optional[float] = None,
    filename: str = "scan_energy.pdf",
    ticks: Optional[dict] = None,
) -> None:
    """
    Plot raw scan energy as a function of chi_s.

    Optional tick dictionary format:
        {
            "x": [...],
            "y": [...],
        }
    """
    if ticks is None:
        ticks = {}

    scan = load_scan_file(scan_file)
    if scan is None:
        print("[WARN] No valid rows found in scan file; skipping energy plot.")
        return

    valid = (scan.is_valid == 1) & np.isfinite(scan.total_energy)
    if not np.any(valid):
        print("[WARN] No valid scan points with finite energy; skipping.")
        return

    chi_surface = scan.chi_surface[valid]
    total_energy = scan.total_energy[valid]

    order = np.argsort(chi_surface)
    chi_surface_sorted = chi_surface[order]
    total_energy_sorted = total_energy[order]

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.2))

    apply_generic_tick_format(
        ax,
        format_x=True,
        format_y=True,
        xticks=ticks.get("x"),
        yticks=ticks.get("y"),
    )

    ax.plot(
        chi_surface_sorted,
        total_energy_sorted,
        marker="o",
        ms=3,
        lw=1,
        ls="-",
        label=r"$\text{valid scan points}$",
    )

    if chi_initial_guess is not None and np.isfinite(chi_initial_guess):
        ax.axvline(
            chi_initial_guess,
            ls="--",
            alpha=0.6,
            label=r"$\chi_{s,\rm initial}$",
        )

    if chi_analytic_guess is not None and np.isfinite(chi_analytic_guess):
        ax.axvline(
            chi_analytic_guess,
            ls=":",
            alpha=0.8,
            label=r"$\chi_{s,\rm analytic}$",
        )

    ax.axvline(
        chi_surface_reference,
        ls="-.",
        alpha=0.85,
        label=r"$\chi_{s,\min}\; \text{(numerical)}$",
    )

    ax.set_xlabel(r"$\chi_s$", fontsize=AXIS_LABELSIZE)
    ax.set_ylabel(r"$E$", fontsize=AXIS_LABELSIZE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=TICK_LABELSIZE)
    ax.legend(loc="best", frameon=False, fontsize=LEGEND_FONTSIZE)

    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, filename), show)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

if __name__ == "__main__":
    output_dir = "BBQ energy-minimiser/Stars/baseline_outputs/example_run"
    ensure_dir(output_dir)

    profile_specs = [
        ProfileSpec(
            path="BBQ energy-minimiser/Stars/baseline_outputs/profiles_best.txt",
            label=r"$\Delta\mathfrak{a}=0$",
        ),
        ProfileSpec(
            path="BBQ energy-minimiser/Stars/bbq_outputs/profiles_best.txt",
            label=r"$\ell_{\mathfrak{a}}=0.4$",
        ),
    ]

    scan_file = "BBQ energy-minimiser/Stars/bbq_outputs/dense_scan.txt"

    # Profile used to define the reference surface value shown on the scan plots.
    reference_profile_index = 1

    chi_initial_guess = None
    chi_analytic_guess = None

    # Optional custom tick locations.
    # Set an entry to None to use matplotlib defaults.

    profile_ticks = {
        "x": [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001],
        "chi": [0.0, 5.0e-8, 1.0e-7, 1.5e-7],
        "a": None,
        "rho": [0.0, 0.25, 0.5, 0.75, 1.0],
        "L": [0.0, 2.5e-8, 5.0e-8, 7.5e-8, 1.0e-7],
    }

    objective_ticks = {
        "x": None,
        "y": None,
    }

    energy_ticks = {
        "x": None,
        "y": None,
    }

    show_plots = not HEADLESS

    plot_profiles(
        profile_specs=profile_specs,
        output_dir=output_dir,
        show=show_plots,
        show_surface_chi_lines=False,
        filename="profiles.pdf",
        ticks=profile_ticks,
    )

    if os.path.exists(scan_file):
        reference_profile = load_profile_file(profile_specs[reference_profile_index].path)
        chi_surface_reference = reference_profile.chi_surface

        plot_normalized_objective(
            scan_file=scan_file,
            output_dir=output_dir,
            chi_surface_reference=chi_surface_reference,
            show=show_plots,
            chi_initial_guess=chi_initial_guess,
            chi_analytic_guess=chi_analytic_guess,
            filename="objective_from_scan.pdf",
            ticks=objective_ticks,
        )

        plot_scan_energy(
            scan_file=scan_file,
            output_dir=output_dir,
            chi_surface_reference=chi_surface_reference,
            show=show_plots,
            chi_initial_guess=chi_initial_guess,
            chi_analytic_guess=chi_analytic_guess,
            filename="scan_energy.pdf",
            ticks=energy_ticks,
        )
    else:
        print("[INFO] Scan file not found; skipping scan plots.")

    if show_plots:
        plt.show()