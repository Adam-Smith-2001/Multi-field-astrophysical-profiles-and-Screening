{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww21960\viewh15140\viewkind0
\deftab720
\pard\pardeftab720\sa321\partightenfactor0

\f0\fs28 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 The main solver (`star_solver.py`) computes the coupled field configuration for a given set of model parameters and determines the preferred surface value of the dilaton by numerically minimising the total energy. In addition to the final energy-minimised solution, the solver performs an energy scan over trial surface values, allowing the energy landscape and stability of the configuration to be analysed. The resulting profiles, energy scans, and diagnostic quantities are automatically written to the `bbq_outputs` directory.\
\
The plotting scripts located in the `plotting_code` directory read the stored outputs from multiple solver runs and generate overlayed, publication-quality figures of the field profiles, scalar charge, and energy curves. A `baseline_outputs` directory is included to provide a reference configuration for comparison, currently corresponding to a single-field (axion-less) dilaton solution.\
}