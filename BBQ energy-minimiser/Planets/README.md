{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 This script solves the coupled BBQ thin-shell plus axion-flux system for a step-function-like planetary density profile, with the dilaton surface value determined by numerical energy minimisation. For a given parameter configuration, the code performs a global search over trial surface values, refines the candidate minima, and selects the configuration with the lowest total energy. Alongside the final energy-minimised solution, it also computes the corresponding field profiles, scalar charge, energy scan, and residual diagnostics used to verify the boundary-value solution directly on the solver mesh.\
\
The plotting for this step-function solver is built directly into the same script. Once the solution has been obtained, the code immediately generates the main diagnostic and publication-style PDF plots from the saved outputs, without requiring a separate plotting script. All numerical text outputs and generated figures are written automatically to the `bbq_outputs` directory created next to the script, so each run produces a self-contained set of profiles, scans, summaries, residual files, and plot PDFs in a single location.\
\
If this code is used in the creation of published material, please include a citation to the paper: https://arxiv.org/pdf/2603.13986\
}