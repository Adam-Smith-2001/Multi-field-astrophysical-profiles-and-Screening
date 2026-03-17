{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid101\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid2}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}}
\paperw11900\paperh16840\margl1440\margr1440\vieww17120\viewh11320\viewkind0
\deftab720
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 \expnd0\expndtw0\kerning0
Axion-Dilaton Planetary Solver (Flux Approximation)\
This script solves the coupled axio-dilaton field equations for a planetary density profile. It is designed to model the screening mechanisms discussed in "Multi-Field Dilaton Screening Beyond the Thin-Shell Mechanism."\
Instead of solving both fields fully, this version uses a reduced description to save computational cost:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls1\ilvl0\cf0 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Dilaton: Solved completely as an Initial Value Problem (IVP) starting from the center of the object.\
\ls1\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Axion: Treated using a conserved-flux approximation. The axion gradient is reconstructed algebraically using the dilaton's profile and a symmetric source window.\
\pard\pardeftab720\sa240\partightenfactor0
\cf0 A damped fixed-point iteration is used to find the self-consistent solution, handling the mutual backreaction between the axion's spatial gradients and the dilaton's dynamics.\
Outputs:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls2\ilvl0\cf0 \kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Generates plots of the dilaton profile, axion profile, and the axion-decay constant product.\
\ls2\ilvl0\kerning1\expnd0\expndtw0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
Saves the full radial profile data to axion_dilaton_full_profile.txt.\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\cf0 If this code is used in the creation of published material, please include a citation to the paper: https://arxiv.org/pdf/2603.13986}