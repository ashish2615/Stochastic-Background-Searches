from __future__ import division, print_function

import os
import sys
import shutil
import bilby
import deepdish
import numpy as np
import logging
import deepdish
import pandas as pd
import json
import math
import sklearn
import seaborn as sns
import pickle as pkl

from bilby.core.utils import speed_of_light

import scipy
from scipy import signal, fftpack
from scipy.fftpack import fft, rfft,ifft,irfft, fftfreq, rfftfreq
from scipy.signal import (periodogram, welch, lombscargle, csd, coherence,
                          spectrogram)
from scipy.signal import *

import matplotlib
#matplotlib.use('tkagg')
#matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from Initial_data import InitialData, InterferometerStrain
from ORF_OF import detector_functions
from plot_data import PSDWelch
from astro_stochastic_background import StochasticBackground
from Sensitivity_curves import SensitivityCurves

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

ifos = ['L1','H1','V1','K1','ET_D_TR','CE']
sampling_frequency = 2048.
start_time = 1198800017
end_time   = 1230336017
n_seg = 10000
Tfft=8

## Initialization of Data
data = InitialData()
data_sets = data.initial_data(ifos, sampling_frequency, start_time, end_time, Tfft, n_seg=n_seg)
sampling_frequency = data_sets[1]
start_time = data_sets[2]
end_time = data_sets[3]
duration = data_sets[4]
duration_seg = data_sets[5]

n_fft = data_sets[6]
N_samples = data_sets[7]
frequency = data_sets[8]
waveform_generator = data_sets[9]
IFOs = data_sets[10]
n_samples = data_sets[18]

function =  detector_functions()
function.initial(ifos,sampling_frequency, start_time, end_time, n_seg, Tfft)
plots_PSD = PSDWelch(IFOs)
Omega_gw = StochasticBackground()
Omega_gw.initial(ifos, sampling_frequency, start_time, end_time, n_seg, Tfft)
sensitivity = SensitivityCurves()

###############################
# Stochastic Backgrounds Plot #
###############################
cosmo_spectrum = Omega_gw.cosmo_spectrum()
cobe_spectrum = Omega_gw.cobe_spectrum()
PGWB = Omega_gw.PGWB()

#######################
# PSD of GW Detectors #
# Given ifos #
#######################
psd_detector = function.psd()

######################
# Sensitivity Curves #
######################
sensitivity.initial(ifos,sampling_frequency, start_time, end_time, n_seg)
sensitivity.sensitivity_curve_from_interpolation(psd=psd_detector,frequency=frequency)
sensitivity.sensitivity_curve_from_files()
sensitivity.sensitivity_curve_bilby()
sensitivity.sensitivity_space_based()
sensitivity.sensitivity_curve_all(PGWB = PGWB)

#################################
# Strain to Omega for a detector #
#################################

## Sensitivity Curves from PSD files
aLIGO = np.loadtxt('./Sensitivity_Curve/aLIGO_ZERO_DET_high_P_psd.txt')
AdV = np.loadtxt('./Sensitivity_Curve/AdV_psd.txt')
KAGRA = np.loadtxt('./Sensitivity_Curve/Kagra_Design.txt')
aplus_LIGO = np.loadtxt('./Sensitivity_Curve/LIGO_Aplusdesign.txt')
aplus_V = np.loadtxt('./Sensitivity_Curve/AVirgo.txt')
ET_D_TR = np.loadtxt('./Sensitivity_Curve/ET_D_psd.txt')
ET_Chiara = np.loadtxt('./Sensitivity_Curve/ET_ChiaraPaper_sensitivity.txt')
CE = np.loadtxt('./Sensitivity_Curve/CEsensitivity.txt')
# CE1 = np.loadtxt('./Sensitivity_Curve/CE1_psd.txt')
# CE2 = np.fromfile('./Sensitivity_Curve/CE2_psd.txt')
BBO = np.loadtxt('./Sensitivity_Curve/BBOtech.dat.txt')
LISA = sensitivity.LISA()
DECIGO = sensitivity.DECIGO()
BBO1 = sensitivity.BBO()
TianQin = sensitivity.TianQin()

psd_curve = plots_PSD.plot_detector_psd_from_file(aLIGO_psd=aLIGO, AdV_psd=AdV, KAGRA_psd=KAGRA,
                                            aplus_LIGO_psd=aplus_LIGO, aplus_V_psd = aplus_V, ET_D_TR_psd=ET_D_TR, CE_psd=CE)#, CE1_psd =CE1_psd, CE2_psd=CE2_psd)

## Observation time period for the detectors
## For one year of observation time period
Obstime = 3.154 * 10**7

##reference frequency for each detector
aLIGO = sensitivity.strain2omega(aLIGO[:, 0], np.sqrt(aLIGO[:, 1]), Obstime, 25)
AdV = sensitivity.strain2omega(AdV[:, 0], np.sqrt(AdV[:, 1]), Obstime, 25)
KAGRA = sensitivity.strain2omega(KAGRA[:, 0], KAGRA[:, 1], Obstime, 25)
aplus_LIGO = sensitivity.strain2omega(aplus_LIGO[:, 0], aplus_LIGO[:,1], Obstime, 15)
aplus_V = sensitivity.strain2omega(aplus_V[:, 0], aplus_V[:,1], Obstime, 15)
ET_D_TR = sensitivity.strain2omega(ET_D_TR[:, 0], np.sqrt(ET_D_TR[:,1]), Obstime, 5)
ET_Chiara1 = sensitivity.strain2omega(ET_Chiara[:, 0], ET_Chiara[:,3], Obstime, 5)
CE = sensitivity.strain2omega(CE[:, 0], CE[:,1], Obstime, 5)
LISA = sensitivity.strain2omega(LISA[0], np.sqrt(LISA[1]), Obstime, 10**-3)
DECIGO = sensitivity.strain2omega(DECIGO[0], np.sqrt(DECIGO[1]), Obstime, 1)
BBO = sensitivity.strain2omega(BBO[:,0], BBO[:,1], Obstime, 0.2)
BBO1 = sensitivity.strain2omega(BBO1[0], np.sqrt(BBO1[1]), Obstime,0.5)
TianQin = sensitivity.strain2omega(TianQin[0], np.sqrt(TianQin[1]), Obstime, 10**-2)

plots_PSD.plot_strain2omega(aLIGO=aLIGO, AdV=AdV, KAGRA=KAGRA, ET_D_TR=ET_D_TR, CE=CE,
                          BBO=BBO, LISA=LISA, DECIGO=DECIGO, BBO1=BBO1, TianQin=TianQin, cosmo_spectrum=cosmo_spectrum)


#########################
# Sensitivity and Omega #
#########################

sen_omega = sensitivity.sensitivity_and_omega(PGWB, cosmo_spectrum)