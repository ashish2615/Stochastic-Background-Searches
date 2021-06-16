from __future__ import division, print_function

import os
import sys
import logging
import deepdish
import numpy as np
import math
import scipy
import json
import math
import sklearn
import seaborn as sns

from Initial_data import InitialData, InterferometerStrain
from ORF_OF import detector_functions
from Cross_Correlation import CrossCorrelation
from plot_data import PSDWelch
from astro_stochastic_background import StochasticBackground
from CSD_Sum import SumCrossCorr

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

## GW detectors priors
ifos = ['CE', 'ET_D_TR'] #, 'L1', 'H1', 'V1', 'K1']
sampling_frequency = 2048.
start_time = 1198800017
end_time   = 1230336017
n_seg = 10000

## Change Tfft = [8, 2]
Tfft= 2

## Setting Up the data directory to store the Overall CSD and corresponding plots.
outdir = 'Overall_CSD_10000_TFFT'+str(Tfft)+'_Plot'
outdir_zero = 'Overall_CSD_1000_TFFT'+str(Tfft)+'_Plot'
outdir_compare = 'Overall_CSD_TFFT'+str(Tfft)+'_Compare'

path = ['Overall_CSD_10000_TFFT'+str(Tfft), 'Overall_CSD_1000_TFFT'+str(Tfft), outdir, outdir_zero, outdir_compare]

for path in path:
    if os.path.exists(path):
        print("{} directory already exist".format(path))
    else :
        print("{} directory does not exist".format(path))
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory {} failed".format(path))
        else:
            print("Successfully created the directory {}".format(path))

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
plots_PSD = PSDWelch(IFOs)
cross_corr = CrossCorrelation()
Omega_gw = StochasticBackground()
sum=SumCrossCorr()

function.initial(ifos,sampling_frequency, start_time, end_time, n_seg, Tfft)
Omega_gw.initial(ifos, sampling_frequency, start_time, end_time, n_seg, Tfft)

#######################
# PSD of GW Detectors #
# Given ifos #
#######################
psd_detector = function.psd()

##########################
# ORF and Optimal_Filter #
##########################
gamma = function.overlap_reduction_function()
optimal_filter = function.optimal_filter_JH(gamma)

#####################
# CSD from Omega_gw #
#####################
cross_corr.initial(ifos, sampling_frequency, start_time, end_time, n_seg, gamma, optimal_filter, Tfft=Tfft)
## CSD from Omega_GW of Astrophysical Origin
CSD_from_Omega_astro = cross_corr.CSD_from_Omega_astro()
## CSD from Omega_GW of Cosmological Origin
CSD_from_Omega_cosmo = cross_corr.CSD_from_Omega_cosmo()
# plot_CSD_from_omega = plots_PSD.plot_CSD_from_Omega(IFOs, frequency, CSD_from_Omega_astro=CSD_from_Omega_astro, CSD_from_Omega_cosmo=CSD_from_Omega_cosmo)

#####################
# Binary Parameters #
#####################
## Number of Binary Signals
n_inj = 100
## Number of Time segments
n_seg = 10000
n_seg_zero = 1000

###################
# Variance in PSD #
###################
"""Variance defined over the Gaussian power spectral densities for each detector pair and normalized over
the number of fft samples for each time segment and total number of time segments."""
variance = np.zeros((len(IFOs),len(IFOs), len(frequency)))
for d1 in np.arange(len(IFOs)):
    detector_1 = psd_detector[d1, :]
    for d2 in range(d1+1, len(IFOs)):
        detector_2 = psd_detector[d2, :]
        variance[d1,d2, :] = np.sqrt(detector_1 * detector_2) / np.sqrt(2 * duration_seg / Tfft * n_seg)

################################
# Sum Over n_seg time segments #
################################

## Read time series for the signals injected into the Noise of detector.
filefolder = 'Sum_CSDs_10000_TFFT'+str(Tfft)

sum_noise, sum_inj, sum_sub, sum_proj = sum.sum(IFOs, frequency=frequency, filefolder=filefolder, n_seg=n_seg)

## Saving the SUm_CSD files for all time segments with noise
np.save('/Overall_CSD_10000_TFFT'+str(Tfft)+'/sum_noise_CSD'+ '.npy',sum_noise)
np.save('/Overall_CSD_10000_TFFT'+str(Tfft)+'/sum_inj_CSD'+ '.npy',sum_inj)
np.save('/Overall_CSD_10000_TFFT'+str(Tfft)+'/sum_sub_CSD'+ '.npy',sum_sub)
np.save('/Overall_CSD_10000_TFFT'+str(Tfft)+'/sum_proj_CSD'+ '.npy',sum_proj)

plot_noise =  plots_PSD.plot_sum_csd_var(IFOs, frequency, sum_series=sum_noise, variance=variance, CSD_from_Omega_astro=CSD_from_Omega_astro,
                                          CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, method='Noise', outdir=outdir)
plot_inj   =  plots_PSD.plot_sum_csd_var(IFOs, frequency, sum_series=sum_inj, variance=variance, CSD_from_Omega_astro=CSD_from_Omega_astro,
                                          CSD_from_Omega_cosmo=CSD_from_Omega_cosmo,  method='Injection',outdir=outdir)
plot_sub   =  plots_PSD.plot_sum_csd_var(IFOs, frequency, sum_series=sum_sub, variance=variance, CSD_from_Omega_astro=CSD_from_Omega_astro,
                                          CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, method='Subtraction',outdir=outdir)
plot_proj  =  plots_PSD.plot_sum_csd_var(IFOs, frequency, sum_series=sum_proj,variance=variance, CSD_from_Omega_astro=CSD_from_Omega_astro,
                                          CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, method='Projection',outdir=outdir)


## Read time series for the Signal Injected into the detector with zero noise.
filefolder_zero = 'Sum_CSDs_1000_TFFT'+str(Tfft)

sum_noise_zero, sum_inj_zero, sum_sub_zero, sum_proj_zero = sum.sum(IFOs, frequency=frequency, filefolder=filefolder_zero, n_seg=n_seg_zero)

## Saving the SUm_CSD files for all time segments with zero noise
np.save('/Overall_CSD_1000_TFFT'+str(Tfft)+'/sum_noise_CSD'+ '.npy',sum_noise_zero)
np.save('/Overall_CSD_1000_TFFT'+str(Tfft)+'/sum_inj_CSD'+ '.npy',sum_inj_zero)
np.save('/Overall_CSD_1000_TFFT'+str(Tfft)+'/sum_sub_CSD'+ '.npy',sum_sub_zero)
np.save('/Overall_CSD_1000_TFFT'+str(Tfft)+'/sum_proj_CSD'+ '.npy',sum_proj_zero)

plot_noise =  plots_PSD.plot_sum_csd_var(IFOs, frequency, sum_series=sum_noise_zero, variance=variance, CSD_from_Omega_astro=CSD_from_Omega_astro,
                                          CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, method='Noise', outdir=outdir_zero)
plot_inj   =  plots_PSD.plot_sum_csd_var(IFOs, frequency, sum_series=sum_inj_zero, variance=variance, CSD_from_Omega_astro=CSD_from_Omega_astro,
                                          CSD_from_Omega_cosmo=CSD_from_Omega_cosmo,  method='Injection',outdir=outdir_zero)
plot_sub   =  plots_PSD.plot_sum_csd_var(IFOs, frequency, sum_series=sum_sub_zero, variance=variance, CSD_from_Omega_astro=CSD_from_Omega_astro,
                                          CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, method='Subtraction',outdir=outdir_zero)
plot_proj  =  plots_PSD.plot_sum_csd_var(IFOs, frequency, sum_series=sum_proj_zero,variance=variance, CSD_from_Omega_astro=CSD_from_Omega_astro,
                                          CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, method='Projection',outdir=outdir_zero)


###Comapre the Overall CSD Plots.
## Plotting both the CSD's for Injection signals  injected into with and without detector noise.

plot_noise =  plots_PSD.plot_sum_csd_compare(IFOs, frequency, sum_series=sum_noise, sum_series_zero=sum_noise_zero, variance=variance, CSD_from_Omega_astro=CSD_from_Omega_astro,
                                          CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, method='Noise', outdir=outdir_comapre)

plot_inj   =  plots_PSD.plot_sum_csd_compare(IFOs, frequency, sum_series=sum_inj, sum_series_zero=sum_inj_zero, variance=variance, CSD_from_Omega_astro=CSD_from_Omega_astro,
                                          CSD_from_Omega_cosmo=CSD_from_Omega_cosmo,  method='Injection',outdir=outdir_comapre)

plot_sub   =  plots_PSD.plot_sum_csd_compare(IFOs, frequency, sum_series=sum_sub, sum_series_zero=sum_sub_zero, variance=variance, CSD_from_Omega_astro=CSD_from_Omega_astro,
                                          CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, method='Subtraction',outdir=outdir_comapre)

plot_proj  =  plots_PSD.plot_sum_csd_compare(IFOs, frequency, sum_series=sum_proj, sum_series_zero=sum_proj_zero, variance=variance, CSD_from_Omega_astro=CSD_from_Omega_astro,
                                          CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, method='Projection',outdir=outdir_comapre)
