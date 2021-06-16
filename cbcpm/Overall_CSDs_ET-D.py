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
import csv

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.font_manager as font_manager

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
Tfft = 2

## Setting Up the data directory to store the plots.
# outmain = 'Output'
outdir = './Uniform_BBH_DATA/BBH_SNRless10g2/Sum_CSD_TFFT'+str(Tfft)+'_Plot'
# outdir = os.path.join(outmain, outdir)
# bilby.utils.setup_logger(outdir=outdir, label=label)

if os.path.exists(outdir):
    print("{} directory already exist".format(outdir))
else :
    print("{} directory does not exist".format(outdir))
    try:
        os.mkdir(outdir)
    except OSError:
        print("Creation of the directory {} failed".format(outdir))
    else:
        print("Successfully created the directory {}".format(outdir))


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

# variance = variance[1,2,:]
# zip(frequency, variance)
# with open('Sigma_Tfft8_ET-D.csv', 'w') as file:
#     write = csv.writer(file, delimiter='\t')
#     write.writerows(zip(frequency, variance))
#
# c = [frequency, variance]
# with open('Sigma_TFFT8_ET-D.txt', 'w') as file:
#     for x in zip(*c):
#         file.write('{0}\t{1}\n'.format(*x))

# ## Plot for ET detector with CSD Astro and CSD Cosmo
# font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
# font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')
#
# for d1 in range(len(IFOs)):
#     for d2 in range(d1 + 1, len(IFOs)):
#         labelstring = IFOs[d1].name + ' & ' + IFOs[d2].name
#
#         plt.loglog(frequency, np.abs(CSD_from_Omega_astro[d1, d2, :]), label='CSD Astro')
#         plt.loglog(frequency, np.abs(CSD_from_Omega_cosmo[d1, d2, :]), label='CSD Cosmo')
#         plt.loglog(frequency, variance[d1, d2, :], label= labelstring + ' for Tfft = '+ str(Tfft))#'ET-D $\sigma$')
#         plt.loglog(frequency, variance1[d1, d2, :], label=labelstring + ' for V1 Tfft =' + str(Tfft))  # 'ET-D $\sigma$')
#
#         legend = plt.legend(loc='best', prop=font1)
#         plt.xlim(1, 1000)
#         # plt.ylim(1*10**-65, 1*10**-40)
#         plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
#         plt.ylabel(r'CSD $~ & ~\sigma$ [1/Hz]', fontdict=font)
#         plt.tight_layout()
#         # plt.title(r'CSD_and_Variance for_' +str(ifo)+'_'+ method, fontdict=font)
#         plt.savefig('./ET_Plots_1/ET-D_Astro_Cosmo_Sigma_'+str(Tfft) +'_' + labelstring, dpi=300)
#         # plt.savefig('./ET_Plots_1/Variance_Compare_' + str(Tfft)+ '_&_'+str(Tfft1)+'_' + labelstring, dpi=300)
#         plt.close()

################################
# Sum Over n_seg time segments #
################################
## FOR ET sum series with detector noise
## For BBH SNR > 10 use case2 directory in Uniform_BBH_DATA/BBH_SNRg10.
# sum_CSD_wn = np.load('./Uniform_BBH_DATA/BBH_SNRless10g2/Overall_CSD_10000_TFFT'+str(Tfft)+'/sum_inj_CSD.npy')
# sum_CSD_wn = np.load('./Uniform_BBH_DATA/BBH_SNRless10g2/Overall_CSD_10000_TFFT'+str(Tfft)+'/sum_sub_CSD.npy')
sum_CSD_wn = np.load('./Uniform_BBH_DATA/BBH_SNRless10g2/Overall_CSD_10000_TFFT'+str(Tfft)+'/sum_proj_CSD.npy')
sum_CSD_wn_ET = np.zeros((1, len(frequency)), dtype=np.complex)
for d1 in np.arange(len(IFOs)):
    if d1 >= 1:
        cnt =0
        for d2 in np.arange(d1+1, len(IFOs)):
            sum_CSD_wn_ET[cnt,:] += sum_CSD_wn[d1, d2, :]/3
        cnt += 1

## sum series without detector noise.
## FOR BBH SNR>10 use test_sum directory in Uniform_BBH_DATA/BBH_SNRg10
# sum_CSD_zero = np.load('./Uniform_BBH_DATA/BBH_SNRless10g2/Overall_CSD_1000_TFFT'+str(Tfft)+'/sum_inj_CSD.npy')
# sum_CSD_zero = np.load('./Uniform_BBH_DATA/BBH_SNRless10g2/Overall_CSD_1000_TFFT'+str(Tfft)+'/sum_sub_CSD.npy')
sum_CSD_zero = np.load('./Uniform_BBH_DATA/BBH_SNRless10g2/Overall_CSD_1000_TFFT'+str(Tfft)+'/sum_proj_CSD.npy')
sum_CSD_zero_ET = np.zeros((1, len(frequency)), dtype=np.complex)
for d1 in np.arange(len(IFOs)):
    if d1 >= 1:
        cnt =0
        for d2 in np.arange(d1+1, len(IFOs)):
            sum_CSD_zero_ET[cnt,:] += sum_CSD_zero[d1, d2, :]/3
        cnt += 1

variance_ET = variance[1, 2, :] * np.sqrt(1/3)

font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

## Change method in accordance to Sum_CSD file.
# method = 'Injection'
# method = 'Subtraction'
method ='Projection'

plt.loglog(frequency, np.abs(sum_CSD_wn_ET[0, :]), label='CSD ' + str(method) +' with Noise')
plt.loglog(frequency, np.abs(sum_CSD_zero_ET[0, :]), label='CSD ' + str(method) + ' in Zero Noise')
plt.loglog(frequency, variance_ET, label='$\sigma$')
plt.loglog(frequency, np.abs(CSD_from_Omega_astro[1, 2, :]), label='CSD Astro')
plt.loglog(frequency, np.abs(CSD_from_Omega_cosmo[1, 2, :]), label='CSD Cosmo')

legend = plt.legend(loc='lower left', prop=font1)
plt.xlim(1, 1000)
# plt.ylim(1*10**-60, 1*10**-35)
plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
plt.ylabel(r'CSD $~ & ~\sigma ~~$ [1/Hz]', fontdict=font)
plt.tight_layout()
# plt.title(r'CSD_and_Variance for_' +str(ifo)+'_'+ method, fontdict=font)
plt.savefig(outdir + '/Sum_CSD_' + str(method) + '_ET_pair', dpi=300)
plt.close()

###########################################
""" Sum Over n_seg time segments for CE """
###########################################

## FOR CE sum series with detector noise
# sum_CSD_wn = np.load('./Uniform_BBH_DATA/BBH_SNRless10g2/Overall_CSD_10000_TFFT'+str(Tfft)+'/sum_inj_CSD.npy')
# sum_CSD_wn = np.load('./Uniform_BBH_DATA/BBH_SNRless10g2/Overall_CSD_10000_TFFT'+str(Tfft)+'/sum_sub_CSD.npy')
sum_CSD_wn = np.load('./Uniform_BBH_DATA/BBH_SNRless10g2/Overall_CSD_10000_TFFT'+str(Tfft)+'/sum_proj_CSD.npy')

sum_CSD_wn_CE = np.zeros((1, len(frequency)), dtype=np.complex)
for d1 in np.arange(len(IFOs)):
    if d1 == 0:
        cnt = 0
        for d2 in np.arange(d1+1, len(IFOs)):
            sum_CSD_wn_CE[cnt,:] += sum_CSD_wn[d1, d2, :]/3
        cnt += 1

## CE sum series without detector noise.
# sum_CSD_zero = np.load('./Uniform_BBH_DATA/BBH_SNRless10g2/Overall_CSD_1000_TFFT'+str(Tfft)+'/sum_inj_CSD.npy')
# sum_CSD_zero = np.load('./Uniform_BBH_DATA/BBH_SNRless10g2/Overall_CSD_1000_TFFT'+str(Tfft)+'/sum_sub_CSD.npy')
sum_CSD_zero = np.load('./Uniform_BBH_DATA/BBH_SNRless10g2/Overall_CSD_1000_TFFT'+str(Tfft)+'/sum_proj_CSD.npy')

sum_CSD_zero_CE = np.zeros((1, len(frequency)), dtype=np.complex)
for d1 in np.arange(len(IFOs)):
    if d1 == 1:
        cnt = 0
        for d2 in np.arange(d1+1, len(IFOs)):
            sum_CSD_zero_CE[cnt,:] += sum_CSD_zero[d1, d2, :]/3
        cnt += 1

variance_CE = variance[1, 2, :] * np.sqrt(1/3)


font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

## Change method in accordance to Sum_CSD file.
# method = 'Injection'
# method = 'Subtraction'
method ='Projection'

plt.loglog(frequency, np.abs(sum_CSD_wn[0, 1, : ]), label='CSD ' + str(method) +' with Noise')
plt.loglog(frequency, np.abs(sum_CSD_zero[0, 1 ,:]), label='CSD ' + str(method) + ' in Zero Noise')
plt.loglog(frequency, variance[0, 1, :], label='$\sigma$')
plt.loglog(frequency, np.abs(CSD_from_Omega_astro[0, 1, :]), label='CSD Astro')
plt.loglog(frequency, np.abs(CSD_from_Omega_cosmo[0, 1, :]), label='CSD Cosmo')

legend = plt.legend(loc='lower left', prop=font1)
plt.xlim(1, 1000)
# plt.ylim(1*10**-60, 1*10**-35)
plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
plt.ylabel(r'CSD $~ & ~\sigma ~~$ [1/Hz]', fontdict=font)
plt.tight_layout()
# plt.title(r'CSD_and_Variance for_' +str(ifo)+'_'+ method, fontdict=font)
plt.savefig(outdir + '/Sum_CSD_' + str(method) + '_CE', dpi=300)
plt.close()
