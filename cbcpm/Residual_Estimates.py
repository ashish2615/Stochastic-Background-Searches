from __future__ import division, print_function

import os
import sys
import bilby
import numpy as np
import logging
import pandas as pd
import json
import math
import sklearn
import seaborn as sns
import pickle
import shutil
from bilby.core.utils import speed_of_light

import matplotlib.pyplot as plt
# plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.font_manager as font_manager

from Initial_data import InitialData
from Injection import InjectionSignal
from Subtraction import SubtractionSignal

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

# Specify the output directory and the label name of the simulation.
outmain = 'Output'
outdir = 'Residual_Est'
label = 'injected_signal'
label1 = 'subtracted_signal'
outdir = os.path.join(outmain, outdir)
bilby.utils.setup_logger(outdir=outdir, label=label)

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

## GW detectors priors
ifos = ['CE', 'ET_D_TR']
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

#####################
# Binary Parameters #
#####################
## Number of Binary Signals
n_inj = 100
## Number of Time segments
n_seg = 1

injection = InjectionSignal()
# injection_param = injection.injections(filename ='./Injection_file/injections_10e6.hdf5')
injection_params = injection.injections_set(filename ='Injection_file/injections_10e6.hdf5')
# injection_params = injection.redshift_cutoff(filename ='./injection_file/injections_10e6.hdf5', cutoff_low=3, cutoff_high=6)
subtraction = SubtractionSignal()
## Reading the pickle file, If you have pickle data file use the following commnad to read the subtraction params pickle data file.
bestfit_params = subtraction.subtraction_params(filename='./Injection_file/bestfit_params.pkl')

# Number of Injections which are not working.
nw = [8,85,86,87,94,102,103]
injection_params = injection_params.drop(nw)
bestfit_params = bestfit_params.drop(nw)

injection_params = injection_params.iloc[0:n_inj]
bestfit_params = bestfit_params.iloc[0:n_inj]

##############################
# Best-Fit params SNR Values #
##############################
# bestfit_SNR = subtraction.SNR_bestfit(filefolder='data', n_inj=110)
# bestfit_SNR =  open('bestfit_SNR.txt','r')
# bestfit_SNR = bestfit_SNR.read()
# print('bestfit_SNR', bestfit_SNR)

# parameters to be included in the Fisher matrix (others were kept fixed at posterior sampling)
parameters = ['mass_1', 'mass_2', 'luminosity_distance', 'theta_jn']
num_parmas = 4

####################################
# Injection-Subtraction-Projection #
####################################
for k in np.arange(n_seg):

    print('k',k)
    print('Data segment: ', 100 * (k + 1) / n_seg, '%')

    ## Setting the seg_start_time and seg_end_time
    seg_start_time = start_time + k * duration_seg
    seg_end_time = seg_start_time + duration_seg

    ## Initializing the Interferometer
    IFOs = bilby.gw.detector.networks.InterferometerList(ifos)

    ## One can choose where you want to Inject and Subtract your Binary Signals.
    ## Choice:1 Setting strain data from zero noise
    # IFOs.set_strain_data_from_zero_noise(sampling_frequency,duration=duration_seg,start_time=start_time)
    ## Choice:2 Setting strain data from power spectral density of the detector.
    IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency,duration=duration_seg,
                                                       start_time=seg_start_time)

    ##################################
    # injection Parameter and Signal #
    ##################################

    t_coalescence, inject_time_series, freq_domain_strain, IFOs = injection.injection_signal(IFOs, sampling_frequency, seg_start_time=seg_start_time,
                                seg_end_time=seg_end_time, n_seg = k, N_samples=N_samples, injection_params = injection_params,
                                waveform_generator=waveform_generator, n_samples=n_samples)

    print('injected time series is',inject_time_series)

    ###########################################
    # Matched Filter SNR for Injected signals #
    # Amplitude of Residual Error Estimates   #
    ###########################################
    ## PRD 73042001 (2006) Eq. 49 and 53-55
    inj_sum_SNR = injection.avg_SNR(IFOs, injection_params, t_coalescence, waveform_generator)
    ## SNR average and fractional error.
    inj_avg_SNR = np.zeros(len(IFOs))
    inj_fractional_error = np.zeros((len(IFOs), n_samples))
    freq_strain = np.zeros((len(IFOs), n_samples))
    for d1 in np.arange(len(IFOs)):
        inj_avg_SNR[d1] = np.sqrt(inj_sum_SNR[d1]**2)
        freq_strain[d1,:], freq = np.abs(bilby.core.utils.nfft(inject_time_series[d1,:], sampling_frequency))
        inj_fractional_error[d1,:] = freq_strain[d1,:] * (np.sqrt(num_parmas) / inj_avg_SNR[d1])

    for d1 in np.arange(len(IFOs)):
        labelstring = IFOs[d1].name

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        frequency = np.array(freq)

        plt.loglog(frequency, (freq_strain[d1,:]), label=labelstring +' Injections Spectrum')
        plt.loglog(frequency, inj_fractional_error[d1,:], label=labelstring + ' Fractional Spectrum')
        legend = plt.legend(loc='lower left', prop=font1)
        plt.xlim(2, 1000)
        plt.ylim(10**-30, 10**-20)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r' Strain 1/$~\rm[Hz]$', fontdict=font)
        plt.tight_layout()
        plt.savefig(outdir+'/Injection_Est_' + labelstring, dpi=300)
        plt.close()
