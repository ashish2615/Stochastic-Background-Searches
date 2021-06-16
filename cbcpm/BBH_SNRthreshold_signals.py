from __future__ import division, print_function

import os
import sys
import bilby
import corner
import logging
from astropy.cosmology import FlatLambdaCDM

import numpy as np
from numpy import loadtxt
import pandas as pd
import h5py
import json
import math
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "Times New Roman"

import platform
import warnings
import Injection
from Injection import InjectionSignal

injection = InjectionSignal()

outmain = 'Output'
if os.path.exists(outmain):
    print("{} directory already exist".format(outmain))
else :
    print("{} directory does not exist".format(outmain))
    try:
        os.mkdir(outmain)
    except OSError:
        print("Creation of the directory {} failed".format(outmain))
    else:
        print("Successfully created the directory {}".format(outmain))

outdir = 'BBH_SNRthreshold_signal_bestfit_individualET'
outdir1 = os.path.join(outmain,outdir)
print(outdir1)
if os.path.exists(outdir1):
    print("{} directory already exist".format(outdir1))
else :
    print("{} directory does not exist".format(outdir1))
    try:
        os.mkdir(outdir1)
    except OSError:
        print("Creation of the directory {} failed".format(outdir1))
    else:
        print("Successfully created the directory {}".format(outdir1))

# def intersection(lst1, lst2):
#     lst3 = [value for value in lst1 if value in lst2]
#     return lst3

# list = open('./Uniform_BBH_DATA/IMRPhenomPv2_SNRL10_Rerun/Uniform_BBH_SNRless10.txt')
# list = list.readline().split()
#
# list1 = open('./Uniform_BBH_DATA/IMRPhenomXP_SNRL10_Rerun/Uniform_BBH_SNRless10.txt')
# list1 = list1.readline().split()
#
# list2 = intersection(list,list1)
# print(list2)
# print(len(list2))
#
# with open("Waveform_Systematics_BBH_SNRless10.txt", "w") as f:
#     for item in list2:
#         f.write('{} '.format(item))


ifos = ['CE', 'ET_D_TR']  ## sys.argv[2].split(',')
sampler = 'dynesty'  ## sys.argv[3] 'dynesty' , 'pymultinest'

outdir = 'outdir'
extension = 'json'

n_cut = 100

## Set the duration and sampling frequency of the data segment that we're going to inject the signal into
duration = 4.  # time duration in sec.
sampling_frequency = 2048.  # Sampling Frequency in Hz

injection_params = injection.injections(filename='./Injection_file/injections_10e6.hdf5')
# signal_list = open('./Uniform_BBH_DATA/BBH_SNRless10/Uniform_BBH_SNRless10_Signals_200.txt')
signal_list = open('./Uniform_BBH_DATA/BBH_SNRless10/IMRPhenomPv2_SNRL10_Rerun/Uniform_BBH_SNRless10.txt')
signal_list = signal_list.readline().split()
injection_params = injection_params.iloc[signal_list]
# injection_params = injection_params.iloc[0:n_cut]
injection_params.index = range(len(injection_params.index))

SNR = np.zeros((len(signal_list), 4))
# SNR = np.zeros(((n_cut), 4))
for index, injection_parameters in injection_params.iterrows():
    # if index == 70:
    print('Index is',index)

    injection_parameters = dict(injection_parameters)

    ## Changing 'iota' to 'theta_jn' to be suitable with bilby
    injection_parameters['theta_jn'] = injection_parameters.pop('iota')

    ## we use start_time=start_time to match the start time and wave interferome time. if we do not this then it will create mismatch between time.
    start_time = injection_parameters['geocent_time']

    ## Redshift to luminosity Distance conversion using bilby
    injection_parameters['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(
        injection_parameters['redshift'])

    # First mass needs to be larger than second mass
    if injection_parameters['mass_1'] < injection_parameters['mass_2']:
        tmp = injection_parameters['mass_1']
        injection_parameters['mass_1'] = injection_parameters['mass_2']
        injection_parameters['mass_2'] = tmp

    ## SEOBNRv4_ROM is a BBH waveform with aligned spin
    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=50., minimum_frequency=2.)

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency, start_time=start_time,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments)

    ## Initialization of GW interferometer
    IFOs = bilby.gw.detector.InterferometerList(ifos)

    # Generates an Interferometer with a power spectral density based on advanced LIGO.
    IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration,
                                                       start_time=start_time)

    IFOs.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

    mf_snr = np.zeros((1, len(IFOs)))[0]
    optimal_snr = np.zeros((1, len(IFOs)))[0]
    waveform_polarizations = waveform_generator.frequency_domain_strain(injection_parameters)
    k = 0
    for ifo in IFOs:
        signal_ifo = ifo.get_detector_response(waveform_polarizations, injection_parameters)
        ## Calculate the _complex_ matched filter snr of a signal.
        ##This is <signal|frequency_domain_strain> / optimal_snr
        mf_snr[k] = ifo.matched_filter_snr(signal=signal_ifo)

        if np.isnan(mf_snr[k]):
            mf_snr[k] = 0.

        # print('{}: SNR = {:02.2f} at z = {:02.2f}'.format(ifo.name, mf_snr[k], injection_parameters['redshift']))

        optimal_snr[k] = ifo.optimal_snr_squared(signal=signal_ifo)
        if np.isnan(optimal_snr[k]):
            optimal_snr[k] = 0.
        print('{}: SNR = {:02.2f} at z = {:02.2f}'.format(ifo.name, np.sqrt(optimal_snr[k]), injection_parameters['redshift']))

        SNR[index, k] = optimal_snr[k]

        k += 1

# --------------------------------------------------------#
""" Plotting the SNR vs Redshift for BBH Injection Signals
 having SNR less than 10."""
# --------------------------------------------------------#


n_signals = len(signal_list)
# n_signals = n_cut

dataframe = pd.DataFrame(SNR, columns=['CE', 'ET1', 'ET2', 'ET3'])
ET1 = dataframe['ET1']
ET2 = dataframe['ET2']
ET3 = dataframe['ET3']
dataframe['ET Total'] = np.sqrt(ET1 + ET2 + ET3)
print(dataframe['ET Total'])

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='small')

## Plotting 2D histogram for CE and ET SNR with redshift.
plt.hist2d(injection_params['redshift'], dataframe['CE'], bins=25)
plt.hist2d(injection_params['redshift'], dataframe['ET Total'], bins=25)
plt.grid()
plt.xlabel(r'$z$')
plt.ylabel(r'SNR')
# plt.xlim(0, 10)
plt.tight_layout()
plt.savefig(outdir1 + '/Uniform_BBH_SNRless10_SNRvsRedshift_plot',dpi=300)
plt.show()

sns.set()
sns.set_style("whitegrid")

bin_edges = list(range(0,21))
print(bin_edges)

## ET total SNR histogram plot
sns.displot(dataframe['ET Total'], bins=bin_edges) #stat="probability",
plt.xlabel(r'SNR')
plt.ylabel(r'Count')
plt.xlim(0, 20)
# plt.legend(loc='best', prop=font1)
plt.tight_layout()
plt.savefig(outdir1 + '/ET_total_SNRn3_histogram_'+str(n_signals)+'_BBH_Signals',dpi=300)
plt.show()

## Normalized Histogram
sns.displot(dataframe['ET Total'], stat="probability", bins=bin_edges)
plt.xlabel(r'SNR')
plt.ylabel(r'Probability')
plt.xlim(0, 20)
# plt.legend(loc='best', prop=font1)
plt.tight_layout()
plt.savefig(outdir1 + '/ET_total_SNRn3_histogramProb_'+str(n_signals)+'_BBH_Signals',dpi=300)
plt.show()

# SNR histogram plot for Individual ET detector.
for key in dataframe.keys():
    sns.histplot(dataframe[key],bins=bin_edges, label=key) #stat="probability",
    plt.legend(loc='best', prop=font1)
    plt.xlabel(r'SNR')
    plt.ylabel(r'Count')
    plt.xlim(0, 20)
    plt.tight_layout()
    plt.savefig(outdir1 + '/'+key+'_individual_SNRn3_histogram_'+str(n_signals)+'_BBH_Signals',dpi=300)
    plt.show()

# Combined SNR histogram plot for Individual ET detector.
sns.distplot(dataframe['ET1'], hist=True, kde=False, bins=bin_edges, label='ET1') #stat="probability",
sns.distplot(dataframe['ET2'], hist=True, kde=False, bins=bin_edges, label='ET2')
sns.distplot(dataframe['ET3'], hist=True, kde=False, bins=bin_edges, label='ET2')
plt.legend(loc='best', prop=font1)
plt.xlabel(r'SNR')
plt.ylabel(r'Count')
plt.xlim(0, 25)
plt.tight_layout()
plt.savefig(outdir1 + '/ET_individual_SNRn3_histogram_'+str(n_signals)+'_BBH_Signals',dpi=300)
plt.show()
