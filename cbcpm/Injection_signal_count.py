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
import scipy as sp
import h5py
import json
import math
import seaborn as sns

import lalsimulation
from lalsimulation import SimIMRPhenomDChirpTime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "Times New Roman"
import platform

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

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

outdir = 'Injection_signal_count_ET_BBH_IMPRPv2_10e4'
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

#-----------------#
""" Main Cource """
#-----------------#

n_signals = 100000

# save_idx = open(outdir1 + '/BBH_Signal_SNRless10.txt', 'w')
# CE_SNR = open(outdir1+ '/CE_SNR_'+str(n_signals)+'.txt', 'w')
# ET1_SNR = open(outdir1+ '/ET1_SNR_'+str(n_signals)+'.txt', 'w')
# ET2_SNR = open(outdir1+ '/ET2_SNR_'+str(n_signals)+'.txt', 'w')
# ET3_SNR = open(outdir1+ '/ET3_SNR_'+str(n_signals)+'.txt', 'w')
# redshift = open(outdir1+ '/Redshift_SNR_'+str(n_signals)+'.txt','w')
# lum_dist = open(outdir1+ '/Luminosity_dist_'+str(n_signals)+'.txt','w')

injectionPE = False

SNR = []
z = []
sig_duration  =  np.zeros(n_signals)

for idx in np.arange(n_signals):

    print('idx is', idx)

    ifos = ['ET_D_TR']
    sampler = 'dynesty'
    ## Set the duration and sampling frequency of the data segment that we're going to inject the signal into
    duration = 4.               ## in sec.
    minimum_frequency = 2.      ## in Hz
    reference_frequency = 20.   ## in Hz
    sampling_frequency = 2048.  ## in Hz

    ## Loading the injection parameters from hdf5 file without any change
    injection_parameters = pd.read_hdf('./Injection_file/injections_10e6.hdf5')
    injection_parameters = dict((injection_parameters).loc[idx])

    ## Changing 'iota' to 'theta_jn' to be suitable with bilby
    injection_parameters['theta_jn'] = injection_parameters.pop('iota')

    # injection_parameters['mass_1'] = 1.4
    # injection_parameters['mass_2'] = 1.4

    ## Injection Parameter Conversion
    # injection_parameters['chirp_mass'] =bilby.gw.conversion.component_masses_to_chirp_mass(injection_parameters['mass_1'],injection_parameters['mass_2'])
    injection_parameters['total_mass'] = bilby.gw.conversion.component_masses_to_total_mass(injection_parameters['mass_1'],injection_parameters['mass_2'])
    # injection_parameters['symmetric_mass_ratio'] = bilby.gw.conversion.component_masses_to_symmetric_mass_ratio(injection_parameters['mass_1'],injection_parameters['mass_2'])
    # injection_parameters['mass_ratio'] = bilby.gw.conversion.component_masses_to_mass_ratio(injection_parameters['mass_1'],injection_parameters['mass_2'])

    ## Define maximum frequency (f_ISCO)
    # c = 2.998 * 10 ** 8      ## in m/s
    # G = 6.674 * 10 ** (-11)  ## in m**3 kg**(-1) s**(-2)
    # fmax = c ** 3 / (6 ** (3/2) * np.pi * G * round(injection_parameters['total_mass'],2) * 1.99 * 10**(31)* (1+injection_parameters['redshift']))
    # fmax = round(fmax,2)

    ## we use start_time=start_time to match the start time and wave interferome time. if we do not this then it will create mismatch between time.
    start_time = injection_parameters['geocent_time']

    ## Redshift to luminosity Distance conversion using bilby
    injection_parameters['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(injection_parameters['redshift'])

    # First mass needs to be larger than second mass
    if injection_parameters['mass_1'] < injection_parameters['mass_2']:
        tmp = injection_parameters['mass_1']
        injection_parameters['mass_1'] = injection_parameters['mass_2']
        injection_parameters['mass_2'] = tmp

    ## Calcualting the duration of BBH signal in the detector
    ## mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, flow=10
    # signal_duration = bilby.gw.detector.get_safe_signal_duration(injection_parameters['mass_1'],
    #                   injection_parameters['mass_2'], injection_parameters['a_1'],
    #                   injection_parameters['a_2'], injection_parameters['tilt_1'],
    #                   injection_parameters['tilt_2'], flow=2.)
    # sig_duration[idx] = signal_duration

    ## Signal Duration from lalsimulation for aligned spin BBHs
    # fHz = 2.
    # chirp_signal = SimIMRPhenomDChirpTime(injection_parameters['mass_1'],
    #                   injection_parameters['mass_2'], injection_parameters['a_1'],
    #                   injection_parameters['a_2'], fHz)

    ## Changing parameters to lalSimulation parameters
    # iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
    #     bilby.gw.conversion.bilby_to_lalsimulation_spins(injection_parameters['theta_jn'], injection_parameters['phi_jl'],
    #     injection_parameters['tilt_1'], injection_parameters['tilt_2'],
    #     injection_parameters['phi_12'], injection_parameters['a_1'],injection_parameters['a_2'], injection_parameters['mass_1'],
    #     injection_parameters['mass_2'], reference_frequency=reference_frequency, phase=injection_parameters['phase'])
    #

    # injection_parameters['theta_jn'] = iota
    # injection_parameters['spin_1x'] = 0. #spin_1x
    # injection_parameters['spin_1y'] = 0. #spin_1y
    # injection_parameters['spin_1z'] = 0. #spin_1z
    # injection_parameters['spin_2x'] = 0. #spin_2x
    # injection_parameters['spin_2y'] = 0. #spin_2y
    # injection_parameters['spin_2z'] = 0. #spin_2z

    injection_parameters['a_1'] = 0.
    injection_parameters['a_2'] = 0.
    injection_parameters['tilt_1'] = 0.
    injection_parameters['tilt_2'] = 0.
    injection_parameters['phi_jl'] = 0.
    injection_parameters['phi_12'] = 0.

    ## Fixed arguments passed into the source model : A dictionary of fixed keyword arguments
    ## to pass to either `frequency_domain_source_model` or `time_domain_source_model`.
    ## SEOBNRv4_ROM , SEOBNRv4PHM ,  TaylorF2 , SEOBNRv3_opt , IMRPhenomPv2 , IMRPhenomXHM
    ## IMRPhenomPv2_NRTidalv2 , IMRPhenomD_NRTidalv2

    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=reference_frequency,
                              minimum_frequency=minimum_frequency) #, maximum_frequency = fmax)
    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency=sampling_frequency, start_time=start_time,
                        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                        waveform_arguments=waveform_arguments)

    print("start_time = {}".format(getattr(waveform_generator, 'start_time')))

    waveform_polarizations = waveform_generator.frequency_domain_strain(injection_parameters)
    ## create the frequency domain signal
    hf_signal = waveform_generator.frequency_domain_strain(parameters=injection_parameters)

    # plt.loglog(waveform_generator.frequency_array, abs(hf_signal['plus']),label='max freq = ' +str(fmax))
    # # plt.loglog(abs(hf_signal['plus']))
    # plt.ylabel(r'$ h_{+}$[f]')
    # plt.xlabel('Frequency (Hz)')
    # plt.legend()
    # plt.savefig(outdir1 + '/abs_hf_signal' + '.png')
    # plt.close()

    # ## Initialization of GW interferometer
    IFOs = bilby.gw.detector.InterferometerList(ifos)
    # Generates an Interferometer with a power spectral density
    IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration,
                                                       start_time=start_time)
    IFOs.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

    # optimal_snr = np.zeros((1, len(IFOs)))[0]
    # cnt = 0
    # for ifo in IFOs:
    #     signal_ifo = ifo.get_detector_response(waveform_polarizations, injection_parameters)
    #     optimal_snr[cnt] = ifo.optimal_snr_squared(signal=signal_ifo)
    #     if np.isnan(optimal_snr[cnt]):
    #         optimal_snr[cnt] = 0.
    #     print('{}: SNR = {:02.2f} at z = {:02.2f}'.format(ifo.name, np.sqrt(optimal_snr[cnt]), injection_parameters['redshift']))

    #     plt.loglog(waveform_generator.frequency_array, abs(signal_ifo),label= 'max freq = '+str(fmax))
    #     plt.ylabel('Strain')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.legend()
    #     plt.savefig(outdir1 + '/absGW_Signal_' + str(cnt) + '_.png')
    #     plt.close()
    #
    #     cnt += 1

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
        print('{}: SNR = {:02.2f} at z = {:02.2f}'.format(ifo.name, np.sqrt(optimal_snr[k]),
                                                          injection_parameters['redshift']))

        k += 1

    total_SNR_ET = np.sqrt(optimal_snr[0] + optimal_snr[1] + optimal_snr[2])
    # total_SNR_ET = optimal_snr

    if total_SNR_ET > 8:
        injectionPE = True
        print(' Hoorahhh! Total SNR = {} is greater than 8 for idx = {}'.format(total_SNR_ET,idx))
        SNR.append(total_SNR_ET)
        z.append(injection_parameters['redshift'])

    else:
        print(' AAhhh! SNR is less than 8 for idx', idx)

    # # Strain of Detector
    # strain = waveform_generator.frequency_domain_strain(parameters=injection_parameters)
    # strain_time = waveform_generator.frequency_domain_strain(parameters=injection_parameters)
    # plt.loglog(waveform_generator.frequency_array, abs(strain['plus']), label='Bilby')
    # plt.ylabel('|hf|')
    # plt.xlabel('Frequency (Hz)')
    # plt.legend()
    # plt.savefig(outdir1 + '/abs_fd_waveform'+str(duration)+'.png')
    # plt.close()

SNR =  np.array(SNR)
z = np.array(z)
# --------------------------------------------------------#
""" Plotting the SNR vs Redshift for Injection Signals """
#--------------------------------------------------------#
## Use this part after running the main course.

# dataframe = pd.DataFrame(SNR, columns=['ET1', 'ET2','ET3'])
# ET1 = dataframe['ET1']
# ET2 = dataframe['ET2']
# ET3 = dataframe['ET3']
# dataframe['ET Total'] = np.sqrt(ET1 + ET2 + ET3)
# print(dataframe['ET Total'])
# dataframe['redshift'] = z
# # dataframe['Signal Duration'] = sig_duration

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='small')

# ## Plotting a 2D histogram between CE and ET SNR with redshift.
# plt.hist2d(dataframe['redshift'], dataframe['CE'], bins=50)
# plt.hist2d(dataframe['redshift'], dataframe['ET Total'],bins=50)
# plt.grid()
# plt.xlabel(r'$z$')
# plt.ylabel(r'SNR')
# plt.ylim(0,50)
# plt.tight_layout()
# plt.savefig(outdir1 + '/SNR_vs_Redshift_n3_'+str(n_signals)+'_BBH_Signals',dpi=300)
# plt.show()

sns.set()
sns.set_style("whitegrid")

bin_edges = np.arange(0,50,1)
print(bin_edges)
bin_edges1 = np.arange(0,10,0.2)
print(bin_edges1)

## ET total SNR histogram plot
plt.hist(SNR, bins=bin_edges, edgecolor='black') #stat="probability",
plt.xlabel(r'SNR')
plt.ylabel(r'Counts')
plt.xlim(0, 50)
plt.legend(loc='best', prop=font1)
plt.tight_layout()
plt.savefig(outdir1 + '/ET_total_SNRn3_histogram_'+str(n_signals)+'_BBH_Signals',dpi=300)
# plt.show()
plt.close()

# ## Normalized Histogram
# sns.histplot(dataframe['ET Total'], stat="probability", bins=bin_edges, color='b')
# plt.xlabel(r'SNR')
# plt.ylabel(r'Probability')
# plt.xlim(0, 10)
# plt.legend(loc='best', prop=font1)
# plt.tight_layout()
# plt.savefig(outdir1 + '/ET_total_SNRn3_histogramProb_'+str(n_signals)+'_BBH_Signals',dpi=300)
# # plt.show()
# plt.close()

## ET total redshift plot
plt.hist(z, bins=bin_edges1, edgecolor='black') #stat="probability",
plt.xlabel(r'RedShift for SNR > 8 BNS')
plt.ylabel(r'Counts for SNR > 8 BNS')
plt.xlim(0, 10)
plt.legend(loc='best', prop=font1)
plt.tight_layout()
plt.savefig(outdir1 + '/ET_total_redshift_histogram_'+str(n_signals)+'_BBH_Signals',dpi=300)
# plt.show()
plt.close()

# ## Signal Duration Plot
# sns.histplot(dataframe['Signal Duration'], bins=bin_edges, edgecolor='black')
# plt.xlabel(r'Signal Duration')
# plt.ylabel(r'Counts')
# # plt.xlim(0, 50)
# # plt.legend(loc='best', prop=font1)
# plt.tight_layout()
# plt.savefig(outdir1 + '/ET_total_Signal_Duration_'+str(n_signals)+'_BBH_Signals',dpi=300)
# plt.show()


# # SNR histogram plot for Individual ET detector.
# for key in dataframe.keys():
#     sns.histplot(dataframe[key], bins=bin_edges, label=key) #stat="probability",
#     plt.legend(loc='best', prop=font1)
#     plt.xlabel(r'SNR')
#     plt.ylabel(r'Counts')
#     plt.xlim(0, 50)
#     plt.tight_layout()
#     plt.savefig(outdir1 + '/'+key+'_individual_SNRn3_histogram_'+str(n_signals)+'_BBH_Signals',dpi=300)
#     plt.show()
#
# # Combined SNR histogram plot for Individual ET detector.
# sns.distplot(dataframe['ET1'], kde=False, bins=bin_edges, label='ET1') #stat="probability",
# sns.distplot(dataframe['ET2'], kde=False, bins=bin_edges, label='ET2')
# sns.distplot(dataframe['ET3'], kde=False, bins=bin_edges, label='ET2')
# plt.legend(loc='best', prop=font1)
# plt.xlabel(r'SNR')
# plt.ylabel(r'Counts')
# plt.xlim(0, 50)
# plt.tight_layout()
# plt.savefig(outdir1 + '/ET_individual_SNRn3_histogram_'+str(n_signals)+'_BBH_Signals',dpi=300)
# plt.show()

