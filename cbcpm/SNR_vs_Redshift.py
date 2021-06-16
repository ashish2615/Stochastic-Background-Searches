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

import matplotlib.style
import matplotlib
inline_rc = dict(matplotlib.rcParams)
import matplotlib.pyplot as plt
import platform
import warnings
import Injection
from Injection import InjectionSignal

injection = InjectionSignal()

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


ifos = ['CE', 'ET_D_TR']   ## sys.argv[2].split(',')
sampler = 'dynesty'       ## sys.argv[3] 'dynesty' , 'pymultinest'

outdir = 'outdir'
extension='json'

## Set the duration and sampling frequency of the data segment that we're going to inject the signal into
duration = 4.                     # time duration in sec.
sampling_frequency = 2048.        # Sampling Frequency in Hz

injection_params = injection.injections(filename ='./Injection_file/injections_10e6.hdf5')
list = open('Uniform_BBH_DATA/BBH_SNRless10/IMRPhenomPv2_SNRL10_Rerun/Uniform_BBH_SNRless10.txt')
list = list.readline().split()
injection_params = injection_params.iloc[list]
injection_params.index = range(len(injection_params.index))

SNR = np.zeros((len(list), 4))
for index, injection_parameters in injection_params.iterrows():

    injection_parameters = dict(injection_parameters)

    ## Changing 'iota' to 'theta_jn' to be suitable with bilby
    injection_parameters['theta_jn'] = injection_parameters.pop('iota')

    ## we use start_time=start_time to match the start time and wave interferome time. if we do not this then it will create mismatch between time.
    start_time = injection_parameters['geocent_time']

    ## Redshift to luminosity Distance conversion using bilby
    injection_parameters['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(injection_parameters['redshift'])

    # First mass needs to be larger than second mass
    if injection_parameters['mass_1'] < injection_parameters['mass_2']:
        tmp = injection_parameters['mass_1']
        injection_parameters['mass_1'] = injection_parameters['mass_2']
        injection_parameters['mass_2'] = tmp

    ## SEOBNRv4_ROM is a BBH waveform with aligned spin
    waveform_arguments = dict(waveform_approximant= 'IMRPhenomPv2', reference_frequency=50., minimum_frequency=2.)
    print(waveform_arguments)

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(
          duration=duration, sampling_frequency=sampling_frequency, start_time=start_time,
          frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
          waveform_arguments=waveform_arguments)
    print(waveform_generator)

    ## Initialization of GW interferometer
    IFOs = bilby.gw.detector.InterferometerList(ifos)

    # Generates an Interferometer with a power spectral density based on advanced LIGO.
    IFOs.set_strain_data_from_power_spectral_densities( sampling_frequency=sampling_frequency, duration=duration,
                            start_time=start_time)

    IFOs.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

    mf_snr = np.zeros((1, len(IFOs)))[0]
    waveform_polarizations = waveform_generator.frequency_domain_strain(injection_parameters)
    k = 0
    for ifo in IFOs:
        signal_ifo = ifo.get_detector_response(waveform_polarizations, injection_parameters)
        ## Calculate the _complex_ matched filter snr of a signal.
        ##This is <signal|frequency_domain_strain> / optimal_snr
        mf_snr[k] = ifo.matched_filter_snr(signal=signal_ifo)

        if np.isnan(mf_snr[k]):
            mf_snr[k] = 0.

        print('{}: SNR = {:02.2f} at z = {:02.2f}'.format(ifo.name, mf_snr[k], injection_parameters['redshift']))

        # snr_file_mf.write('{} '.format(ifo.name))
        # snr_file_mf.write('has matched filter SNR = {}'.format(mf_snr[k]))
        # snr_file_mf.write(' at z = {}'.format(str(injection_parameters['redshift'])) + '\n')

        SNR[index,k] = mf_snr[k]

        k += 1

dataframe = pd.DataFrame(SNR, columns=['CE', 'ET1', 'ET2','ET3'])
dataframe['ET Total'] = dataframe['ET1'] + dataframe['ET2'] + dataframe['ET3']

plt.hist2d(injection_params['redshift'], dataframe['CE'], bins=25)
plt.hist2d(injection_params['redshift'],dataframe['ET Total']/3,bins=25)
plt.grid()
plt.xlabel(r'$z$')
plt.ylabel(r'SNR')
plt.ylim(0,11.5)
plt.tight_layout()
plt.savefig('Uniform_BBH_SNRless10_plot')
plt.show()

