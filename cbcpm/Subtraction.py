from __future__ import division, print_function

import os
import sys
import bilby
import numpy as np
import pandas as pd
import json
import pickle

from time import process_time

from Initial_data import InitialData, InterferometerStrain
from Injection import InjectionSignal

current_direc = os.getcwd()
# print("Current Working Directory is :", current_direc)

## Specify the output directory and the name of the simulation.
outmain = 'Output'
outdir = 'Subtractions'
label = 'Subtractions'
outdir = os.path.join(outmain, outdir)

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

# bilby.utils.setup_logger(outdir=outdir, label=label)


class SubtractionSignal():

    def __init__(self):
        pass

    def subtraction_parameters(self, filefolder=None, n_inj=None):
        """
        Parameters
        -----------
        filefolder: str
                location of subtraction/best fit parameter estimation data files
        n_inj = int
                Number of subtraction signal required.

        Return: Dict
            dictionary of max likelihood of all best fit parameter estimation.

        """
        ## reading the directory in whcih data files are stored.
        basedir = os.path.join(os.path.dirname(__file__), filefolder)
        # max_likelihood_data = os.path.join(current_direc, 'data')
        max_likelihood_data = os.listdir(basedir)
        max_likelihood_data_file_select = [x for x in max_likelihood_data if x.startswith('71')]

        ## for List of Keys.
        subtraction_param_list = []
        ## For making a dict of all values and their keys.
        subtraction_params_dict = []
        i = 0
        for max_likelihood_data_direc in sorted(max_likelihood_data_file_select):
            if i < n_inj:

                current_max_likelihood_data_direc = os.path.join(basedir, max_likelihood_data_direc)
                max_likelihood_sample_data_file_name = os.path.join(current_max_likelihood_data_direc,
                                                                    'sample_param_' + str(i) + '_result.json')

                max_likelihood_sample_data_file_open = json.load(open(max_likelihood_sample_data_file_name))

                max_likelihood_sample_data_file_open_read = max_likelihood_sample_data_file_open['posterior']['content']

                key_list = []
                dict = {}
                for sub_sub_keys in max_likelihood_sample_data_file_open_read:
                    key_list.append(sub_sub_keys)
                    dict[sub_sub_keys] = max_likelihood_sample_data_file_open_read[sub_sub_keys][-1]

                ## Deleting the maximum log_likelihood and log_prior parameter from max_likelihood_sample_data_file_open_read .
                del (dict['log_likelihood'])
                del (dict['log_prior'])

                subtraction_param_list.append(key_list)
                subtraction_params_dict.append(dict)

                i += 1

        ## Convertind dictionary to dataframe
        subtraction_params = pd.DataFrame.from_dict(subtraction_params_dict)

        return subtraction_params

    def subtraction_params_Uniform(self, filefolder=None, Injection_file=None, n_inj=None):
        """
        Parameters
        -----------
        filefolder: str
                location of subtraction/best fit parameter estimation data files

        Injection_file : txt file
                Injection signal Index file, containing the index number of all Injections
                injetced into the detector to perform parameter estimation.

        n_inj = int
                Total number of signals Injected.

        Return: Dict
            dictionary of max likelihood of all best fit parameter estimation.

        """
        ## reading the directory in whcih data files are stored.
        basedir = os.path.join(os.path.dirname(__file__), filefolder)
        # max_likelihood_data = os.path.join(current_direc, 'data')
        max_likelihood_data = os.listdir(basedir)
        max_likelihood_data_file_select = [x for x in max_likelihood_data if x.startswith('77')]

        ## for List of Keys.
        subtraction_param_list = []
        ## For making a dict of all values and their keys.
        subtraction_params_dict = []

        list = open(Injection_file)
        list = list.readline().split()
        print(list)

        # n_inj = len(list)

        cnt = 0
        for max_likelihood_data_direc in sorted(max_likelihood_data_file_select):
            for i in list:
                if i == list[cnt]:
                    print('i now', i)
                    current_max_likelihood_data_direc = os.path.join(basedir, max_likelihood_data_direc)
                    max_likelihood_sample_data_file_name = os.path.join(current_max_likelihood_data_direc, 'sample_param_' + str(i) + '_result.json')
                    max_likelihood_sample_data_file_open = json.load(open(max_likelihood_sample_data_file_name))
                    max_likelihood_sample_data_file_open_read = max_likelihood_sample_data_file_open['posterior']['content']

                    key_list = []
                    dict = {}
                    for sub_sub_keys in max_likelihood_sample_data_file_open_read:
                        key_list.append(sub_sub_keys)
                        dict[sub_sub_keys] = max_likelihood_sample_data_file_open_read[sub_sub_keys][-1]

                    ## Deleting the maximum log_likelihood and log_prior parameter from max_likelihood_sample_data_file_open_read .
                    del (dict['log_likelihood'])
                    del (dict['log_prior'])

                    subtraction_param_list.append(key_list)
                    subtraction_params_dict.append(dict)

            cnt +=1

        ## Convertind dictionary to dataframe
        subtraction_params = pd.DataFrame.from_dict(subtraction_params_dict)

        return subtraction_params


    def subtraction_params_IMR(self, filefolder=None, Injection_file=None, n_inj=None):
        """
        Parameters
        -----------
        filefolder: str
                location of subtraction/best fit parameter estimation data files

        Injection_file : txt file
                Injection signal Index file, containing the index number of all Injections
                injetced into the detector to perform parameter estimation.

        n_inj = int
                Total number of signals Injected.

        Return: Dict
            dictionary of max likelihood of all best fit parameter estimation.

        """
        ## reading the directory in whcih data files are stored.
        basedir = os.path.join(os.path.dirname(__file__), filefolder)
        # max_likelihood_data = os.path.join(current_direc, 'data')
        max_likelihood_data = os.listdir(basedir)

        ## for List of Keys.
        subtraction_param_list = []
        ## For making a dict of all values and their keys.
        subtraction_params_dict = []

        list = open(Injection_file)
        list = list.readline().split()

        cnt = 0
        # for max_likelihood_data_direc in sorted(max_likelihood_data):
        for i in list:
            if i == list[cnt]:
                print('i now', i)
                # current_max_likelihood_data_direc = os.path.join(basedir, max_likelihood_data_direc)
                max_likelihood_sample_data_file_name = os.path.join(basedir, 'sample_param_' + str(i) + '_result.json')
                max_likelihood_sample_data_file_open = json.load(open(max_likelihood_sample_data_file_name))
                max_likelihood_sample_data_file_open_read = max_likelihood_sample_data_file_open['posterior']['content']

                key_list = []
                dict = {}
                for sub_sub_keys in max_likelihood_sample_data_file_open_read:
                    key_list.append(sub_sub_keys)
                    dict[sub_sub_keys] = max_likelihood_sample_data_file_open_read[sub_sub_keys][-1]

                ## Deleting the maximum log_likelihood and log_prior parameter from max_likelihood_sample_data_file_open_read .
                del (dict['log_likelihood'])
                del (dict['log_prior'])

                subtraction_param_list.append(key_list)
                subtraction_params_dict.append(dict)

            cnt +=1

        ## Convertind dictionary to dataframe
        subtraction_params = pd.DataFrame.from_dict(subtraction_params_dict)

        return subtraction_params

    def sub_posterior_params(self, filefolder=None, n_inj=None):
        """
        Parameters
        -----------
        filefolder: str
                location of subtraction/best fit parameter estimation data files
        n_inj = int
                Number of subtraction signal required.

        Return: Dict
            dictionary of max likelihood of all best fit parameter estimation.

        """
        ## reading the directory in whcih data files are stored.
        basedir = os.path.join(os.path.dirname(__file__), filefolder)
        # max_likelihood_data = os.path.join(current_direc, 'data')
        max_posterior_data = os.listdir(basedir)
        max_posterior_data_file_select = [x for x in max_posterior_data if x.startswith('76')]

        ## for List of Keys.
        subtraction_param_list = []
        ## For making a dict of all values and their keys.
        subtraction_params_dict = []
        i = 2
        for max_posterior_data_direc in sorted(max_posterior_data_file_select):
            if i < n_inj:

                current_max_posterior_data_direc = os.path.join(basedir, max_posterior_data_direc)
                max_posterior_sample_data_file_name = os.path.join(current_max_posterior_data_direc,'sample_param_' + str(i) + '_result.json')
                print(max_posterior_sample_data_file_name)

                with open (max_posterior_sample_data_file_name, 'r') as max_posterior_sample_data_file_open:
                    max_posterior_sample_data_file = json.load(max_posterior_sample_data_file_open)

                max_posterior_sample_data_file_read = max_posterior_sample_data_file['posterior']['content']

                key_list = []
                dict = {}
                for sub_sub_keys in max_posterior_sample_data_file_read:
                    key_list.append(sub_sub_keys)
                    dict[sub_sub_keys] = max_posterior_sample_data_file_read[sub_sub_keys][-1]

                ## Deleting the maximum log_likelihood and log_prior parameter from max_likelihood_sample_data_file_open_read .
                del (dict['log_likelihood'])
                del (dict['log_prior'])

                subtraction_param_list.append(key_list)
                subtraction_params_dict.append(dict)

                i += 1

        ## Convertind dictionary to dataframe
        subtraction_params = pd.DataFrame.from_dict(subtraction_params_dict)

        return subtraction_params

    def subtraction_params(self, filename):
        """
        Parameters
        -----------
        Return: Panda data frame
            Panda data frame of max likelihood parameter values

        """

        file = open(filename, 'rb')
        subtraction_params = pd.DataFrame(pickle.load(file))

        return subtraction_params

    def SNR_bestfit(self,filefolder, n_inj=None):
        """
        Parameters
        -----------
        filefolder: str
                location of subtraction/best fit parameter estimation data files
        Return: float
            SNR values for each ifo corresponding to the injected Signal into the Interferometer.
        """

        ## reading the data directory in whcih files are stored.
        basedir = os.path.join(os.path.dirname(__file__), filefolder)
        # max_likelihood_data = os.path.join(current_direc, 'data')
        data = os.listdir(basedir)
        data_file_select = [x for x in data if x.startswith('76')]

        snr_file_save = open(r'Uniform_bestfit_SNRg10.txt', 'w')

        list = [14, 22, 25, 30, 31, 32, 50, 53, 54, 55,
                56, 64, 65, 79, 81, 82, 83, 91, 93, 113,
                115, 116, 119, 126, 127, 130, 144, 157, 162, 165,
                166, 171, 182, 185, 186, 187, 190, 191, 199, 203,
                204, 208, 219, 220, 229, 230, 231, 235, 237, 242,
                244, 258, 260, 264, 269, 270, 271, 273, 274, 279,
                285, 291, 294, 296, 298, 300, 308, 311, 314, 318,
                328, 333, 345, 350, 363, 367, 372, 375, 380, 391,
                392, 407, 409, 411, 414, 416, 418, 440, 447, 452,
                453, 455, 456, 470, 473, 478, 479, 488, 489, 491,
                493, 496, 498, 500, 505, 513, 536, 539, 543, 549,
                551, 562, 563, 567, 574, 582]

        cnt = 0
        for data_direc in sorted(data_file_select):
            for i in list:
                if i == list[cnt]:

                    current_data_direc = os.path.join(basedir, data_direc)
                    sample_data_file_name = os.path.join(current_data_direc,'sample_param_' + str(i) + '_result.json')

                    sample_data_file_open = json.load(open(sample_data_file_name))
                    ## Reading the meta_data inside the JSON file, got after parameter estimation.
                    ## Accessing the dictionaries 'likelihood' and 'interferometers' inside meta_data.
                    ## To get the optimal SNR values for each Injected signal into each IFO during Parameter estimation.
                    sample_data_file_open_read = sample_data_file_open['meta_data']['likelihood']['interferometers']

                    ## Writing the In jection Signal number into the text file.
                    snr_file_save.write('Bestfit Signal {}'.format(str(i)) + '\n')

                    ## reading the Interferometer
                    for ifo in sample_data_file_open_read:
                        ## Now to read the keys inside the Interferometer dictionary.
                        SNRkeys = sample_data_file_open_read[ifo]
                        for SNR in SNRkeys:
                            ## look out for perticular key to read SNR values.
                            if SNR == 'optimal_SNR':
                                snr_value = SNRkeys[SNR]

                                snr_file_save.write('{} '.format(str(ifo)))
                                snr_file_save.write(' {} = '.format(str(SNR)))
                                snr_file_save.write(str(snr_value) + '\n')

            cnt += 1

        # i = 0
        # for data_direc in sorted(data_file_select):
        #     if i < n_inj:
        #
        #         current_data_direc = os.path.join(basedir, data_direc)
        #         sample_data_file_name = os.path.join(current_data_direc,'sample_param_' + str(i) + '_result.json')
        #
        #         sample_data_file_open = json.load(open(sample_data_file_name))
        #         ## Reading the meta_data inside the JSON file, got after parameter estimation.
        #         ## Accessing the dictionaries 'likelihood' and 'interferometers' inside meta_data.
        #         ## To get the optimal SNR values for each Injected signal into each IFO during Parameter estimation.
        #         sample_data_file_open_read = sample_data_file_open['meta_data']['likelihood']['interferometers']
        #
        #         ## Writing the In jection Signal number into the text file.
        #         snr_file_save.write('Bestfit Signal {}'.format(str(i)) + '\n')
        #
        #         ## reading the Interferometer
        #         for ifo in sample_data_file_open_read:
        #             ## Now to read the keys inside the Interferometer dictionary.
        #             SNRkeys = sample_data_file_open_read[ifo]
        #             for SNR in SNRkeys:
        #                 ## look out for perticular key to read SNR values.
        #                 if SNR == 'optimal_SNR':
        #                     snr_value = SNRkeys[SNR]
        #
        #                     snr_file_save.write('{} '.format(str(ifo)))
        #                     snr_file_save.write(' {} = '.format(str(SNR)))
        #                     snr_file_save.write(str(snr_value) + '\n')
        #         i += 1

        snr_file_save.close()

        return snr_file_save


    def subtraction_signal(self, IFOs, sampling_frequency = None, seg_start_time = None, seg_end_time = None, t_coalescence=None,
                           n_seg=None, N_samples = None, bestfit_params=None,waveform_generator=None, n_samples=None):
        """
        Parameters
        ----------

        ifos: iterable
            The list of interferometers.
        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.
        sampling_frequency: float
            The sampling frequency (in Hz).
        seg_start_time: duration
            The GPS start-time of the data for n_seg.
        seg_end_time: duration
            The GPS end-time of the data for n_seg.
        t_coalescence : array type
            Array represent a choice of coalescence time for binary signal.
            for information see Injection.py.
        n_seg: int
            number of time segment in total time duration of operation of detector
        N_samples: int
            Number of samples in each segment of duration_duration_seg.
        bestfit_param : Dict
                Dictionary of max likelihood  of all best fit parameters
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.

        Return:

        Calculating residual_doamin_strain for all detectors.
        Check bilby/gw/detector/strain_data.py for more details

        residual_noise_data : array_like
                    Detector's frequency domain Residual Data after Subtracting the best fit of each Signal.
        sub_time_series: array like
                Array of time series of subtraction residuals.
                Real valued array of time series calculated by taking inverse fft of residual_noise_data.
        """

        print('Subtraction script starting (', process_time(), ')')

        subtracted = False
        tcnt = 0
        for index, single_params in bestfit_params.iterrows():

            ## We used t_coalescence from the Injection.py script to have the same coalescence time for an Injected and Subtracted binary signal.
            t_c = t_coalescence[tcnt]

            single_params['geocent_time'] = t_c

            if seg_start_time < t_c < seg_end_time:

                subtracted = True

                print("Number of Subtraction signal is :", index)
                print('seg_start_time', seg_start_time)
                print('geocent_time', t_c)
                print('seg_end_time', seg_end_time)

                single_params['luminosity_distance'] = float(single_params['luminosity_distance'])

                ## First mass needs to be larger than second mass (just to cross check)
                if single_params['mass_1'] < single_params["mass_2"]:
                    tmp = single_params['mass_1']
                    single_params['mass_1'] = single_params['mass_2']
                    single_params['mass_2'] = tmp

                subtracted_signal = IFOs.subtract_signal(parameters=single_params.to_dict(), waveform_generator=waveform_generator)

                tcnt += 1

        print('Signals subtracted (', process_time(), ')')

        if subtracted:
            label = 'sub_segment_' + str(n_seg)
            # IFOs.save_data(outdir=outdir, label=label)
            IFOs.plot_data(signal=None, outdir=outdir, label=label)

            print('Subtraction plots saved (', process_time(), ')')

        residual_noise_data = np.zeros((len(IFOs), n_samples), dtype=np.complex)
        cnt = 0
        for ifo in IFOs:
            residual_noise_data[cnt, :] = ifo.strain_data.frequency_domain_strain
            cnt += 1

        ## Using freq_domain_strain/residual_noise_data to calculate the time_domain_strain by using bilby infft
        sub_time_series = np.zeros((len(IFOs), (N_samples)))
        cnt = 0
        for ifo in IFOs:
            sub_time_series[cnt, :] = bilby.core.utils.infft(ifo.strain_data.frequency_domain_strain, sampling_frequency)
            cnt += 1

        print('Residual time series calculated (', process_time(), ')')

        return residual_noise_data, sub_time_series, IFOs

    def subtraction_signal_SNR(self, IFOs, sampling_frequency = None, seg_start_time = None, seg_end_time = None, t_coalescence=None,
                           n_seg=None, N_samples = None, bestfit_params=None,waveform_generator=None, n_samples=None, SNR_cut=None):
        """

        This part of script subtract only those CBC signals which lies above a selected SNR threshold value.


        Parameters
        ----------

        ifos: iterable
            The list of interferometers.
        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.
        sampling_frequency: float
            The sampling frequency (in Hz).
        seg_start_time: duration
            The GPS start-time of the data for n_seg.
        seg_end_time: duration
            The GPS end-time of the data for n_seg.
        t_coalescence : array type
            Array represent a choice of coalescence time for binary signal.
            for information see Injection.py.
        n_seg: int
            number of time segment in total time duration of operation of detector
        N_samples: int
            Number of samples in each segment of duration_duration_seg.
        bestfit_param : Dict
                Dictionary of max likelihood  of all best fit parameters
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.

        SNR_cut: int
            Number to set as an SNR threshold value to subtract the CBC signals.

        Return:

        Calculating residual_doamin_strain for all detectors.
        Check bilby/gw/detector/strain_data.py for more details

        residual_noise_data : array_like
                    Detector's frequency domain Residual Data after Subtracting the best fit of each Signal.
        sub_time_series: array like
                Array of time series of subtraction residuals.
                Real valued array of time series calculated by taking inverse fft of residual_noise_data.
        subtracted_signal_idx: array like
                Array of index corresponding to the subtracted CBC signals. This array of subtracted signals will be
                used in projection script to porject out the corresponding residuals.
        """

        print('Subtraction script starting (', process_time(), ')')

        ## List of all subtracted signals above SNR_cut threshold value.
        subtracted_signal_idx = []

        subtracted = False
        tcnt = 0
        for index, single_params in bestfit_params.iterrows():

            ## We used t_coalescence from the Injection.py script to have the same coalescence time for an Injected and Subtracted binary signal.
            t_c = t_coalescence[tcnt]
            print('t_c is',t_c)

            single_params['geocent_time'] = t_c

            if seg_start_time < t_c < seg_end_time:

                subtracted = True
                print("Number of Subtraction signal is :", index)
                print('seg_start_time', seg_start_time)
                print('geocent_time', t_c)
                print('seg_end_time', seg_end_time)

                single_params['luminosity_distance'] = float(single_params['luminosity_distance'])

                ## First mass needs to be larger than second mass (just to cross check)
                if single_params['mass_1'] < single_params["mass_2"]:
                    tmp = single_params['mass_1']
                    single_params['mass_1'] = single_params['mass_2']
                    single_params['mass_2'] = tmp

                ## Check the SNR of estimated best-fit signals to be subtracted
                mf_snr = np.zeros((1, len(IFOs)))[0]
                waveform_polarizations = waveform_generator.frequency_domain_strain(single_params.to_dict())
                k = 0
                for ifo in IFOs:
                    signal_ifo = ifo.get_detector_response(waveform_polarizations, single_params.to_dict())
                    mf_snr[k] = np.sqrt(ifo.optimal_snr_squared(signal=signal_ifo))
                    if np.isnan(mf_snr[k]):
                        mf_snr[k] = 0.
                    print('{}: SNR = {:02.2f} at Luminosity Distance = {:02.2f}'.format(ifo.name, mf_snr[k],single_params['luminosity_distance']))
                    k += 1

                ## If SNR is above the SNR_cut threshold value.
                if np.all(mf_snr > SNR_cut):
                    subtracted_signal_idx.append(index)
                    subtracted_signal = IFOs.subtract_signal(parameters=single_params.to_dict(), waveform_generator=waveform_generator)
                else:
                    print('SNR of Subtracted Signal is less than {}.'.format(SNR_cut))

                tcnt += 1

        print('Signals subtracted (', process_time(), ')')

        if subtracted:
            label = 'sub_segment_SNR_' + str(n_seg)
            # IFOs.save_data(outdir=outdir, label=label)
            IFOs.plot_data(signal=None, outdir=outdir, label=label)

            print('Subtraction plots saved (', process_time(), ')')

        residual_noise_data = np.zeros((len(IFOs), n_samples), dtype=np.complex)
        cnt = 0
        for ifo in IFOs:
            residual_noise_data[cnt, :] = ifo.strain_data.frequency_domain_strain
            cnt += 1

        ## Using freq_domain_strain/residual_noise_data to calculate the time_domain_strain by using bilby infft
        sub_time_series = np.zeros((len(IFOs), (N_samples)))
        cnt = 0
        for ifo in IFOs:
            sub_time_series[cnt, :] = bilby.core.utils.infft(ifo.strain_data.frequency_domain_strain, sampling_frequency)
            cnt += 1

        print('Residual time series calculated (', process_time(), ')')

        return residual_noise_data, sub_time_series, IFOs, subtracted_signal_idx
