from __future__ import division, print_function

import os
import sys
import bilby
import random
import numpy as np
import pandas as pd
from time import process_time

current_direc = os.getcwd()
# print("Current Working Directory is :", current_direc)

## Specify the output directory and the name of the simulation.
outmain = 'Output'
outdir = 'Injections'
label = 'Injections'
outdir = os.path.join(outmain, outdir)
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

class InjectionSignal:

    def __init__(self):
        pass

    def injections(self,filename=None):
        """
        filename: str, Data file
            Injection signal Data file

        Return: data file, hdf5
                Injection signals.
        """
        # basedir = os.path.join(os.path.dirname(__file__), 'Injection_file')
        filename = filename
        # filename = os.path.join(basedir, filename)
        injections = pd.read_hdf(filename)
        # if filename is None:
        #     print('No Injection  Signal File is given. Using default filename')
        #     fname = 'injections_10e6.hdf5'
        #     filename = os.path.join(basedir, fname)
        #     injections = pd.read_hdf(filename)
        #
        # elif filename is not None:
        #     if not os.path.isfile(filename):
        #         filename = os.path.join(os.path.dirname(__file__), 'Injection_file', filename)
        #         # filename = os.path.join(basedir, filename)
        #         injections = pd.read_hdf(filename)
        # else:
        #     print('Check for the Bug')

        return injections

    def injections_set(self, filename=None):
        """
        We set the injections signals according to the increase in redshift.

        filename: str, Data file
            Injection signal Data file

        Return: data file, hdf5
                Injection signals in asecnding order in redshift.
        """

        # injections = self.injections(filename=filename)
        injections = pd.read_hdf(filename)
        injections = injections.sort_values('redshift', ascending=True)
        injections.index = range(len(injections.index))

        return injections

    def redshift_cutoff(self, filename, cutoff_low=None, cutoff_high=None):
        """
        Setting a redshift cutoff over the BBH signal to do the Parameter estimation.
        test the low SNR effectivness of Subtraction-noise Projection Method.

        filename: str, Data file
            Injection signal Data file

        Return: data file, hdf5
                Injection signals in asecnding order in redshift above redshift threshold.

        """
        injections = pd.read_hdf(filename)
        injections = injections.sort_values('redshift', ascending=True)
        injections_redshift = injections[(injections['redshift'] > cutoff_low) & (injections['redshift'] < cutoff_high)]
        injections_redshift.index = range(len(injections_redshift.index))

        return injections_redshift

    def injection_signal(self, IFOs,sampling_frequency, seg_start_time=None, seg_end_time=None,
                         n_seg=None,  N_samples = None, injection_params=None, waveform_generator=None, n_samples=None):
        """
        Injection signals : Signals which are known and identified inspiralling compact binaries from the data_stream.

        Parameters
        -------------
        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.
        sampling_frequency: float
            The sampling frequency (in Hz).
        seg_start_time: duration
            The GPS start-time of the data for n_seg.
        seg_end_time: duration
            The GPS end-time of the data for n_seg.
        n_seg: int
            number of time segment in total time duration of operation of detector
        N_samples: int
            Number of samples in each segment of duration_duration_seg.
        injection_params: dict
            Parameters of the injection.
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.

        *** Note:  If waveform generator is given, injection polarization is not required and vice versa. ***

        Return:
        -----------
        Calculating time_doamin_strain for all detectors.
        Check bilby/gw/detector/strain_data.py for more details

        inj_time_series : array_like
                    time_domain_strain for injection signals in units of strain.
        """

        print('Injection script starting (', process_time(), ')')

        ## Changing 'iota' to 'theta_jn' to be suitable with bilby
        injection_params['theta_jn'] = injection_params['iota']

        injected = False

        ## Defining a t_coalescence array to store the coalescence time of a binary signal genrated randomly
        ## between the seg_start_time and seg_end_time.
        t_coalescence = np.zeros(len(injection_params))
        tcnt = 0

        for index, single_params in injection_params.iterrows():

            ## Generating a random array of coalescence time between the seg_start_time and seg_end_time of the time_seg
            t_c = random.uniform(np.float(seg_start_time + 10), np.float(seg_end_time - 10))  ## to generate a float array.
            single_params['geocent_time'] = t_c

            if seg_start_time < t_c < seg_end_time:

                injected = True
                print("Number of Injection signal is :", index)
                print('seg_start_time', seg_start_time)
                print('geocent_time', t_c)
                print('seg_end_time', seg_end_time)

                ## Saving t_coalescence as an array for Subtraction part
                t_coalescence[tcnt] = t_c
                ## Redshift to luminosity Distance conversion using bilby
                single_params['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(single_params['redshift'])

                ## First mass needs to be larger than second mass
                if single_params['mass_1'] < single_params['mass_2']:
                    tmp = single_params['mass_1']
                    single_params['mass_1'] = single_params['mass_2']
                    single_params['mass_2'] = tmp

                injected_signal = IFOs.inject_signal(parameters=single_params.to_dict(), waveform_generator=waveform_generator)

                tcnt += 1

        print('Signals injected (', process_time(), ')')

        # freq_domain_strain = np.zeros((len(IFOs), (n_samples)), dtype=np.complex)
        # ci = 0
        # for ifo in IFOs:
        #     freq_domain_strain[ci, :] = ifo.strain_data.frequency_domain_strain
        #     ci += 1
        #
        # sum_SNR = self.avg_SNR(IFOs, injection_params, t_coalescence, waveform_generator)
        #
        # avg_SNR = np.zeros(len(IFOs))
        # inj_fractional = np.zeros((len(IFOs), n_samples))
        # for d1 in np.arange(len(IFOs)):
        #     avg_SNR[d1] = np.sqrt(sum_SNR[d1] ** 2)
        #     inj_fractional[d1, :] = np.abs(freq_domain_strain[d1, :]) * (np.sqrt(4) / avg_SNR[d1])
        #
        # label1 = 'Fractional_Spectrum'
        # for d1 in np.arange(len(IFOs)):
        #     ifo.plot_data(signal=inj_fractional[d1, :], outdir=outdir, label=label1)

        if injected:
            label = 'inj_segment_' + str(n_seg)
            # IFOs.save_data(outdir=outdir, label=label)
            IFOs.plot_data(signal=None, outdir=outdir, label=label)
            print('Injection plots saved (', process_time(), ')')

        # inj_time_series = np.zeros((len(IFOs), (N_samples)))
        # ci = 0
        # for ifo in IFOs:
        #     inj_time_series[ci, :] = ifo.strain_data.time_domain_strain
        #     ci += 1
        # print('inj_time_series', inj_time_series)
        # print(np.shape(inj_time_series))

        inj_time_series = np.zeros((len(IFOs), (N_samples)))
        ci = 0
        for ifo in IFOs:
            inj_time_series[ci, :] = bilby.core.utils.infft(ifo.strain_data.frequency_domain_strain, sampling_frequency)
            ci += 1

        print('Time series calculated (', process_time(), ')')

        return t_coalescence, inj_time_series, IFOs

    def injection_signal_SNRcut(self,IFOs,sampling_frequency, seg_start_time=None, seg_end_time=None, n_seg=None,
                                    N_samples = None, injection_params=None, waveform_generator=None, n_samples=None,
                                    SNR_cut=None):
        """
        :param IFOs:
        :param sampling_frequency:
        :param seg_start_time:
        :param seg_end_time:
        :param n_seg:
        :param N_samples:
        :param injection_params:
        :param waveform_generator:
        :param n_samples:
        :param SNR_cut:
        :param subtracted_signal_idx:
        :return:
        """

        print('Injection script above SNR threshold starting (', process_time(), ')')

        ## Changing 'iota' to 'theta_jn' to be suitable with bilby
        injection_params['theta_jn'] = injection_params['iota']

        injected = False

        ## List of all Injected signals above SNR threshold i.e. detected binary signals
        injected_signal_idx = []

        ## Defining a t_coalescence array to store the coalescence time of a binary signal genrated randomly
        ## between the seg_start_time and seg_end_time.
        t_coalescence = np.zeros(len(injection_params))
        tcnt = 0

        for index, single_params in injection_params.iterrows():

            ## Generating a random array of coalescence time between the seg_start_time and seg_end_time of the time_seg
            t_c = random.uniform(np.float(seg_start_time + 10), np.float(seg_end_time - 10))  ## to generate a float array.

            single_params['geocent_time'] = t_c

            if seg_start_time < t_c < seg_end_time:

                injected = True
                print("Number of Injection signal is :", index)
                print('seg_start_time', seg_start_time)
                print('geocent_time', t_c)
                print('seg_end_time', seg_end_time)

                ## Saving t_coalescence as an array for Subtraction part
                t_coalescence[tcnt] = t_c

                ## Redshift to luminosity Distance conversion using bilby
                single_params['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(single_params['redshift'])

                ## First mass needs to be larger than second mass
                if single_params['mass_1'] < single_params['mass_2']:
                    tmp = single_params['mass_1']
                    single_params['mass_1'] = single_params['mass_2']
                    single_params['mass_2'] = tmp

                ## Check the SNR of bianry signals to be injected
                mf_snr = np.zeros((1, len(IFOs)))[0]
                waveform_polarizations = waveform_generator.frequency_domain_strain(single_params.to_dict())
                k = 0
                for ifo in IFOs:
                    signal_ifo = ifo.get_detector_response(waveform_polarizations, single_params.to_dict())
                    mf_snr[k] = np.sqrt(ifo.optimal_snr_squared(signal=signal_ifo))
                    if np.isnan(mf_snr[k]):
                        mf_snr[k] = 0.
                    print('{}: SNR = {:02.2f} at Luminosity Distance = {:02.2f}'.format(ifo.name, mf_snr[k],
                                                                                        single_params[
                                                                                            'luminosity_distance']))
                    k += 1

                if np.all(mf_snr > SNR_cut):
                    print(mf_snr)
                    injected_signal_idx.append(index)
                    injected_signal = IFOs.inject_signal(parameters=single_params.to_dict(),waveform_generator=waveform_generator)
                else:
                    print('SNR of detected Signal is less than SNR threshold value {} .'.format(SNR_cut))

                tcnt += 1

        print('Signals injected (', process_time(), ')')

        if injected:
            label = 'inj_segment_' + str(n_seg)
            # IFOs.save_data(outdir=outdir, label=label)
            IFOs.plot_data(signal=None, outdir=outdir, label=label)

            print('Injection plots saved (', process_time(), ')')

        inj_time_series = np.zeros((len(IFOs), (N_samples)))
        ci = 0
        for ifo in IFOs:
            inj_time_series[ci, :] = bilby.core.utils.infft(ifo.strain_data.frequency_domain_strain, sampling_frequency)
            ci += 1

        print('Time series calculated (', process_time(), ')')

        return injected_signal_idx, t_coalescence, inj_time_series,  IFOs


    def unresolved_injection_signal(self, IFOs,sampling_frequency, seg_start_time=None, seg_end_time=None, n_seg=None,
                                    N_samples = None, injection_params=None, waveform_generator=None, n_samples=None,
                                    SNR_cut=None, subtracted_signal_idx=None, t_coalescence=None):

        """
        Injection signals : Signals which are known and identified inspiralling compact binaries from the data_stream.

        Parameters
        -------------
        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.
        sampling_frequency: float
            The sampling frequency (in Hz).
        seg_start_time: duration
            The GPS start-time of the data for n_seg.
        seg_end_time: duration
            The GPS end-time of the data for n_seg.
        n_seg: int
            number of time segment in total time duration of operation of detector
        N_samples: int
            Number of samples in each segment of duration_duration_seg.
        injection_params: dict
            Parameters of the injection.
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.
        subtracted_signal_idx: list
            A list of signals which are identified and subtracted from the data stream of the GW detector.

        *** Note:  If waveform generator is given, injection polarization is not required and vice versa. ***

        Return:
        -----------
        Calculating time_doamin_strain for all detectors.
        Check bilby/gw/detector/strain_data.py for more details

        unresolved_time_series : array_like
                    time_domain_strain for unresolved injection signals in units of strain.
        """

        print('Unresolved Injection script starting (', process_time(), ')')

        ## Changing 'iota' to 'theta_jn' to be suitable with bilby
        injection_params['theta_jn'] = injection_params['iota']

        ## List of all unresolved signals below SNR_cut threshold value.
        # unresolved_signal_idx = []

        unresolved_injected = False
        tcnt = 0
        for index, single_params in injection_params.iterrows():
            ## Check if the unresolved signal index is not in the  subtracted_signal_idx.
            if index not in subtracted_signal_idx:

                print('Signal is an unresolved binary signal')

                ## We used t_coalescence from the Injection.py script to have the same coalescence time for an Injected and Subtracted binary signal.
                t_c = t_coalescence[tcnt]
                print('t_c is',t_c)
                single_params['geocent_time'] = t_c

                if seg_start_time < t_c < seg_end_time:

                    unresolved_injected = True
                    print("Number of unresolved signal is :", index)
                    print('seg_start_time', seg_start_time)
                    print('geocent_time', t_c)
                    print('seg_end_time', seg_end_time)

                    ## Redshift to luminosity Distance conversion using bilby
                    single_params['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(single_params['redshift'])

                    ## First mass needs to be larger than second mass
                    if single_params['mass_1'] < single_params['mass_2']:
                        tmp = single_params['mass_1']
                        single_params['mass_1'] = single_params['mass_2']
                        single_params['mass_2'] = tmp

                    # unresolved_signal_idx.append(index)
                    unresolved_inject_signal = IFOs.inject_signal(parameters=single_params.to_dict(), waveform_generator=waveform_generator)

            tcnt += 1

        print('Unresolved Signals injected (', process_time(), ')')

        unresolved_time_series = np.zeros((len(IFOs), (N_samples)))
        ci = 0
        for ifo in IFOs:
            unresolved_time_series[ci, :] = bilby.core.utils.infft(ifo.strain_data.frequency_domain_strain, sampling_frequency)
            ci += 1

        print('Time series calculated (', process_time(), ')')

        return unresolved_time_series, IFOs

    def avg_SNR(self, IFOs, injection_params, t_coalescence, waveform_generator):
        """
        Calculation of avergae SNR of all Injected Signal into a GW detector

        injection_params : dict
            Parameters of the injection signals.
        t_coalescence : array type
            Array represent a choice of coalescence time for binary signal.
            for information see Injection.InjectionSIgnal.injection_signal().


        :return:
        matched filter SNR : array
            Average SNR of all injected signal.
        """

        print(' Sum SNR calculation starting (', process_time(), ')')

        mf_snr = np.zeros((len(IFOs)))

        tcnt = 0
        for index, single_params in injection_params.iterrows():

            if 'iota' in single_params:
                ## Changing 'iota' to 'theta_jn' to be suitable with bilby
                single_params['theta_jn'] = single_params['iota']

            if 'luminosity_distance' in single_params:
                single_params['luminosity_distance'] = float(single_params['luminosity_distance'])
            else:
                ## Redshift to Luminosity Distance.
                single_params['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(
                    single_params['redshift'])

            if 'redshift' not in single_params:
                ## Luminosity Distance to Redshift.
                single_params['redshift'] = float(
                    bilby.gw.conversion.luminosity_distance_to_redshift(single_params['luminosity_distance']))

            ## We used t_coalescence to have the same coalescence time as for an Injected and Subtracted binary signal.
            t_c = t_coalescence[tcnt]
            single_params['geocent_time'] = t_c

            ## First mass needs to be larger than second mass
            if single_params['mass_1'] < single_params['mass_2']:
                tmp = single_params['mass_1']
                single_params['mass_1'] = single_params['mass_2']
                single_params['mass_2'] = tmp

            waveform_polarizations = waveform_generator.frequency_domain_strain(dict(single_params))
            z = 0
            for ifo in IFOs:
                signal_ifo = ifo.get_detector_response(waveform_polarizations, single_params)
                ## Calculate the _complex_ matched filter snr of a signal.
                ##This is <signal|frequency_domain_strain> / optimal_snr
                mf_snr[z] += ifo.matched_filter_snr(signal=signal_ifo)
                if np.isnan(mf_snr[z]):
                    mf_snr[z] = 0.

                print('{}: SNR = {:02.2f} at z = {:02.2f}'.format(ifo.name, mf_snr[z], (single_params['redshift'])))

                z += 1

            tcnt += 1

        print(' Sum SNR calculated (', process_time(), ')')

        return mf_snr