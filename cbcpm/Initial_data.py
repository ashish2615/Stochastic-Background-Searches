from __future__ import division, print_function

import os
import sys
import bilby
from bilby.core.utils import speed_of_light
import numpy as np
from numpy.core._multiarray_umath import dtype
import scipy
from scipy import signal, fftpack
from scipy.fftpack import fft, rfft,ifft,irfft, fftfreq, rfftfreq

import astropy
from astropy import constants as const
from astropy.constants import G, pc, c

class InitialData:

    def __init__(self):
        pass

    def initial_data(self, ifos, sampling_frequency, start_time, end_time, Tfft, n_seg=None):

        """
        Parameteres
        -----------
        ifos: List
            List of all detectors
        sampling_frequency: float
            The sampling frequency (in Hz).
        start_time: Duration
            data taking or operation of period detector
            The GPS start-time of the data
        end_time:  The GPS end-time of the data
        n_seg:  number of segment in total time duration of operation of detector


        Return:
        -------------
        ifos: List
            List of all detectors.
        sampling_frequency: float
               The sampling frequency (in Hz).
        start_time: Duration
               data taking or operation of period detector. The GPS start-time of the data.
        duration_seg: float
            The data duration of a segment (in s).
        n_seg: int
           number of segment in total time duration of operation of detector.
        n_samples : int
                Number of samples in each segment of duration (i.e. duration_seg).
        N_samples: int
            Number of samples in each segment of duration_duration_seg.
        n_samples_fft: int
                Total number of samples in T_seg of FFT bins. Number of seconds in a single fft.
        frequency: array_like
           Real FFT of sample frequencies.

        IFOs: Initialization of GW interferometer.
           Generates an Interferometer with a power spectral density.

        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.

        Gravitational Constant G: int
        one_pc: int
            distance, one persec in meters
        H0: int
            Hubble constant
        speed_of_light: int
            constant
        rho_c: int
            critical energy density required to close the Universe
        omega_gw:int
            Energy density of Stochastic Cosmological background Gravitational Wave. omega_gw is a dimensionless quantity.

        """

        self.ifos = ifos
        self.sampling_frequency = sampling_frequency
        self.Nquist = self.sampling_frequency / 2
        self.start_time = start_time
        self.end_time = end_time

        if n_seg is None:
            n_seg = 10000
        else:
            n_seg = n_seg

        self.n_seg = n_seg

        self.duration = self.end_time - self.start_time
        duration_seg = self.duration / n_seg

        self.duration_seg = 2**(int(duration_seg) - 1).bit_length()

        n_seg = np.trunc(self.duration / self.duration_seg)

        self.n_samples = int(self.sampling_frequency * self.duration_seg / 2) + 1
        self.N_samples = int(self.sampling_frequency * self.duration_seg )

        delta_T = self.duration_seg / self.N_samples
        frequency_resolution = 1 / self.duration_seg

        self.n_fft = int(np.round(Tfft * self.sampling_frequency))

        n_frequencies = int(np.round(self.n_fft / 2) + 1)
        freq_series = np.linspace(start=0, stop = self.sampling_frequency / 2, num = n_frequencies)

        freq_rfft = rfftfreq(self.n_fft, d=1. / self.sampling_frequency)

        self.frequency = np.append([0], freq_rfft[1::2])

        self.waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=50.,minimum_frequency=2.)

        self.waveform_generator = bilby.gw.WaveformGenerator(duration=self.duration_seg, sampling_frequency=self.sampling_frequency,
                                                                   frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                                   waveform_arguments=self.waveform_arguments)

        self.IFOs = bilby.gw.detector.networks.InterferometerList(ifos)

        self.modes = ['plus', 'cross']  # ,'breathing']

        ## Hubble constant H0 = (67.4±0.5) km s−1Mpc−1
        self.G = 6.67408 * 10 ** -11  ## units = m**3/ (kg* sec**2)
        self.one_pc = 3.0856775814671916 * 10 ** 16  ## units = m
        self.H0 = 67.9 * 10 ** 3 * 10 ** -6 * self.one_pc ** -1  ## units = 1/sec
        # self.h0 = 0.6766
        # self.H0 = h0*3.24*10**-18
        self.speed_of_light = bilby.gw.utils.speed_of_light

        ## rho_c = (3 * c ** 2 * H0 ** 2)/(8 * np.pi * G)
        self.rho_c = (3 * self.speed_of_light**2 * self.H0**2) / (8 * np.pi * self.G)  ## units = erg/cm**3
        self.omega_gw = 10**-15

        data_set = ( self.ifos, self.sampling_frequency, self.start_time, self.end_time, self.duration, self.duration_seg,
                     self.n_fft, self.N_samples, self.frequency, self.waveform_generator, self.IFOs, self.modes,
                     self.G, self.one_pc, self.H0, self.speed_of_light, self.rho_c, self.omega_gw, self.n_samples)

        return data_set

class InterferometerStrain:

    def interferometer_strain_data(self,IFOs=None,sampling_frequency=None,duration=None, start_time=None):
        """
        Set the Interferometer strain data from the power spectral densities of the detectors.
        This uses theinterferometer power_spectral_density object to set the strain_data to a noise realization.
        bilby.gw.detector.InterferometerStrainData` for further information.

        Parameters
        ----------
        sampling_frequency: float
            The sampling frequency (in Hz)
        duration: float
            The data segment duration (in s)
        start_time: float
            The GPS segment start_time of the data.

        """
        # if IFOs is not None:
        #     return IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency,
        #                                                         duration=duration,
        #                                                         start_time=start_time)
        # elif IFOs is None:
        #     return self.IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=self.sampling_frequency,
        #                                                                    duration=self.duration_seg,
        #                                                                    start_time=self.seg_start_time)

        IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency,
                                                           duration=duration, start_time=start_time)
        return IFOs