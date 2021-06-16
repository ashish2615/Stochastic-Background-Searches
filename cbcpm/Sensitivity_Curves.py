from __future__ import division, print_function

import os
import sys
import bilby
import numpy as np
import logging
import pandas as pd
import math
import gwpy

from bilby.core.utils import speed_of_light


import matplotlib
#matplotlib.use('tkagg')
#matplotlib.use('agg')
import matplotlib.pyplot as plt

from Initial_data import InitialData, InterferometerStrain
from ORF_OF import detector_functions
from plot_data import PSDWelch
from astro_stochastic_background import StochasticBackground

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

# Specify the output directory and the label name of the simulation.
outdir = 'Sensitivity_Curve'
# label = 'injected_signal'
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

ifos = ['L1','H1','V1','K1','ET_D_TR','CE']
sampling_frequency = 2048.
start_time = 1198800017
end_time   = 1230336017
n_seg = 10000
Tfft=8

## Initialization of Data
data = InitialData()
data_sets = data.initial_data(ifos, sampling_frequency, start_time, end_time, Tfft, n_seg=n_seg)
IFOs = data_sets[10]

function =  detector_functions()
plots_PSD = PSDWelch(IFOs)
Omega_gw = StochasticBackground()

class SensitivityCurves:

    def __init__(self):
        pass

    def initial(self, ifos, sampling_frequency, start_time, end_time, n_seg):
        """
        Initialize/access the data from Initial_data script.
        check cbcpm.InitialData.initial_data

        ifos: iterable
            The list of interferometers
        sampling_frequency: float
            The sampling frequency (in Hz).
        duration_seg: float
            The data duration of a segment (in s).
        N_samples: int
            Number of samples in each segment of duration_duration_seg.
        frequency: array_like
           Real FFT of sample frequencies.

        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.
        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.

        """
        data = InitialData()
        data_sets = data.initial_data(ifos, sampling_frequency, start_time, end_time, n_seg)
        self.sampling_frequency = data_sets[1]
        self.start_time = data_sets[2]
        self.end_time = data_sets[3]
        self.duration = data_sets[4]

        self.duration_seg = data_sets[5]
        self.N_samples = data_sets[7]
        self.frequency = data_sets[8]
        self.waveform_generator = data_sets[9]
        self.IFOs = data_sets[10]

        self.modes = data_sets[11]
        self.G = data_sets[12]
        self.one_pc = data_sets[13]
        self.H0 = data_sets[14]
        self.speed_of_light = data_sets[15]
        self.rho_c = data_sets[16]
        self.omega_gw = data_sets[17]
        self.Tfft = Tfft

    def LISA(self):
        """
        LISA sensitivity Curve as given in ref. LISA Science Requirements Document

        f : array type
             Frequency range in which LISA is sensitive for GWs signal.
        S1 :
        S2 :
        R  :

        Return
        --------------
        Sh : array type
            power spectral density for LISA GW detector.
        """

        f = np.linspace(start=10**-4, stop = 10**0, num=10000)

        S1 = 5.76 * 10 ** (-48) * (1 + (4.0 * 10 ** (-4) / f) ** 2)
        S2 = 3.6 * 10 ** (-41)
        R = 1.0 + (f / (25.0 * 10 ** (-3))) ** 2

        Sh = (10.0 / 3.0) * (S1 / (2 * np.pi * f) ** 4 + S2) * R

        return f, Sh

    def DECIGO(self):
        """
        arXiv:1101.3940v2 [astro-ph.CO] check eq. 5

        f : array type
             Frequency range in which DECIGO is sensitive for GWs signal.

        fp : int
            Pivot frequency, 7.36

        Return
        --------------
        Sh : array type
            power spectral density for DECIGO GW detector.


        """
        f = np.linspace(start = 10**-3, stop= 10**2, num=10000)

        Sh = 7.05 * 10**-48 * (1 + (f / 7.36)**2) + 4.8 * 10**-51 * f**-4 * (1 / (1 + (f / 7.36)**2)) + 5.33 * 10**-52 * f**-4
        return f, Sh

    def BBO(self):
        """
        arXiv:1101.3940v2 [astro-ph.CO] Check eq. 6


        Note :  BBO/BBO1 and psd file of BBO (./Sensitivity/BBOtech.dat.txt) does not lead to same result/same plot.

        f : array type
             Frequency range in which BBO is sensitive for GWs signal.

        Return
        ---------
        Sh : array type
            power spectral density for BBO GW detector.

        """
        f = np.linspace(start=10 ** -3, stop=10 ** 2, num=10000)
        Sh = 2. * 10**-49 * f**2 + 4.58 * 10**-49 + 1.26 * 10**-51 * f**-4

        return f, Sh

    def TianQin(self):
        """
        Class. Quantum Grav. 33 (2016) 035010 ;
        Class. Quantum Grav. 35 (2018) 095008 ;
        arXiv:1510.04754v1 [gr-qc] and
        PHYSICAL REVIEW D 100, 043003 (2019)

        f : array
            frequecy range for the TianQin GW detector. 0.1 -10 mHz

        Sx : int
            Position noise power density. (Accuracy requirements for the postion of TianQin mission)
        Sa : int
            residual acceleration noise power density.
            (Accuracy requirements for the residual acceleration measurements of TianQin mission)

        w : int
            w is the angular orbital frequency of the source.

        tau : int
            Is the light travel time for a TianQin arm length.
        g
        L0 : int
            Arm Length of TianQin GW detector
        R :
            Is the transfer function

        Return
        --------------
        Sn : array like
            power spectral density for TianQin GW detector.

        """

        f = np.linspace(start=10**-4, stop=10**1, num=10000)

        L0 = 10**8  ## units m
        delta = 38  ## degrees
        tau = L0 / self.speed_of_light
        Sa = (1 * 10**-15)**2   ## units m**2 s**-4 Hz**-1
        Sx = (1 * 10**-12)**2    ## units m**2 Hz**-1

        # ## For LISA like configuration, transfer function is
        w = 2 * np.pi * f

        ## From PRD 100, 043003 (2019)
        x = w * tau
        gx = 1
        SN = (1 / L0**2) * ((4 * Sa)/(w**4) * (1 + (10**-4) / f) + Sx)
        R = 3 / 10  * gx / (1 + 0.6 * x**2)
        Sn = SN / R


        ## for w = 2*np.pi*f = 0 , use
        # R = 8/15
        ## when w is not zero, use
        # R = (8 / 15) * (1 + ((2 * np.pi * f * L0) / (0.41 * np.pi)) ** 2) ** -1
        ## when independ of w, use
        # R = 3 - 2.6 * np.sin(delta) ** 2
        # hf = (2 / np.sqrt(R)) * ((Sx / L0 ** 2) + ((Sa / ((2 * np.pi * f) ** 4 * L0 ** 2)) * (1 + 10 ** -4 / f))) ** (
        #             1 / 2)


        # w = 0
        # R = 8 / 15 * (1 + ((w * L0) / (0.41 * np.pi)) ** 2) ** -1
        # Sh = 2 /(np.sqrt(R * (2 * np.pi * f))) * ((Sx / L0**2) + (Sa / ((2 * np.pi * f)**4 * L0**2)) * (1 + (10**-4 / f)))**(1/2)

        return f, Sn

    def sensitivity_curve_from_interpolation(self, psd=None, frequency=None):
        """
        Plot  sensitivity curve for all Ground based GW detectors from bilby using interpolation function.

        psd : array like
             Power Spectral Density for all given ifos

        :return:
        """
        ## Plot Sensitivity Curve for each ifos
        psd_curve = plots_PSD.plot_detector_psd(psd=psd, frequency=frequency)

    def sensitivity_curve_from_files(self):
        """
        Plot  sensitivity curve for all Ground based GW detectors from given PSD files.

        :return:
        """

        """ PSD from psd files """
        # ## Sensitivity Curves from PSD files
        aLIGO_psd = np.loadtxt('./Sensitivity_Curve/aLIGO_ZERO_DET_high_P_psd.txt')
        # frequecy = aLIGO_psd[:,0]
        # psd = np.sqrt(aLIGO_psd[:,1])
        AdV_psd = np.loadtxt('./Sensitivity_Curve/AdV_psd.txt')
        KAGRA_design_psd = np.loadtxt('./Sensitivity_Curve/Kagra_Design.txt')
        aplus_LIGO_psd = np.loadtxt('./Sensitivity_Curve/LIGO_Aplusdesign.txt')
        aplus_V_psd = np.loadtxt('./Sensitivity_Curve/AVirgo.txt')
        ET_D_TR_psd = np.loadtxt('./Sensitivity_Curve/ET_D_psd.txt')
        # CE_psd = np.loadtxt('./Sensitivity_Curve/CEsensitivity.txt')
        CE1 = np.loadtxt('./Sensitivity_Curve/CE1_psd.txt')
        CE2 = np.loadtxt('./Sensitivity_Curve/CE2_psd.txt')

        psd_curve = plots_PSD.plot_detector_psd_from_file(aLIGO_psd=aLIGO_psd, AdV_psd=AdV_psd, KAGRA_psd=KAGRA_design_psd,
                                                          aplus_LIGO_psd=aplus_LIGO_psd,aplus_V_psd=aplus_V_psd,
                                                          ET_D_TR_psd=ET_D_TR_psd, CE1_psd=CE1, CE2_psd=CE2)#, CE_psd=CE_psd, )


        # psd_curve = plots_PSD.plot_detector_psd_from_file(aLIGO_psd=aLIGO_psd, aplus_LIGO_psd=aplus_LIGO_psd, AdV_psd=AdV_psd,
        #                                                   KAGRA_design_psd=KAGRA_design_psd)#, CE1_psd=CE1, CE2_psd=CE2)

    def sensitivity_curve_bilby(self):
        """
        Plot  sensitivity curve for all Ground based GW detectors from given bilby PSD files.

        :return:
        """
        """ PSD from bilby psd files """
        # ## Sensitivity Curves from PSD files from bilby
        aLIGO_early_high = np.loadtxt('./noise_curves/aLIGO_ZERO_DET_high_P_psd.txt')
        aLIGO_early = np.loadtxt('./noise_curves/aLIGO_early_psd.txt')
        aLIGO_mid = np.loadtxt('./noise_curves/aLIGO_mid_psd.txt')
        aLIGO_late = np.loadtxt('./noise_curves/aLIGO_late_psd.txt')
        aLIGO = np.loadtxt('./noise_curves/aLIGO_ZERO_DET_high_P_psd.txt')
        aplus = np.loadtxt('./noise_curves/aplus.txt')
        aVIRGO = np.loadtxt('./noise_curves/AdV_psd.txt')
        kagra = np.loadtxt('./noise_curves/KAGRA_design_psd.txt')
        ligo_srd = np.loadtxt('./noise_curves/LIGO_srd_psd.txt')
        ET = np.loadtxt('./noise_curves/ET_D_psd.txt')
        CE = np.loadtxt('./noise_curves/CE_psd.txt')
        CE1 = np.loadtxt('./noise_curves/CE1_2030.txt')
        CE2 = np.loadtxt('./noise_curves/CE2_2040.txt')

        plot_psd = plots_PSD.plot_detector_psd_bilby_file(aLIGO_early_high=aLIGO_early_high, aLIGO_early=aLIGO_early,aLIGO_mid=aLIGO_mid,
                                        aLIGO_late=aLIGO_late, aLIGO=aLIGO, aVIRGO=aVIRGO, kagra=kagra, ligo_srd=ligo_srd, ET=ET, CE=CE, CE1=CE1,CE2=CE2)


    def sensitivity_space_based(self):
        """
        :param BBO_psd:
        :param LISA_psd:
        :param DECIGO_psd:
        :param BBO1_psd:
        :param TianQin_psd:
        :return:
        """

        LISA_psd = self.LISA()
        TianQin_psd = self.TianQin()
        DECIGO_psd = self.DECIGO()
        BBO_psd = np.loadtxt('./Sensitivity_Curve/BBOtech.dat.txt')
        BBO1_psd = self.BBO()

        plot_psd = plots_PSD.plot_detector_spaced_based(LISA_psd=LISA_psd, TianQin_psd=TianQin_psd,DECIGO_psd=DECIGO_psd,BBO_psd=BBO_psd,BBO1_psd=BBO1_psd)

    def sensitivity_curve_all(self,PGWB = None, cosmo_spectrum=None):
        """
        Plot the sensitivity curve for all Space based and ground based detectors.

        :return:
        """

        """ PSD from psd files """
        ## Sensitivity Curves from PSD files
        aLIGO_psd = np.loadtxt('./Sensitivity_Curve/aLIGO_ZERO_DET_high_P_psd.txt')
        # frequecy = aLIGO_psd[:,0]
        # psd =aLIGO_psd[:,1]
        AdV_psd = np.loadtxt('./Sensitivity_Curve/AdV_psd.txt')
        KAGRA_design_psd = np.loadtxt('./Sensitivity_Curve/Kagra_Design.txt')
        aplus_LIGO_psd = np.loadtxt('./Sensitivity_Curve/LIGO_Aplusdesign.txt')
        aplus_V_psd = np.loadtxt('./Sensitivity_Curve/AVirgo.txt')
        ET_D_TR_psd = np.loadtxt('./Sensitivity_Curve/ET_D_psd.txt')
        CE_psd = np.loadtxt('./Sensitivity_Curve/CEsensitivity.txt')
        BBO_psd = np.loadtxt('./Sensitivity_Curve/BBOtech.dat.txt')
        LISA_psd = self.LISA()
        DECIGO_psd =self.DECIGO()
        BBO1_psd = self.BBO()
        TianQin_psd = self.TianQin()
        PGWB = PGWB
        cosmo_spectrum = cosmo_spectrum

        # psd_curve = plots_PSD.plot_detector_psd_All(aLIGO_psd=aLIGO_psd, AdV_psd=AdV_psd, KAGRA_psd=KAGRA_design_psd,
        #                          ET_D_TR_psd=ET_D_TR_psd, CE_psd=CE_psd, BBO_psd=BBO_psd,
        #                         LISA_psd=LISA_psd, DECIGO_psd=DECIGO_psd, BBO1_psd=BBO1_psd, TianQin_psd=TianQin_psd,PGWB=PGWB)

        psd_curve = plots_PSD.plot_detector_psd_All(ET_D_TR_psd=ET_D_TR_psd, CE_psd=CE_psd, BBO_psd=BBO_psd,
                                                    LISA_psd=LISA_psd, DECIGO_psd=DECIGO_psd, BBO1_psd=BBO1_psd,
                                                    TianQin_psd=TianQin_psd, PGWB=PGWB)



    def sensitivity_curve_IFOs(self, aLIGO=None, AdV=None, KAGRA=None, aplus_LIGO=None, aplus_V=None, ET_D_TR=None, CE=None, CE1=None, CE2=None):
        """
        Plot  sensitivity curve for all Ground based GW detectors from given PSD files.

        :return:
        """

        """ PSD from psd files """
        # ## Sensitivity Curves from PSD files
        aLIGO_psd = np.loadtxt(aLIGO,dtype=np.float)
        print(aLIGO_psd)
        # frequecy = aLIGO_psd[:,0]
        # psd =aLIGO_psd[:,1]
        AdV_psd = np.loadtxt(AdV)
        KAGRA_psd = np.loadtxt(KARGRA)
        aplus_LIGO_psd = np.loadtxt(aplus_LIGO)
        aplus_V_psd = np.loadtxt(aplus_V)
        ET_D_TR_psd = np.loadtxt(ET_D_TR)
        CE_psd  = np.loadtxt(CE)

        # CE1_psd = np.loadtxt(CE1)
        # CE2_psd = np.loadtxt(CE2)
        # print(CE1)
        # print(CE2)

        psd_curve = plots_PSD.plot_detector_psd_from_file(aLIGO_psd=aLIGO_psd, AdV_psd=AdV_psd, KAGRA_psd=KAGRA_psd,
                                            aplus_LIGO_psd=aplus_LIGO_psd, aplus_V_psd = aplus_V_psd, ET_D_TR_psd=ET_D_TR_psd, CE_psd=CE_psd)#, CE1_psd =CE1_psd, CE2_psd=CE2_psd)

        # psd_curve = plots_PSD.plot_detector_psd_from_file(aLIGO_psd=aLIGO_psd, aplus_V_psd=aLIGO_psd,
        #                                                  CE1_psd=CE1_psd,
        #                                                   CE2_psd=CE2_psd)

    # def sensitivity_curve_detector(self, detector=None):
    #     """
    #     Plot  sensitivity curve for all Ground based GW detectors from given PSD files.
    #
    #     :return:
    #     """
    #
    #     """ PSD from psd files """
    #     # ## Sensitivity Curves from PSD files
    #     detector_psd = np.loadtxt(detector)
    #     # frequecy = aLIGO_psd[:,0]
    #     # psd =aLIGO_psd[:,1]
    #
    #     return detector_psd

    def strain2omega(self, frequency, strain_detector, Obstime, ref_freq):
        """
        GW energy density parameter Ω_GW is from

        https://arxiv.org/pdf/1801.04268.pdf    Check Eq. 90 and 149
        https://arxiv.org/abs/1911.09745
        https://arxiv.org/pdf/2002.04615.pdf

        Sh : ndim array
            Given Interferometer Strain Sensitivity.
        F : ndim array
            Antenna Pattern function Over all direaction of arrival of the GW wave.
        alpha : int
            Angle between the two arms of a GW detector.

        frequency: array
            frequecy range for the given GW detector.

        strain_detector : array
            power spectral density for given GW detector.

        Obstime: int/float
            Observation time period for the given detector

        ref_frequency : int

            reference frequency for the given GW detector.

         Ω_GW = (10 * np.pi**2 / (3 * H0**2)) * self.frequency **3 * Sh
         ## Note :  use sh**2 if given strain_detector is already a square root quantity.

        :return:
        """
        # self.one_pc = 3.0856775814671916 * 10**16  ## units = meters
        # one_pc2sec = 1.029 * 10**8  ## units in seconds
        # h0 = 0.6766
        # H0 = 67.9 * 10**3 * 10**-6 * one_pc**-1  ## units = 1/sec

        # F = 2/5 * np.sin(alpha)**2
        # Omega_IFOs = 10**-2 / F * (self.frequency/10)**3 * (Sh/(10**-22))**2

        ## Integrating Over the Observation time of the detector.
        int_time = np.sqrt(Obstime * ref_freq)
        Omega_IFOs = (10 * np.pi**2 * frequency**3 * strain_detector**2)/(3* self.H0**2)/int_time

        return frequency, Omega_IFOs

    def sensitivity_and_omega(self, PGWB, cosmo_spectrum):
        """
        Plot  sensitivity curve for all Ground based GW detectors from given PSD files.

        :return:
        """

        """ PSD from psd files """

        ET_D_TR_psd = np.loadtxt('./Sensitivity_Curve/ET_D_psd.txt')
        CE_psd = np.loadtxt('./Sensitivity_Curve/CEsensitivity.txt')

        # psd_curve = plots_PSD.plot_detector_psd_from_file(ET_D_TR_psd=ET_D_TR_psd, CE_psd=CE_psd, PGWB, cosmo_spectrum)
