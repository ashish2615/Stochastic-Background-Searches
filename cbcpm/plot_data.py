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

from bilby.core.utils import speed_of_light

import scipy
from scipy import signal, fftpack
from scipy.fftpack import fft, rfft,ifft,irfft, fftfreq, rfftfreq
from scipy.signal import (periodogram, welch, lombscargle, csd, coherence,
                          spectrogram)
from scipy.signal import welch
from scipy.signal import *

import matplotlib
#matplotlib.use('tkagg')
#matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.font_manager as font_manager

current_direc = os.getcwd()
# print("Current Working Directory is :", current_direc)
## Specify the output directory and the name of the simulation.
outmain = 'Output'
outdir = 'Plot_PSD'
label = 'Plot_PSD'
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

class PSDWelch():

    def __init__(self,IFOs):
        self.IFOs = IFOs
        self.n_det = len(self.IFOs)

        pass

    def plot_detector_psd(self, psd=None, frequency=None):
        """
        Sensitivity cruve for ground based GW detectors from bilby.
        we used bilby to interpolate the psd for each detector.

        psd : array like
            Power Spectral Density Calculated by using bilby.
        frequency : array like
            Frequecy range for each detector.
        """

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='small')

        plt.loglog(frequency, np.sqrt(psd[0, :]), ':', label='aLIGO')
        # plt.loglog(frequency, np.sqrt(psd[1, :]), label='H1')
        plt.loglog(frequency, np.sqrt(psd[2, :]), '--', label='AdV')
        plt.loglog(frequency, np.sqrt(psd[3, :]), '-.', label='kAGRA')
        plt.loglog(frequency, np.sqrt(psd[4, :]), label='ET-D')
        # plt.loglog(frequency, np.sqrt(psd[5, :]), label='ET_D_TR_2')
        # plt.loglog(frequency, np.sqrt(psd[6, :]), label='ET_D_TR_3')
        plt.loglog(frequency, np.sqrt(psd[7, :]), label='CE1')

        legend = plt.legend(loc='upper right', prop=font1)
        # plt.xscale('log')
        plt.xlim(2, 1000)
        plt.ylim(10**-25, 10**-18)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)
        # plt.tick_params(axis='both', direction='in')
        # plt.title(r'Sensitivity Curve for GW Detectors', fontdict=font)
        plt.tight_layout()
        plt.savefig('./Sensitivity_Curve/Sensitivity_Curve', dpi=300)
        plt.close()

    def plot_detector_psd_from_file(self, aLIGO_psd=None, AdV_psd=None, KAGRA_psd=None, aplus_LIGO_psd=None,
                                    aplus_V_psd=None, ET_D_TR_psd=None, CE_psd=None, CE1_psd=None, CE2_psd=None):
        """
        Sensitivity curve for ground based detectors from given PSD files.

        #detector_psd : array like
            PSD files for each given ground based GW detector. Contains frequency and corresponding PSD values.

        """

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='small')
        # plt.figure(figsize=(10, 8))
        plt.loglog(aLIGO_psd[:, 0], np.sqrt(aLIGO_psd[:, 1]), ':', label='aLIGO')
        plt.loglog(AdV_psd[:, 0], np.sqrt(AdV_psd[:, 1]), '--', label='AdV')
        plt.loglog(KAGRA_psd[:, 0], (KAGRA_psd[:, 1]), '-.', label='kAGRA')
        # plt.loglog(aplus_LIGO_psd[:, 0], (aplus_LIGO_psd[:, 1]), label='$A^{+}$ LIGO')
        # plt.loglog(aplus_V_psd[:, 0], (aplus_V_psd[:, 1]), label='$A^{+}$ VIRGO')
        plt.loglog(ET_D_TR_psd[:, 0], np.sqrt(ET_D_TR_psd[:, 1]), label='ET-D')
        # plt.loglog(CE_psd[:, 0], (CE_psd[:, 1]), label='CE')
        plt.loglog(CE1_psd[:, 0], (CE1_psd[:, 1]), label='CE1')
        plt.loglog(CE2_psd[:, 0], (CE2_psd[:, 1]), label='CE2')

        legend = plt.legend(loc='upper right', prop=font1)
        # plt.grid(True, which="majorminor", ls="-", color='0.5')
        # plt.xscale('log')
        plt.xlim(1, 2 * 1000)
        plt.ylim(10**-25, 10**-18)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'Strain  [1/$\sqrt{\rm Hz}$] ', fontdict=font)
        # plt.tick_params(axis='both', direction='in')
        # plt.title(r'Sensitivity Curve for GW Detectors', fontdict=font)
        plt.tight_layout()
        plt.savefig('./Sensitivity_Curve/Sensitivity_Curve_PSDs', dpi=300)
        plt.close()

    def plot_detector_psd_bilby_file(self, aLIGO_early_high=None, aLIGO_early=None, aLIGO_mid=None, aLIGO_late=None,
                                     aLIGO=None, aVIRGO=None, kagra=None, aplus=None, ligo_srd=None, ET=None, CE=None, CE1=None,CE2=None):
        """
        Sensitivity curve for ground based detectors from given PSD files in bilby.

        #detector_psd : array like
            PSD files for each given ground based GW detector. Contains frequency and corresponding PSD values.

        """

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='small')

        # plt.loglog(aLIGO_early_high[:,0], np.sqrt(aLIGO_early_high[:,1]), ':', label='aLIGO_early_high')
        # plt.loglog(aLIGO_early[:, 0], np.sqrt(aLIGO_early[:, 1]), '--', label='aLIGO_early')
        # plt.loglog(aLIGO_mid[:, 0], np.sqrt(aLIGO_mid[:, 1]), '-.', label='aLIGO_mid')
        # plt.loglog(aLIGO_late[:, 0], np.sqrt(aLIGO_late[:, 1]), label='aLIGO_late')
        plt.loglog(aLIGO[:, 0], np.sqrt(aLIGO[:, 1]), ':', label='aLIGO')
        plt.loglog(aVIRGO[:, 0], np.sqrt(aVIRGO[:, 1]), '--', label='aVIRGO')
        plt.loglog(kagra[:, 0], np.sqrt(kagra[:, 1]),'-.', label='kAGRA')
        # plt.loglog(aplus[:, 0], (aplus[:, 1]), label='$A^{+}$')
        # plt.loglog(ligo_srd[:, 0], np.sqrt(ligo_srd[:, 1]), label='ligo_srd')
        plt.loglog(ET[:, 0], np.sqrt(ET[:, 1]), label='ET-D')
        plt.loglog(CE[:, 0], np.sqrt(CE[:, 1]), label='CE1')

        # plt.loglog(CE1[:, 0], (CE1[:, 1]), label='CE1')
        # plt.loglog(CE2[:, 0], (CE2[:, 1]), label='CE2')

        legend = plt.legend(loc='upper right', prop=font1)
        # plt.xscale('log')
        plt.xlim(1, 2 * 1000)
        plt.ylim(10**-25, 10**-18)
        plt.xlabel(r' Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r' Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)
        # plt.tick_params(axis='both', direction='in')
        # plt.title(r'Sensitivity Curve for GW Detectors Bilby', fontdict=font)
        plt.tight_layout()
        plt.savefig('./Sensitivity_Curve/Sensitivity_PSD_Curve_Bilby', dpi=300)
        plt.close()

    def plot_detector_spaced_based(self, LISA_psd=None, TianQin_psd=None, DECIGO_psd=None, BBO_psd=None, BBO1_psd=None):

        """
        Sensitivity curve for Space based GW detectors.

        LISA_psd : array like
            PSD (see Sensitivity curve SensitivityCurves.LISA)
        TianQin_psd: array like
            PSD   (see Sensitivity curve SensitivityCurves.TianQin)
        DECIGO_psd: array like
            PSD (see Sensitivity curve SensitivityCurves.DECIGO)
        BBO_psd: array like
            PSD from given PSD file
        BBO1_psd: array like
            PSD (see Sensitivity curve SensitivityCurves.BBO)

        """
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='small')

        plt.loglog(LISA_psd[0], np.sqrt(LISA_psd[1]), label='LISA')
        plt.loglog(TianQin_psd[0], np.sqrt(TianQin_psd[1]), label='TianQin')
        plt.loglog(DECIGO_psd[0], np.sqrt(DECIGO_psd[1]), label='DECIGO')
        ## BBO PSD has been multiplied by sqrt(frequecy)
        plt.loglog(BBO_psd[:, 0], np.sqrt(BBO_psd[:,0]) * (BBO_psd[:, 1]), label='BBO')
        plt.loglog(BBO1_psd[0], np.sqrt(BBO1_psd[1]), label='BBO1')

        legend = plt.legend(loc='upper right', prop=font1)
        # plt.grid(True, which="majorminor", ls="-", color='0.5')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlim(10 ** -5, 6 * 10**2)
        plt.ylim(10 ** -25, 10 ** -14)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)
        # plt.title(r'Sensitivity Curve for GW Detectors', fontdict=font)
        plt.tight_layout()
        plt.savefig('./Sensitivity_Curve/Sensitivity_PSD_Curve_Spaced', dpi=300)
        plt.close()

    def plot_detector_psd_All(self, aLIGO_psd=None, AdV_psd=None, KAGRA_psd=None, aplus_LIGO_psd=None, aplus_V_psd=None,
                              ET_D_TR_psd=None, CE_psd=None, BBO_psd=None, LISA_psd=None, DECIGO_psd=None,
                              BBO1_psd=None, TianQin_psd=None,PGWB=None, cosmo_spectrum=None):
        """
        Sensitivity curve for All Ground and Space based GW detectors.

        """

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 20}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size=12)
        # plt.rcParams["figure.figsize"] = (13,6)
        plt.figure(figsize=(12, 8))

        # plt.loglog(aLIGO_psd[:, 0], np.sqrt(aLIGO_psd[:, 1]), ':', label='aLIGO')
        # plt.loglog(AdV_psd[:, 0], np.sqrt(AdV_psd[:, 1]), '--', label='AdV')
        # plt.loglog(KAGRA_psd[:, 0], (KAGRA_psd[:, 1]), '-.', label='kAGRA')
        # plt.loglog(aplus_LIGO_psd[:, 0], (aplus_LIGO_psd[:, 1]), label='$A^{+}$ LIGO')
        # plt.loglog(aplus_V_psd[:, 0], (aplus_V_psd[:, 1]), label='$A^{+}$ VIRGO')
        plt.loglog(ET_D_TR_psd[:, 0], np.sqrt(ET_D_TR_psd[:, 1]), label='ET-D')
        plt.loglog(CE_psd[:, 0], (CE_psd[:, 1]), label='CE1')

        ## BBO PSD has been multiplied by sqrt(frequecy)
        # plt.loglog(BBO_psd[:, 0], np.sqrt(BBO_psd[:, 0]) * (BBO_psd[:, 1]), label='BBO')
        plt.loglog(DECIGO_psd[0], np.sqrt(DECIGO_psd[1]), label='DECIGO')
        plt.loglog(BBO1_psd[0], np.sqrt(BBO1_psd[1]), label='BBO1')

        plt.loglog(LISA_psd[0], np.sqrt(LISA_psd[1]), label='LISA')
        plt.loglog(TianQin_psd[0], np.sqrt(TianQin_psd[1]), label='TianQin')

        # plt.loglog(PGWB[0], PGWB[1], '--' , label='Primordial GWB')
        # plt.text(0.5, 1*10**-26, 'Primordial GWB', horizontalalignment='right',color='green',fontsize=16, weight='bold')

        # frequency = np.logspace(-18, 4, num=8193)
        # plt.loglog(frequency, cosmo_spectrum, label='$\Omega_{GW}^{cosmo}$')

        legend = plt.legend(loc='upper right', prop=font1)
        # plt.grid(True, which="majorminor", ls="-", color='0.5')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlim(10 ** -4,  10000)
        plt.ylim(10 ** -26, 10 ** -14)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)

        # plt.rcParams["figure.figsize"] = (20,20)
        # plt.figure(figsize=(1, 1))
        # plt.tick_params(axis='both', direction='in')
        # plt.title(r'Sensitivity Curve for Space and Ground based GW Detectors', fontdict=font)
        plt.tight_layout()
        plt.savefig('./Sensitivity_Curve/Sensitivity_PSD_NASA', dpi=300)
        plt.close()

    def plot_strain2omega(self,aLIGO=None, AdV=None, KAGRA=None, aplus_LIGO=None, aplus_V=None, ET_D_TR=None, CE=None,
                          BBO=None, LISA=None, DECIGO=None, BBO1=None, TianQin=None, cosmo_spectrum=None):
        """
        Sensitivity curve for ground and spaced based GW detectors towards the the Omega_gw calculated from detector strain of each detector.
        Check Sensitivity_curve  SensitivityCurve.strain2omega()

        """

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 14}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size=10)
        # plt.rcParams["figure.figsize"] = (13,6)
        # plt.figure(figsize=(12, 8))

        plt.loglog(aLIGO[0], (aLIGO[1]), ':', label='aLIGO')
        plt.loglog(AdV[0], (AdV[1]), '--', label='AdV')
        plt.loglog(KAGRA[0], (KAGRA[1]), '-.', label='kAGRA')
        # plt.loglog(aplus_LIGO[0], (aplus_LIGO[1]), label='$A^{+}$ LIGO')
        # plt.loglog(aplus_V[0], (aplus_V[1]), label='$A^{+}$ VIRGO')
        plt.loglog(ET_D_TR[0], ET_D_TR[1], label='ET-D')
        plt.loglog(CE[0], (CE[1]), label='CE')

        ## BBO PSD has been multiplied by sqrt(frequecy)
        # plt.loglog(BBO[0], np.sqrt(BBO[0])* (BBO[1]), label='BBO')
        # plt.loglog(DECIGO[0], (DECIGO[1]), label='DECIGO')
        # plt.loglog(BBO1[0], (BBO1[1]), label='BBO1')
        #
        # plt.loglog(LISA[0], (LISA[1]), label='LISA')
        # plt.loglog(TianQin[0], (TianQin[1]), label='TianQin')

        # frequency = np.logspace(-18, 4, num=8193)
        # plt.loglog(frequency, cosmo_spectrum, label='$\Omega_{GW}^{cosmo}$')

        legend = plt.legend(loc='lower right', prop=font1)
        # plt.grid(True, which="majorminor", ls="-", color='0.5')
        # plt.xscale('log')
        # plt.yscale('log')

        ##Ground Based Limit
        plt.xlim(1, 2 * 1000)
        plt.ylim(10 ** -14, 10 ** -1)
        ## Space based limit
        # plt.xlim(10**-4, 100)
        # plt.ylim(10 ** -20, 10 ** -1)
        ## X and Y limit for All
        # plt.xlim(10**-5, 4 * 1000)
        # plt.ylim(10 ** -20, 10 ** -1)

        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'$\Omega_{GW}(f)$ ', fontdict=font)
        plt.tick_params(axis='both', direction='in')

        # plt.rcParams["figure.figsize"] = (20,20)
        # plt.figure(figsize=(1, 1))
        plt.tick_params(axis='both', direction='in')
        # plt.title(r'Strain to Omega', fontdict=font)
        plt.tight_layout()
        plt.savefig('./Sensitivity_Curve/Strain_to_Omega_Ground', dpi=300)
        plt.close()

    def plot_data(self, IFOs, sampling_frequency=None, n_seg =None, inj_time_series = None, sub_time_series=None, proj_time_series=None, frequency=None, nperseg=None,Tfft=None):
        """

        Detector strain after Injection, Subtraction and Projection method.
        We used scipy.signal.welch method to compute the PSD for each method and for each detector.

        Welch method computes an estimate of the power spectral density by dividing the data into overlapping segments,
        computing a modified periodogram for each segment and averaging the periodograms.

        Further we compared between PSD, using welch method,to comapre the difference in PSD after the implementation of
        Injection, Subtraction and Projection method.

        IFOs: Initialization of GW interferometer.
           Generates an Interferometer with a power spectral density.
        sampling_frequency: float
            The sampling frequency (in Hz).
        n_seg: int
            number of time segment in total time duration of operation of detector.
        inj_time_series : array_like
            time_domain_strain for injection signals in units of strain.
        sub_time_series: array like
            Real valued array of time series calculated by taking inverse fft of residual_noise_data.
        proj_data_stream: array like
            Real valued array of time series calculated by taking inverse fft of residual_noise_data.
        frequency: array_like
            Real FFT of sample frequencies.
        Tfft: int
            Duration of FFT (in sec.).

        nperseg: int
            Length of each segment.
        fft_seg: int
            Length of the FFT used.
        Note: check scipy.signal.welch for to use default values.

        Return:

        *** = inj, sub, proj

        freq_*** : ndarray
            Array of sample frequencies.
        ***_welch : ndarray
            Power spectral density or power spectrum of ***_time_series.

        """

        for detector in range(len(IFOs)):

            font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
            font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

            nperseg = sampling_frequency * Tfft

            if detector == 0:
                ifo = 'CE'
                label = 'CE After Injections'
                label1 = 'CE After Subtraction'
                label2 = 'CE After Projection'

            if detector ==1:
                ifo = 'ET-D_1'
                label = 'ET-D_1 After Injections'
                label1 = 'ET-D_1 After Subtraction'
                label2 = 'ET-D_1 After Projection'

            if detector ==2:
                ifo = 'ET-D_2'
                label = 'ET-D_2 After Injections'
                label1 = 'ET-D_2 After Subtraction'
                label2 = 'ET-D_2 After Projection'

            if detector ==3:
                ifo = 'ET-D_3'
                label = 'ET-D_3 After Injections'
                label1 = 'ET-D_3 After Subtraction'
                label2 = 'ET-D_3 After Projection'

            inj_freq, inj_welch = scipy.signal.welch(inj_time_series[detector, :], fs = sampling_frequency, nperseg=nperseg)
            # print('inj_welch',inj_welch)
            sub_freq, sub_welch = scipy.signal.welch(sub_time_series[detector, :], fs = sampling_frequency, nperseg=nperseg)
            # print('sub_welch',sub_welch)
            proj_freq, proj_welch =  scipy.signal.welch(proj_time_series[detector, :], fs = sampling_frequency, nperseg=nperseg)
            # print('proj_welch',proj_welch)

            plt.subplot(3, 1, 1)
            plt.semilogy(inj_freq, np.sqrt(inj_welch), label=label)
            legend = plt.legend(loc='lower left', prop=font1)
            plt.xscale('log')
            plt.xlim(10, 1000)
            # plt.ylim(10 ** -20, 10 ** -32)
            # plt.autoscale(enable=True, axis='y', tight=False)
            # plt.xlabel(r'f (Hz)')
            # plt.ylabel(r'PSD_Welch(f)')
            #plt.title(r' PSD Spectrum for ' + ifo, fontdict=font)

            plt.subplot(3, 1, 2)
            plt.semilogy(sub_freq, np.sqrt(sub_welch), label=label1)
            legend = plt.legend(loc='lower left', prop=font1)
            plt.xscale('log')
            plt.xlim(10, 1000)
            # plt.ylim(10 ** -20, 10 ** -32)
            # plt.autoscale(enable=True, axis='y', tight=False)
            # plt.xlabel(r'f (Hz)')
            plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)

            plt.subplot(3, 1, 3)
            plt.semilogy(proj_freq, np.sqrt(proj_welch), label=label2)
            legend = plt.legend(loc='lower left', prop=font1)
            plt.xscale('log')
            plt.xlim(10, 1000)
            # plt.ylim(10**-20, 10**-32)
            # plt.autoscale(enable=True, axis='y', tight=False)
            plt.xlabel(r'Frequency $~\rm[Hz]$',fontdict=font)
            plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.5)
            # plt.ylabel(r'PSD_Welch(f)')
            plt.savefig(outdir+'/PSD_Welch_'+ifo+'_'+str(n_seg)+'.png', dpi=300)
            # plt.savefig('./PSD_without Noise/PSD_Welch_{}_{}_{}_{}'.format(ifo, str(n_seg), detector, nperseg)) #+ifo+str(int(n_seg))+'_'+str(detector)+str(nperseg)
            plt.close()


            ###################
            # PSD Comparision #
            ###################

            plt.loglog(inj_freq, np.sqrt(inj_welch), 'r-',  label=label)
            plt.loglog(sub_freq, np.sqrt(sub_welch), 'b-', label=label1)
            plt.loglog(proj_freq, np.sqrt(proj_welch), 'g-', label=label2)
            legend = plt.legend(loc='best', prop=font1)
            # plt.xscale('log')
            plt.xlim(1, 1000)
            # plt.ylim(10 ** -20, 10 ** -34)
            # plt.autoscale(enable=True, axis='y', tight=False)
            plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
            plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)
            plt.tight_layout()
            #plt.title(r' PSD Spectrum for ' + ifo, fontdict=font)
            plt.savefig(outdir+'/PSD_Welch_comparision_'+ifo+'_'+str(n_seg)+'.png',dpi=300)
            plt.close()

    def plot_wunresolved_data(self, IFOs, sampling_frequency=None, n_seg =None, inj_time_series = None, sub_time_series=None,
                              unresolved_time_series =None, proj_time_series=None, frequency=None, nperseg=None,Tfft=None,
                              SNR_cut=None):
        """
        For details of variable check last function

        unresolved_time_series: array like
            Time series data of unresolved binary signals.
        SNR_cut: int
            SNR value below which binary signals are unresolvable.
        """

        for detector in range(len(IFOs)):

            font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
            font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

            nperseg = sampling_frequency * Tfft

            if detector == 0:
                ifo = 'CE'
                label = 'CE After Injections'
                label1 = 'CE After Subtracting resolved binaries upto SNR > '+str(SNR_cut)
                label2 = 'CE After Projecting out residuals of resolved binaries'
                label3 = 'CE Spectrum of Unresolved binaries SNR < '+str(SNR_cut)

            if detector ==1:
                ifo = 'ET-D_1'
                label = 'ET-D_1 After Injections'
                label1 = 'ET-D_1 After Subtracting resolved binaries upto SNR > '+str(SNR_cut)
                label2 = 'ET-D_1 Projecting out residuals of resolved binaries'
                label3 = 'ET-D_1 Spectrum of unresolved binaries below SNR < ' + str(SNR_cut)

            if detector ==2:
                ifo = 'ET-D_2'
                label = 'ET-D_2 After Injections'
                label1 = 'ET-D_2 After Subtracting resolved binaries upto SNR > '+str(SNR_cut)
                label2 = 'ET-D_2 After Projecting out residuals of resolved binaries'
                label3 = 'ET-D_2 Spectrum of unresolved binaries below SNR < ' + str(SNR_cut)

            if detector ==3:
                ifo = 'ET-D_3'
                label = 'ET-D_3 After Injections'
                label1 = 'ET-D_3 After Subtracting resolved binaries upto SNR > '+str(SNR_cut)
                label2 = 'ET-D_3 After Projecting out residuals of resolved binaries'
                label3 = 'ET-D_3 Spectrum of unresolved binaries below SNR < ' + str(SNR_cut)

            inj_freq, inj_welch = scipy.signal.welch(inj_time_series[detector, :], fs = sampling_frequency, nperseg=nperseg)
            # print('inj_welch',inj_welch)
            sub_freq, sub_welch = scipy.signal.welch(sub_time_series[detector, :], fs = sampling_frequency, nperseg=nperseg)
            # print('sub_welch',sub_welch)
            unresolved_freq, unresolved_welch = scipy.signal.welch(unresolved_time_series[detector, :],fs=sampling_frequency, nperseg=nperseg)
            #print('unresolved_welch',unresolved_welch)
            proj_freq, proj_welch =  scipy.signal.welch(proj_time_series[detector, :], fs = sampling_frequency, nperseg=nperseg)
            # print('proj_welch',proj_welch)

            ###################
            # PSD Comparision #
            ###################

            plt.loglog(inj_freq, np.sqrt(inj_welch), 'r-',  label=label)
            plt.loglog(sub_freq, np.sqrt(sub_welch), 'b-', label=label1)
            plt.loglog(proj_freq, np.sqrt(proj_welch), 'g-', label=label2)
            plt.loglog(unresolved_freq, np.sqrt(unresolved_welch),'C1',label=label3)
            legend = plt.legend(loc='best', prop=font1)
            # plt.xscale('log')
            plt.xlim(1, 1000)
            # plt.ylim(10 ** -20, 10 ** -34)
            # plt.autoscale(enable=True, axis='y', tight=False)
            plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
            plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)
            plt.tight_layout()
            #plt.title(r' PSD Spectrum for ' + ifo, fontdict=font)
            plt.savefig(outdir+'/PSD_Welch_comparision_'+ifo+'_'+str(n_seg)+'.png',dpi=300)
            plt.close()

    def plot_data_SNR(self, IFOs, sampling_frequency=None, n_seg=None, inj_time_series=None, sub_time_series=None,
                  proj_time_series=None, sub_time_series_SNR=None, proj_time_series_SNR=None, frequency=None, nperseg=None,
                  Tfft=None, SNR_cut=None):
        """

        Detector strain after Injection, Subtraction and Projection method.
        We used scipy.signal.welch method to compute the PSD for each method and for each detector.

        Welch method computes an estimate of the power spectral density by dividing the data into overlapping segments,
        computing a modified periodogram for each segment and averaging the periodograms.

        Further we compared between PSD, using welch method,to comapre the difference in PSD after the implementation of
        Injection, Subtraction and Projection method.

        IFOs: Initialization of GW interferometer.
           Generates an Interferometer with a power spectral density.
        sampling_frequency: float
            The sampling frequency (in Hz).
        n_seg: int
            number of time segment in total time duration of operation of detector.
        inj_time_series : array_like
            time_domain_strain for injection signals in units of strain.
        sub_time_series: array like
            Real valued array of time series calculated by taking inverse fft of residual_noise_data.
        proj_data_stream: array like
            Real valued array of time series calculated by taking inverse fft of residual_noise_data.
        frequency: array_like
            Real FFT of sample frequencies.
        Tfft: int
            Duration of FFT (in sec.).

        nperseg: int
            Length of each segment.
        fft_seg: int
            Length of the FFT used.
        Note: check scipy.signal.welch for to use default values.

        Return:

        *** = inj, sub, proj

        freq_*** : ndarray
            Array of sample frequencies.
        ***_welch : ndarray
            Power spectral density or power spectrum of ***_time_series.

        """

        for detector in range(len(IFOs)):

            font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
            font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

            nperseg = sampling_frequency * Tfft

            if detector == 0:
                ifo = 'CE'
                label = 'CE After Injections'
                label1 = 'CE After Subtraction'
                label2 = 'CE After Projection'
                label3 = 'CE After Subtraction SNR > '+str(SNR_cut)
                label4 = 'CE After Projection SNR > '+str(SNR_cut)

            if detector == 1:
                ifo = 'ET-D_1'
                label = 'ET-D_1 After Injections'
                label1 = 'ET-D_1 After Subtraction'
                label2 = 'ET-D_1 After Projection'
                label3 = 'ET-D_1 After Subtraction SNR > '+str(SNR_cut)
                label4 = 'ET-D_1 After Projection SNR > '+str(SNR_cut)

            if detector == 2:
                ifo = 'ET-D_2'
                label = 'ET-D_2 After Injections'
                label1 = 'ET-D_2 After Subtraction'
                label2 = 'ET-D_2 After Projection'
                label3 = 'ET-D_2 After Subtraction SNR > '+str(SNR_cut)
                label4 = 'ET-D_2 After Projection SNR > '+str(SNR_cut)

            if detector == 3:
                ifo = 'ET-D_3'
                label = 'ET-D_3 After Injections'
                label1 = 'ET-D_3 After Subtraction'
                label2 = 'ET-D_3 After Projection'
                label3 = 'ET-D_3 After Subtraction SNR > '+str(SNR_cut)
                label4 = 'ET-D_3 After Projection SNR > '+str(SNR_cut)

            inj_freq, inj_welch = scipy.signal.welch(inj_time_series[detector, :], fs=sampling_frequency,
                                                     nperseg=nperseg)
            # print('inj_welch',inj_welch)
            sub_freq, sub_welch = scipy.signal.welch(sub_time_series[detector, :], fs=sampling_frequency,
                                                     nperseg=nperseg)
            # print('sub_welch',sub_welch)
            proj_freq, proj_welch = scipy.signal.welch(proj_time_series[detector, :], fs=sampling_frequency,
                                                       nperseg=nperseg)
            # print('proj_welch',proj_welch0

            sub_freq_SNR, sub_welch_SNR = scipy.signal.welch(sub_time_series_SNR[detector, :], fs=sampling_frequency,
                                                     nperseg=nperseg)
            proj_freq_SNR, proj_welch_SNR = scipy.signal.welch(proj_time_series_SNR[detector, :], fs=sampling_frequency,
                                                       nperseg=nperseg)

            ###################
            # PSD Comparision #
            ###################

            plt.loglog(inj_freq, np.sqrt(inj_welch), '#8B0000', label=label)
            plt.loglog(sub_freq, np.sqrt(sub_welch), 'b-', label=label1)
            plt.loglog(proj_freq, np.sqrt(proj_welch), 'g-', label=label2)
            plt.loglog(sub_freq_SNR, np.sqrt(sub_welch_SNR), '#800080', label=label3)
            plt.loglog(proj_freq_SNR, np.sqrt(proj_welch_SNR), '#DC143C', label=label4)
            legend = plt.legend(loc='best', prop=font1)
            # plt.xscale('log')
            plt.xlim(1, 1000)
            # plt.ylim(10 ** -20, 10 ** -34)
            # plt.autoscale(enable=True, axis='y', tight=False)
            plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
            plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)
            plt.tight_layout()
            # plt.title(r' PSD Spectrum for ' + ifo, fontdict=font)
            plt.savefig(outdir + '/PSD_Welch_SNR_comparision_' + ifo + '_' + str(n_seg) + '.png', dpi=300)
            plt.close()

    def plot_data_SNRcut_all(self, IFOs, sampling_frequency=None, n_seg=None, inj_time_series=None,
                sub_time_series=None, proj_time_series=None,
                sub_time_series_SNRcut05=None, sub_time_series_SNRcut1=None,sub_time_series_SNRcut3=None,
                proj_time_series_SNRcut05=None, proj_time_series_SNRcut1=None, proj_time_series_SNRcut3=None,
                Tfft=None,SNR_cut=None):
        """

        Detector strain after Injection, Subtraction and Projection method.
        We used scipy.signal.welch method to compute the PSD for each method and for each detector.

        Welch method computes an estimate of the power spectral density by dividing the data into overlapping segments,
        computing a modified periodogram for each segment and averaging the periodograms.

        Further we compared between PSD, using welch method,to comapre the difference in PSD after the implementation of
        Injection, Subtraction and Projection method.

        IFOs: Initialization of GW interferometer.
           Generates an Interferometer with a power spectral density.
        sampling_frequency: float
            The sampling frequency (in Hz).
        n_seg: int
            number of time segment in total time duration of operation of detector.
        inj_time_series : array_like
            time_domain_strain for injection signals in units of strain.
        sub_time_series: array like
            Real valued array of time series calculated by taking inverse fft of residual_noise_data.
        proj_data_stream: array like
            Real valued array of time series calculated by taking inverse fft of residual_noise_data.
        frequency: array_like
            Real FFT of sample frequencies.
        Tfft: int
            Duration of FFT (in sec.).

        nperseg: int
            Length of each segment.
        fft_seg: int
            Length of the FFT used.
        Note: check scipy.signal.welch for to use default values.

        Return:

        *** = inj, sub, proj

        freq_*** : ndarray
            Array of sample frequencies.
        ***_welch : ndarray
            Power spectral density or power spectrum of ***_time_series.

        """

        for detector in range(len(IFOs)):

            font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
            font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

            nperseg = sampling_frequency * Tfft

            if detector == 0:
                ifo = 'CE'
                label = 'CE After Injections'
                label1 = 'CE After Subtraction'
                label2 = 'CE After Projection'
                label3 = 'CE After Subtraction SNR > '
                label4 = 'CE After Projection SNR > '

            if detector == 1:
                ifo = 'ET-D_1'
                label = 'ET-D_1 After Injections'
                label1 = 'ET-D_1 After Subtraction'
                label2 = 'ET-D_1 After Projection'
                label3 = 'ET-D_1 After Subtraction SNR > '
                label4 = 'ET-D_1 After Projection SNR > '

            if detector == 2:
                ifo = 'ET-D_2'
                label = 'ET-D_2 After Injections'
                label1 = 'ET-D_2 After Subtraction'
                label2 = 'ET-D_2 After Projection'
                label3 = 'ET-D_2 After Subtraction SNR > '
                label4 = 'ET-D_2 After Projection SNR > '

            if detector == 3:
                ifo = 'ET-D_3'
                label = 'ET-D_3 After Injections'
                label1 = 'ET-D_3 After Subtraction'
                label2 = 'ET-D_3 After Projection'
                label3 = 'ET-D_3 After Subtraction SNR > '
                label4 = 'ET-D_3 After Projection SNR > '

            ## Time Series without SNR threshold
            inj_freq, inj_welch = scipy.signal.welch(inj_time_series[detector, :], fs=sampling_frequency,nperseg=nperseg)
            sub_freq, sub_welch = scipy.signal.welch(sub_time_series[detector, :], fs=sampling_frequency,nperseg=nperseg)
            proj_freq, proj_welch = scipy.signal.welch(proj_time_series[detector, :], fs=sampling_frequency,nperseg=nperseg)

            ## Time Series with SNR threshold 0.5
            sub_freq_SNRcut05, sub_welch_SNRcut05 = scipy.signal.welch(sub_time_series_SNRcut05[detector, :], fs=sampling_frequency,
                                                             nperseg=nperseg)
            proj_freq_SNRcut05, proj_welch_SNRcut05 = scipy.signal.welch(proj_time_series_SNRcut05[detector, :], fs=sampling_frequency,
                                                               nperseg=nperseg)

            ## Time Series with SNR threshold 1
            sub_freq_SNRcut1, sub_welch_SNRcut1 = scipy.signal.welch(sub_time_series_SNRcut1[detector, :],
                                                                       fs=sampling_frequency, nperseg=nperseg)
            proj_freq_SNRcut1, proj_welch_SNRcut1 = scipy.signal.welch(proj_time_series_SNRcut1[detector, :],
                                                                      fs=sampling_frequency, nperseg=nperseg)

            ## Time Series with SNR threshold 3
            sub_freq_SNRcut3, sub_welch_SNRcut3 = scipy.signal.welch(sub_time_series_SNRcut3[detector, :],
                                                                       fs=sampling_frequency, nperseg=nperseg)
            proj_freq_SNRcut3, proj_welch_SNRcut3 = scipy.signal.welch(proj_time_series_SNRcut3[detector, :],
                                                                      fs=sampling_frequency, nperseg=nperseg)



            ###################
            # PSD Comparision #
            ###################

            plt.loglog(inj_freq, np.sqrt(inj_welch), 'r-', label=label)
            plt.loglog(sub_freq, np.sqrt(sub_welch), 'b-', label=label1)
            plt.loglog(proj_freq, np.sqrt(proj_welch), 'g-', label=label2)

            plt.loglog(sub_freq_SNRcut05, np.sqrt(sub_welch_SNRcut05), '#01386a', label=label3+str(SNR_cut[0]))
            plt.loglog(proj_freq_SNRcut05, np.sqrt(proj_welch_SNRcut05), '#75bbfd', label=label4+str(SNR_cut[0]))

            plt.loglog(sub_freq_SNRcut1, np.sqrt(sub_welch_SNRcut1), '#1f6357', label=label3+str(SNR_cut[1]))
            plt.loglog(proj_freq_SNRcut1, np.sqrt(proj_welch_SNRcut1), '#13eac9', label=label4+str(SNR_cut[1]))

            plt.loglog(sub_freq_SNRcut3, np.sqrt(sub_welch_SNRcut3), '#80013f', label=label3+str(SNR_cut[2]))
            plt.loglog(proj_freq_SNRcut3, np.sqrt(proj_welch_SNRcut3), '#f36196', label=label4+str(SNR_cut[2]))

            legend = plt.legend(loc='best', prop=font1)
            # plt.xscale('log')
            plt.xlim(1, 1000)
            # plt.ylim(10 ** -20, 10 ** -34)
            # plt.autoscale(enable=True, axis='y', tight=False)
            plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
            plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)
            plt.tight_layout()
            # plt.title(r' PSD Spectrum for ' + ifo, fontdict=font)
            plt.savefig('./Output/Plot_InjSubProj_SNRcut/PSD_Welch_SNR_cut_comparison_' + ifo + '_' + str(n_seg) + '.png', dpi=300)
            plt.close()

    def plot_one_psd(self, IFOs, sampling_frequency=None, n_seg =None, time_series = None, frequency=None, nperseg=None,Tfft=None):
        """

        :param IFOs:
        :param sampling_frequency:
        :param n_seg:
        :param time_series:
        :param frequency:
        :param nperseg:
        :param Tfft:
        :return:
        """

        nperseg = int(sampling_frequency * Tfft)

        for detector in range(len(IFOs)):
            if detector == 0:
                ifo = 'CE'
                label = 'CE After Injections'
                label1 = 'CE After Subtraction'
                label2 = 'CE After Projection'

            if detector ==1:
                ifo = 'ET-D_1'
                label = 'ET-D_1 After Injections'
                label1 = 'ET-D_1 After Subtraction'
                label2 = 'ET-D_1 After Projection'

            if detector ==2:
                ifo = 'ET-D_2'
                label = 'ET-D_2 After Injections'
                label1 = 'ET-D_2 After Subtraction'
                label2 = 'ET-D_2 After Projection'

            if detector ==3:
                ifo = 'ET-D_3'
                label = 'ET-D_3 After Injections'
                label1 = 'ET-D_3 After Subtraction'
                label2 = 'ET-D_3 After Projection'

            freq, welch = scipy.signal.welch(time_series[detector, :], fs = sampling_frequency, nperseg=nperseg, nfft = nperseg)
            # print('inj_welch',inj_welch)

            font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal','size': 12}
            font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

            plt.loglog(freq, np.sqrt(welch), label=label)
            legend = plt.legend(loc='lower left', prop=font1)
            plt.xlim(1, 1000)
            # plt.ylim(10 ** -20, 10 ** -32)
            # plt.autoscale(enable=True, axis='y', tight=False)
            # plt.xlabel(r'f (Hz)')
            # plt.ylabel(r'PSD_Welch(f)')
            plt.xlabel(r' Frequency $~\rm[Hz]$', fontdict=font)
            plt.ylabel(r'Strain [1/$~\sqrt{\rm Hz}$]', fontdict=font)
            # plt.title(r' PSD Spectrum for ' + ifo, fontdict=font)
            # plt.ylabel(r'PSD_Welch(f)')
            plt.tight_layout()
            plt.savefig(outdir+'/PSD_'+ifo+'_'+str(n_seg)+'.png', dpi=300)
            # plt.savefig(outdir+'/PSD_Welch_{}_{}_{}_{}'.format(ifo, str(n_seg), detector, nperseg)) #+ifo+str(int(n_seg))+'_'+str(detector)+str(nperseg)
            plt.close()

            # plt.loglog(freq, np.sqrt(welch), 'r-',  label=label)
            # legend = plt.legend(loc='lower left', fontsize='small')
            # # plt.xscale('log')
            # plt.xlim(1, 1000)
            # # plt.ylim(10 ** -20, 10 ** -34)
            # # plt.autoscale(enable=True, axis='y', tight=False)
            # plt.xlabel(r'frequency [Hz]', fontdict=font)
            # plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)
            # plt.title(r' PSD Spectrum for ' + ifo, fontdict=font)
            # plt.savefig(outdir+'/PSD_for_'+ifo+'_'+str(n_seg)+'.png',dpi=300)
            # plt.close()


    def plots_csd(self,IFOs, frequency, psd_series=None, cc_inj=None, cc_sub=None, cc_proj=None, n_seg=None):
        """
        Cross-Correlation between the CSD calculated using optimal filter for each detector pair.

        IFOs: Initialization of GW interferometer.
           Generates an Interferometer with a power spectral density.

        frequency: array_like
            Real FFT of sample frequencies.
        psd_series: array like
            PSD for each detector.

        cc_inj: array like
            Cross-Correlation for Injection part using optimal filter.

        cc_sub: array like
            Cross-Correlation for Subtraction part using optimal filter.
        cc_proj: array like
            Cross-Correlation for Projection part using optimal filter.
        n_seg: int
            time segment.
        """
        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        for d1 in range(self.n_det):
            for d2 in range(d1 + 1, self.n_det):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name

                plt.loglog(frequency, np.sqrt(cc_inj[d1,d2,:]), 'r-', label='injection')
                plt.loglog(frequency, np.sqrt(cc_sub[d1,d2,:]), 'b-', label='subtraction')
                plt.loglog(frequency, np.sqrt(cc_proj[d1,d2,:]), 'g-', label='projection')
                legend = plt.legend(loc='lower left', prop=font1)
                # plt.xscale('log')
                plt.xlim(1, 1000)
                # plt.ylim(10 ** -20, 10 ** -34)
                plt.autoscale(enable=True, axis='y', tight=False)
                plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
                plt.ylabel(r'CSD  [1/$\sqrt{\rm Hz}$]', fontdict=font)
                plt.tight_layout()
                # plt.title(r' CSD Spectrum for_' + str(detector), fontdict=font)
                plt.savefig(outdir+'/CSD_comparision_'+str(n_seg)+ '_' + labelstring +'.png', dpi=300)
                plt.close()

    def plot_avg_csd(self, IFOs, frequency, sum_series=None, method=None, n_seg=None):
        """
        :param IFOs:
        :param frequency:
        :param sum_series:
        :param method:
        :param n_seg:
        :return:
        """
        one_pc = 3.0856775814671916 * 10 ** 16  ## units = m
        H0 = 67.9 * 10 ** 3 * 10 ** -6 * one_pc ** -1  ## units = 1/sec

        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        for d1 in range(self.n_det):
            for d2 in range(d1 + 1, self.n_det):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name

                plt.loglog(frequency, (np.abs(sum_series[d1,d2, :])),label=method)
                legend = plt.legend(loc='lower left', prop= font1)
                plt.xlim(1, 1000)
                plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
                plt.ylabel(r' CSD [1/$\sqrt{\rm Hz}$]', fontdict=font)
                plt.tight_layout()
                #plt.title(r'Avg Cross_Correlation for ' +method, fontdict=font)
                plt.savefig(outmain+'/Cross_Corr/Avg_Cross_Correlation_' + labelstring + '_' + method, dpi=300)
                plt.close()

        # omega_gw = np.abs(sum_series[detector, :]) * (frequency ** 3 * 4 * np.pi ** 2) / (3. * H0 ** 2)
        # plt.loglog(frequency, omega_gw)
        # plt.title(r'Omega is '+ method + '_'+ str(detector))
        # # plt.xlim(2, 1000)
        # plt.xlabel(r'frequency [Hz]')
        # plt.ylabel(r'omega')
        # plt.savefig(outmain+'/Cross_Corr/omega is_' + str(detector) + '_' +method, dpi=300)
        # plt.show()
        # plt.close()

    def plot_variance(self, IFOs, frequency, variance=None, n_seg=None, method=None):
        """
        :param frequency:
        :param variance:
        :return:
        """
        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        for d1 in range(self.n_det):
            for d2 in range(d1 + 1, self.n_det):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name

                plt.loglog(frequency, variance[d1,d2, :], label = '$\sigma$')
                plt.xlim(1, 1000)
                plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
                plt.ylabel(r'$ \sigma ~~ \rm[1/Hz]$', fontdict=font)
                plt.tight_layout()
                # plt.title(r'Variance Between_' + str(ifo) + '_' + method, fontdict=font)
                plt.savefig(outmain+'/Cross_Corr/variance_noise_' + labelstring + '_' + str(n_seg) + '_' + method, dpi=300)
                plt.close()

    def plot_csd_var(self, IFOs, frequency, sum_series=None, variance=None, CSD_from_Omega_astro=None, CSD_from_Omega_cosmo=None, n_seg=None, method=None):

        """
        IFOs: Initialization of GW interferometer.
           Generates an Interferometer with a power spectral density.
        frequency: array_like
            Real FFT of sample frequencies.
        :sum_series: array like
            time series calculated by taking sum over cross-correaltion for all time segments for Injection, Subtraction and projection part.
        variance: array like
            Gaussian power spectral densities for each detector pair
        n_seg: int
            time_seg
        method: str
            for Injection, Subtraction and projection Method.

        """
        one_pc = 3.0856775814671916 * 10 ** 16  ## units = m
        H0 = 67.9 * 10 ** 3 * 10 ** -6 * one_pc ** -1  ## units = 1/sec

        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        for d1 in range(self.n_det):
            for d2 in range(d1 + 1, self.n_det):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name

                plt.loglog(frequency, np.abs(sum_series[d1, d2, :]), label='CSD ' + str(method))
                plt.loglog(frequency, variance[d1, d2, :], label='Sigma')
                plt.loglog(frequency, np.abs(CSD_from_Omega_astro[d1, d2, :]), label='CSD Astro')
                plt.loglog(frequency, np.abs(CSD_from_Omega_cosmo[d1, d2, :]), label='CSD Cosmo')

                legend = plt.legend(loc='lower left', prop=font1)
                plt.xlim(1, 1000)
                # plt.ylim(1*10**-65, 1*10**-40)
                plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
                plt.ylabel(r'CSD $~ & ~\sigma$ [1/Hz]', fontdict=font)
                plt.tight_layout()
                # plt.title(r'CSD_and_Variance for_' +str(ifo)+'_'+ method, fontdict=font)
                plt.savefig(outmain+'/Cross_Corr/CSD_Variance_Astro_Cosmo_' + labelstring + '_' + method + '_' + str(n_seg),dpi=300)
                plt.close()

    def plot_sum_csd_var(self, IFOs, frequency, sum_series=None, variance=None, CSD_from_Omega_astro=None, CSD_from_Omega_cosmo=None, method=None, outdir=None):
        """
        :param IFOs:
        :param frequency:
        :param sum_series:
        :param variance:
        :param n_seg:
        :param method:
        :return:
        """
        one_pc = 3.0856775814671916 * 10 ** 16  ## units = m
        H0 = 67.9 * 10 ** 3 * 10 ** -6 * one_pc ** -1  ## units = 1/sec

        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        for d1 in np.arange(self.n_det):
            for d2 in np.arange(d1 + 1, self.n_det):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name

                plt.loglog(frequency, np.abs(sum_series[d1,d2, :]), label='CSD ' + str(method))
                plt.loglog(frequency, variance[d1,d2, :], label='$\sigma$')
                plt.loglog(frequency, np.abs(CSD_from_Omega_astro[d1,d2, :]), label='CSD Astro')
                plt.loglog(frequency, np.abs(CSD_from_Omega_cosmo[d1,d2, :]), label='CSD Cosmo')

                legend = plt.legend(loc='lower left', prop=font1)
                plt.xlim(1, 1000)
                # plt.ylim(1*10**-60, 1*10**-35)
                # plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
                plt.ylabel(r'CSD $~ & ~\sigma ~~$ [1/Hz]', fontdict=font)
                plt.tight_layout()
                # plt.title(r'CSD_and_Variance for_' +str(ifo)+'_'+ method, fontdict=font)
                plt.savefig(outdir + '/Sum_CSD_Variance_Astro_Cosmo_' + labelstring + '_' + method,dpi=300)
                plt.close()

    def plot_sum_csd_compare(self, IFOs, frequency, sum_series=None, sum_series_zero=None, variance=None, CSD_from_Omega_astro=None, CSD_from_Omega_cosmo=None, method=None, outdir=None):
        """
        :param IFOs:
        :param frequency:
        :param sum_series:
        :param variance:
        :param n_seg:
        :param method:
        :return:
        """
        one_pc = 3.0856775814671916 * 10 ** 16  ## units = m
        H0 = 67.9 * 10 ** 3 * 10 ** -6 * one_pc ** -1  ## units = 1/sec

        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        for d1 in np.arange(self.n_det):
            for d2 in np.arange(d1 + 1, self.n_det):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name

                plt.loglog(frequency, np.abs(sum_series[d1,d2, :]), label='CSD ' + str(method))
                plt.loglog(frequency, np.abs(sum_series_zero[d1, d2, :]), label='CSD ' + str(method) + ' in Zero Noise')
                plt.loglog(frequency, variance[d1,d2, :], label='$\sigma$')
                plt.loglog(frequency, np.abs(CSD_from_Omega_astro[d1,d2, :]), label='CSD Astro')
                plt.loglog(frequency, np.abs(CSD_from_Omega_cosmo[d1,d2, :]), label='CSD Cosmo')

                legend = plt.legend(loc='lower left', prop=font1)
                plt.xlim(1, 1000)
                # plt.ylim(1*10**-60, 1*10**-35)
                # plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
                plt.ylabel(r'CSD $~ & ~\sigma ~~$ [1/Hz]', fontdict=font)
                plt.tight_layout()
                # plt.title(r'CSD_and_Variance for_' +str(ifo)+'_'+ method, fontdict=font)
                plt.savefig(outdir + '/Sum_CSD_Compare_Variance_Astro_Cosmo_' + labelstring + '_' + method,dpi=300)
                plt.close()

    def plot_CSD_from_Omega(self, IFOs, frequency, CSD_from_Omega_astro=None, CSD_from_Omega_cosmo=None):
        """
        :param IFOs:
        :param frequency:
        :param CSD_from_Omega:
        :return:
        """

        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size=5.5)

        for d1 in range(self.n_det):
            for d2 in range(d1 + 1, self.n_det):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name
                plt.loglog(frequency, (np.abs(np.real(CSD_from_Omega_astro[d1, d2, :]))), label=labelstring +' CSD Astro')
                plt.loglog(frequency, (np.abs(np.real(CSD_from_Omega_cosmo[d1, d2, :]))), label=labelstring +' CSD Cosmo')

        legend = plt.legend(loc='lower left', prop=font1)
        plt.xlim(1, 1000)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r' CSD [1/$\sqrt{\rm Hz}$]', fontdict=font)
        plt.tight_layout()
        #plt.title(r'CSD_from_Omega_' + str(ifo) + background, fontdict=font)
        plt.savefig(outmain+'/Omega_gw/CSD_from_Omega', dpi=300)
        plt.close()


    def plot_comso_omega_gw(self, frequency, comso_omega_gw=None, cobe_spectrum=None, PGWB=None):
        """
        :param comso_omega_gw:
        :param frequency:
        :return:
        """
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        one_pc = 3.0856775814671916 * 10**16  ## units = m
        H00 = 67.9 * 10**3 * 10**-6 * one_pc**-1  ## units = 1/sec
        H0 = 30 * H00
        freq = np.logspace(-18, -16, num=len(frequency))
        frequency = np.logspace(-18, 4, num=len(frequency))

        plt.loglog(frequency, comso_omega_gw, label='$\Omega_{GW}^{cosmo}$')
        plt.loglog(freq, cobe_spectrum, label ='COBE')
        plt.loglog(PGWB[0], PGWB[1], '--', label='Primordial GWB')
        plt.text(0.8, 10 **-18, 'Primordial GWB', horizontalalignment='right', color='green', fontsize=12)

        legend = plt.legend(loc='lower left', prop=font1)
        plt.xlim(10**-18, 10**4)
        plt.ylim(10**-20, 10**-10)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'$\Omega(f)$', fontdict=font)
        plt.tight_layout()
        #plt.title(r'$\Omega_{GW}^{cosmo}$', fontdict=font)
        plt.savefig(outmain+'/Omega_gw/Cosmo_Omega_GW_COBE', dpi=300)
        plt.close()

    def plot_omega_gw_astro_cosmo(self, frequency, astro_omega_gw, cosmo_omega_gw):
        """
        frequency: array
            Frequency array for which Sh_astro and Sh_cosmo are calculated
        astro_omega_gw: array
            Spectral Energy Density for Astrophysical Sources
        comso_omega_gw:  array
            Spectral Energy Density for Cosmological Sources.

        """
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        plt.loglog(frequency, astro_omega_gw, label='$\Omega_{GW}^{astro}$')
        plt.loglog(frequency, cosmo_omega_gw, label='$\Omega_{GW}^{cosmo}$')
        legend = plt.legend(loc='best', prop=font1)
        plt.xlim(1, 1000)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'$ \Omega_{GW} $', fontdict=font)
        plt.tight_layout()
        # plt.title(r'CSD_from_Omega_' + str(ifo) + background, fontdict=font)
        plt.savefig(outmain+'/Omega_gw/Omega_Astro_Cosmo', dpi=300)
        plt.close()

    def plot_sh_from_omega(self, frequency, Sh_astro, Sh_comso):
        """

        frequency: array
            Frequency array for which Sh_astro and Sh_cosmo are calculated
        Sh_astro: array
            Spectral Energy Density for Astrophysical Sources
        Sh_comso:  array
            Spectral Energy Density for Cosmological Sources.
        """
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        plt.loglog(frequency, Sh_astro, label='Sh astro')
        plt.loglog(frequency, Sh_comso, label='Sh comso')
        plt.xlim(1, 1000)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r' Sh 1/$~\rm[Hz]$', fontdict=font)
        plt.tight_layout()
        # plt.title(r'CSD_from_Omega_' + str(ifo) + background, fontdict=font)
        plt.savefig(outmain+'/Omega_gw/Sh from Omega', dpi=300)
        plt.close()

    def CSD_Omega(self, IFOs, frequency, CSD_astro_omega, CSD_cosmo_omega):
        """
        frequency : array like
            Frequency array for which Sh_astro and Sh_cosmo are calculated
        CSD_astro_omega : array like
            Cross power spectral desnity for a detector pair for a astro-physical energy density.
        CSD_cosmo_omega : array like
            Cross power spectral desnity for a detector pair for a cosmological energy density.

        """
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        for d1 in range(self.n_det):
            for d2 in range(d1 + 1, self.n_det):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name

                plt.loglog(frequency, np.sqrt(np.abs(np.real(CSD_astro_omega[d1,d2, :]))), label='CSD Astro')  #
                plt.loglog(frequency, np.sqrt(np.abs(np.real(CSD_cosmo_omega[d1,d2, :]))), label='CSD Cosmo')
                legend = plt.legend(loc='lower left', prop=font1)
                plt.xlim(1, 1000)
                # plt.ylim(10**-70, 10**-20)
                # plt.autoscale(enable=True, axis='y', tight=True)
                plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
                plt.ylabel(r' CSD [1/$\sqrt{\rm Hz}$]', fontdict=font)
                plt.tight_layout()
                # plt.title(r'CSD_from_Omega_' + str(ifo) + background, fontdict=font)
                plt.savefig(outmain+'/Omega_gw/CSD Omega without norm ' + labelstring, dpi=300)
                plt.close()
