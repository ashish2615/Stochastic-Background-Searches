from __future__ import division, print_function

import os
import sys
import bilby
import numpy as np
import scipy
from scipy import signal
import scipy
from scipy.signal import csd

import matplotlib
#matplotlib.use('tkagg')
#matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.font_manager as font_manager

from Initial_data import InitialData
from ORF_OF import detector_functions

current_direc = os.getcwd()
# print("Current Working Directory is :", current_direc)

## Specify the output directory and the name of the simulation.
outmain = 'Output'
outdir = 'Cross_Corr'
label = 'Cross_Corr'
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

class CrossCorrelation:

    def __init__(self):
        pass

    def initial(self, ifos, sampling_frequency, start_time, end_time, n_seg, gamma, optimal_filter, Tfft): #seg_start_time, seg_end_time,
        """
        Initialize/access the data from Initial_data script.
        check cbcpm.InitialData.initial_data

        ifos: iterable
            The list of interferometers
        sampling_frequency: float
            The sampling frequency (in Hz).
        start_time: GPS time
            data taking or operation of period detector. The GPS start-time of the data.
        end_time: GPS time
            end of operation period of a detector. The GPS end-time of the data.
        n_seg: int
           number of segment in total time duration of operation of detector.
        seg_start_time : int
            segment start time
        seg_end_time : int
            segment end time.
        gamma : array like
            overlap reduction function for a detector pair
        optimal_filter: array like
            optimal filter for a detector pair.
        frequency: array_like
           Real FFT of sample frequencies.
        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.
        Tfft : int
            time duration for fft.

        """

        data = InitialData()
        data_sets = data.initial_data(ifos, sampling_frequency, start_time, end_time, Tfft, n_seg)
        self.sampling_frequency = data_sets[1]
        self.start_time = data_sets[2]
        self.end_time = data_sets[3]
        self.duration = data_sets[4]
        duration_seg = data_sets[5]

        self.n_fft = data_sets[6]
        self.N_samples = data_sets[7]
        self.frequency = data_sets[8]
        self.IFOs = data_sets[10]

        self.modes = data_sets[11]
        self.G = data_sets[12]
        self.one_pc = data_sets[13]
        self.H0 = data_sets[14]
        self.speed_of_light = data_sets[15]
        self.rho_c = data_sets[16]
        self.omega_gw = data_sets[17]

        self.gamma = gamma
        self.optimal_filter = optimal_filter
        self.Tfft = Tfft

    def cross_correlation(self, time_series=None, method = None, n_seg=None):
        """
        time_series: array like.
            time_doamin_strain for all detectors. time_domain_strain in units of strain.

        gamma: array_like
            overlap reduction function for all detectors.

        optimal_filter: array_like
            optimal filter for the signal

        return:
        cross_correlation: array like,
            Cross corelation between the signal of detector pair.
        """

        cross_correl = np.zeros((len(self.IFOs), len(self.IFOs), len(self.frequency)),dtype=np.complex)
        cross_correlation = np.zeros((len(self.IFOs), len(self.IFOs), len(self.frequency)), dtype=np.complex)
        ## spectral shape of stochastic background is
        # Sb = np.dot((3. * self.H0 ** 2) / (4 * np.pi ** 2), np.divide(self.omega_gw, self.frequency ** 3))

        for d1 in range(len(self.IFOs)):
            detector_1 = time_series[d1, :]
            for d2 in range(d1 + 1, len(self.IFOs)):
                detector_2 = time_series[d2, :]

                freq_cc, cross_correl[d1, d2, :] = scipy.signal.csd(detector_1, detector_2, fs=self.sampling_frequency,nperseg=self.n_fft)
                cross_correlation[d1, d2, :] = np.real(cross_correl[d1, d2, :] * self.optimal_filter[d1, d2, :]) * (freq_cc[1] - freq_cc[0])

        return cross_correl, cross_correlation

    def CSD_from_Omega_astro(self):
        """
        Calculating CSD from Omega_gw using power spectral density of Astrophysical signal background.
        :return:
        Sb = array like
            Power spectral density of Astrophyscial signal background
        """
        ## Omega_GW Power Law Spectrum
        omega_gw = 7 * 10**(-10) * (self.frequency / 10)**(2/3)   ## units = Independent of frequency  / no units
        ## Background spectral density S^b(frequency)
        # Sb = np.dot((3. * self.H0**2) / (10 * np.pi**2), np.divide(omega_gw, self.frequency**3))
        Sb = ((3 * self.H0**2)/(10 * np.pi**2)) * self.frequency**(-3) * omega_gw       ## units =  1/Hz

        n_det = len(self.IFOs)

        CSD_astro = np.zeros((n_det, n_det, len(self.frequency)))  ## units =  1/Hz

        for d1 in range(n_det):
            for d2 in range(d1 + 1, n_det):
                CSD_astro[d1, d2, :] = Sb * self.gamma[d1, d2, :]

        return CSD_astro

    def CSD_from_Omega_cosmo(self):
        """
        Calculating CSD from Omega_gw using power spectral density of Cosmological signal background.

        Sb = array like
            Spectrum of energy density of cosmological signal background

        Return:
        CSD_cosmo : array like
            CSD for energy density spectrum of cosmological origin.

        """
        ## Background spectral density S^b(frequency)
        ## omega_gw = 5 * 10**-15
        # Sb = np.dot((3. * self.H0**2) / (10 * np.pi**2), np.divide(self.omega_gw, self.frequency**3))
        Sb = (3 * self.H0**2)/(10 * np.pi**2) * self.frequency**(-3) * self.omega_gw      ## units =  1/Hz

        n_det = len(self.IFOs)

        CSD_cosmo = np.zeros((n_det, n_det, len(self.frequency)))
        for d1 in range(n_det):
            for d2 in range(d1 + 1, n_det):
                CSD_cosmo[d1, d2, :] = Sb * self.gamma[d1, d2, :]

        return CSD_cosmo

    def CSD_Omega(self, Sh_astro, Sh_cosmo, gamma=None):
        """
        CSD calculated from the spectrum of enery density for astrophyscial and cosmological backgrounds using
        un-normalized overlap reduction function.

        https://arxiv.org/pdf/1911.09745.pdf
        Check equations 10 and 19.

        :Sh_astro : array like
            Background Spectral Density for Astrophysical Sources

        Sh_cosmo : array like
            Background Spectral Desnity for Cosmological Sources

        gamma: array like
             overlap reduction function

        ----------------------
        Return

        CSD : array like
            Cross power spectral desnity for a detector pair

        """

        ## CSD for astrophyscial background
        CSD_astro_omega = np.zeros((len(self.IFOs), len(self.IFOs), len(self.frequency)))  ## units =  1/Hz
        ## CSD for cosmological background
        CSD_cosmo_omega = np.zeros((len(self.IFOs), len(self.IFOs), len(self.frequency)))  ## units =  1/Hz

        for d1 in range(len(self.IFOs)):
            for d2 in range(d1 + 1, len(self.IFOs)):

                ## Note : A factor of 5 introduced in denominator to make ORF un-normalized.
                ## In accordance to Mingarelli 2019 paper (link given above).
                CSD_astro_omega[d1, d2, :] = (Sh_astro/(5 * 8 *np.pi)) * gamma[d1, d2, :]
                CSD_cosmo_omega[d1, d2, :] = (Sh_cosmo/(5 * 8 *np.pi)) * gamma[d1, d2, :]

        return CSD_astro_omega, CSD_cosmo_omega


    def cros_correl_mean(self, cross_correl=None, n_seg=None):
        """
        Calculating the mean of CSD over all time segments given by n_seg.

        cross_correl : array like
            Cross Power spectral density for each detector pair.
            Note : Here we used the real CSD without using the optimnal filter to calculated the CSD.
                   We can also use optimally filtered CSD to calculate the mean of CSD.
        n_seg : int
            Total number of time segments.

        :return:
        cross_correlation_mean: array like
            mean of CSD for each detector pair
        """

        cross_correlation_mean =  np.zeros((len(self.IFOs), len(self.IFOs), len(self.frequency)))

        for d1 in range(len(self.IFOs)):
            for d2 in range(d1 + 1, len(self.IFOs)):
                cross_correlation_mean[d1, d2, :] = 1/n_seg * (cross_correl[d1, d2, :])

        return cross_correlation_mean

    def cross_corr_var(self, cross_correl=None, n_seg=None):
        """
        Defines the variance of cross-correlation signal between multiple detector pair.
        Return:
        cross_correlation_var: float
            Variance of correlation noise is
        """

        mean = self.cros_correl_mean(cross_correl=cross_correl, n_seg=n_seg)

        cross_correlation_var = np.zeros((len(self.IFOs), len(self.IFOs), len(self.frequency)))

        for d1 in range(len(self.IFOs)):
            for d2 in range(d1 + 1, len(self.IFOs)):
                cross_correlation_var[d1, d2, :]  = 1/n_seg * (cross_correl[d1, d2, :] ** 2) - mean[d1, d2, :]**2

        return cross_correlation_var

    def cross_corr_expect(self):
        """
        Sb: float
            Stochastic GW background power spectral density

        Return:
        cross_correlation_expect: array like
            Expectation valiue of cross-corealtion function of detector pairs.
        """

        cross_correlation_expect = np.zeros((len(self.IFOs), len(self.IFOs), 1))

        for d1 in range(len(self.IFOs)):
            for d2 in range(d1 + 1, len(self.IFOs)):
                cross_correlation_expect[d1, d2, :] = np.sum(self.gamma[d1, d2, :] * Sb * self.optimal_filter[d1, d2, :]) \
                                                    * (self.frequency[1] - self.frequency[0])

        return cross_correlation_expect

    def optimal_estimator(self,time_series=None,method = None, n_seg=None):
        """
        optimal estimator is given as (average results from detector pairs weighted by their variances)
        estimator which brings the mean square error to minimum i.e.  variance-weighted mean and
        variance weighted mean is always a minimum variance estimator
        Eq. 6 (PRL 113, 231101 (2014))

        Return
        --------
        Optimal estimator: int
            real value
        """

        cross_correlation = self.cross_correlation(time_series=time_series,method = method,n_seg=n_seg)
        cross_correlation_var = self.cross_corr_var()

        optimal_estimator = np.zeros((1))
        for d1 in range(len(self.IFOs)):
            for d2 in range(d1 + 1, len(self.IFOs)):
                optimal_estimator += np.sum(np.real(cross_correlation[d1, d2, :]) * cross_correlation_var[d1, d2, :]**-2) / np.sum(
                    cross_correlation_var[d1, d2, :]**-2)

        return optimal_estimator

    def plot_cc(self, cross_corr_= None, n_seg=None, method = None):

        """
        cross_corr_: array like
            cross-corelation between detector pairs.
        n_seg: int
           number of segment in total time duration of operation of detector.
        method: str
            Define the method for which cross-corelation corresponds out of Injections, Subtractions and Projection.
        inj: int
            Number of signal injected, subtracted or projected.

        Return
            Combined Cross-corelation plot for Injections, Subtractions and Projection.
        """
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 10}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size=5)

        plt.subplot(2, 1, 1)
        for d1 in range(len(self.IFOs)):
            for d2 in range(d1 + 1, len(self.IFOs)):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name
                plt.plot(self.frequency, cross_corr_[d1, d2, :], label=labelstring)

        legend = plt.legend(loc='lower right', prop=font1)
        plt.xscale('log')
        plt.xlim(1, 1000)
        # plt.ylim(-0.2*10**-8, 0.2*10**-8)
        plt.autoscale(enable=True, axis='y',tight=False)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r' Cross Correlation (f)', fontdict=font)
        # plt.title(r'Cross Correlation Between CE and ET_D_TR', fontdict=font)
        # plt.savefig('/storage/gpfs_small_files/VIRGO/users/ashish/cbcpm/Cross_Corr/Cross Correlation_log')
        # plt.show()
        # plt.close()

        plt.subplot(2, 1, 2)
        for d1 in range(len(self.IFOs)):
            for d2 in range(d1 + 1, len(self.IFOs)):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name
                plt.plot(self.frequency, cross_corr_[d1, d2, :], label=labelstring)

        legend = plt.legend(loc='lower right', prop=font1)
        plt.xlim(30, 100)
        # plt.ylim(-0.05*10**-8, 0.05*10**-8)
        plt.autoscale(enable=True, axis='y', tight=False)
        # plt.axhline(y=0)
        # plt.axvline(x=0)
        plt.xlabel(r'Frequency $~\rm[Hz]$',fontdict=font)
        plt.ylabel(r'Cross Correlation (f) ',fontdict=font)
        # plt.title(r'Cross Corr Between CE and ET_D_TR',fontdict=font)
        plt.tight_layout()
        plt.savefig(outdir + '/Cross Correlation_norm_'+str(n_seg)+'_'+method, dpi = 300)
        plt.close()