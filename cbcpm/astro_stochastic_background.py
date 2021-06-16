from __future__ import division, print_function

import os
import sys
from tkinter import font

import bilby
from bilby.core.utils import speed_of_light
import numpy as np
import scipy
from scipy import integrate
import scipy.integrate as integrate

from Initial_data import InitialData

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
outdir = 'Omega_gw'
label = 'Omega_gw'
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

# plots_PSD = PSDWelch()
# function =  detector_functions()
# orf = function.initial(ifos,sampling_frequency, start_time, end_time, n_seg, Tfft)
# gamma = function.overlap_reduction_function()
# optimal_filter = function.optimal_filter_JH()

class StochasticBackground:

    def __init__(self):
        pass

    def initial(self, ifos, sampling_frequency, start_time, end_time, n_seg, Tfft):
        """
        Initialize/access the data from Initial_data script.
        check cbcpm.InitialData.initial_data

        ifos: iterable
            The list of interferometers
        sampling_frequency: float
            The sampling frequency (in Hz).
        duration_seg: float
            The data duration of a segment (in s).
        frequency: array_like
           Real FFT of sample frequencies.

        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.

        """
        data = InitialData()
        data_sets = data.initial_data(ifos, sampling_frequency, start_time, end_time, Tfft, n_seg=n_seg)
        self.sampling_frequency = data_sets[1]
        self.start_time = data_sets[2]
        self.end_time = data_sets[3]
        self.duration = data_sets[4]
        self.duration_seg = data_sets[5]

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

    def astro_Omega_gw(self):
        """
        Simple Interpolation of Astrophysical Omega_gw from Power law formula
        Omega_GW = Omega_alpha * (frequecy / frequency_ref) ** alpha
        """
        ## Astrophysical Binary Background
        astro_omega_gw = 7 * 10**(-10) * (self.frequency / 10)**(2/3)   ## units = Independent of frequency
        return astro_omega_gw

    def cosmo_Omega_gw(self):
        """
        Cosmological energy density, Omega_gw
        """
        ## Cosmological Spectrum
        # cosmo_omega_gw = self.omega_gw  ## units = Independent of frequency
        cosmo_omega_gw = self.omega_gw * (self.frequency / 10)**0
        return cosmo_omega_gw

    def sh_from_Omega(self):
        """
        Calculating Sh, one sided power spectral density for a detector from Omega_gw modesl for astrophysical and cosmological signals
        Check eq.7 of paper https://arxiv.org/pdf/1911.09745.pdf
        ----------
        Returns:

        Sh_astro : array like
            Background Spectral Density for Astrophysical Sources
        Sh_cosmo : array like
            Background Spectral Desnity for Cosmological Sources
        """
        omega_gw_astro = 5 * 10**(-9) * (self.frequency / 10)**(2/3)   ## units = independent of frequency

        ## Sh for astrophusical sources
        Sh_astro = ((3 * self.H0**2)/(10 * np.pi**2)) * self.frequency**(-3) * omega_gw_astro   ## units = 1/Hz
        ## Sh for cosmological sources
        Sh_cosmo = (3 * self.H0**2)/(10 * np.pi**2) * self.frequency**(-3) * self.omega_gw      ## units = 1/Hz

        return Sh_astro, Sh_cosmo

    def Omega_from_CSD(self, CSD=None, method=None, n_seg=None):
        """
        Interpolation of Omega_gw to CSD i.e. Cross-Correlation between two detectors.

        :param self:
        :param CSD:
        :return:
        """
        astro_Omega_gw = self.astro_Omega_gw()

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')


        for d1 in np.arange(len(self.IFOs)):
            for d2 in np.arange(d1 + 1, len(self.IFOs)):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name

                omega_gw = np.abs(CSD[d1,d2, :]) * (self.frequency ** 3 * 10 * np.pi ** 2) / (3. * self.H0 ** 2)

            plt.loglog(self.frequency, omega_gw, label='$\Omega_{GW}$ from CSD')
            # plt.loglog(self.frequency, omega_astro_gw, label='Omega_gw')
            legend = plt.legend(loc='lower left', prop=font1)
            # plt.title(r'Omega_gw is ' + str(ifo) + '_' + method,fontdict=font)
            plt.xlim(1, 1000)
            plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
            plt.ylabel(r'$\Omega_{GW}$', fontdict=font)
            plt.tight_layout()
            plt.savefig(outdir + '/Omega_sum_CSD_is_'+labelstring +'_' + method, dpi=300)
            plt.close()

    def acoustic_spectrum(self):
        """
        Acoustic production of GWs
        https://arxiv.org/pdf/1512.06239v2.pdf
        eqution 13-15
        https://iopscience.iop.org/article/10.1088/1742-6596/840/1/012031/pdf
        eqution 3-4

        k_nu : int
            κv = ρ_v/ρ_vac
            fraction of latent heat that is transformed into bulk motion of the fluid, and depends on the expansion mode of the bubble
        v_w : int
            acoustic wall speed
        Ssw : array like
            Acoustic GW spectral shape
        fsw : int
            Acoustic GW peak frequency

        alpha : int
            the ratio of vacuum energy to radiation energy is α = ρ_vac/ρ_rad
        beta : int
            nucleation rate


        :return:
        """
        v_w = 0.83 ##detonation
        # v_w =0.44 ##deflagration
        fsw = (2/np.sqrt(3)) * (beta/v_w)
        fsw = 1.9 * 10 **(-2) * (1/ v_w) * (beta/H) * (T/100) * (g/100)**(1/6)

        #q = 3
        #q = 4
        # Ssw = ((p + q)*(self.frequency/fsw)**q)/(1 + q * (self.frequency/fsw)**(p+q))

        Ssw = (self.frequency/fsw)**3 * (7 / (4 + 3*(self.frequency/fsw)**2))**(7/2)
        h2Omega_sw = 2.65 * 10**(-6) *(H/beta) * ((k_nu * alpha)/(1 + alpha))**2 * (100/g)**(1/3) * v_w * Ssw

        return
    def BBHGWSpect(self):
        """
        T : int
            Normalization factor defines the lenght of the data sample.
        :return:
        """
        ## Redshift probability distribution
        modes = ['plus','cross']
        sig_sum = np.zeros((len(n_in),len(self.frequency)))
        # cidx = 0
        # for inj in np.arange(n_inj):
        #     sig_sum[cidx, :] +=
        flux = T**-1 * (np.pi *self.speed_of_light**3)/(2*self.G) * self.frequency**2 * np.sum
        omega_astro = 1/(rho_c * self.speed_of_light) * self.frequency * flux

    def IMR_spectrum(self):
        """
        :return:
        """
        chirp_mass = m1 * m2(m1+m2)**(-1/3)
        IMR_specrtrum = ((self.G * np.pi)**(2/3) * chirp_mass**(5/3))/3 * freq

    def cosmo_spectrum(self):
        """
        GW energy density fraction today
        https://arxiv.org/pdf/1801.04268.pdf    Check Eq. 139-140.
        f_ref : int
            pivot frequency at which the primordial scalar amplitude is normalised.
        k_p : int
            pivot scale
        P_r : int
            primordial curvature power spectrum amplitude at the pivot scale k∗ = 0.05 Mpc−1
        n_T : int
            tensor spectral index, depends upon the choice of tensor to scalar ratio r.
        r : int
            tensor to scalar ratio
        f_eq : int
            frequency entering the horizon at matter-radiation equality.

        :return:
        """

        frequency = np.logspace(-18, 4,num=len(self.frequency))
        omega_rad = 2.47 * 10**-5 ## Radiation Energy Density Today
        omega_matt = 0.26         ## Matter Energy Density Today

        ## For Dark Energy dominated phase we have scale factor as
        ## a(t) propt to exp(H0 * sqrt(Omega_DE) * t), t is choice to time
        ## for present time t = 0 thus we have
        a_0 = 1
        k_p = 0.05 * (self.one_pc * 10**6)**(-1)
        freq_ref = k_p/(2 * np.pi * a_0)
        P_r = 2*10**-9
        f_eq = (self.H0 * omega_matt) / (np.pi * np.sqrt(2 * omega_rad))
        r = 0.07
        n_T = -r/8
        # n_T = 0.2

        cosmo_omega_gw = (3/128)* omega_rad * r * P_r *(frequency/freq_ref)**n_T * ((1/2)*(f_eq/frequency)**2 + (4/9)*(np.sqrt(2) -1))

        return cosmo_omega_gw

    def cobe_spectrum(self):
        ##  COBE observational limits on the different multipole moments (2 ≤ l ≤ 30)
        """Note that this limit does not apply to any gravitational waves, but only to those of cosmological origin,
        which were already present at the time of last scattering of the CMBR. This limit applies only over a
        narrow band of frequencies at the very lowest limits """
        ## https://arxiv.org/pdf/gr-qc/9604033.pdf

        H00 = self.H0
        H0 = 30 * self.H0
        frequency = np.logspace(-18, -16, num=len(self.frequency))

        cobe_cosmo_limit = []
        for x in frequency:
            cobe_limit = 7 * 10**-11*(H00/x)**2
            cobe_cosmo_limit.append(cobe_limit)
        cobe_cosmo_limit = np.array(cobe_cosmo_limit)

        return cobe_cosmo_limit

    def PGWB(self):
        """
        PGWB = Premordial Gravitatioanl Wave Background
        Ref : arXiv:1101.3940v2 [astro-ph.CO] Check eq. 14
        :return:
        """
        h0 = 0.6766
        f = np.linspace(start=10**-6, stop=10, num=10000)
        # omega_gw = 10**-15
        Sh_PGWB = 8.85 * 10**-27 * h0 * (self.omega_gw / 10 **-15) ** (1/2) * f**(-3/2)   ## units =

        return f, Sh_PGWB

    def Pivot_frequency(self, nt=None):
        """
        ## The power spectrum of  the primordial curvature fluctuations is

        Pt = As (k/k0)**(ns-1) # k0 is pivot wave number, ns is spectral index

        Pt(k) as the ‘primordial’ power spectrum as it described density fluctuations at time t = t. These perturbations with
        different wave numbers evolved differently in the very early Universe.  This modifies the matter power spectrum from the
        power law form given above. These modifications are encoded in the  so-called ‘transfer funtion’ T(k). This transfer function T(k)
        encodes the information on the evolution of some density perturbation δ(k), and therefore affects the power spectrum as

        Pt = As (k/k0)**(ns-1) * Tk**2

        :return:
        """
        frequency = np.logspace(-18, 4, num=10000)
        ## Pivot wave number k = 0.05 Mpc**-1 from Planck
        f_cmb = (self.speed_of_light / 2 * np.pi) * 0.05  ## units Mpc **-1,

        ## r = At/As
        ## from Planck Data
        As = 2.196 * 10**-9
        r = 0.11
        At = r * As

        Pt = At * (frequency/f_cmb)**nt

        return frequency, Pt

    def Omega_CMB(self):
        """
        :return:
        """
        ## r = At/As, From Planck Data
        r = 0.11
        As = 2.196 * 10**-9  ## from Planck
        Omega_r = 2.47 * 10**-5
        Omega_m = 0.308
        Omega_CMB = (3 * r * As * Omega_r) / 128

        return Omega_CMB

    def Omega_GW_today(self, nt=None):
        """
        At :
            Tensor
        As :
            Scalar, is the amplitude of the primordial power spectrum of density perturbations, evaluated at the pivot scale
        r : int
            defines the amplitude of the primordial GW spectrum in terms of thescalar to tensor ratio.
        omega_r :
            radiation energy density today
        omega_m : int
            Matter energy density today
        omega_cmb :
        f_eq : int
            is the frequency of the mode whose corresponding wavelength is equal to the size of the Universe at the time
            of matter–radiation equality.
        H0 :
        f_cmb : int
            Pivot frequency
        :return:
        """
        frequency = np.logspace(-15, 4, num=10000)
        ## r = At/As, From Planck Data
        r = 0.11
        # nt = [0.68, 0.54, 0.36, 0.34, -r/8]
        As = 2.196 * 10**-9 ## from Planck
        Omega_r = 2.47 * 10**-5
        Omega_m = 0.308

        f_eq = np.sqrt(2) * self.speed_of_light * self.H0 * (Omega_m / 2 * np.pi) * np.sqrt(Omega_r)
        ## pivot wave number k = 0.05 Mpc**-1 from Planck
        f_CMB = (self.speed_of_light / (2 * np.pi)) * 0.05  ## units Mpc **-1,

        Omega_CMB = self.Omega_CMB()
        Omega_GW = Omega_CMB * (frequency / f_CMB)**nt * (1/2 * (f_eq/frequency)**2 + 16/9)

        return frequency, Omega_GW

    def rho_GW_today(self):
        """
        Tf = T(f) : array
            Transfer function which encodes information about how GWs change as a function of frequency.
        :return:
        """
        rho_GW = ((self.frequency**4 * (2*np.pi)**3) / (self.speed_of_light**5)) * Pt * Tf**2 * (self.frequency[1] - self.frequency[0])

        return rho_GW

    def plot_all_backgrounds(self):
        """

        :return:
        """

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        r = 0.11
        nt_range = [0.68, 0.54, 0.36, 0.34, -r/8]
        # frequency = np.logspace(-18, 4, num=len(self.frequency))

        for nt in nt_range:
            frequency ,Omega_GW = self.Omega_GW_today(nt=nt)
            plt.loglog(frequency, Omega_GW, label='PGWB'+' for nt = '+ str(nt))
            plt.xlim(10 ** -18, 10 ** 4)
            plt.ylim(10 ** -16, 10 ** -3)

        legend = plt.legend(loc='upper center', prop=font1)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'$\Omega_{GW}~ (f)$', fontdict=font)
        # plt.title(r'All_background_GW', fontdict=font)
        plt.tight_layout()
        plt.savefig(outdir+'/All_background_GW', dpi=300)
        plt.close()


    # def gw_from_cosmic_strings(self,):
    #
    #     """
    #     PHYSICAL REVIEW D 97, 123505 (2018)
    #
    #     rho_c : int
    #         critical energy density of Universe today.
    #
    #     f : array.
    #         frequecy array.
    #
    #     k : int
    #         is the number of modes.
    #
    #     t_F : int
    #         time scale at which cosmic strings network reached scaling, shortly after the formation of network.
    #
    #     gamma : int
    #          is a dimensionless constant.
    #
    #     ti : int
    #         cosmic string time of formation.
    #
    #     li : int
    #         lenght of cosmic string.
    #
    #     alpha : int
    #         loop size parameter.
    #
    #     a(t) : int
    #         scale factor for matter and radiation dominated phases.
    #
    #     f_alpha : int
    #         Energy released by long comsic strings.
    #
    #     t_0 : int
    #         current time.
    #
    #     t_tilde : int
    #         Represent the GW emission time.
    #
    #     tik(t_tilde, f)  : int
    #         Represent the formation time of loops conrtibuting with mode number k and t_F
    #
    #     C_eff : float.
    #         Is a function depends on the redshift scaling of the domianted energy density rho_c of the Universe.
    #
    #     :return:
    #     """
    #
    #     alpha = 10**-1
    #     g_mu = 5*10**-12
    #     g_mu = 5*10**-15
    #     g_mu = 5*10**-17
    #     C_eff = [0.41, 5.5, 30]
    #     k =
    #     f_alpha =
    #     gamma = 50
    #     gamma_k = gamma / (3.60 * k ** (4 / 3))
    #
    #
    #     ti =
    #     t_tilde =
    #     t_f =
    #
    #     a(t_tilde) =
    #     a(t0) =
    #     a(ti) =
    #
    #
    #
    #     tik(t_tilde, f) = (1/(alpha + gamma_k * g_mu)) *((2*k)/self.frequency * a_t/a_t0 + gamma * g_mu * t_tilde)
    #
    #     cs_omega_gw = (1/self.rho_c) * (2*k/self.frequency) * (f_alpha * gamma_k * g_mu)/(alpha * (alpha + gamma * g_mu)) \
    #                   * dt * (C_eff/ti**4) * (a_t/a_t0)**5 * (a_tik/a_t)**3 * theta(tik - t_f)