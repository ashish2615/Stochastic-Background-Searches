from __future__ import division, print_function

import os
import sys
import bilby
import logging

import numpy as np
import pandas as pd
import scipy
import json
import math
import sklearn
import pickle
import shutil
from bilby.core.utils import speed_of_light

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.font_manager as font_manager

from multiprocessing import Process

# path = ['./Cross_Corr', './Injections','./Plot_PSD', './Projections','./Subtractions']
# for path in path:
#     try:
#         if os.path.exists(path):
#             shutil.rmtree(path)
#             print("Directory {} has been Deleted".format(path))
#         else:
#             print('Nothing to Delete')
#     except OSError:
#         print('Deleting Error')

from Initial_data import InitialData
from Injection import InjectionSignal
from Subtraction import SubtractionSignal
from Projection import ProjectionSignal
from ORF_OF import detector_functions
from Cross_Correlation import CrossCorrelation
from plot_data import PSDWelch
from astro_stochastic_background import StochasticBackground
from distribution import Distribution

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

# Specify the output directory and the label name of the simulation.
outdir = 'Output'
label = 'injected_signal'
label1 = 'subtracted_signal'
label2 = 'projected_signal'
label3 = 'unresolved_signal'
bilby.utils.setup_logger(outdir=outdir, label=label)

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

if os.path.exists(os.path.join(outdir, 'Sum')):
    print("Sum directory already exist")
else :
    print("Sum directory does not exist")
    try:
        os.mkdir(os.path.join(outdir, 'Sum'))
    except OSError:
        print("Creation of the directory Sum failed")
    else:
        print("Successfully created the directory Sum")

## GW detectors priors
ifos = ['CE', 'ET_D_TR'] #, 'L1', 'H1', 'V1', 'K1']
sampling_frequency = 2048.
start_time = 1198800017
end_time   = 1230336017
n_seg = 10000
Tfft = 8

# parameters to be included in the Fisher matrix (others were kept fixed at posterior sampling)
parameters = ['mass_1', 'mass_2', 'luminosity_distance', 'theta_jn']

## Initialization of Data
data = InitialData()
data_sets = data.initial_data(ifos, sampling_frequency, start_time, end_time, Tfft, n_seg=n_seg)
sampling_frequency = data_sets[1]
start_time = data_sets[2]
end_time = data_sets[3]
duration = data_sets[4]
duration_seg = data_sets[5]

n_fft = data_sets[6]
N_samples = data_sets[7]
frequency = data_sets[8]
waveform_generator = data_sets[9]
IFOs = data_sets[10]
n_samples = data_sets[18]

function =  detector_functions()
plots_PSD = PSDWelch(IFOs)
cross_corr = CrossCorrelation()
Omega_gw = StochasticBackground()
distribution = Distribution()

function.initial(ifos,sampling_frequency, start_time, end_time, n_seg, Tfft)
Omega_gw.initial(ifos, sampling_frequency, start_time, end_time, n_seg, Tfft)

##########################################
""" PSD of GW detectors for given ifos """
##########################################
psd_detector = function.psd()

##############################
""" ORF and Optimal Filter """
##############################
gamma = function.overlap_reduction_function()
optimal_filter = function.optimal_filter_JH(gamma)
# function.plot_orf(gamma=gamma)
# function.plot_of(optimal_filter=optimal_filter)

#######################################
""" Stochastic Backgrounds and Plot """
#######################################
""" Astro (Omega_{GW}^{astro}) and Cosmological (Omega_{GW}^{cosmo})Backgrounds """
astro_omega_gw = Omega_gw.astro_Omega_gw()
cosmo_omega_gw = Omega_gw.cosmo_Omega_gw()
# plots_PSD.plot_omega_gw_astro_cosmo(frequency, astro_omega_gw, cosmo_omega_gw)

""" Cosmological Spectrum and COBE Spectrum """
cosmo_spectrum = Omega_gw.cosmo_spectrum()
cobe_spectrum = Omega_gw.cobe_spectrum()
PGWB = Omega_gw.PGWB()
# plots_PSD.plot_comso_omega_gw(frequency, comso_omega_gw=cosmo_spectrum, cobe_spectrum=cobe_spectrum, PGWB=PGWB)
# plot_all_gw = Omega_gw.plot_all_backgrounds()

#########################
""" CSD from Omega_gw """
#########################
cross_corr.initial(ifos, sampling_frequency, start_time, end_time, n_seg, gamma, optimal_filter, Tfft=Tfft)
## CSD from Omega_GW of Astrophysical Origin
CSD_from_Omega_astro = cross_corr.CSD_from_Omega_astro()
## CSD from Omega_GW of Cosmological Origin
CSD_from_Omega_cosmo = cross_corr.CSD_from_Omega_cosmo()
# plot_CSD_from_omega = plots_PSD.plot_CSD_from_Omega(IFOs, frequency, CSD_from_Omega_astro=CSD_from_Omega_astro, CSD_from_Omega_cosmo=CSD_from_Omega_cosmo)

########################################
""" Binary Signals and time segments """
########################################
## Number of Binary Signals
n_inj = 100
## Number of Time segments
n_seg = 1

""" Initializing the Injection, Subtraction and Projection Scripts """
injection = InjectionSignal()
subtraction = SubtractionSignal()
projection = ProjectionSignal()

###############################################################################
""" ## Reading the Injection signals to create the astrophysical forground. """
###############################################################################
"""## This one used for High SNR BBH signals. """
# injection_params = injection.injections_set(filename ='./Injection_file/injections_10e6.hdf5')

""" ## This one is used for BBH signals in between a redshift interval """
# injection_params = injection.redshift_cutoff(filename ='./injection_file/injections_10e6.hdf5', cutoff_low=3, cutoff_high=6)

""" ## This one is used to read Injection BBH signals in a normal/usual way. 
It can be used for Uniform BBH signals with SNR > 10 and SNR < 10. """
injection_params = injection.injections(filename ='./Injection_file/injections_10e6.hdf5')

"""## For Waveform Systematics study use these injection parameters."""
# injection_params = subtraction.subtraction_params(filename ='./Uniform_BBH_DATA/BBH_SNRg10/Uniform_bestfit_BBH_ML_SNRg10.pkl')

""" When using Uniform BBH Signals for SNR > 10 or SNR < 10, the following lines are must """
## Change the path with respective case.
list = open('./Uniform_BBH_DATA/BBH_SNRless10/IMRPhenomPv2_SNRL10_Rerun/Uniform_BBH_SNRless10.txt')
list = list.readline().split()
injection_params = injection_params.iloc[list]
injection_params.index = range(len(injection_params.index))

####################################################################################
"""  Reading estimated best-fit parameters corresponding to the Injected signals """
####################################################################################
""" If you only have the json data files, use the following command to read the max likelihood data files. 
One can also use the script "convert_to_pickle.py" for this purpose. Else go to line number 206."""

""" For High SNR BBH signals."""
# bestfit_params = subtraction.subtraction_parameters(filefolder='data', n_inj=n_inj)

""" For Uniform BBH signals having SNR > 10 or SNR < 10."""
# bestfit_params = subtraction.subtraction_params_Uniform(filefolder='Uniform_BBH_DATA/BBH_SNRless10/IMRPhenomPv2_SNRL10_Rerun/', Injection_file='./Uniform_BBH_DATA/BBH_SNRg10/Uniform_BBH_SNRless10.txt', n_inj=n_inj)

""" saving the best fit max likelihood after the parameter estimation as a pickle file."""
# best_fit_params = open("Uniform_bestfit_BBH_ML_SNRg10.pkl", 'wb')
# pickle.dump(bestfit_params, best_fit_params)
# best_fit_params.close()

""" Reading the pickle file, If you have pickle data file use the following commnad to read the subtraction params pickle data file. """
""" For High SNR BBH signals. """
# bestfit_params = subtraction.subtraction_params(filename='./Injection_file/bestfit_highSNR_maxlhood.pkl')

"""  For Uniform BBH signals having SNR > 10 or SNR < 10 """
bestfit_params = subtraction.subtraction_params(filename='./Uniform_BBH_DATA/BBH_SNRless10/IMRPhenomPv2_SNRL10_Rerun/Uniform_bestfit_BBH_ML_SNRless10.pkl')

""" Number of Injections which are not working for High SNR case i.e. PE didn't work well for these perticular signals. """
# nw = [8,85,86,87,94,102,103]
# injection_params = injection_params.drop(nw)
# bestfit_params = bestfit_params.drop(nw)
# for i in nw:
#     injection_params = injection_params.drop(i)
#     bestfit_params = bestfit_params.drop(i)

""" ## Selecting the first 100 BBHs from Injected and estimated best-fits signals. """
injection_params = injection_params.iloc[0:n_inj]
bestfit_params = bestfit_params.iloc[0:n_inj]

# print(injection_params)
# print(bestfit_params)

#####################################################################
""" Plotting the parameter estimation errors and distribution of BBH
parameters used to Inject into the Detector. """
#####################################################################
# plot_dist = distribution.plot_dist(data=injection_params)
# injection_params['mass_1'] = injection_params.pop('$m_{1}$')
# injection_params['mass_2'] = injection_params.pop('$m_{2}$')
# plot_errors = distribution.errors(parameters, injection_params, bestfit_params)

""" ## Defining sum arrays for CSD to take sum over all time segment """
sum_noise = np.zeros((len(IFOs), len(IFOs), len(frequency)), dtype = np.complex)
sum_inj = np.zeros((len(IFOs),len(IFOs), len(frequency)), dtype = np.complex)
sum_sub = np.zeros((len(IFOs),len(IFOs), len(frequency)), dtype = np.complex)
sum_proj = np.zeros((len(IFOs),len(IFOs), len(frequency)), dtype = np.complex)
sum_unres = np.zeros((len(IFOs),len(IFOs), len(frequency)), dtype = np.complex)
variance = np.zeros((len(IFOs),len(IFOs), len(frequency)))

#######################
""" Variance in PSD """
#######################
"""Variance defined over the Gaussian power spectral densities for each detector pair and normalized over
the number of fft samples for each time segment and total number of time segments."""
for d1 in np.arange(len(IFOs)):
    detector_1 = psd_detector[d1, :]
    for d2 in range(d1+1, len(IFOs)):
        detector_2 = psd_detector[d2, :]
        variance[d1,d2, :] = np.sqrt(detector_1 * detector_2) / np.sqrt(2 * duration_seg / Tfft * n_seg)
# plot_variance = plots_PSD.plot_variance(IFOs, frequency, variance, n_seg=None, method='noise_'+str(n_seg))

########################################
""" Injection-Subtraction-Projection """
########################################

for k in np.arange(1):

    print('k',k)
    print('Data segment: ', 100 * (k + 1) / n_seg, '%')

    """ ## Setting the seg_start_time and seg_end_time """
    seg_start_time = start_time + k * duration_seg
    seg_end_time = seg_start_time + duration_seg

    """# Initializing the Interferometer """
    IFOs = bilby.gw.detector.networks.InterferometerList(ifos)

    """ One can choose where you want to Inject and Subtract your Binary Signals.  """
    """ Choice1 : Setting strain data from zero noise """
    IFOs.set_strain_data_from_zero_noise(sampling_frequency,duration=duration_seg,start_time=seg_start_time)

    """# Choice2 : Setting strain data from power spectral density of the detector."""
    # IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency,duration=duration_seg,
    #                                                    start_time=seg_start_time)

    """## Time series for each detector"""
    time_series = np.zeros((len(IFOs), (N_samples)))
    ci = 0
    for ifo in IFOs:
        time_series[ci, :] = ifo.strain_data.time_domain_strain
        ci += 1

    ######################################
    """ injection Parameter and Signal """
    ######################################

    t_coalescence, inject_time_series, IFOs = injection.injection_signal(IFOs, sampling_frequency, seg_start_time=seg_start_time,
                                seg_end_time=seg_end_time, n_seg = k, N_samples=N_samples, injection_params = injection_params,
                                waveform_generator=waveform_generator, n_samples=n_samples)
    print('injected time series is',inject_time_series)

    # ########################################
    # """ Subtraction Parameter and Signal """
    # ########################################
    #
    # residual_noise_data, sub_time_series, IFOs = subtraction.subtraction_signal(IFOs, sampling_frequency=sampling_frequency,
    #                                     seg_start_time=seg_start_time, seg_end_time=seg_end_time, t_coalescence=t_coalescence, n_seg=k,
    #                                     N_samples=N_samples, bestfit_params=bestfit_params, waveform_generator=waveform_generator,
    #                                     n_samples=n_samples)
    #
    # print('subtract time series is', sub_time_series)
    # # sub_residual = np.save(outdir + '/Sum/sub_time_series_' + str(k) + '.npy', sub_time_series)
    #
    # ##################################################
    # """ Projection Parameter Derivative and Signal """
    # ##################################################
    #
    # proj_time_series = projection.projection_signal(IFOs, sampling_frequency=sampling_frequency,
    #                    seg_start_time=seg_start_time, seg_end_time=seg_end_time, t_coalescence=t_coalescence, n_seg=k, N_samples=N_samples,
    #                    bestfit_params=bestfit_params, waveform_generator=waveform_generator, residual_noise_data=residual_noise_data,
    #                    sub_time_series=sub_time_series, parameters=parameters, n_samples=n_samples)
    #
    # print('projected time series is', proj_time_series)
    #proj_residual = np.save(outdir + '/Sum/proj_time_series_' + str(k) + '.npy', proj_time_series)

    # ## Variance in projected time series is
    # proj_time_series_variance = np.zeros((len(IFOs)))
    # ci = 0
    # for ifo in IFOs:
    #     proj_time_series_variance[ci] = np.var(proj_time_series[ci,:])
    #     ci += 1
    # print('variance in projected time series is', proj_time_series_variance)

    # ########################
    # """ PSD_Welch Plots """
    # ########################
    #
    # ## For injection,subtraction and projection together
    # plot_data = plots_PSD.plot_data(IFOs, sampling_frequency=sampling_frequency, n_seg=k,
    #                                 inj_time_series=inject_time_series,
    #                                 sub_time_series=sub_time_series,
    #                                 proj_time_series=proj_time_series,
    #                                 Tfft=Tfft)


    #------------------------------------------------------------------------------------------------------------------#
    """ Apply an SNR cut to estimated best-fit signals to subtract the astrophysical foreground and then to project out
        the corresponding subtraction residuals. i.e. instead of remvoing all the identified signals from the detector
        data stream we are removing only those signals which lies above a selected SNR threshold value. """
    #------------------------------------------------------------------------------------------------------------------#

    ## Selected SNR threshold value
    SNR_cut = 2

    ######################################################################
    """ Subtraction Parameter and Signal above a particular SNR values """
    ######################################################################

    residual_noise_data_SNR, sub_time_series_SNR, IFOs, subtracted_signal_idx = subtraction.subtraction_signal_SNR(IFOs,
                                        sampling_frequency=sampling_frequency, seg_start_time=seg_start_time, seg_end_time=seg_end_time,
                                        t_coalescence=t_coalescence, n_seg=k, N_samples=N_samples, bestfit_params=bestfit_params,
                                        waveform_generator=waveform_generator, n_samples=n_samples, SNR_cut=SNR_cut)

    print('subtract time series for SNR is', sub_time_series_SNR)
    # sub_time_series_SNR_save = np.save(outdir + '/Sum/sub_time_series_SNR_cut_'+str(SNR_cut)+'_seg_'+ str(k) + '.npy', sub_time_series_SNR)

    ############################################
    """ Unresolved Signal or Confusion Noise """
    ############################################

    unresolved_time_series, IFOs = injection.unresolved_injection_signal(IFOs, sampling_frequency, seg_start_time=seg_start_time,
                                seg_end_time=seg_end_time, n_seg=k, N_samples=N_samples, injection_params=injection_params,
                                t_coalescence=t_coalescence, waveform_generator=waveform_generator, n_samples=n_samples,
                                SNR_cut=SNR_cut, subtracted_signal_idx=subtracted_signal_idx)

    print('Unresolved signals time series is', inject_time_series)
    print('Both time series are equal', (sub_time_series_SNR == unresolved_time_series).all())

    ###############################################################################
    """ Projection Parameter Derivative and Signal above a particular SNR value """
    ###############################################################################

    proj_time_series_SNR = projection.projection_signal_SNR(IFOs, sampling_frequency=sampling_frequency,
                           seg_start_time=seg_start_time, seg_end_time=seg_end_time, t_coalescence=t_coalescence, n_seg=k,
                           N_samples=N_samples, bestfit_params=bestfit_params, waveform_generator=waveform_generator,
                           residual_noise_data=residual_noise_data_SNR, sub_time_series=sub_time_series_SNR,
                           parameters=parameters, n_samples=n_samples, subtracted_signal_idx=subtracted_signal_idx)

    print('projected time series for SNR is', proj_time_series_SNR)
    # proj_time_series_SNR_save = np.save(outdir + '/Sum/proj_time_series_SNR_cut_'+str(SNR_cut)+'_seg_' + str(k) + '.npy', proj_time_series_SNR)

    print('Sub and Proj time series are equal', (sub_time_series_SNR == proj_time_series_SNR).all())
    print(' Proj and Unresolved time series are equal', (proj_time_series_SNR == unresolved_time_series).all())

    ########################
    """ PSD_Welch Plots """
    ########################

    ## For injection,subtraction and projection together
    plot_data = plots_PSD.plot_wunresolved_data(IFOs, sampling_frequency=sampling_frequency, n_seg=k,
                                    inj_time_series=inject_time_series, sub_time_series=sub_time_series_SNR,
                                    unresolved_time_series = unresolved_time_series,
                                    proj_time_series=proj_time_series_SNR,
                                    Tfft=Tfft, SNR_cut=SNR_cut)

    exit()

    # ###########################################################
    # """ PSD_Welch Plots altogether with and without SNR_cut """
    # ###########################################################
    #
    # sub_time_series = np.load('./Output/Sum/sub_time_series_'+str(k)+'.npy')
    # proj_time_series = np.load('./Output/Sum/proj_time_series_'+str(k)+'.npy')
    # # sub_time_series_cut1 = np.load('./Output/Sum/sub_time_series_SNR_cut_1_'+str(k)+'.npy')
    # # proj_time_series_cut1 = np.load('./Output/Sum/proj_time_series_SNR_cut_1_'+str(k)+'.npy')
    #
    # plot_data = plots_PSD.plot_data_SNR(IFOs, sampling_frequency=sampling_frequency, n_seg=k, inj_time_series=inject_time_series,
    #                                     sub_time_series=sub_time_series, sub_time_series_SNR=sub_time_series_SNR,
    #                                     proj_time_series=proj_time_series, proj_time_series_SNR=proj_time_series_SNR,
    #                                     Tfft=Tfft,SNR_cut=SNR_cut)

    #######################################################################
    """ Cross-Correlation between the time series for different signals """
    #######################################################################
    #-----------------------------------#
    """" Cross-Correlation for Noise """
    #-----------------------------------#
    cc_noise, cc_noise_1 = cross_corr.cross_correlation(time_series=time_series, method='noise', n_seg=k)
    cross_corr.plot_cc(cross_corr_=cc_noise_1, n_seg=k, method='noise')
    # cross_corr.plot_cc(cross_corr_=cc_noise, n_seg=k, method='noise_wo_OF', inj='none')

    """Taking sum over all measurement for noise cross-correlation i.e. cc_noise to calculate the mean deviation."""
    for d1 in np.arange(len(IFOs)):
        for d2 in np.arange(d1 + 1, len(IFOs)):
            sum_noise[d1, d2, :] += (cc_noise[d1, d2, :] / n_seg)

    """Calculate Combined sample mean of estimator for a multi-detector pair"""
    ## Plot sum of Noise time series with Cross-Correlation, Variance, CSD from astro and cosmo backgrounds
    if k == range(n_seg)[-1]:
        #plot_avg_csd = plots_PSD.plot_avg_csd(IFOs,frequency,sum_series=sum_noise,n_seg=None, method='Noise')
        plot_csd_var = plots_PSD.plot_csd_var(IFOs, frequency, sum_series=sum_noise, variance=variance,
                                              CSD_from_Omega_astro=CSD_from_Omega_astro,
                                              CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, n_seg=k, method='Noise')


    #----------------------------------------#
    """" Cross-Correlation for Injections """
    #----------------------------------------#
    cc_inj, cc_inj_1 = cross_corr.cross_correlation(time_series=inject_time_series, method = label, n_seg = k)
    cross_corr.plot_cc(cross_corr_=cc_inj_1, n_seg=k, method=label)
    # cross_corr.plot_cc(cross_corr_=cc_inj, n_seg=k, method='injected_wo_OF', inj=injected_i)

    """Taking sum over all measurement for injection cross-correlation i.e. cc_noise to calculate the mean deviation."""
    for d1 in np.arange(len(IFOs)):
        for d2 in np.arange(d1 + 1, len(IFOs)):
            sum_inj[d1, d2, :] += cc_inj[d1, d2, :] / n_seg

    # sum_inj_save = np.save(outdir + '/Sum/sum_inj_wo2_' + str(k) + '.npy', sum_inj)
    ## Plot sum of Injection time series with Cross-Correlation, Variance, CSD from astro and cosmo backgrounds
    if k == range(n_seg)[-1]:
        # plot_avg_csd = plots_PSD.plot_avg_csd(IFOs, frequency, sum_series=sum_injection, n_seg=None, method='Injections')
        plot_csd_var = plots_PSD.plot_csd_var(IFOs, frequency, sum_series=sum_inj, variance=variance,
                                              CSD_from_Omega_astro=CSD_from_Omega_astro,
                                              CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, n_seg=k, method='Injections')


    #-----------------------------------------#
    """ Cross-Correlation for Subtractions """
    #-----------------------------------------#
    cc_sub, cc_sub_1 = cross_corr.cross_correlation(time_series=sub_time_series_SNR, method = label1, n_seg = k)
    cross_corr.plot_cc(cross_corr_=cc_sub_1, n_seg = k, method = label1)
    # cross_corr.plot_cc(cross_corr_=cc_sub_1, n_seg=k, method='subtraction_wo_OF', inj=subtracted_x)

    """Taking sum over all measurement for subtraction cross-correlation i.e. cc_noise to calculate the mean deviation."""
    for d1 in np.arange(len(IFOs)):
        for d2 in np.arange(d1 + 1, len(IFOs)):
            sum_sub[d1, d2, :] += cc_sub[d1, d2, :] / n_seg

    ## Plot sum of Subtraction time series with Cross-Correlation, Variance, CSD from astro and cosmo backgrounds
    if k == range(n_seg)[-1]:
        # plot_avg_csd = plots_PSD.plot_avg_csd(IFOs, frequency, sum_series=sum_sub, n_seg=None, method='Subtraction')
        plot_csd_var = plots_PSD.plot_csd_var(IFOs, frequency, sum_series=sum_sub, variance=variance,
                                              CSD_from_Omega_astro=CSD_from_Omega_astro,
                                              CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, n_seg=k, method='Subtraction')

    # ---------------------------------------------#
    """ Cross-Correlation for unresolved signals """
    # ---------------------------------------------#
    cc_unres, cc_unres_1 = cross_corr.cross_correlation(time_series=unresolved_time_series, method=label3, n_seg=k)
    cross_corr.plot_cc(cross_corr_=cc_unres_1, n_seg=k, method=label3)
    # cross_corr.plot_cc(cross_corr_=cc_sub_1, n_seg=k, method='subtraction_wo_OF', inj=subtracted_x)

    """Taking sum over all measurement for unresolved signals cross-correlation i.e. cc_unres to calculate the mean deviation."""
    for d1 in np.arange(len(IFOs)):
        for d2 in np.arange(d1 + 1, len(IFOs)):
            sum_unres[d1, d2, :] += cc_unres[d1, d2, :] / n_seg

    ## Plot sum of Subtraction time series with Cross-Correlation, Variance, CSD from astro and cosmo backgrounds
    if k == range(n_seg)[-1]:
        # plot_avg_csd = plots_PSD.plot_avg_csd(IFOs, frequency, sum_series=sum_unres, n_seg=None, method='Unresolved')
        plot_csd_var = plots_PSD.plot_csd_var(IFOs, frequency, sum_series=sum_unres, variance=variance,
                                              CSD_from_Omega_astro=CSD_from_Omega_astro,
                                              CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, n_seg=k, method='Unresolved')


    #---------------------------------------#
    """ Cross-Correlation for Projections """
    #---------------------------------------#
    cc_proj, cc_proj_1 = cross_corr.cross_correlation(time_series=proj_time_series_SNR,method = label2, n_seg = k)
    cross_corr.plot_cc(cross_corr_=cc_proj_1, n_seg = k, method = label2)
    # cross_corr.plot_cc(cross_corr_=cc_proj, n_seg = k, method = 'projection_w/o_OF', inj = projected_z)

    """Taking sum over all measurement for projection cross-correlation i.e. cc_noise to calculate the mean deviation."""
    for d1 in np.arange(len(IFOs)):
        for d2 in np.arange(d1 + 1, len(IFOs)):
            sum_proj[d1, d2, :] += cc_proj[d1, d2, :] / n_seg

    ## Plot sum of Projection time series with Cross-Correlation, Variance, CSD from astro and cosmo backgrounds
    if k == range(n_seg)[-1]:
        # plot_avg_csd = plots_PSD.plot_avg_csd(IFOs, frequency, sum_series=sum_proj, n_seg=None, method='Projection')
        plot_csd_var = plots_PSD.plot_csd_var(IFOs, frequency, sum_series=sum_proj, variance=variance,
                                              CSD_from_Omega_astro=CSD_from_Omega_astro,
                                              CSD_from_Omega_cosmo=CSD_from_Omega_cosmo, n_seg=k, method='Projection')

    #################
    """ Plot CSDs """
    #################
    # plots_CSD = plots_PSD.plots_csd(IFOs, frequency, psd_series=psd_detector, cc_inj=cc_inj_1, cc_sub=cc_sub_1, cc_proj=cc_proj_1, n_seg=k)
