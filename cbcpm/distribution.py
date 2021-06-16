from __future__ import division, print_function

import os
import sys
import bilby
import deepdish
import numpy as np
import logging
import pandas as pd
import json
import math
import sklearn
import seaborn as sns
from scipy.stats import norm

import matplotlib
#matplotlib.use('tkagg')
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "Times New Roman"

current_direc = os.getcwd()
# print("Current Working Directory is :", current_direc)
## Specify the output directory and the name of the simulation.
outmain = 'Output'
outdir = 'New_Plot_Dist_highSNR'
label = 'Plot_Dist'
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

outdir1 = 'Errors'
outdir1 = os.path.join(outmain, outdir1)

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


class Distribution():

    def __init__(self):
        pass

    def plot_dist(self, data=None):
        """
        data : array type
            BBH signals Injected into the Detector

        """

        data.mass_1, data.mass_2 = np.where(data.mass_1 < data.mass_2, [data.mass_2, data.mass_1],
                                            [data.mass_1, data.mass_2])

        data['$m_{1}$'] = data.pop('mass_1')
        data['$m_{2}$'] = data.pop('mass_2')

        for key in data.keys():

            ## For just 100 signals Use seaborn distplot
            ## For all injection signal use seaborn kde and rug plot commands.

            font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
            font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='small')

            with sns.axes_style('white'):

                m1 = (data['$m_{1}$'])
                m2 = (data['$m_{2}$'])
                m1, m2 = np.where(m1 < m2, [m2, m1], [m1, m2])

                ## Total Mass M = m1 + m2
                M_tot = m1 + m2
                ## Chirp Mass
                M_chirp = (m1 * m2) ** (3 / 5) * M_tot ** (-1 / 5)

                sns.set()
                sns.set_style("whitegrid")
                # sns.set_style("white")
                sns.set_style("ticks")
                # ## Plot is kind = 'hex', Plot1 is for kind = 'kde'
                plot = sns.jointplot(m1, m2, kind='hex')
                cax = plot.fig.add_axes([.92, .4, .02, .3]) # [.98, .4, .01, .2])
                plt.colorbar(cax=cax)
                plot.ax_joint.set_xlabel('$m_1 [M_{\odot}]$', fontsize=14)
                plot.ax_joint.set_ylabel('$m_2 [M_{\odot}]$', fontsize=14)
                # plt.xlim(5, 70)
                # plt.ylim(5, 70)
                plt.tight_layout()
                plt.savefig(outdir + '/masses_1&2_plot', dpi=300)
                plt.close()

            if key == '$m_{1}$':

                sns.set()
                sns.set_style("whitegrid")
                # sns.set_style("white")
                # sns.set_style("ticks")
                # mass_1_plot  = plt.hist(data[key], bins = 500, density=True )
                ## set rug = False for all injections
                mass_1 = sns.distplot(data[key], hist=False, rug=True, color='g')
                ## comment sns.rugplot for all 10**6 signals.
                # mass_1 = sns.kdeplot(data[key], shade=True)
                # mass_1 = sns.rugplot(data[key])
                plt.xlim(5, 70)
                plt.xlabel('$ m_1 ~[M_{\odot}]$', fontsize=14)
                plt.ylabel('Probability Distribution P($m_{1}$)', fontsize=14)
                plt.tight_layout()
                plt.savefig(outdir + '/mass_1', dpi=300)
                plt.close()

            if key == '$m_{2}$':

                sns.set()
                sns.set_style("whitegrid")
                # sns.set_style("white")
                # sns.set_style("ticks")
                # mass_2_plot  = plt.hist(data[key], bins = 500, density=True )
                ## set rug = False for all 10**6 injection signals.
                mass_2 = sns.distplot(data[key], hist=False, rug=True, color='g')
                # mass_2 = sns.kdeplot(data[key], shade=True)
                # mass_2 = sns.rugplot(data[key])
                plt.xlim(5, 70)
                plt.xlabel('$ m_{2}~ [M_{\odot}]$', fontsize=14)
                plt.ylabel('Probability Distribution P($m_{2}$)', fontsize=14)
                plt.tight_layout()
                plt.savefig(outdir + '/mass_2', dpi=300)
                plt.close()

            if key == 'redshift':

                sns.set()
                sns.set_style("whitegrid")
                # sns.set_style("ticks")

                # redshift = plt.hist(data[key], bins = 500, density=True)
                redshift = sns.distplot(data[key], hist=False, rug=True)
                # redshift = sns.kdeplot(data[key], shade=True)
                # redshift = sns.rugplot(data[key])
                # plt.xlim(0,0.1)
                plt.xlabel('Redshift (z)', fontsize=14)
                # plt.ylabel('bins')
                plt.ylabel('Probability Distribution P(z)', fontsize=14)
                plt.tight_layout()
                plt.savefig(outdir + '/redshift', dpi=300)
                plt.close()

                ## Luminosity Distance.
                luminsoity_dist = bilby.gw.conversion.redshift_to_luminosity_distance(data['redshift'])
                ## divied by 1000 to remove the factor of 10^{5} on x-axis. So we can use Gpc instead of Mpc.
                luminsoity_dist = luminsoity_dist / 1000
                # luminsoity= plt.hist(luminsoity_dist, bins = 500, density=True )
                luminosity = sns.distplot(luminsoity_dist, hist=False, rug=True)
                # luminosity  = sns.kdeplot(luminsoity_dist, shade=True)
                # luminosity  = sns.rugplot(data[key])
                # plt.xlim(0,0.1)
                plt.xlabel('Luminosity Distance $d_{L}~$[Gpc]', fontsize=14)
                # plt.ylabel('bins')
                plt.ylabel('Probability Distribution P($d_{L}$)', fontsize=14)
                plt.tight_layout()
                plt.savefig(outdir + '/luminosity_distance', dpi=300)
                plt.close()

                sns.set()
                sns.set_style("whitegrid")
                sns.set_style("ticks")
                tot_mass = sns.jointplot(M_tot, data[key], kind='kde', color="#4CB391", shade=True)
                tot_mass.plot_joint(plt.scatter, c='black', s=100, linewidth=1, marker='+')
                tot_mass.ax_joint.collections[0].set_alpha(0.8)
                # tot_mass = sns.jointplot(M_tot, data[key], kind="hex", color="#4CB391")
                plt.xlabel('Total Mass $~ M_{tot} ~[M_{\odot}]$', fontsize=20)
                plt.ylabel('Redshift z', fontsize=20)
                tot_mass.fig.set_size_inches(10, 8)
                # tot_mass.add_gridspecs(2,2)
                plt.tight_layout()
                plt.savefig(outdir + '/Total_Mass_vs_Redshift', dpi=300)
                plt.close()

            if key == 'iota':

                sns.set()
                sns.set_style("whitegrid")
                # sns.set_style("ticks")
                # iota_plot  = plt.hist(data[key], bins = 500, density=True )
                iota_plot = sns.distplot(data[key], hist=False, rug=True)
                # iota = sns.kdeplot(data[key], shade=True)
                # iota = sns.rugplot(data[key])
                plt.xlabel('Inclination Angle $\iota$ [rad]', fontsize=14)
                # plt.ylabel('Probability')
                plt.ylabel('Probability Distribution P($\iota$)', fontsize=14)
                plt.tight_layout()
                plt.savefig(outdir + '/iota', dpi=300)
                plt.close()

    def errors(self, parameters, injection_params, bestfit_params):
        """
        To plot the histogram for Injected parameters, bestfit parameters and the difference between these two for each BBH signal.

        parameters: list
            List of parameter used to perform PE for best-fit parameters.
        injection_params: DataFrame
            DataFrame of all BBH Injection signals used to Inject into the Detector
        bestfit_params:  DataFrame
            DataFrame of all best-fit estimated parameters correspoing to Injection signals.
        """

        injection_params['theta_jn'] = injection_params['iota']
        injection_params['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(injection_params['redshift'])

        # print(injection_params['luminosity_distance'])
        # print('bestfit_params', bestfit_params['luminosity_distance'])

        mass1 = []
        mass2 = []
        iota = []
        dl = []

        for key in parameters:

            if key == 'mass_1':

                m1 = injection_params['mass_1'] - bestfit_params['mass_1']
                mass1.append(m1)

            if key == 'mass_2':
                m2 = injection_params['mass_2'] - bestfit_params['mass_2']
                mass2.append(m2)

            if key == 'theta_jn':
                iota = injection_params['theta_jn'] - bestfit_params['theta_jn']
                iota.append(iota)

            if key == 'luminosity_distance':
                injection_params['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(injection_params['redshift'])
                dl = injection_params['luminosity_distance'] - bestfit_params['luminosity_distance']
                dl.append(dl)

        print(mass1)
        print(mass2)
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        ## Mass_1
        sns.distplot(injection_params['mass_1'], bins=20, hist=True, kde=True, label='True $~m_{1}$')
        sns.distplot(bestfit_params['mass_1'], bins=20, hist=True, kde=True, label='Best-fit $~m_{1}$')
        sns.distplot(mass1, bins=20, hist=True, kde=True, label='Diff in $~m_{1}$')
        legend = plt.legend(loc='best', prop=font1)
        plt.xlabel('$ m_{1}~ [M_{\odot}]$', fontsize=14)
        plt.ylabel('Probability P($m_{1}$)', fontsize=14)
        plt.tight_layout()
        plt.savefig(outdir1 + '/mass_1', dpi=300)
        plt.close()

        ## Mass_2
        sns.distplot(injection_params['mass_2'], bins=20, hist=True, kde=True, label='True $~m_{2}$')
        sns.distplot(bestfit_params['mass_2'], bins=20, hist=True, kde=True, label='Best-fit $~m_{2}$')
        sns.distplot(mass2, bins=20, hist=True, kde=True, label='Diff in $~m_{2}$')
        legend = plt.legend(loc='best', prop=font1)
        plt.xlabel('$ m_{2}~ [M_{\odot}]$', fontsize=14)
        plt.ylabel('Probability  P($m_{2}$)', fontsize=14)
        plt.tight_layout()
        plt.savefig(outdir1 + '/mass_2', dpi=300)
        plt.close()

        ## Inclination Angle
        sns.histplot(injection_params['iota'], bins=20, stat="probability", kde=True, color='r', label='True' +'$~$'+ r'$\theta_{jn}$')
        sns.histplot(bestfit_params['theta_jn'], bins=20, stat="probability", kde=True, color='b', label='Best-fit'+ '$~$' + r'$\theta_{jn}$')
        sns.histplot(iota, bins=20, stat="probability", kde=True, color='g', label='Diff in'+ '$~$' + r'$\theta_{jn}$')
        legend = plt.legend(loc='best', prop=font1)
        plt.xlabel(r'$\theta_{jn}$ [rad]', fontsize=14)
        plt.ylabel(r'Probability P($\theta_{jn}$)', fontsize=14)
        plt.xlim(-0.2, 3)
        # plt.ylim(0, 10)
        plt.tight_layout()
        plt.savefig(outdir1 + '/theta_jn', dpi=300)
        plt.close()

        ## Luminosity Distance
        sns.histplot(injection_params['luminosity_distance'] / 1000, bins=20, stat="probability", kde=True, color='r', label='True $D_{L}~$[Gpc]')
        sns.histplot(bestfit_params['luminosity_distance'] / 1000, bins=20, stat="probability",  kde=True, color='b', label='Best-fit $D_{L}~$[Gpc]')
        sns.histplot(dl / 1000, bins=20, stat="probability",  kde=True, color='g', label='Diff in $D_{L}~$[Gpc]')
        legend = plt.legend(loc='best', prop=font1)
        plt.xlabel(' Luminosity Distance $D_{L}~[Gpc]$', fontsize=14)
        plt.ylabel('Probability P($D_{L}$)', fontsize=14)
        plt.xlim(-2, 20)
        # plt.ylim(0, 2)
        plt.tight_layout()
        plt.savefig(outdir1 + '/DL', dpi=300)
        plt.close()