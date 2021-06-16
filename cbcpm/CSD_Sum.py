from __future__ import division, print_function

import os
import sys
import logging
import numpy as np
import math

class SumCrossCorr():

    def __init__(self):
        pass

    def sum(self, IFOs, frequency=None, filefolder=None, n_seg=None):
        """

        :return:
        """

        ## reading the Sum directory.
        basedir = os.path.join(os.path.dirname(__file__), filefolder)
        ## list of all files
        sum_data = os.listdir(basedir)
        sum_file_select = [x for x in sum_data if x.startswith('sum_')]

        sum_noise = np.zeros((len(IFOs), len(IFOs), len(frequency)), dtype=np.complex)
        sum_inj   = np.zeros((len(IFOs), len(IFOs), len(frequency)), dtype=np.complex)
        sum_sub   = np.zeros((len(IFOs), len(IFOs), len(frequency)), dtype=np.complex)
        sum_proj  = np.zeros((len(IFOs), len(IFOs), len(frequency)), dtype=np.complex)

        i = 0
        for sum_file in (sum_file_select):
            if i < n_seg:

                sum_data_direc = os.path.join(basedir)

                ## Read Sum Noise file
                sum_data_file_noise = os.path.join(sum_data_direc,'sum_noise_' + str(i) + '.npy')
                ## Taking sum of sum noise for all time segment given by n_seg
                if os.path.isfile(sum_data_file_noise):
                    sum_noise_read = np.load(sum_data_file_noise)
                    # print('sum_noise', sum_noise_read)
                    # print(np.shape(sum_noise_read))

                    for d1 in np.arange(len(IFOs)):
                        for d2 in np.arange(d1 + 1,len(IFOs)):
                            sum_noise[d1, d2, :] += sum_noise_read[d1, d2, :]

                ## Read Sum Injection file
                sum_data_file_inj = os.path.join(sum_data_direc, 'sum_inj_' + str(i) + '.npy')
                ## Taking sum of Sum Injections for all time segment given by n_seg
                print('check', sum_data_file_inj)
                if os.path.isfile(sum_data_file_inj):
                    sum_inj_read = np.load(sum_data_file_inj)
                    # print('sum_inj', sum_inj_read)
                    # print(np.shape(sum_inj_read))

                    for d1 in np.arange(len(IFOs)):
                        for d2 in np.arange(d1 + 1, len(IFOs)):
                            sum_inj[d1, d2, :] += sum_inj_read[d1, d2, :]

                # else:
                #     sum_data_file_inj = os.path.join(sum_data_direc, 'sum_injection_' + str(i) + '.npy')
                #     sum_inj_read = np.load(sum_data_file_inj)
                #
                #     for d1 in np.arange(len(IFOs)):
                #         for d2 in np.arange(d1 + 1, len(IFOs)):
                #             sum_inj[d1, d2, :] += sum_inj_read[d1, d2, :]

                ## Read Sum Subtraction file
                sum_data_file_sub = os.path.join(sum_data_direc, 'sum_sub_' + str(i) + '.npy')
                ## Taking sum of Sum Subtractions for all time segment given by n_seg
                if os.path.isfile(sum_data_file_sub):
                    sum_sub_read = np.load(sum_data_file_sub)
                    # print('sum_sub', sum_sub_read)

                    for d1 in np.arange(len(IFOs)):
                        for d2 in np.arange(d1 + 1, len(IFOs)):
                            sum_sub[d1, d2, :] += sum_sub_read[d1, d2, :]

                ## Read Sum Projection file
                sum_data_file_proj = os.path.join(sum_data_direc, 'sum_proj_' + str(i) + '.npy')
                ## Taking sum of Sum Projections for all time segment given by n_seg
                if os.path.isfile(sum_data_file_proj):
                    sum_proj_read = np.load(sum_data_file_proj)
                    # print('sum_proj', sum_proj_read)

                    for d1 in np.arange(len(IFOs)):
                        for d2 in np.arange(d1 + 1, len(IFOs)):
                            sum_proj[d1, d2, :] += sum_proj_read[d1, d2, :]

                i += 1

        return sum_noise, sum_inj, sum_sub, sum_proj
