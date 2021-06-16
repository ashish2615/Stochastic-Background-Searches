from __future__ import division, print_function

import os
import sys
import bilby
import numpy as np
import pandas as pd
import json
import pickle

from Subtraction import SubtractionSignal

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

# Specify the output directory and the label name of the simulation. BBH_SNRless10/Uniform_BBH_SNRless10.txt
outdir = './Uniform_BBH_DATA/'
label = 'Uniform_BBH_SNRless10'
bilby.utils.setup_logger(outdir=outdir, label=label)

injection_file = './Uniform_BBH_DATA/IMRPhenomPv2_SNRL10_Rerun/Uniform_BBH_SNRless10.txt'
inj_signal = open('./Uniform_BBH_DATA/IMRPhenomPv2_SNRL10_Rerun/Uniform_BBH_SNRless10.txt')
n_inj = inj_signal.readline().split()
print(len(n_inj))

subtraction = SubtractionSignal()
# saving the best-fit max-likelihood after the parameter estimation as a pickle file.
bestfit_params = subtraction.subtraction_params_IMR(filefolder='./Uniform_BBH_DATA/IMRPhenomPv2_SNRL10_Rerun/outdir/json', Injection_file=injection_file, n_inj=n_inj)
best_fit_params = open("Uniform_bestfit_BBH_ML_SNRless10.pkl", 'wb')
pickle.dump(bestfit_params, best_fit_params)
# bestfit_params.close()