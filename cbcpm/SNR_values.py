from __future__ import division, print_function

import os
import sys
import bilby
import numpy as np

from Injection import InjectionSignal
from Subtraction import SubtractionSignal

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

###############################################################
# Reading the Optimal SNR values of each Injected singal into #
# the Interferometer for Parameter Estimation.                #
###############################################################
## Total number of signals.
n_inj = 101
subtraction = SubtractionSignal()
optmal_snr_values = subtraction.SNR_bestfit('uniform_max_lhood_SNRg10', n_inj=n_inj)