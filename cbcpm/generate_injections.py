###############################################
# BBH injections
# generate_injections.py 100 binary_black_holes_cosmo.prior injections_test.hdf5
# No of injections = 1000000 ; Redshift = 10 ; mass = {min=5, max = 60}
###############################################
# __future__ package give new -- or different -- meaning to words or symbols in the file i.e.
# used to enable new language features which are not compatible with the current interpreter

from __future__ import division, print_function
import os
import bilby
import sys
import pandas as pd
import numpy as np

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

# Specify the output directory and the label name of the simulation.
outdir = 'outdir'
label = 'injected_signal'
bilby.utils.setup_logger(outdir=outdir, label=label)

n_inject = 1000000 # int(sys.argv[1])

#try:
#   n_inject = int(float(sys.argv[1]))
#except ValueError:
#   print ("Stupid user, please enter a number")
#   sys.exit(1)
#print(n_inject)

#if sys.argv[1].isdigit():
#    a = int(sys.argv[1])
#else:
#    print ("First argument is not a digit")
#    sys.exit(1)
#print(a)

prior_file = 'binary_black_holes_comso_uniform.prior'     # sys.argv[2]
out_file =   './Injection_file/injections_10e6.hdf5' # sys.argv[3]

## BBH prior file for a Uniform distribution.
prior = bilby.gw.prior.BBHPriorDict(filename= 'binary_black_holes_cosmo_uniform.prior')
pzBH = np.loadtxt("PzBH.txt")
#pzBH = np.loadtxt("PzBH.txt")
prior['redshift'] = bilby.core.prior.Interped(
                                              name='redshift', xx=pzBH[:, 0], yy=pzBH[:, 1], minimum=0, maximum=10)
prior['mass_1'] = bilby.core.prior.PowerLaw(
                                            name='mass_1', alpha=-1.6, minimum=5, maximum=60, unit='$M_{\\odot}$')
prior['mass_2'] = bilby.core.prior.PowerLaw(
    name='mass_2', alpha=-1.6, minimum=5, maximum=60, unit='$M_{\\odot}$')
print('Prior created')
injections = prior.sample(n_inject)
samples = pd.DataFrame(injections)
print('Samples generated')
samples.to_hdf(out_file, key='injections')
print('Samples saved to {}'.format(out_file))
