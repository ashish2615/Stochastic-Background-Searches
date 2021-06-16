from __future__ import division, print_function

import os
import sys
import bilby
import numpy as np
import scipy.signal as sp
from time import process_time

from Initial_data import InitialData
from Injection import  InjectionSignal
from Subtraction import SubtractionSignal

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "Times New Roman"

current_direc = os.getcwd()
# print("Current Working Directory is :", current_direc)

## Specify the output directory.
outmain = 'Output'
outdir = 'Projections'
label = 'Projections'
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

outdir1 = 'Fisher_Matrices'
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


class ProjectionSignal:

    def __init__(self):
        pass

    def projection_derivatives(self, ifo, vals_dict, waveform_generator, i_params, releps=1e-4):
        """
        Calculate the partial derivatives of a function at a set of values. The
        derivatives are calculated using the central difference, using an iterative
        method to check that the values converge as step size decreases.

        Parameters
        ----------
        ifos: iterable
            The list of interferometers.
        vals: array_like
            A set of values, that are passed to a function, at which to calculate
            the gradient of that function
        i_params: int
            Indices of parameters with respect to which derivatives are to be calculated
        releps: float, array_like, 1e-3
            The initial relative step size for calculating the derivative.

        Returns
        -------
        grads: array_like
            An array of gradients for each non-fixed value.
        """

        vals = list(vals_dict.values())
        keys = vals_dict.keys()

        # set steps
        if isinstance(releps, float):
            eps = np.abs(vals) * releps
            eps[eps == 0.] = releps  # if any values are zero set eps to releps
        elif isinstance(releps, (list, np.ndarray)):
            if len(releps) != len(vals):
                raise ValueError("Problem with input relative step sizes")
            eps = np.multiply(np.abs(vals), releps)
            eps[eps == 0.] = np.array(releps)[eps == 0.]
        else:
            raise RuntimeError("Relative step sizes are not a recognised type!")

        ## waveform as function of parameters
        def waveform(parameters):
            polarizations = waveform_generator.frequency_domain_strain(parameters)
            return ifo.get_detector_response(polarizations, parameters)

        i = i_params[0]

        # initial parameter diffs
        leps = eps[i]

        # get central finite difference
        fvals = np.copy(vals)
        bvals = np.copy(vals)

        # central difference
        fvals[i] += 0.5 * leps  # change forwards distance to half eps
        bvals[i] -= 0.5 * leps  # change backwards distance to half eps

        fvals_dict = dict(zip(keys, fvals))
        bvals_dict = dict(zip(keys, bvals))

        grads = (waveform(fvals_dict) - waveform(bvals_dict)) / leps

        if len(i_params) > 1:
            for i in i_params[1:]:
                # initial parameter diffs
                leps = eps[i]

                # get central finite difference
                fvals = np.copy(vals)
                bvals = np.copy(vals)

                # central difference
                fvals[i] += 0.5 * leps  # change forwards distance to half eps
                bvals[i] -= 0.5 * leps  # change backwards distance to half eps

                fvals_dict = dict(zip(keys, fvals))
                bvals_dict = dict(zip(keys, bvals))
                cdiff = (waveform(fvals_dict) - waveform(bvals_dict)) / leps

                grads = np.vstack((grads, cdiff))

        return grads

    def invertMatrixSVD(self, matrices, threshold=1e-16):
        """"
        This method avoids problems with degeneracy or numerical errors.

        For a given matrix X of order m*n we can decompose it in to U, diagS and V^T
        U and V^T are resultant real Unitary matrices.

        X (m * n) = U (n * n) diagS (n * m) V^T (m * m)

        U : ndarray
        Unitary matrix having left singular vectors as columns.
        diagS : ndarray
            The singular values, sorted in non-increasing order.
        V^T : ndarray
            Unitary matrix having right singular vectors as rows.
        thresh : int
            least value bound.
        """

        n_comp = len(matrices[:, 0])

        ## normalize matrices
        matrices_norm = np.zeros((n_comp, n_comp))
        for q in range(n_comp):
            for p in range(n_comp):
                matrices_norm[q, p] = matrices[q, p] / np.sqrt(matrices[p, p] * matrices[q, q])

        ## Inverse of matrix using singular value decomposition (SVD).
        [U, diagS, VT] = np.linalg.svd(matrices_norm)
        kVal = np.sum(diagS > threshold)
        iU = np.conjugate(U).T
        iV = np.conjugate(VT).T
        matrices_inverse = (iV[:, 0:kVal] @ np.diag(1 / diagS[0:kVal]) @ iU[0:kVal, :])

        ## Normalizing the inverse matrices
        for q in range(n_comp):
            for p in range(n_comp):
                matrices_inverse[q, p] = matrices_inverse[q, p] / np.sqrt(
                    matrices[p, p] * matrices[q, q])

        return matrices_inverse

    def projection_signal(self, IFOs, sampling_frequency=None, seg_start_time = None, seg_end_time = None, t_coalescence=None,n_seg=None, N_samples = None,
                         bestfit_params = None, waveform_generator = None, residual_noise_data=None, sub_time_series=None, parameters=None, n_samples=None):

        """
        Parameters
        ----------

        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.
        sampling_frequency: float
            The sampling frequency (in Hz).
        seg_start_time: duration
            The GPS start-time of the data for n_seg.
        seg_end_time: duration
            The GPS end-time of the data for n_seg.
        t_coalescence : array type
            Array represent a choice of coalescence time for binary signal.
            for information see Injection.py.
        n_seg: int
            number of time segment in total time duration of operation of detector.
        N_samples: int
            Number of samples in each segment of duration_duration_seg.
        bestfit_param : Dict
            Dictionary of max likelihood  of all best fit parameters.

        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.

        residual_noise_data: array like
                 Detector residual Data after Subtracting the best fit of each Signal.
        sub_time_series: array like
            Array of time series of subtraction residuals.
            Real valued array of time series calculated by taking inverse fft of residual_noise_data.
        parameters: list
            A list of parameter used to perform the parameter estimation and will be used to calculated
            the projection derivatives with respect to them.

        Return:

        proj_data_stream: array like
                Time series obtained by projecting the residual time series.
                Real valued array of time series calculated by taking inverse fft of residual_noise_data.

        """
        print('Projection script starting (', process_time(), ')')

        # check if Fisher-matrix calculation is contrained to a subset of parameters
        if parameters == None:
            n_params = len(bestfit_params.loc[0])
            i_params = range(n_params)
        else:
            n_params = len(parameters)
            keys = bestfit_params.loc[0].to_dict().keys()
            i_params = np.zeros(n_params, dtype=int)
            for k in range(n_params):
                i_params[k] = list(keys).index(parameters[k])

        proj_time_series = np.zeros((len(IFOs), N_samples))
        proj_data_stream = np.zeros((len(IFOs), n_samples))

        icnt = 0
        for ifo in IFOs:
            PSD = ifo.power_spectral_density
            sub_fourier = ifo.strain_data.frequency_domain_strain
            proj_fourier= 0

            tcnt = 0
            projected = False
            # loop over all signals
            for index, single_params in bestfit_params.iterrows():

                ## We used t_coalescence from Injection.py script to have the same coalescence time for an injected and subtracted binary signal from the data of the detector..
                t_c = t_coalescence[tcnt]
                single_params['geocent_time'] = t_c

                if seg_start_time < t_c < seg_end_time:

                    projected = True
                    print("Number of Projection signal is :", index)
                    print('seg_start_time', seg_start_time)
                    print('geocent_time', t_c)
                    print('seg_end_time', seg_end_time)

                    ## Change the luminosity_distance for reducing the signal from Noise of detector.
                    single_params['luminosity_distance'] = float(single_params['luminosity_distance'])

                    ## First mass needs to be larger than second mass (just to cross check)
                    if single_params['mass_1'] < single_params["mass_2"]:
                        tmp = projection_parameters['mass_1']
                        single_params['mass_1'] = single_params['mass_2']
                        single_params['mass_2'] = tmp

                    waveform_derivatives = self.projection_derivatives(ifo, single_params.to_dict(), waveform_generator, i_params)

                    ## Defining the Fisher Matrix
                    fisher_matrices = np.zeros((n_params, n_params))
                    ## Calculation of Fisher Matrix : A scalar product of signal model w.r.t. model parameters.
                    ## For every parameter in n_params
                    ## iterate through rows
                    for q in range(n_params):
                        ## iterate through columns
                        for p in range(q, n_params):

                            prod = bilby.gw.utils.inner_product(waveform_derivatives[q,:], waveform_derivatives[p,:],
                                                                      waveform_generator.frequency_array, PSD)
                            fisher_matrices[q, p] = prod
                            fisher_matrices[p, q] = prod

                    ## Defining the Correlation matrix = Inverse of Fisher Matrix
                    correlation_matrices = self.invertMatrixSVD(fisher_matrices, threshold=1e-14)

                    # print('Fisher Matrix is :', fisher_matrices)
                    # print(np.shape(fisher_matrices))
                    # print('Co-relation Matrix is :', correlation_matrices)
                    # print(np.shape(correlation_matrices))

                    # fisher_matrix_save = np.save(outdir1 + '/fisher_matrix_' + ifo.name + '_' + str(n_seg) + '_' + str(index) + '.npy', fisher_matrices)

                    ## Calculating the scalar product of derivatives of data signal w.r.t number of parameters and residual_noise_data
                    ## i.e. Noise Projection
                    scalar_product = np.zeros((n_params))
                    count = 0
                    ## Calculation of Scalar Product : A scalar product between two signals and defined by the inner product on the vector space of signals
                    ## For every detector in IFOs iteration is
                    for q in range(n_params):
                        scalar_product[q] = bilby.gw.utils.inner_product(waveform_derivatives[q,:], sub_fourier, waveform_generator.frequency_array, PSD)

                    # print('Scalar Product is ', scalar_product)

                    # add projections with respect to all signals in the data
                    proj_fourier += np.matmul(np.matmul(correlation_matrices, scalar_product), waveform_derivatives)

                    tcnt += 1

            proj_time_series[icnt, :] = sub_time_series[icnt, :] - bilby.core.utils.infft(proj_fourier, sampling_frequency)
            proj_data_stream[icnt,:] =  residual_noise_data[icnt,:] - proj_fourier

            print('Projection finished for detector ', ifo.name, ' (', process_time(), ')')

            icnt += 1

        if projected:
            label = 'Project_Seg_' + str(n_seg)
            # IFOs.save_data(outdir=outdir, label=label)
            cnt = 0
            for ifo in IFOs:
                ifo.plot_data(signal=proj_data_stream[cnt,:], outdir=outdir, label=label)
                cnt += 1

        # proj_time_series = np.zeros((len(IFOs), N_samples))
        # cnt = 0
        # for d in range(len(IFOs)):
        #     proj_time_series[cnt, :] = bilby.core.utils.infft(proj_data_stream[d, :], sampling_frequency)
        #     cnt += 1

        print('Projection script finished (', process_time(), ')')

        return  proj_time_series

    def projection_signal_SNR(self, IFOs, sampling_frequency=None, seg_start_time = None, seg_end_time = None, t_coalescence=None,
                              n_seg=None, N_samples = None, bestfit_params = None, waveform_generator = None, residual_noise_data=None,
                              sub_time_series=None,parameters=None, n_samples=None,subtracted_signal_idx=None):

        """

        This part of the script project out the residuals of only those CBC signals which are above the selected SNR threshold value.


        Parameters
        ----------

        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.
        sampling_frequency: float
            The sampling frequency (in Hz).
        seg_start_time: duration
            The GPS start-time of the data for n_seg.
        seg_end_time: duration
            The GPS end-time of the data for n_seg.
        t_coalescence : array type
            Array represent a choice of coalescence time for binary signal.
            for information see Injection.py.
        n_seg: int
            number of time segment in total time duration of operation of detector.
        N_samples: int
            Number of samples in each segment of duration_duration_seg.
        bestfit_param : Dict
            Dictionary of max likelihood  of all best fit parameters.

        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.
        residual_noise_data: array like
            Detector residual Data after Subtracting the best fit of each Signal.
        sub_time_series: array like
            Array of time series of subtraction residuals.
            Real valued array of time series calculated by taking inverse fft of residual_noise_data.
        parameters: list
            A list of parameter used to perform the parameter estimation and will be used to calculated
            the projection derivatives with respect to them.
        subtracted_signal_idx: array like
            An array on all those subtracted CBC signals which lies above the selected SNR threshold value.
            These signal index will be used to project out the residual of subtracted CBC signals.


        Return:

        proj_data_stream: array like
                Time series obtained by projecting the residual time series.
                Real valued array of time series calculated by taking inverse fft of residual_noise_data.

        """
        print('Projection script starting (', process_time(), ')')

        # check if Fisher-matrix calculation is contrained to a subset of parameters
        if parameters == None:
            n_params = len(bestfit_params.loc[0])
            i_params = range(n_params)
        else:
            n_params = len(parameters)
            keys = bestfit_params.loc[0].to_dict().keys()
            i_params = np.zeros(n_params, dtype=int)
            for k in range(n_params):
                i_params[k] = list(keys).index(parameters[k])

        proj_time_series = np.zeros((len(IFOs), N_samples))
        proj_data_stream = np.zeros((len(IFOs), n_samples))

        icnt = 0
        for ifo in IFOs:
            PSD = ifo.power_spectral_density
            sub_fourier = ifo.strain_data.frequency_domain_strain
            proj_fourier= 0

            tcnt = 0
            projected = False
            # loop over all signals
            for index, single_params in bestfit_params.iterrows():
                ## Check if the index is in the  subtracted_signal_idx.
                if index in subtracted_signal_idx:

                    ## We used t_coalescence from Injection.py script to have the same coalescence time for an injected and subtracted binary signal from the data of the detector..
                    t_c = t_coalescence[tcnt]
                    print('tc is',t_c)
                    single_params['geocent_time'] = t_c

                    if seg_start_time < t_c < seg_end_time:

                        projected = True
                        print("Number of Projection signal is :", index)
                        print('seg_start_time', seg_start_time)
                        print('geocent_time', t_c)
                        print('seg_end_time', seg_end_time)

                        ## Change the luminosity_distance for reducing the signal from Noise of detector.
                        single_params['luminosity_distance'] = float(single_params['luminosity_distance'])

                        ## First mass needs to be larger than second mass (just to cross check)
                        if single_params['mass_1'] < single_params["mass_2"]:
                            tmp = projection_parameters['mass_1']
                            single_params['mass_1'] = single_params['mass_2']
                            single_params['mass_2'] = tmp

                        waveform_derivatives = self.projection_derivatives(ifo, single_params.to_dict(), waveform_generator, i_params)

                        ## Defining the Fisher Matrix
                        fisher_matrices = np.zeros((n_params, n_params))
                        ## Calculation of Fisher Matrix : A scalar product of signal model w.r.t. model parameters.
                        ## For every parameter in n_params
                        ## iterate through rows
                        for q in range(n_params):
                            ## iterate through columns
                            for p in range(q, n_params):

                                prod = bilby.gw.utils.inner_product(waveform_derivatives[q,:], waveform_derivatives[p,:],
                                                                          waveform_generator.frequency_array, PSD)
                                fisher_matrices[q, p] = prod
                                fisher_matrices[p, q] = prod

                        ## Defining the Correlation matrix = Inverse of Fisher Matrix
                        correlation_matrices = self.invertMatrixSVD(fisher_matrices, threshold=1e-14)

                        # print('Fisher Matrix is :', fisher_matrices)
                        # print(np.shape(fisher_matrices))
                        # print('Co-relation Matrix is :', correlation_matrices)
                        # print(np.shape(correlation_matrices))

                        # fisher_matrix_save = np.save(outdir1 + '/fisher_matrix_' + ifo.name + '_' + str(n_seg) + '_' + str(index) + '.npy', fisher_matrices)

                        ## Calculating the scalar product of derivatives of data signal w.r.t number of parameters and residual_noise_data
                        ## i.e. Noise Projection
                        scalar_product = np.zeros((n_params))
                        count = 0
                        ## Calculation of Scalar Product : A scalar product between two signals and defined by the inner product on the vector space of signals
                        ## For every detector in IFOs iteration is
                        for q in range(n_params):
                            scalar_product[q] = bilby.gw.utils.inner_product(waveform_derivatives[q,:], sub_fourier, waveform_generator.frequency_array, PSD)

                        # print('Scalar Product is ', scalar_product)

                        # add projections with respect to all signals in the data
                        proj_fourier += np.matmul(np.matmul(correlation_matrices, scalar_product), waveform_derivatives)

                tcnt += 1

            proj_time_series[icnt, :] = sub_time_series[icnt, :] - bilby.core.utils.infft(proj_fourier, sampling_frequency)
            proj_data_stream[icnt,:] =  residual_noise_data[icnt,:] - proj_fourier

            print('Projection finished for detector ', ifo.name, ' (', process_time(), ')')

            icnt += 1

        if projected:
            label = 'Project_Seg_' + str(n_seg)
            # IFOs.save_data(outdir=outdir, label=label)
            cnt = 0
            for ifo in IFOs:
                ifo.plot_data(signal=proj_data_stream[cnt,:], outdir=outdir, label=label)
                cnt += 1

        # proj_time_series = np.zeros((len(IFOs), N_samples))
        # cnt = 0
        # for d in range(len(IFOs)):
        #     proj_time_series[cnt, :] = bilby.core.utils.infft(proj_data_stream[d, :], sampling_frequency)
        #     cnt += 1

        print('Projection script finished (', process_time(), ')')

        return  proj_time_series
