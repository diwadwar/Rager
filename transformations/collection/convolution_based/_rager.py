import multiprocessing
from itertools import combinations

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.validation import check_n_jobs


class Rager(BaseCollectionTransformer):
    """Rager is a modified MultiRocket transformer for human action and gesture recognition
    based on skeletal data. Changes from original MultiRocket:
    1. Removed first-order differenced series calculation to be invariant to execution speed.
    2. Fixed to 6 features per kernel: PPV, MPV, LSPV, TVA, PNV, MNV.
    3. Zero-Sum Dipole (ZSD) spatial transformation:
       Before any spatial aggregation, every selected channel combination is forced to have 
       about half of its channels multiplied by +1, and the other half by -1.
    4. Hybrid Spatial Extraction with Mean Absolute Deviatio (MAD): 
       - Features 0-3 alternate 50/50 between Zero-Sum Dipole (Signed Spatial Sum) 
         and Signed Spatial MAD.
       - Features 4-5 ALWAYS use the ZSD because MAD is always positive 
         and theese features operate on negative values.

    Parameters
    ----------
    n_kernels : int, default = 8,333
       Number of random convolutional kernels. The calculated number of features is the
       nearest multiple of ``n_features_per_kernel(default 6)*84=336 < 50,000``
       (``n_features_per_kernel(default 6)*n_kernels(default 8,333)``).
    max_dilations_per_kernel : int, default = 32
        Maximum number of dilations per kernel.
    normalise : bool, default False
        Whether or not to normalise the input time series per instance.
    n_jobs : int, default=1
        The number of jobs to run in parallel for `transform`. ``-1`` means using all
        processors.
    random_state : None or int, default = None
        Seed for random number generation.

    Attributes
    ----------
    parameter : tuple
        Parameter (dilations, n_features_per_dilation, biases) for
        transformation of input `X`.
    parameter1 : tuple
        Parameter (dilations, n_features_per_dilation, biases) for
        transformation of input ``X1 = np.diff(X, 1)``.


    See Also
    --------
    Rocket, MiniRocket, MultiRocket, HydraTransformer
    
    Examples
    --------
    Please refer to "example_validation.py"
    """

    _tags = {
        "output_data_type": "Tabular",
        "algorithm_type": "convolution",
        "capability:multivariate": True,
        "capability:multithreading": True,
    }
    # indices for the 84 kernels
    _indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype=np.int32)

    def __init__(
        self,
        n_kernels=8_333,
        max_dilations_per_kernel=32,
        normalise=False,
        n_jobs=1,
        random_state=None,
    ):
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = 6
        self.n_kernels = n_kernels

        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.parameter = None

        super().__init__()

    def _fit(self, X, y=None):
        """Fit dilations and biases to input time series.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.random_state_ = (
            np.int32(self.random_state) if isinstance(self.random_state, int) else None
        )
        if self.random_state_ is not None:
            np.random.seed(self.random_state_)

        _, n_channels, n_timepoints = X.shape
        if n_timepoints < 9:
            raise ValueError(
                f"n_timepoints must be >= 9, but found {n_timepoints};"
                " zero pad shorter series so that n_timepoints == 9"
            )
        X = X.astype(np.float32)
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )
            
        if n_channels == 1:
            X = X.squeeze()
            self.parameter = self._fit_univariate(X)
        else:
            self.parameter = self._fit_multivariate(X)

        return self

    def _transform(self, X, y=None):
        """Transform input time series using random convolutional kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            Collection of time series to transform.
        y : ignored argument for interface compatibility

        Returns
        -------
        pandas DataFrame, transformed features
        """
        _, n_channels, n_timepoints = X.shape
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )
        
        # change n_jobs depending on value and existing cores
        prev_threads = get_num_threads()
        if self._n_jobs < 1 or self._n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self._n_jobs
        set_num_threads(n_jobs)

        X = X.astype(np.float32)
        
        if n_channels > 1:
            X = _transform_multi(
                X,
                self.parameter,
                self.n_features_per_kernel,
                Rager._indices,
                self.random_state_,
            )
        else:
            X = X.reshape(X.shape[0], X.shape[2])
            X = _transform_uni(
                X,
                self.parameter,
                self.n_features_per_kernel,
                Rager._indices,
                self.random_state_,
            )

        X = np.nan_to_num(X)

        set_num_threads(prev_threads)
        return X

    def _fit_univariate(self, X):
        _, input_length = X.shape
        n_kernels = 84
        dilations, n_features_per_dilation = _fit_dilations(
            input_length, self.n_kernels, self.max_dilations_per_kernel
        )
        n_features_per_kernel = np.sum(n_features_per_dilation)
        quantiles = _quantiles(n_kernels * n_features_per_kernel)

        biases, biases_sum = _fit_biases_univariate(
            X,
            dilations,
            n_features_per_dilation,
            quantiles,
            Rager._indices,
            self.random_state_,
        )

        return dilations, n_features_per_dilation, biases, biases_sum

    def _fit_multivariate(self, X):
        _, n_channels, input_length = X.shape
        n_kernels = 84

        dilations, n_features_per_dilation = _fit_dilations(
            input_length, self.n_kernels, self.max_dilations_per_kernel
        )
        n_features_per_kernel = np.sum(n_features_per_dilation)
        quantiles = _quantiles(n_kernels * n_features_per_kernel)

        n_dilations = len(dilations)
        n_combinations = n_kernels * n_dilations

        max_n_channels = min(n_channels, 9)
        max_exponent = np.log2(max_n_channels + 1)

        n_channels_per_combination = (
            2 ** np.random.uniform(0, max_exponent, n_combinations)
        ).astype(np.int32)

        channel_indices = np.zeros(n_channels_per_combination.sum(), dtype=np.int32)
        channel_signs = np.zeros(n_channels_per_combination.sum(), dtype=np.float32)

        n_channels_start = 0
        for combination_index in range(n_combinations):
            n_channels_this_combination = n_channels_per_combination[combination_index]
            n_channels_end = n_channels_start + n_channels_this_combination
            channel_indices[n_channels_start:n_channels_end] = np.random.choice(
                n_channels, n_channels_this_combination, replace=False
            )

            n_neg = n_channels_this_combination // 2
            signs = np.ones(n_channels_this_combination, dtype=np.float32)
            signs[:n_neg] = -1.0
            np.random.shuffle(signs)
            
            channel_signs[n_channels_start:n_channels_end] = signs
            n_channels_start = n_channels_end

        biases, biases_sum = _fit_biases_multivariate(
            X,
            n_channels_per_combination,
            channel_indices,
            channel_signs,
            dilations,
            n_features_per_dilation,
            quantiles,
            Rager._indices,
            self.random_state_,
        )

        return (
            n_channels_per_combination,
            channel_indices,
            channel_signs,
            dilations,
            n_features_per_dilation,
            biases,
            biases_sum,
        )


@njit(
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform_uni(
    X, parameters, n_features_per_kernel, indices, seed
):
    if seed is not None:
        np.random.seed(seed)
    n_cases, n_timepoints = X.shape

    dilations, n_features_per_dilation, biases, biases_sum = parameters
    n_kernels = len(indices)
    n_dilations = len(dilations)

    n_features = n_kernels * np.sum(n_features_per_dilation)

    features = np.zeros(
        (n_cases, n_features * n_features_per_kernel),
        dtype=np.float32,
    )

    for example_index in prange(n_cases):
        _X = X[example_index]

        A = -_X # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        # Base series
        feature_index_start = 0

        for dilation_index in range(n_dilations):
            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            n_features_this_dilation = n_features_per_dilation[dilation_index]

            C_alpha = np.zeros(n_timepoints, dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(n_kernels):
                feature_index_end = feature_index_start + n_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(n_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]
                        _bias_sum = biases_sum[feature_index]
                        
                        ppv = 0
                        mean_pos = 0.0
                        last_val_pos = 0
                        max_stretch_pos = 0.0

                        pnv = 0
                        mean_neg = 0.0
                        
                        tva_sum = 0.0
                        prev_c = C[0]

                        for j in range(C.shape[0]):
                            val = C[j]
                            tva_sum += abs(val - prev_c)
                            prev_c = val
                            
                            if val > _bias:
                                ppv += 1
                                mean_pos += val
                            elif val < _bias:
                                stretch = j - last_val_pos
                                if stretch > max_stretch_pos:
                                    max_stretch_pos = stretch
                                last_val_pos = j

                            if val < -_bias_sum:
                                pnv += 1
                                mean_neg += val

                        stretch_pos = C.shape[0] - 1 - last_val_pos
                        if stretch_pos > max_stretch_pos:
                            max_stretch_pos = stretch_pos
                            
                        end = feature_index
                        features[example_index, end] = ppv / C.shape[0]
                        end += n_features
                        features[example_index, end] = mean_pos / ppv if ppv > 0 else 0.0
                        end += n_features
                        features[example_index, end] = max_stretch_pos
                        end += n_features
                        features[example_index, end] = tva_sum / C.shape[0]
                        end += n_features
                        features[example_index, end] = pnv / C.shape[0]
                        end += n_features
                        features[example_index, end] = mean_neg / pnv if pnv > 0 else 0.0

                else:
                    _c = C[padding:-padding]

                    for feature_count in range(n_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]
                        _bias_sum = biases_sum[feature_index]

                        ppv = 0
                        mean_pos = 0.0
                        last_val_pos = 0
                        max_stretch_pos = 0.0

                        pnv = 0
                        mean_neg = 0.0
                        
                        tva_sum = 0.0
                        prev_c = _c[0]

                        for j in range(_c.shape[0]):
                            val = _c[j]
                            tva_sum += abs(val - prev_c)
                            prev_c = val
                            
                            if val > _bias:
                                ppv += 1
                                mean_pos += val
                            elif val < _bias:
                                stretch = j - last_val_pos
                                if stretch > max_stretch_pos:
                                    max_stretch_pos = stretch
                                last_val_pos = j

                            if val < -_bias_sum:
                                pnv += 1
                                mean_neg += val

                        stretch_pos = _c.shape[0] - 1 - last_val_pos
                        if stretch_pos > max_stretch_pos:
                            max_stretch_pos = stretch_pos

                        end = feature_index
                        features[example_index, end] = ppv / _c.shape[0]
                        end += n_features
                        features[example_index, end] = mean_pos / ppv if ppv > 0 else 0.0
                        end += n_features
                        features[example_index, end] = max_stretch_pos
                        end += n_features
                        features[example_index, end] = tva_sum / _c.shape[0]
                        end += n_features
                        features[example_index, end] = pnv / _c.shape[0]
                        end += n_features
                        features[example_index, end] = mean_neg / pnv if pnv > 0 else 0.0

                feature_index_start = feature_index_end

    return features


@njit(
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform_multi(
    X, parameters, n_features_per_kernel, indices, seed
):
    n_cases, n_channels, n_timepoints = X.shape
    (
        n_channels_per_combination,
        channel_indices,
        channel_signs,
        dilations,
        n_features_per_dilation,
        biases,
        biases_sum,
    ) = parameters
    
    if seed is not None:
        np.random.seed(seed)

    n_kernels = len(indices)
    n_dilations = len(dilations)

    n_features = n_kernels * np.sum(n_features_per_dilation)

    features = np.zeros(
        (n_cases, n_features * n_features_per_kernel),
        dtype=np.float32,
    )

    for example_index in prange(n_cases):
        _X = X[example_index]

        A = -_X # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0
        combination_index = 0
        n_channels_start = 0

        for dilation_index in range(n_dilations):
            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            n_features_this_dilation = n_features_per_dilation[dilation_index]

            C_alpha = np.zeros((n_channels, n_timepoints), dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, n_channels, n_timepoints), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = n_timepoints - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(n_kernels):
                feature_index_end = feature_index_start + n_features_this_dilation

                n_channels_this_combination = n_channels_per_combination[
                    combination_index
                ]

                n_channels_end = n_channels_start + n_channels_this_combination

                channels_this_combination = channel_indices[
                    n_channels_start:n_channels_end
                ]
                
                signs_this_combination = channel_signs[
                    n_channels_start:n_channels_end
                ]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C_raw = (
                    C_alpha[channels_this_combination]
                    + C_gamma[index_0][channels_this_combination]
                    + C_gamma[index_1][channels_this_combination]
                    + C_gamma[index_2][channels_this_combination]
                )
                
                # Calculating ZSD and MAD
                n_t = C_raw.shape[1]
                C_sum = np.zeros(n_t, dtype=np.float32)
                C_mad = np.zeros(n_t, dtype=np.float32)
                
                if n_channels_this_combination > 1:
                    inv_N = np.float32(1.0 / n_channels_this_combination)
                    # ZSD
                    for c_idx in range(n_channels_this_combination):
                        sign = signs_this_combination[c_idx]
                        for t_idx in range(n_t):
                            C_sum[t_idx] += C_raw[c_idx, t_idx] * sign
                    
                    # MAD
                    for c_idx in range(n_channels_this_combination):
                        sign = signs_this_combination[c_idx]
                        for t_idx in range(n_t):
                            diff = (C_raw[c_idx, t_idx] * sign) - (C_sum[t_idx] * inv_N)
                            C_mad[t_idx] += abs(diff)
                            
                    for t_idx in range(n_t):
                        C_mad[t_idx] = C_mad[t_idx] * inv_N
                else:
                    sign = signs_this_combination[0]
                    for t_idx in range(n_t):
                        C_sum[t_idx] = C_raw[0, t_idx] * sign

                if _padding1 == 0:
                    for feature_count in range(n_features_this_dilation):
                        is_mad = (feature_count % 2 == 1)
                        if is_mad and n_channels_this_combination > 1:
                            C_target = C_mad
                        else:
                            C_target = C_sum
                            
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]
                        _bias_sum = biases_sum[feature_index]

                        ppv = 0
                        mean_pos = 0.0
                        last_val_pos = 0
                        max_stretch_pos = 0.0

                        pnv = 0
                        mean_neg = 0.0
                        
                        tva_sum = 0.0
                        prev_c = C_target[0]

                        for j in range(C_target.shape[0]):
                            val = C_target[j]
                            tva_sum += abs(val - prev_c)
                            prev_c = val
                            
                            if val > _bias:
                                ppv += 1
                                mean_pos += val
                            elif val < _bias:
                                stretch = j - last_val_pos
                                if stretch > max_stretch_pos:
                                    max_stretch_pos = stretch
                                last_val_pos = j

                            val_sum = C_sum[j]
                            if val_sum < -_bias_sum:
                                pnv += 1
                                mean_neg += val_sum

                        stretch_pos = C_target.shape[0] - 1 - last_val_pos
                        if stretch_pos > max_stretch_pos:
                            max_stretch_pos = stretch_pos
                            
                        end = feature_index
                        features[example_index, end] = ppv / C_target.shape[0]
                        end += n_features
                        features[example_index, end] = mean_pos / ppv if ppv > 0 else 0.0
                        end += n_features
                        features[example_index, end] = max_stretch_pos
                        end += n_features
                        features[example_index, end] = tva_sum / C_target.shape[0]
                        end += n_features
                        features[example_index, end] = pnv / C_sum.shape[0]
                        end += n_features
                        features[example_index, end] = mean_neg / pnv if pnv > 0 else 0.0

                else:
                    _c_sum = C_sum[padding:-padding]
                    _c_mad = C_mad[padding:-padding]
                    
                    for feature_count in range(n_features_this_dilation):
                        is_mad = (feature_count % 2 == 1)
                        if is_mad and n_channels_this_combination > 1:
                            C_target = _c_mad
                        else:
                            C_target = _c_sum
                            
                        feature_index = feature_index_start + feature_count
                        _bias = biases[feature_index]
                        _bias_sum = biases_sum[feature_index]

                        ppv = 0
                        mean_pos = 0.0
                        last_val_pos = 0
                        max_stretch_pos = 0.0

                        pnv = 0
                        mean_neg = 0.0
                        
                        tva_sum = 0.0
                        prev_c = C_target[0]

                        for j in range(C_target.shape[0]):
                            val = C_target[j]
                            tva_sum += abs(val - prev_c)
                            prev_c = val
                            
                            if val > _bias:
                                ppv += 1
                                mean_pos += val
                            elif val < _bias:
                                stretch = j - last_val_pos
                                if stretch > max_stretch_pos:
                                    max_stretch_pos = stretch
                                last_val_pos = j

                            val_sum = _c_sum[j]
                            if val_sum < -_bias_sum:
                                pnv += 1
                                mean_neg += val_sum

                        stretch_pos = C_target.shape[0] - 1 - last_val_pos
                        if stretch_pos > max_stretch_pos:
                            max_stretch_pos = stretch_pos

                        end = feature_index
                        features[example_index, end] = ppv / C_target.shape[0]
                        end += n_features
                        features[example_index, end] = mean_pos / ppv if ppv > 0 else 0.0
                        end += n_features
                        features[example_index, end] = max_stretch_pos
                        end += n_features
                        features[example_index, end] = tva_sum / C_target.shape[0]
                        end += n_features
                        features[example_index, end] = pnv / _c_sum.shape[0]
                        end += n_features
                        features[example_index, end] = mean_neg / pnv if pnv > 0 else 0.0

                feature_index_start = feature_index_end

                combination_index += 1
                n_channels_start = n_channels_end

    return features


@njit(
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases_univariate(
    X, dilations, n_features_per_dilation, quantiles, indices, seed
):
    if seed is not None:
        np.random.seed(seed)

    n_cases, input_length = X.shape
    n_kernels = len(indices)
    n_dilations = len(dilations)

    n_features = n_kernels * np.sum(n_features_per_dilation)

    biases = np.zeros(n_features, dtype=np.float32)
    biases_sum = np.zeros(n_features, dtype=np.float32)

    feature_index_start = 0

    for dilation_index in range(n_dilations):
        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        n_features_this_dilation = n_features_per_dilation[dilation_index]

        for kernel_index in range(n_kernels):
            feature_index_end = feature_index_start + n_features_this_dilation

            _X = X[np.random.randint(n_cases)]

            A = -_X # A = alpha * X = -X
            G = _X + _X + _X # G = gamma * X = 3X

            C_alpha = np.zeros(input_length, dtype=np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype=np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            q = np.quantile(C, quantiles[feature_index_start:feature_index_end])
            biases[feature_index_start:feature_index_end] = q
            biases_sum[feature_index_start:feature_index_end] = q

            feature_index_start = feature_index_end

    return biases, biases_sum


@njit(
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases_multivariate(
    X,
    n_channels_per_combination,
    channel_indices,
    channel_signs,
    dilations,
    n_features_per_dilation,
    quantiles,
    indices,
    seed,
):
    if seed is not None:
        np.random.seed(seed)

    n_cases, n_channels, input_length = X.shape

    n_kernels = len(indices)
    n_dilations = len(dilations)

    n_features = n_kernels * np.sum(n_features_per_dilation)

    biases = np.zeros(n_features, dtype=np.float32)
    biases_sum = np.zeros(n_features, dtype=np.float32)

    feature_index_start = 0

    combination_index = 0
    n_channels_start = 0

    for dilation_index in range(n_dilations):
        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        n_features_this_dilation = n_features_per_dilation[dilation_index]

        for kernel_index in range(n_kernels):
            feature_index_end = feature_index_start + n_features_this_dilation

            n_channels_this_combination = n_channels_per_combination[combination_index]

            n_channels_end = n_channels_start + n_channels_this_combination

            channels_this_combination = channel_indices[n_channels_start:n_channels_end]
            signs_this_combination = channel_signs[n_channels_start:n_channels_end]

            _X = X[np.random.randint(n_cases)][channels_this_combination]

            A = -_X # A = alpha * X = -X
            G = _X + _X + _X # G = gamma * X = 3X

            C_alpha = np.zeros(
                (n_channels_this_combination, input_length), dtype=np.float32
            )
            C_alpha[:] = A

            C_gamma = np.zeros(
                (9, n_channels_this_combination, input_length), dtype=np.float32
            )
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C_raw = (
                C_alpha
                + C_gamma[index_0]
                + C_gamma[index_1]
                + C_gamma[index_2]
            )
            
            n_t = C_raw.shape[1]
            C_sum = np.zeros(n_t, dtype=np.float32)
            C_mad = np.zeros(n_t, dtype=np.float32)
            
            if n_channels_this_combination > 1:
                inv_N = np.float32(1.0 / n_channels_this_combination)
                for c_idx in range(n_channels_this_combination):
                    sign = signs_this_combination[c_idx]
                    for t_idx in range(n_t):
                        C_sum[t_idx] += C_raw[c_idx, t_idx] * sign
                
                for c_idx in range(n_channels_this_combination):
                    sign = signs_this_combination[c_idx]
                    for t_idx in range(n_t):
                        diff = (C_raw[c_idx, t_idx] * sign) - (C_sum[t_idx] * inv_N)
                        C_mad[t_idx] += abs(diff)
                        
                for t_idx in range(n_t):
                    C_mad[t_idx] = C_mad[t_idx] * inv_N
            else:
                sign = signs_this_combination[0]
                for t_idx in range(n_t):
                    C_sum[t_idx] = C_raw[0, t_idx] * sign

            for feature_count in range(n_features_this_dilation):
                is_mad = (feature_count % 2 == 1)
                
                if is_mad and n_channels_this_combination > 1:
                    C_target = C_mad
                else:
                    C_target = C_sum
                    
                idx = feature_index_start + feature_count
                biases[idx] = np.quantile(C_target, quantiles[idx])
                biases_sum[idx] = np.quantile(C_sum, quantiles[idx])

            feature_index_start = feature_index_end

            combination_index += 1
            n_channels_start = n_channels_end

    return biases, biases_sum


def _fit_dilations(input_length, n_features, max_dilations_per_kernel):
    n_kernels = 84

    n_features_per_kernel = n_features // n_kernels
    true_max_dilations_per_kernel = min(n_features_per_kernel, max_dilations_per_kernel)
    multiplier = n_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, n_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
            np.int32
        ),
        return_counts=True,
    )
    n_features_per_dilation = (n_features_per_dilation * multiplier).astype(
        np.int32
    )

    remainder = n_features_per_kernel - np.sum(n_features_per_dilation)
    i = 0
    while remainder > 0:
        n_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(n_features_per_dilation)

    return dilations, n_features_per_dilation

# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )