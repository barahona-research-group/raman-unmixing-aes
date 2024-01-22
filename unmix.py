import os
import glob
import pickle
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import ramanspy as rp

import autoencoders


def set_seeds(seed: int):
    """
    A function that sets relevant seeds for reproducibility.


    Parameters
    ----------
    seed: int
        The seed to use for reproducibility.
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _compute_sad_matrix(matrix1, matrix2):
    # Normalize matrices
    matrix1 = matrix1 / np.linalg.norm(matrix1, axis=1, keepdims=True)
    matrix2 = matrix2 / np.linalg.norm(matrix2, axis=1, keepdims=True)

    # Compute the dot product between each pair of vectors in matrix1 and matrix2
    dot_product = np.dot(matrix1, matrix2.T)

    # Clip the dot product to avoid numerical errors
    dot_product = np.clip(dot_product, -1, 1)

    # Compute the spectral angle distance
    sad = np.arccos(dot_product)

    return sad


def _pearson(A, B):
    # Compute mean values
    mean_A, mean_B = np.mean(A), np.mean(B)

    # Compute PCC
    numerator = np.sum((A - mean_A) * (B - mean_B))
    denominator = np.sqrt(np.sum((A - mean_A) ** 2) * np.sum((B - mean_B) ** 2))

    # Protect against division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)

    pcc = numerator / denominator

    return 1 - pcc


def _compute_pearson_matrix(A, B):
    num_gt = A.shape[0]
    num_derived = B.shape[0]

    # Initialize the distance matrix with zeros
    distance_matrix = np.zeros((num_gt, num_derived))

    # Compute PCC for each pair of endmembers
    for i in range(num_gt):
        for j in range(num_derived):
            distance_matrix[i, j] = _pearson(A[i], B[j])

    return distance_matrix


def _compare_endmembers(ground_truth_endmembers, predicted_endmembers):
    dist_matrix_sad = _compute_sad_matrix(ground_truth_endmembers, predicted_endmembers)
    dist_matrix_pcc = _compute_pearson_matrix(ground_truth_endmembers, predicted_endmembers)

    # Use the Hungarian algorithm to find the optimal assignment based in SAD
    row_ind, col_ind = linear_sum_assignment(dist_matrix_sad)

    return dist_matrix_sad[row_ind, col_ind].mean(), dist_matrix_pcc[row_ind, col_ind].mean(), row_ind, col_ind


def _mse(image1, image2):
    return np.mean((image1 - image2) ** 2)


def evaluate(predicted_endmembers, predicted_abundances, gt_endmembers, gt_abundances):
    """
    A function that evaluates the performance of endmember and abundance estimation algorithms.


    Parameters
    ----------
    predicted_endmembers: np.ndarray
        The predicted endmembers.
    predicted_abundances: np.ndarray
        The predicted abundances.
    gt_endmembers: np.ndarray
        The ground truth endmembers.
    gt_abundances: np.ndarray
        The ground truth abundances.

    Returns
    -------
    mse: float
        The mean squared error of the abundances.
    sad: float
        The mean spectral angle distance between the ground truth and predicted endmembers.
    pcc: float
        The mean Pearson correlation coefficient between the ground truth and predicted endmembers.
    ordered_endmembers: np.ndarray
        The predicted endmembers ordered according to the ground truth endmembers.
    ordered_abundances: np.ndarray
        The predicted abundances ordered according to the ground truth endmembers.
    """

    # Compute the mean spectral angle distance
    sad, pcc, row_ind, col_ind = _compare_endmembers(gt_endmembers, predicted_endmembers)

    # Reorder the predicted endmembers and abundances based on the optimal assignment
    ordered_endmembers = predicted_endmembers[col_ind, ...]
    ordered_abundances = predicted_abundances[col_ind, ...]

    # Compute the mean squared error of the abundances
    mse = _mse(gt_abundances, np.moveaxis(ordered_abundances, 0, -1))

    return mse, sad, pcc, ordered_endmembers, ordered_abundances


def load_data(dataset_folder):
    """
    A function that loads a dataset stored in a folder.


    Parameters
    ----------
    dataset: str
        The path to the dataset.

    Returns
    -------
    gt_endmembers: np.ndarray
        The ground truth endmembers.
    gt_abundances: np.ndarray
        The ground truth abundances.
    datasets: np.ndarray or list[np.ndarray]
        A list of mixed images.
    spectral_axis: np.ndarray
        The spectral axis in wavenumbers (cm^-1).
    """
    with open(os.path.join(dataset_folder, 'gt_endmembers.pkl'), 'rb') as f:
        gt_endmembers = pickle.load(f)
    with open(os.path.join(dataset_folder, 'gt_abundance_image.pkl'), 'rb') as f:
        gt_abundances = pickle.load(f)
    with open(os.path.join(dataset_folder, 'spectral_axis.pkl'), 'rb') as f:
        spectral_axis = pickle.load(f)

    files = [f.split('/')[-1] for f in glob.glob(os.path.join(dataset_folder, '*.pkl'))]

    dataset_files = [d for d in files if
                     d.split('.')[0] not in ['gt_abundance_image', 'gt_endmembers', 'spectral_axis', 'parameters']]

    datasets = []
    for dataset_file in dataset_files:
        with open(os.path.join(dataset_folder, dataset_file), 'rb') as f:
            mixed_image = pickle.load(f)
            datasets.append(mixed_image)

    if len(datasets) == 1:
        return gt_endmembers, gt_abundances, datasets[0], spectral_axis

    return gt_endmembers, gt_abundances, datasets, spectral_axis


def init_models(spectral_size, num_endmembers, decoder_type='linear', include_conventional=True, asc=True):
    """
    A function that initializes the autoencoder models used in the experiments.


    Parameters
    ----------
    spectral_size: int
        The number of spectral bands in the hyperspectral data.
    num_endmembers: int
        The number of endmembers to estimate.
    decoder_type: str, optional
        The type of decoder to use in the autoencoders. Default is 'linear'.
    include_conventional: bool, optional
        Whether to include conventional methods in the experiments. Default is True.
    asc: bool, optional
        Whether to apply the abundance sum-to-one constraint. Default is True.

    Returns
    -------
    methods: list
        A list of tuples containing the names and models.
    """
    methods = []

    if include_conventional:
        abundance_method = 'fcls' if asc else 'nnls'

        # Add conventional methods
        methods += [
            ('PCA', rp.analysis.decompose.PCA(n_components=num_endmembers)),
            (f'N-FINDR + {abundance_method.upper()}',
             rp.analysis.unmix.NFINDR(n_endmembers=num_endmembers, abundance_method=abundance_method)),
            (f'VCA + {abundance_method.upper()}',
             rp.analysis.unmix.VCA(n_endmembers=num_endmembers, abundance_method=abundance_method)),
        ]

    # Add autoencoder methods
    methods += [
        ('Dense AE', autoencoders.DenseAE(
            input_dim=spectral_size, bottleneck_dim=num_endmembers, decoder_type=decoder_type,
            encoder_hidden_dims=[128], asc=asc)),
        ('Convolutional AE', autoencoders.ConvolutionalAE(
            input_dim=spectral_size, bottleneck_dim=num_endmembers, decoder_type=decoder_type, kernel_sizes=[3, 5],
            num_filters=[16, 16], encoder_hidden_dims=[128], asc=asc)),
        ('Transformer AE', autoencoders.TransformerAE(
            input_dim=spectral_size, bottleneck_dim=num_endmembers, decoder_type=decoder_type, d_model=32, num_heads=2,
            num_layers=1, asc=asc)),
        ('Convolutional Transformer AE', autoencoders.ConvolutionalTransformerAE(
            input_dim=spectral_size, bottleneck_dim=num_endmembers, decoder_type=decoder_type, d_model=32, num_heads=2,
            num_layers=1, kernel_sizes=[3, 5], num_filters=[16, 16], asc=asc)),
    ]

    return methods


def _ae_unmixing(model, spectra, *, epochs, verbose, loss=autoencoders.SAD, learning_rate=0.001):
    if not isinstance(spectra, list):
        spectra = [spectra]

    spectral_data_concat = np.concatenate([v.flat.spectral_data for v in spectra])

    # shuffle the data
    np.random.shuffle(spectral_data_concat)

    model.compile(optimizer=tf.optimizers.legacy.Adam(learning_rate=learning_rate), loss=loss)
    model.fit(spectral_data_concat, spectral_data_concat, epochs=epochs, verbose=verbose)

    # get endmembers
    endmembers = model.get_endmembers()

    # get abundances
    abundances = [model.get_abundances(scan.flat.spectral_data).numpy().reshape(list(scan.shape) + [-1]) for scan in
                  spectra]

    return abundances, endmembers


def unmixing(model, data, epochs=10, verbose=0, ae_loss=autoencoders.SAD):
    """
    A function that performs unmixing using either an autoencoder or a conventional method.


    Parameters
    ----------
    model: ramanspy.analysis.AnalysisStep or UnmixingAE
        The model to use for unmixing.
    data: ramanspy.SpectralContainer or list[ramanspy.SpectralContainer]
        The data to unmix.
    epochs: int, optional
        The number of epochs to train the autoencoder for. Default is 10.
    verbose: int, optional
        The verbosity level. Default is 0.
    ae_loss: callable, optional
        The loss function to use for the autoencoder. Default is autoencoders.SAD.

    Returns
    -------
    endmembers: list[np.ndarray]
        The estimated endmembers.
    abundance_maps: np.ndarray or list[np.ndarray]
        The estimated abundance maps.
    """
    if isinstance(model, autoencoders._UnmixingAE):
        abundance_maps, endmembers = _ae_unmixing(model, data, epochs=epochs, verbose=verbose, loss=ae_loss)

        # move channels to the last axis to comply with ramanspy unmixing
        abundance_maps = [np.moveaxis(abundance_map, -1, 0) for abundance_map in abundance_maps]

        if len(abundance_maps) == 1:
            abundance_maps = abundance_maps[0]

    elif isinstance(model, rp.analysis.unmix.AnalysisStep):
        abundance_maps, endmembers = model.apply(data)
    else:
        raise ValueError('Model must be either an AnalysisStep or an UnmixingAE object.')

    return endmembers, abundance_maps
