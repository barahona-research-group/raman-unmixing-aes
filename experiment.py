import argparse
import json
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import ramanspy as rpy

import autoencoders
import unmix


def _summarise_results(gt_endmembers, gt_abundances, predicted_endmembers, predicted_abundances, model_names, colors, endmembers_normalize=False, experiment_folder=None):
    os.makedirs(os.path.join(experiment_folder, 'endmembers'), exist_ok=True)
    os.makedirs(os.path.join(experiment_folder, 'abundances'), exist_ok=True)

    N = len(gt_endmembers)
    M = 1 + len(model_names)

    metrics = {}
    ordered_endmembers = []
    ordered_abundances = []
    for m, model_name in enumerate(model_names):
        if np.isnan(predicted_endmembers[m]).any() or np.isnan(predicted_abundances[m]).any():
            mse = sad = pcc = np.nan
            endmembers = np.zeros_like(gt_endmembers)
            abundances = np.zeros_like(gt_abundances)
        else:
            mse, sad, pcc, endmembers, abundances = unmix.evaluate(predicted_endmembers[m], predicted_abundances[m], gt_endmembers, gt_abundances)

        metrics[model_name] = [sad, pcc, mse]

        ordered_endmembers.append(endmembers)
        ordered_abundances.append(abundances)

        # Save the derived endmembers
        np.save(os.path.join(experiment_folder, 'endmembers', f'{model_name}.npy'), np.array(endmembers))

        # Save the derived abundances
        np.save(os.path.join(experiment_folder, 'abundances', f'{model_name}.npy'), np.array(abundances))

    # Save the GT endmembers and abundances
    np.save(os.path.join(experiment_folder, 'abundances', f'gt.npy'), np.array(gt_abundances))
    np.save(os.path.join(experiment_folder, 'endmembers', f'gt.npy'), np.array(gt_endmembers))

    # Save metrics
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['SAD', 'PCC', 'MSE'])
    metrics_df.to_csv(os.path.join(experiment_folder, 'metrics.csv'))

    for i in range(len(model_names)):
        if model_names[i] == 'Convolutional Transformer AE':
            model_names[i] = 'Convolutional\nTransformer AE'

    if len(gt_abundances.shape) == 3:
        # Plot the fractional abundances
        fig, axs = plt.subplots(N, M, sharex='col', sharey='row', figsize=(M*1.5, N*1.5))
        for i in range(N):
            cmap = LinearSegmentedColormap.from_list('custom', ['white', colors(i)], N=256)
            for j in range(M):
                if j == 0:  # First column for ground truth
                    axs[i, j].imshow(gt_abundances[..., i], cmap=cmap, vmin=0, vmax=1)
                else:  # Model predictions
                    axs[i, j].imshow(ordered_abundances[j-1][i], cmap=cmap, vmin=0, vmax=1)
                if i == N - 1:  # Labels for the models at the bottom
                    if j == 0:
                        axs[i, j].set_xlabel('Ground truth')
                    else:
                        axs[i, j].set_xlabel(model_names[j-1])

                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        fig.savefig(os.path.join(experiment_folder, 'abundances.png'), dpi=1200, bbox_inches='tight')
        plt.close(fig)

    # Plot the endmembers
    fig, axs = plt.subplots(N, M, sharex='col', sharey='row', figsize=(M*1.5, N*1.5))
    for i in range(N):
        for j in range(M):
            endmember_to_plot = gt_endmembers[i] if j == 0 else ordered_endmembers[j-1][i]
            if endmembers_normalize:
                endmember_to_plot = endmember_to_plot/np.max(endmember_to_plot)

            axs[i, j].plot(endmember_to_plot, color=colors(i))

            # Labels for the models at the bottom
            if i == N - 1:
                if j == 0:
                    axs[i, j].set_xlabel('Ground truth')
                else:
                    axs[i, j].set_xlabel(model_names[j-1])

            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    fig.savefig(os.path.join(experiment_folder, 'endmembers.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)

    return metrics_df


def _experiment(dataset, num_endmembers, *, epochs, experiment_folder, normalize, preprocess, decoder_type, include_conventional, asc, ae_loss):
    # Load data
    gt_endmembers, gt_abundance_image, mixed_image, spectral_axis = unmix.load_data(dataset)

    rp_mixed_image = rpy.SpectralImage(mixed_image, spectral_axis)

    if normalize:
        normalizer = rpy.preprocessing.normalise.Vector(pixelwise=False)
        rp_mixed_image, gt_endmembers = normalizer.apply([rp_mixed_image, rpy.SpectralImage(gt_endmembers, spectral_axis)])
        gt_endmembers = gt_endmembers.spectral_data

    if preprocess:
        preprocesser = rpy.preprocessing.Pipeline([
            rpy.preprocessing.misc.Cropper(region=(400, 1800)),
            rpy.preprocessing.baseline.ASPLS(),
            rpy.preprocessing.normalise.Vector(pixelwise=False),
        ])
        rp_mixed_image, gt_endmembers = preprocesser.apply([rp_mixed_image, rpy.SpectralImage(gt_endmembers, spectral_axis)])
        gt_endmembers = gt_endmembers.spectral_data

    # Define the models
    models = unmix.init_models(
        len(rp_mixed_image.spectral_axis), num_endmembers=num_endmembers, decoder_type=decoder_type,
        include_conventional=include_conventional, asc=asc)

    # Collect the results
    pred_endmembers = []
    pred_abundances = []
    for name, model in models:
        print(f'Running model {name}')
        endmembers, abundance_maps = unmix.unmixing(model, rp_mixed_image, epochs=epochs, ae_loss=ae_loss)

        pred_endmembers.append(endmembers)
        pred_abundances.append(abundance_maps)

    # Set the colors for the plots
    num_endmembers = len(gt_endmembers)
    colors = plt.cm.get_cmap('viridis', num_endmembers)

    # Plot the results
    model_names = [name for name, _ in models]
    metrics = _summarise_results(
        gt_endmembers, gt_abundance_image, np.array(pred_endmembers), np.array(pred_abundances),  model_names, colors,
        endmembers_normalize=True, experiment_folder=experiment_folder)

    return metrics


def run(
    datasets,
    num_endmembers,
    model_replicates=1,
    epochs=10,
    normalize=False,
    preprocess=False,
    decoder_type='linear',
    experiment_folder=None,
    include_conventional=True,
    asc=True,
    ae_loss=autoencoders.SAD,
    seed=42,
):
    """
    A function that performs a benchmarking experiment on a set of datasets.

    Parameters
    ----------
    datasets : list of str
        A list of paths to the datasets to be used in the experiment.
    num_endmembers : int
        The number of endmembers to be used in the experiment.
    model_replicates : int, optional
        The number of replicates of each model to be used in the experiment. The default is 1.
    epochs : int, optional
        The number of epochs to be used for autoencoder models in the experiment. The default is 10.
    normalize : bool, optional
        Whether to normalize the data before running the experiment. The default is False.
    preprocess : bool, optional
        Whether to preprocess the data before running the experiment. The default is False.
    decoder_type : str, optional
        The type of decoder to be used in the autoencoder models. The default is 'linear'.
    experiment_folder : str, optional
        The folder in which to save the results of the experiment. The default is None.
    include_conventional : bool, optional
        Whether to include the conventional model in the experiment. The default is True.
    asc : bool, optional
        Whether to apply the abundance sum-to-one constraint to the abundance maps. The default is True.
    ae_loss : function, optional
        The loss function to be used for the autoencoder models. The default is autoencoders.SAD.
    seed : int, optional
        The random seed to be used for the experiment. The default is 42.

    Returns
    -------
    metrics_df : pandas.DataFrame
        A dataframe containing the metrics for each model and dataset.
    """
    if experiment_folder is None:
        experiment_folder = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    os.makedirs(experiment_folder, exist_ok=True)

    # Log the experiment parameters
    with open(os.path.join(experiment_folder, 'params.json'), 'w') as f:
        params = {
            'datasets': datasets,
            'num_endmembers': num_endmembers,
            'model_replicates': model_replicates,
            'epochs': epochs,
            'normalize': normalize,
            'preprocess': preprocess,
            'seed': seed,
        }
        json.dump(params, f, indent=2)

    metrics_total = []
    for dataset in datasets:
        print(f'Running experiment on {dataset} ...')

        datafolder = os.path.join(experiment_folder, dataset.split('/')[-1])
        os.makedirs(datafolder, exist_ok=True)

        np.random.seed(seed)
        seeds = np.random.randint(0, 10000, model_replicates)

        metrics = []
        for i, seed in enumerate(seeds):
            print(f'Replicate {i+1}/{model_replicates}')

            unmix.set_seeds(int(seed))

            subfolder = os.path.join(datafolder, f'replicate_{i+1}')
            os.makedirs(subfolder, exist_ok=True)

            # Run the experiment
            replicate_metrics = _experiment(
                dataset, experiment_folder=subfolder, num_endmembers=num_endmembers, normalize=normalize, preprocess=preprocess,
                epochs=epochs, decoder_type=decoder_type, include_conventional=include_conventional, asc=asc, ae_loss=ae_loss)

            metrics.append(replicate_metrics)

        # Get confidence intervals over the replicates
        metrics_df = pd.concat(metrics).groupby(level=0).agg([np.mean, np.std])
        metrics_df.columns = metrics_df.columns.map('_'.join)
        metrics_df = metrics_df.reset_index()

        # Save the results
        metrics_df.to_csv(os.path.join(datafolder, 'metrics.csv'), index=False)

        metrics_total.append(metrics)
        print('=======================================================================================================')

    # Get confidence intervals over the replicates
    metrics_total = sum(metrics_total, [])
    metrics_total = pd.concat(metrics_total).groupby(level=0).agg([np.mean, np.std])
    metrics_total.columns = metrics_total.columns.map('_'.join)
    metrics_total = metrics_total.reset_index()

    # Save the results
    metrics_total.to_csv(os.path.join(experiment_folder, 'metrics_total.csv'), index=False)

    return metrics_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an unmixing analysis experiment.")
    parser.add_argument("--datasets", type=str, required=True, nargs='+', help="The paths to the dataset(s) to load.")
    parser.add_argument("--replicates", type=int, default=1, help="The number of times to replicate the experiment for each model.")
    parser.add_argument("--num_endmembers", type=int, default=5, help="The number of endmembers to unmix.")
    parser.add_argument("--decoder_type", type=str, default='linear', help="The type of decoder to use. Can be 'linear' or 'nonlinear'.")
    parser.add_argument("--epochs", type=int, default=10, help="The number of epochs to train the models for.")
    parser.add_argument("--include_conventional", action='store_true', help="Whether to include the conventional model in the analysis.")
    parser.add_argument("--asc", action='store_true', help="Whether to use the ASC model.")
    parser.add_argument("--experiment_folder", type=str, default=None, help="The folder to save the experiment results to.")
    parser.add_argument("--seed", type=int, default=42, help="The random seed to use.")
    parser.add_argument("--normalize", action='store_true', help="Whether to normalize the data before unmixing.")
    parser.add_argument("--preprocess", action='store_true', help="Whether to preprocess the data before unmixing.")

    args = parser.parse_args()

    run(
        args.datasets,
        args.num_endmembers,
        model_replicates=args.replicates,
        epochs=args.epochs,
        normalize=args.normalize,
        preprocess=args.preprocess,
        decoder_type=args.decoder_type,
        experiment_folder=args.experiment_folder,
        include_conventional=args.include_conventional,
        asc=args.asc,
        seed=args.seed,
    )
