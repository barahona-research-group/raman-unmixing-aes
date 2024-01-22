import argparse
from datetime import datetime
import json
import time
import os
import ramanspy as rp

import unmix


def run(
        scene_sizes,
        scene_type,
        num_endmembers,
        spectral_size,
        *,
        replicates,
        training_epochs,
        noise=False,
        baseline=False,
        cosmic_spikes=False,
        decoder_type='linear',
        asc=True,
        seed=None,
        output_dir=None):
    """
    A function that measures the computational time for each model on a given type of synthetic data.

    Parameters
    ----------
    scene_sizes : list of int
        The image sizes to use for the experiment.
    scene_type : str
        The type of scenes to generate.
    num_endmembers : int
        The number of endmembers to be used in the experiment.
    spectral_size : int
        The number of spectral bands to be used in the experiment.
    replicates : int
        The number of replicates of each model to be used in the experiment.
    training_epochs : int, optional
        The number of epochs to be used for autoencoder models in the experiment.
    noise : bool, optional
        Whether to add noise to the mixtures. The default is False.
    baseline : bool, optional
        Whether to add a baseline to the mixtures. The default is False.
    cosmic_spikes : bool, optional
        Whether to add cosmic spikes to the mixtures. The default is False.
    decoder_type : str, optional
        The type of decoder to be used in the autoencoder models. The default is 'linear'.
    output_dir : str, optional
        The folder in which to save the results of the experiment. The default is None.
    asc : bool, optional
        Whether to apply the abundance sum-to-one constraint to the abundance maps. The default is True.
    seed : int, optional
        The random seed to be used for the experiment. The default is 42.

    Returns
    -------
    metrics_df : pandas.DataFrame
        A dataframe containing the metrics for each model and dataset.
    """

    # Create the experiment folder
    if output_dir is None:
        output_dir = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    os.makedirs(output_dir, exist_ok=True)

    # Log the experiment parameters
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        params = {
            'scene_sizes': scene_sizes,
            'scene_type': scene_type,
            'num_endmembers': num_endmembers,
            'spectral_size': spectral_size,
            'training_epochs': training_epochs,
            'noise': noise,
            'baseline': baseline,
            'cosmic_spikes': cosmic_spikes,
            'replicates': replicates,
            'seed': seed,
        }
        json.dump(params, f, indent=2)

    # Set the random seed for reproducibility
    unmix.set_seeds(seed)

    # Measure the wall time for each model and dataset
    results = {}
    for scene_size in scene_sizes:
        print(f'Running experiment for a {scene_type} scene of size {scene_size}x{scene_size}')

        # Generate data
        mixed_image, _, _ = rp.synth.generate_mixture_image(
            num_endmembers,
            spectral_size,
            scene_size,
            scene_type,
            noise=noise,
            baseline=baseline,
            cosmic_spikes=cosmic_spikes,
            seed=seed)

        # Collect the results for this image size over multiple replicates
        results[scene_size] = {}
        for i in range(replicates):

            # Initialize the models
            models = unmix.init_models(spectral_size, num_endmembers=num_endmembers, decoder_type=decoder_type, asc=asc)

            # Perform unmixing using each model
            for model_name, model in models:
                print(f'Running {model_name}, replicate: {i + 1}')

                # Measure the wall time
                start = time.time()
                unmix.unmixing(model, mixed_image, epochs=training_epochs)
                end = time.time()

                wall_time = end - start

                # Save the results
                if model_name in results[scene_size]:
                    results[scene_size][model_name].append(wall_time)
                else:
                    results[scene_size][model_name] = [wall_time]

        # Save the results for current image size
        with open(os.path.join(output_dir, f"{scene_size}.json"), "w") as outfile:
            json.dump(results[scene_size], outfile)

    # Save combined results
    with open(os.path.join(output_dir, f"all.json"), "w") as outfile:
        json.dump(results, outfile)

    return results


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Run a profiling analysis experiment.")
    parser.add_argument("--replicates", type=int, default=5, help="The number of times to replicate the experiment for each model.")
    parser.add_argument("--scene_sizes", type=int, nargs='+', default=[50, 100, 200, 300, 400, 500], help="The image sizes to use for the experiment.")
    parser.add_argument("--scene_type", type=str, default='gaussian', help="The type of scenes to generate.")
    parser.add_argument("--spectral_size", type=int, default=1000, help="The number of spectral bands to use for the experiment.")
    parser.add_argument("--num_endmembers", type=int, default=5, help="The number of endmembers to unmix.")
    parser.add_argument("--training_epochs", type=int, default=10, help="The number of epochs to train the models for.")
    parser.add_argument("--noise", action='store_true', help="Whether to add noise to the mixtures.")
    parser.add_argument("--baseline", action='store_true', help="Whether to add a baseline to the mixtures.")
    parser.add_argument("--cosmic_spikes", action='store_true', help="Whether to add cosmic spikes to the mixtures.")
    parser.add_argument("--decoder_type", type=str, default='linear', help="The type of decoder to use.")
    parser.add_argument("--asc", action='store_true', help="Whether to use ASC.")
    parser.add_argument("--seed", type=int, default=42, help="The random seed to use for the experiment.")
    args = parser.parse_args()

    # Run the experiment
    run(
        args.scene_sizes,
        args.scene_type,
        args.num_endmembers,
        args.spectral_size,
        training_epochs=args.training_epochs,
        noise=args.noise,
        baseline=args.baseline,
        cosmic_spikes=args.cosmic_spikes,
        replicates=args.replicates,
        decoder_type=args.decoder_type,
        asc=args.asc,
        seed=args.seed
    )
