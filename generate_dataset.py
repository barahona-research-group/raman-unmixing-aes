import json
import os
import pickle
import argparse
import numpy as np
import ramanspy as rp


def run(
        image_type,
        num_endmembers=5,
        spectral_size=1000,
        mixture_mode='linear',
        image_size=100,
        realistic_endmembers=False,
        noise=False,
        noise_amplitude=0.1,
        baseline=False,
        baseline_amplitude=2,
        baseline_probability=0.25,
        cosmic_spikes=False,
        cosmic_spikes_amplitude=5,
        cosmic_spikes_probability=0.1,
        output_dir='data',
        seed=42
):
    """
    A function that generates a synthetic Raman mixture dataset based on the RamanPy package. For more information about
    the parameters, please refer to the RamanPy documentation at https://ramanspy.readthedocs.io/en/latest/synth.html#ramanspy.synth.generate_mixture_image.
    """

    # Generate the synthetic data
    mixed_image, endmembers, abundance_image = rp.synth.generate_mixture_image(
        num_endmembers,
        spectral_size,
        image_size,
        image_type,
        mixture_mode=mixture_mode,
        realistic_endmembers=realistic_endmembers,
        noise=noise,
        noise_amplitude=noise_amplitude,
        baseline=baseline,
        baseline_amplitude=baseline_amplitude,
        baseline_probability=baseline_probability,
        cosmic_spikes=cosmic_spikes,
        cosmic_spike_amplitude=cosmic_spikes_amplitude,
        cosmic_spikes_probability=cosmic_spikes_probability,
        seed=seed)

    # Save the generated data
    os.makedirs(os.path.join(output_dir, ), exist_ok=True)
    with open(os.path.join(output_dir, f'gt_endmembers.pkl'), 'wb') as f:
        pickle.dump(np.array([e.spectral_data for e in endmembers]), f)
    with open(os.path.join(output_dir, f'gt_abundance_image.pkl'), 'wb') as f:
        pickle.dump(abundance_image, f)
    with open(os.path.join(output_dir, f'spectral_axis.pkl'), 'wb') as f:
        pickle.dump(endmembers[0].spectral_axis, f)
    with open(os.path.join(output_dir, f'{image_type}.pkl'), 'wb') as f:
        pickle.dump(mixed_image.spectral_data, f)

    # Save the parameters
    with open(os.path.join(output_dir, "parameters.json"), "w") as outfile:
        params = {
            'image_type': image_type,
            'num_endmembers': num_endmembers,
            'spectral_size': spectral_size,
            'mixture_mode': mixture_mode,
            'image_size': image_size,
            'realistic_endmembers': realistic_endmembers,
            'noise': noise,
            'noise_amplitude': noise_amplitude,
            'baseline': baseline,
            'baseline_amplitude': baseline_amplitude,
            'baseline_probability': baseline_probability,
            'cosmic_spikes': cosmic_spikes,
            'cosmic_spikes_amplitude': cosmic_spikes_amplitude,
            'cosmic_spikes_probability': cosmic_spikes_probability,
            'seed': seed,
            'output_dir': output_dir

        }
        json.dump(params, outfile)

    return mixed_image, endmembers, abundance_image


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Generate synthetic spectral unmixing dataset.")
    parser.add_argument("--num_endmembers", type=int, default=5, help="Number of endmembers.")
    parser.add_argument("--spectral_size", type=int, default=1000, help="Spectral size for the endmembers.")
    parser.add_argument("--mixture_mode", type=str, default='linear', help="Mixture mode (linear or non-linear).")
    parser.add_argument("--image_size", type=int, default=100, help="Size of the image.")
    parser.add_argument("--image_type", type=str, required=True, help="Type of the image.")
    parser.add_argument("--realistic_endmembers", action='store_true', help="Whether to generate 'more realistic' endmembers.")
    parser.add_argument("--noise", action='store_true', help="Whether to add noise to the mixtures.")
    parser.add_argument("--noise_amplitude", type=float, default=0.1, help="Amplitude of the noise.")
    parser.add_argument("--baseline", action='store_true', help="Whether to add baseline to the mixtures.")
    parser.add_argument("--baseline_amplitude", type=float, default=2, help="Amplitude of the baseline.")
    parser.add_argument("--baseline_probability", type=float, default=0.25, help="Probability of the baseline occurrence.")
    parser.add_argument("--cosmic_spikes", action='store_true', help="Whether to add cosmic spikes to the mixtures.")
    parser.add_argument("--cosmic_spikes_amplitude", type=float, default=5, help="Amplitude of the cosmic spikes.")
    parser.add_argument("--cosmic_spikes_probability", type=float, default=0.1, help="Probability of the cosmic spikes occurrence.")
    parser.add_argument("--output_dir", type=str, default='data', help="Path to save the generated mixture data.")
    parser.add_argument("--seed", type=int, default=42, help="The random seed to use.")
    args = parser.parse_args()

    run(
        args.image_type,
        args.num_endmembers,
        args.spectral_size,
        args.mixture_mode,
        args.image_size,
        args.realistic_endmembers,
        args.noise,
        args.noise_amplitude,
        args.baseline,
        args.baseline_amplitude,
        args.baseline_probability,
        args.cosmic_spikes,
        args.cosmic_spikes_amplitude,
        args.cosmic_spikes_probability,
        args.output_dir,
        args.seed
    )
