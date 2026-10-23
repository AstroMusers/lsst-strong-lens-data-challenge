import os
import yaml

import numpy as np

DATASET_KEYS = ('hsc_lens', 'hsc_nonlens', 'slsim_lens', 'slsim_nonlens')


def read_dataset_archive(filename, keys=DATASET_KEYS):
    """Read requested image datasets into arrays detached from the closed archive."""
    print(f'Reading from {filename}...')
    with np.load(filename) as archive:
        return {key: archive[key] for key in keys}


def main():
    # read configuration file
    print('Reading from config.yml...')
    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    data_dir = config['data_dir']

    npz_dict = {}
    for key in DATASET_KEYS:
        filename = os.path.join(data_dir, f'{key}.npz')
        npz_dict.update(read_dataset_archive(filename, (key,)))
        print(f'Loaded {key}es: {npz_dict[key].shape}')

    # write as single npz file
    filepath = os.path.join(data_dir, "dataset.npz")
    print(f'Writing to {filepath}...')
    np.savez_compressed(filepath, **npz_dict)
    print(f'Wrote combined dataset to {filepath}')


if __name__ == '__main__':
    main()
