import os
import yaml

import numpy as np


def main():
    # read configuration file
    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    data_dir = config['data_dir']

    # load the four separate npz files
    with np.load(os.path.join(data_dir, 'hsc_lens.npz')) as data:
        hsc_lenses = data['hsc_lens']
    with np.load(os.path.join(data_dir, 'hsc_nonlens.npz')) as data:
        hsc_nonlenses = data['hsc_nonlens']
    with np.load(os.path.join(data_dir, 'slsim_lens.npz')) as data:
        slsim_lenses = data['slsim_lens']
    with np.load(os.path.join(data_dir, 'slsim_nonlens.npz')) as data:
        slsim_nonlenses = data['slsim_nonlens']

    # combine the images into a single dictionary
    npz_dict = {
        'hsc_lens': hsc_lenses,
        'hsc_nonlens': hsc_nonlenses,
        'slsim_lens': slsim_lenses,
        'slsim_nonlens': slsim_nonlenses
    }

    # write as single npz file
    filename = "dataset.npz"
    np.savez_compressed(os.path.join(data_dir, filename), **npz_dict)


if __name__ == '__main__':
    main()
    