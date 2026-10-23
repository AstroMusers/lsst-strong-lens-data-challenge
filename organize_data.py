import os
import yaml
from glob import glob

import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize

from rgb_jpgs.image_utils import get_image_data, normalize_image


def process_image(image):
    # normalize the image to the range [0, 1]
    image = normalize_image(image)
    if not np.any(image):
        return image

    # apply gamma correction
    gamma = find_gamma(image, desired_median=0.2)
    return np.power(image, gamma)


def objective_function(gamma, image, desired_median):
    gamma_corrected_image = np.power(image, gamma)
    median_value = np.median(gamma_corrected_image)
    return (median_value - desired_median)**2


def find_gamma(image, desired_median=0.2, initial_gamma=0.7):
    result = minimize(objective_function, initial_gamma, args=(image, desired_median), method='Nelder-Mead')
    return result.x[0]


def main():
    # set which bands and datasets to use
    bands = ['g', 'r', 'i', 'z', 'y']
    datasets = ['slsim']  # 'hsc', 
    kinds = ['lens']  # 'nonlens', 
    len_dataset = 50000
    dev = False

    # read configuration file
    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    data_dir = config['data_dir']

    # read the fits files, produce npz files, and write to appropriate directories
    npz_dict = {}
    for dataset in tqdm(datasets, desc="Datasets"):    
        for kind in tqdm(kinds, desc='Kind', leave=False):
            image_list = []

            # TODO TEMP
            # if dataset == "slsim" and kind == "lens":
            #     break

            for i in tqdm(range(len_dataset), desc="Images", leave=False):
                try:
                    ids, images = [], []
                    for band in bands:
                        fits_file = glob(os.path.join(data_dir, f'{dataset}_{kind}es', f'*_{str(i).zfill(8)}_{band}.fits'))[0]
                        id, image = get_image_data(fits_file)
                        ids.append(id)

                        processed_image = process_image(image)
                        images.append(processed_image)

                    if len(set(ids)) != 1:
                        raise ValueError(f"Lens IDs do not match across bands {bands}: {ids}")

                    combined_image = np.stack(images, axis=-1)
                    image_list.append(combined_image)
                    
                except Exception as e:
                    print(f"Error processing {i} for {kind} in {dataset}: {e}")
                    continue

            npz_dict[f"{dataset}_{kind}"] = image_list

            filename = f"{dataset}_{kind}.npz" if not dev else f"{dataset}_{kind}_dev.npz"
            np.savez_compressed(os.path.join(data_dir, filename), **npz_dict)

        # filename = "dataset.npz" if not dev else "dataset_dev.npz"
        # np.savez_compressed(os.path.join(data_dir, filename), **npz_dict)


if __name__ == '__main__':
    main()
