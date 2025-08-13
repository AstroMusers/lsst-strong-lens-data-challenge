import os
import yaml
from glob import glob

import numpy as np
from tqdm import tqdm
from astropy.io import fits
from PIL import Image


def get_image_data(f):
    """
    Returns a tuple (Lens ID, image)
    """
    fits_data = fits.getdata(f, ext=1)
    return fits_data[0][0], fits_data[0][1]


def main():
    # set which bands and datasets to use
    bands = ['g', 'i', 'y']
    datasets = ['hsc', 'slsim']
    len_dataset = 50000
    train_frac = 0.9
    dev = False

    # read configuration file
    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    data_dir = config['data_dir']

    # create the output directories if they don't exist; otherwise, delete existing files
    train_output_dir = os.path.join(data_dir, 'train')
    test_output_dir = os.path.join(data_dir, 'test')
    output_dirs = [train_output_dir, test_output_dir]
    if dev: output_dirs = [d + '_dev' for d in output_dirs]
    for output_dir in output_dirs:
        lens_output_dir = os.path.join(output_dir, '1_lens')
        nonlens_output_dir = os.path.join(output_dir, '0_nonlens')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(lens_output_dir)
            os.makedirs(nonlens_output_dir)
        else:
            for subdir in [lens_output_dir, nonlens_output_dir]:
                if os.path.exists(subdir):
                    for f in os.listdir(subdir):
                        file_path = os.path.join(subdir, f)
                        if os.path.isfile(file_path):
                            os.remove(file_path)

    # read the fits files, produce jpgs, and write to appropriate directories
    for dataset in tqdm(datasets, desc="Datasets"):    
        for kind in ['0_nonlens', '1_lens']:
            output_dir = output_dirs[0]

            for i in tqdm(range(len_dataset), desc="Images", leave=False):
                if i == train_frac * len_dataset:
                    output_dir = output_dirs[1]

                try:
                    fits_r = glob(os.path.join(data_dir, f'{dataset}_{kind[2:]}es', f'*_{str(i).zfill(8)}_{bands[2]}.fits'))[0]
                    fits_g = glob(os.path.join(data_dir, f'{dataset}_{kind[2:]}es', f'*_{str(i).zfill(8)}_{bands[1]}.fits'))[0]
                    fits_b = glob(os.path.join(data_dir, f'{dataset}_{kind[2:]}es', f'*_{str(i).zfill(8)}_{bands[0]}.fits'))[0]

                    id_r, image_r = get_image_data(fits_r)
                    id_g, image_g = get_image_data(fits_g)
                    id_b, image_b = get_image_data(fits_b)

                    assert id_r == id_g == id_b, f"Lens IDs do not match across bands: {id_r}, {id_g}, {id_b}"

                    rgb_image = np.stack([
                        (image_r - np.min(image_r)) / (np.max(image_r) - np.min(image_r)),
                        (image_g - np.min(image_g)) / (np.max(image_g) - np.min(image_g)),
                        (image_b - np.min(image_b)) / (np.max(image_b) - np.min(image_b))
                    ], axis=-1)

                    rgb_uint8 = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_uint8)
                    img.save(os.path.join(output_dir, kind, f"{id_r}.jpg"))
                except Exception as e:
                    print(f"Error processing {i} for {kind[2:]} in {dataset}: {e}")
                    continue


if __name__ == '__main__':
    main()
    