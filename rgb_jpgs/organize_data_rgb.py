import os
import yaml
from glob import glob

import numpy as np
from tqdm import tqdm
from PIL import Image

if __package__:
    from .image_utils import get_image_data, normalize_image
else:
    from image_utils import get_image_data, normalize_image


def main():
    # set which bands and datasets to use
    bands = ['g', 'r', 'i']
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
        for kind in ['1_lens', '0_nonlens']:
            subdir = os.path.join(output_dir, kind)
            os.makedirs(subdir, exist_ok=True)
            for entry in os.listdir(subdir):
                file_path = os.path.join(subdir, entry)
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

                    if not id_r == id_g == id_b:
                        raise ValueError(f"Lens IDs do not match across bands: {id_r}, {id_g}, {id_b}")

                    rgb_image = np.stack([
                        normalize_image(image_r),
                        normalize_image(image_g),
                        normalize_image(image_b)
                    ], axis=-1)

                    rgb_uint8 = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_uint8)
                    output_file = os.path.join(output_dir, kind, f"{dataset}_{i:08d}_{id_r}.jpg")
                    print(f'Writing to {output_file}...')
                    img.save(output_file)
                except Exception as e:
                    print(f"Error processing {i} for {kind[2:]} in {dataset}: {e}")
                    continue


if __name__ == '__main__':
    main()
