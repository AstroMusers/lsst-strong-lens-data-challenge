"""Load and preprocess Roman lens-search cutouts.

Shared by roman.ipynb (training) and roman_load_and_predict.ipynb (evaluation), so
both notebooks feed the classifier identically preprocessed images.
"""
from pathlib import Path

import numpy as np
from astropy.table import Table


def process_cutouts(images, desired_median=0.2):
    """Apply organize_data.process_image to every band of (N, y, x, band) images.

    Gamma has a closed form here because the median commutes with a monotonic
    power on images with an odd number of pixels.
    """
    minimum = images.min(axis=(1, 2), keepdims=True)
    maximum = images.max(axis=(1, 2), keepdims=True)
    images = (images - minimum) / (maximum - minimum)
    gamma = np.log(desired_median) / np.log(np.median(images, axis=(1, 2), keepdims=True))
    return (images ** gamma).astype(np.float32)


def load_cutouts(cutouts_dir):
    """Read a cutouts directory, drop partial cutouts, and preprocess the rest.

    The directory holds cutouts.npz and index.ecsv, which has one row per cutout
    in the same order. Partial cutouts at the detector edge contain NaN pixels,
    so they are dropped.

    Returns (index, images, bands): the index rows of the kept cutouts, their
    (N, y, x, band) float32 images, and the band names.
    """
    cutouts_dir = Path(cutouts_dir)
    print(f'Reading from {cutouts_dir}...')
    index = Table.read(cutouts_dir / 'index.ecsv', format='ascii.ecsv')
    with np.load(cutouts_dir / 'cutouts.npz') as cutouts:
        data = cutouts['data']  # (N, band, y, x)
        object_ids = cutouts['object_id']
        bands = cutouts['bands']

    # callers pair images with index columns such as is_lens by row
    if not np.array_equal(object_ids, np.array(index['object_id'])):
        raise ValueError(f'{cutouts_dir}: cutouts.npz and index.ecsv list different object_ids')

    keep = ~np.array(index['partial'])
    print(f'dropped {np.sum(~keep)} partial cutouts')
    images = process_cutouts(np.moveaxis(data[keep], 1, -1))
    return index[keep], images, bands
