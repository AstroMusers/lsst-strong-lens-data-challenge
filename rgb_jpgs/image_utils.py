import numpy as np
from astropy.io import fits


def get_image_data(filename=None, **keywords):
    """Return the challenge lens ID and image from the first extension row."""
    if filename is None and 'f' in keywords:
        filename = keywords.pop('f')
    if keywords:
        raise TypeError(f"Unexpected keyword arguments: {', '.join(keywords)}")
    print(f'Reading from {filename}...')
    data = fits.getdata(filename, ext=1)
    return data[0][0], data[0][1]


def normalize_image(image):
    """Normalize finite pixels, using unsigned offsets before integer-to-float conversion."""
    image = np.asarray(image)
    if image.dtype.kind not in 'biuf' or not image.size or not np.isfinite(image).all():
        raise ValueError('Images must contain nonempty finite real pixels')
    if image.dtype.kind in 'biu':
        minimum = np.min(image)
        span = int(np.max(image)) - int(minimum)
        if span == 0:
            return np.zeros_like(image, dtype=float)
        unsigned_dtype = np.dtype(f'u{image.dtype.itemsize}')
        offsets = image.astype(unsigned_dtype, copy=False) - np.asarray(minimum).astype(unsigned_dtype)
        return offsets.astype(float) / span
    minimum, maximum = np.min(image), np.max(image)
    if minimum == maximum:
        return np.zeros_like(image)
    with np.errstate(over='ignore'):
        span = maximum - minimum
    if not np.isfinite(span):
        return (image / 2 - minimum / 2) / (maximum / 2 - minimum / 2)
    return (image - minimum) / span
