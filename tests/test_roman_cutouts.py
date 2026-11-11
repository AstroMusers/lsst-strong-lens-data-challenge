import importlib
import json
from pathlib import Path

from astropy.table import Table
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ROMAN = ROOT / 'dev' / 'roman'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.syspath_prepend(str(ROMAN))
    return importlib.import_module('roman_cutouts'), importlib.import_module('organize_data')


def write_cutouts(directory, object_ids, partial, data):
    filename = directory / 'cutouts.npz'
    print(f'Writing to {filename}...')
    np.savez_compressed(filename, data=data, object_id=np.array(object_ids), bands=np.array(['F106', 'F129']))
    index = Table({
        'object_id': sorted(object_ids),
        'is_lens': [True, False, True],
        'partial': partial,
    })
    filename = directory / 'index.ecsv'
    print(f'Writing to {filename}...')
    index.write(filename, format='ascii.ecsv')


def test_closed_form_gamma_matches_the_shared_per_band_preprocessing(modules):
    roman_cutouts, organize_data = modules
    rng = np.random.default_rng(0)
    images = rng.lognormal(size=(2, 9, 9, 3))  # odd pixel count, as in the 93 px cutouts

    processed = roman_cutouts.process_cutouts(images)

    assert processed.dtype == np.float32
    np.testing.assert_allclose(np.median(processed, axis=(1, 2)), 0.2, rtol=1e-6)
    for n in range(images.shape[0]):
        for band in range(images.shape[-1]):
            expected = organize_data.process_image(images[n, :, :, band])
            np.testing.assert_allclose(processed[n, :, :, band], expected, atol=1e-3)


def test_load_cutouts_drops_partial_rows_and_keeps_index_alignment(modules, tmp_path):
    roman_cutouts, _ = modules
    rng = np.random.default_rng(1)
    data = rng.lognormal(size=(3, 2, 5, 5))  # (N, band, y, x), as written by the cutout pipeline
    data[1, :, 0, 0] = np.nan  # partial cutouts contain NaN pixels
    write_cutouts(tmp_path, ['a', 'b', 'c'], [False, True, False], data)

    index, images, bands = roman_cutouts.load_cutouts(tmp_path)

    assert list(index['object_id']) == ['a', 'c']
    assert list(index['is_lens']) == [True, True]
    assert list(bands) == ['F106', 'F129']
    assert images.shape == (2, 5, 5, 2)
    assert np.isfinite(images).all()
    np.testing.assert_array_equal(images, roman_cutouts.process_cutouts(np.moveaxis(data[[0, 2]], 1, -1)))


def test_load_cutouts_rejects_files_with_different_row_orders(modules, tmp_path):
    roman_cutouts, _ = modules
    data = np.random.default_rng(2).lognormal(size=(3, 2, 5, 5))
    write_cutouts(tmp_path, ['b', 'a', 'c'], [False, False, False], data)

    with pytest.raises(ValueError, match='different object_ids'):
        roman_cutouts.load_cutouts(tmp_path)


@pytest.mark.parametrize('notebook', ['roman.ipynb', 'roman_load_and_predict.ipynb'])
def test_roman_notebooks_share_the_cutout_loader(notebook):
    path = ROMAN / notebook
    print(f'Reading from {path}...')
    cells = json.loads(path.read_text(encoding='utf-8'))['cells']
    sources = [''.join(cell['source']) for cell in cells if cell['cell_type'] == 'code']
    assert any('from roman_cutouts import load_cutouts' in source for source in sources)
    assert not any('def process_cutouts(' in source or 'cutouts.npz' in source for source in sources)
