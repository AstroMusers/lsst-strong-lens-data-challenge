import ast
import copy
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from astropy.io import fits
import numpy as np
import pytest
import yaml
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def organizers(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return (
        importlib.import_module('organize_data'),
        importlib.import_module('rgb_jpgs.organize_data_rgb'),
        importlib.import_module('rgb_jpgs.image_utils'),
    )


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int16, np.uint16, np.int64, np.uint64, np.bool_])
def test_constant_images_have_finite_zero_contrast_without_mutating_input(organizers, dtype):
    organizer, _, helpers = organizers
    image = np.full((3, 4), 7, dtype=dtype)
    original = image.copy()

    with np.errstate(all='raise'):
        normalized = helpers.normalize_image(image)
        processed = organizer.process_image(image)

    np.testing.assert_array_equal(normalized, np.zeros(image.shape))
    np.testing.assert_array_equal(processed, normalized)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_nonconstant_normalization_matches_existing_formula_exactly(organizers, dtype):
    _, _, helpers = organizers
    image = np.array([[1, 3, 8], [-2, 0, 4]], dtype=dtype)
    expected = (image - np.min(image)) / (np.max(image) - np.min(image))

    actual = helpers.normalize_image(image)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize('image', [
    np.array([[-30000, 0, 30000]], dtype=np.int16),
    np.array([[-1e308, 0, 1e308]]),
])
def test_normalization_avoids_overflow_in_the_pixel_range(organizers, image):
    _, _, helpers = organizers
    with np.errstate(all='raise'):
        actual = helpers.normalize_image(image)
    np.testing.assert_array_equal(actual, [[0, 0.5, 1]])


@pytest.mark.parametrize('dtype', ['<i8', '>i8', '<u8', '>u8'])
@pytest.mark.parametrize('case', ['near_minimum', 'near_maximum', 'full_range'])
def test_integer_normalization_preserves_large_pixel_contrast_and_byte_order(organizers, dtype, case):
    organizer, _, helpers = organizers
    limits = np.iinfo(np.dtype(dtype))
    minimum, maximum = int(limits.min), int(limits.max)
    if case == 'near_minimum':
        values = [minimum, minimum + 1, minimum + 2]
    elif case == 'near_maximum':
        values = [maximum - 2, maximum - 1, maximum]
    else:
        values = [minimum, 0 if minimum < 0 else maximum // 2, maximum]
    image = np.array([values], dtype=dtype)
    original = image.copy()
    expected = [[(value - min(values)) / (max(values) - min(values)) for value in values]]

    with np.errstate(all='raise'):
        actual = helpers.normalize_image(image)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, original)
    assert actual.dtype == np.float64
    if case == 'near_maximum':
        processed = organizer.process_image(image)
        assert np.isfinite(processed).all()
        assert processed[0, 0] == 0.0
        assert processed[0, -1] == 1.0


def test_boolean_normalization_retains_false_true_contrast(organizers):
    _, _, helpers = organizers
    image = np.array([[False, True, False]])
    np.testing.assert_array_equal(helpers.normalize_image(image), [[0.0, 1.0, 0.0]])


@pytest.mark.parametrize('image', [np.array([]), np.array([np.nan]), np.array([np.inf]), np.array([1j])])
def test_invalid_images_raise_instead_of_silently_writing_invalid_pixels(organizers, image):
    _, _, helpers = organizers
    with pytest.raises(ValueError, match='nonempty finite real pixels'):
        helpers.normalize_image(image)


@pytest.mark.parametrize('legacy_keyword', [False, True])
def test_both_organizers_share_the_actual_fits_reader(organizers, tmp_path, capsys, legacy_keyword):
    organizer, rgb_organizer, helpers = organizers
    assert organizer.get_image_data is helpers.get_image_data
    assert rgb_organizer.get_image_data is helpers.get_image_data
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    columns = [
        fits.Column(name='lens_id', format='K', array=[42]),
        fits.Column(name='image', format='6E', dim='(3,2)', array=[image]),
    ]
    filename = tmp_path / 'challenge_image.fits'
    print(f'Writing to {filename}...')
    fits.BinTableHDU.from_columns(columns).writeto(filename)

    lens_id, restored = organizer.get_image_data(f=filename) if legacy_keyword else organizer.get_image_data(filename)

    assert lens_id == 42
    np.testing.assert_array_equal(restored, image)
    assert f'Reading from {filename}...' in capsys.readouterr().out


def test_rgb_script_can_import_its_helpers_when_launched_from_its_own_directory():
    result = subprocess.run(
        [sys.executable, '-c', 'import organize_data_rgb; assert callable(organize_data_rgb.normalize_image)'],
        cwd=ROOT / 'rgb_jpgs', capture_output=True, text=True, check=True,
    )
    assert result.stdout == ''


@pytest.mark.parametrize(('workflow', 'existing_layout', 'repeated_ids'), [
    ('npz', None, False), ('rgb', 'fresh', False), ('rgb', 'parents', False),
    ('rgb', 'partial', False), ('rgb', 'complete', False), ('rgb', 'fresh', True),
])
@pytest.mark.parametrize('mismatch', [False, True])
def test_actual_organizers_preserve_constant_images_and_reject_mixed_lens_ids(organizers, monkeypatch, tmp_path, capsys, workflow, existing_layout, repeated_ids, mismatch):
    organizer, rgb_organizer, _ = organizers
    preserved = []
    if workflow == 'rgb' and existing_layout != 'fresh':
        for split in ('train', 'test'):
            parent = tmp_path / split
            parent.mkdir()
            kinds = ['1_lens', '0_nonlens'] if existing_layout == 'complete' else ['1_lens'] if existing_layout == 'partial' else []
            for kind in kinds:
                folder = parent / kind
                folder.mkdir()
                filename = folder / 'stale.txt'
                print(f'Writing to {filename}...')
                filename.write_text('replace this old output')
                nested = folder / 'nested'
                nested.mkdir()
                filename = nested / 'keep.txt'
                print(f'Writing to {filename}...')
                filename.write_text('preserve nested content')
                preserved.append(filename)
    datasets = ['slsim'] if workflow == 'npz' else ['hsc', 'slsim']
    kinds = ['lens'] if workflow == 'npz' else ['lens', 'nonlens']
    bands = ['g', 'r', 'i', 'z', 'y'] if workflow == 'npz' else ['g', 'r', 'i']
    image_count = 2 if repeated_ids else 1
    image = np.full((8, 8), 7, dtype=np.float32)
    for dataset_index, dataset in enumerate(datasets):
        for kind in kinds:
            folder = tmp_path / f'{dataset}_{kind}es'
            folder.mkdir()
            for image_index in range(image_count):
                for band_index, band in enumerate(bands):
                    lens_id = (42 if repeated_ids else 100 + dataset_index) + (1000 if mismatch and band_index else 0)
                    columns = [
                        fits.Column(name='lens_id', format='K', array=[lens_id]),
                        fits.Column(name='image', format='64E', dim='(8,8)', array=[image]),
                    ]
                    filename = folder / f'synthetic_{image_index:08d}_{band}.fits'
                    print(f'Writing to {filename}...')
                    fits.BinTableHDU.from_columns(columns).writeto(filename)
    filename = tmp_path / 'config.yml'
    print(f'Writing to {filename}...')
    with filename.open('w') as stream:
        yaml.safe_dump({'data_dir': str(tmp_path)}, stream)
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()
    module = organizer if workflow == 'npz' else rgb_organizer
    namespace = vars(module).copy()
    namespace['range'] = lambda count: range(image_count)
    namespace['tqdm'] = lambda values, **kwargs: values
    source = Path(module.__file__)
    print(f'Reading from {source}...')
    main = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), 'exec', optimize=2), namespace)

    namespace['main']()

    output = capsys.readouterr().out
    assert ('Lens IDs do not match across bands' in output) == mismatch
    if workflow == 'npz':
        filename = tmp_path / 'slsim_lens.npz'
        print(f'Reading from {filename}...')
        with np.load(filename) as archive:
            if mismatch:
                assert archive['slsim_lens'].shape == (0,)
            else:
                np.testing.assert_array_equal(archive['slsim_lens'], np.zeros((1, 8, 8, 5)))
    else:
        for split in ('train', 'test'):
            for kind in ('1_lens', '0_nonlens'):
                assert (tmp_path / split / kind).is_dir()
                assert not (tmp_path / split / kind / 'stale.txt').exists()
        for filename in preserved:
            print(f'Reading from {filename}...')
            assert filename.read_text() == 'preserve nested content'
        outputs = sorted((tmp_path / 'train').glob('*/*.jpg'))
        assert len(outputs) == (0 if mismatch else 4 * image_count)
        expected_names = set() if mismatch else {
            f'{dataset}_{image_index:08d}_{42 if repeated_ids else 100 + dataset_index}.jpg'
            for dataset_index, dataset in enumerate(datasets) for image_index in range(image_count)
        }
        for kind in ('0_nonlens', '1_lens'):
            assert {filename.name for filename in (tmp_path / 'train' / kind).glob('*.jpg')} == expected_names
        for filename in outputs:
            print(f'Reading from {filename}...')
            with rgb_organizer.Image.open(filename) as restored:
                np.testing.assert_array_equal(np.asarray(restored), np.zeros((8, 8, 3)))


class IterationDataset:
    """Explicit tf.data double using rotations to distinguish fixed and changing orders."""

    def __init__(self, rows, shuffle_calls):
        self.rows = rows
        self.shuffle_calls = shuffle_calls

    def __iter__(self):
        return iter(self.rows())

    def shuffle(self, buffer_size, reshuffle_each_iteration):
        self.shuffle_calls.append((buffer_size, reshuffle_each_iteration))
        iteration = 0

        def rows():
            nonlocal iteration
            values = list(self)
            if reshuffle_each_iteration:
                iteration += 1
            offset = (iteration if reshuffle_each_iteration else 1) % len(values)
            return values[offset:] + values[:offset]

        return IterationDataset(rows, self.shuffle_calls)

    def take(self, count):
        return IterationDataset(lambda: list(self)[:count], self.shuffle_calls)

    def skip(self, count):
        return IterationDataset(lambda: list(self)[count:], self.shuffle_calls)


@pytest.mark.parametrize('count', [20, 100, 103])
def test_training_source_uses_stable_disjoint_row_partitions_with_training_only_reshuffle(count):
    path = ROOT / 'train.py'
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    end = next(node.lineno for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'data_augmentation')
    names = {'ds', 'total_size', 'train_size', 'val_size', 'train_ds', 'val_ds'}
    statements = [node for node in tree.body if isinstance(node, ast.Assign) and node.lineno < end and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]

    def execute(nodes):
        shuffle_calls = []
        constructor = lambda tensors: IterationDataset(lambda: list(zip(*tensors)), shuffle_calls)
        namespace = {'data': np.arange(count), 'labels': np.arange(count) % 2, 'tf_data': SimpleNamespace(Dataset=SimpleNamespace(from_tensor_slices=constructor))}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        return namespace, shuffle_calls

    namespace, calls = execute(statements)
    training_orders = []
    validation_orders = []
    for _ in range(4):
        training = list(namespace['train_ds'])
        validation = list(namespace['val_ds'])
        assert len(training) == int(0.95 * count)
        assert len(validation) == count - len(training)
        assert set(training).isdisjoint(validation)
        assert set(training) | set(validation) == set(zip(np.arange(count), np.arange(count) % 2))
        training_orders.append(training)
        validation_orders.append(validation)
    assert calls == [(count, False), (int(0.95 * count), True)]
    assert all(set(order) == set(training_orders[0]) for order in training_orders)
    assert training_orders[0] != training_orders[1]
    assert all(order == validation_orders[0] for order in validation_orders)

    unsafe = copy.deepcopy(statements)
    initial_shuffle = next(node.value for node in unsafe if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'shuffle')
    next(keyword for keyword in initial_shuffle.keywords if keyword.arg == 'reshuffle_each_iteration').value = ast.Constant(True)
    ast.fix_missing_locations(ast.Module(body=unsafe, type_ignores=[]))
    previous, _ = execute(unsafe)
    assert not set(previous['train_ds']).isdisjoint(previous['val_ds'])


@pytest.mark.parametrize('empty_key', [None, 'hsc_lens'])
def test_real_npz_packaging_round_trips_through_training_image_and_label_loading(organizers, monkeypatch, tmp_path, capsys, empty_key):
    packer = importlib.import_module('package_npzs')
    datasets = {}
    for index, key in enumerate(packer.DATASET_KEYS):
        count = 0 if key == empty_key else index + 1
        images = np.arange(count * 30, dtype=np.float32).reshape(count, 2, 3, 5)
        datasets[key] = images
        filename = tmp_path / f'{key}.npz'
        print(f'Writing to {filename}...')
        np.savez_compressed(filename, **{key: images})
    config = tmp_path / 'config.yml'
    print(f'Writing to {config}...')
    with config.open('w') as stream:
        yaml.safe_dump({'data_dir': str(tmp_path)}, stream)
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()

    packer.main()

    filename = tmp_path / 'dataset.npz'
    assert f'Writing to {filename}...' in capsys.readouterr().out
    print(f'Reading from {filename}...')
    with np.load(filename) as archive:
        assert archive.files == list(packer.DATASET_KEYS)
        for key in packer.DATASET_KEYS:
            np.testing.assert_array_equal(archive[key], datasets[key])
            assert archive[key].dtype == np.float32
    path = ROOT / 'train.py'
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    names = {'image_datasets', 'hsc_lenses', 'hsc_nonlenses', 'slsim_lenses', 'slsim_nonlenses', 'data', 'labels'}
    end = next(node.lineno for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'ds' for target in node.targets))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == 'package_npzs']
    statements = [node for node in tree.body if isinstance(node, ast.Assign) and node.lineno < end and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    namespace = {'np': np, 'os': os, 'data_dir': str(tmp_path)}
    exec(compile(ast.Module(body=imports + statements, type_ignores=[]), str(path), 'exec'), namespace)
    order = ['hsc_lens', 'slsim_lens', 'hsc_nonlens', 'slsim_nonlens']
    np.testing.assert_array_equal(namespace['data'], np.concatenate([datasets[key] for key in order]))
    np.testing.assert_array_equal(namespace['labels'], np.concatenate([np.full(len(datasets[key]), int(key in order[:2]), dtype=np.uint8) for key in order]))


def test_shared_npz_reader_keeps_missing_dataset_errors_explicit(organizers, tmp_path):
    packer = importlib.import_module('package_npzs')
    filename = tmp_path / 'missing_class.npz'
    print(f'Writing to {filename}...')
    np.savez_compressed(filename, hsc_lens=np.zeros((1, 2, 3, 5)))
    with pytest.raises(KeyError, match='hsc_nonlens'):
        packer.read_dataset_archive(filename)


@pytest.mark.parametrize(('logits', 'truth'), [
    ([-0.2, 0.2, 0.4, 0.8], [0, 1, 1, 1]),
    ([-1000.0, -1.0, 0.0, 1.0, 1000.0], [0, 0, 0, 1, 1]),
    ([-0.2, 0.2, 0.4, 0.8], [1, 0, 1, 0]),
])
def test_actual_training_compile_uses_a_logit_accuracy_threshold(logits, truth):
    """Execute the actual compile call using explicit Keras configuration doubles."""
    path = ROOT / 'train.py'
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    compile_call = next(node for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'compile')
    model = Mock()
    accuracy = Mock(side_effect=lambda name, threshold=0.5: SimpleNamespace(name=name, threshold=threshold))
    loss = Mock(return_value=object())
    optimizer = Mock(return_value=object())
    namespace = {'model': model, 'lr_max': 1e-3, 'keras': SimpleNamespace(optimizers=SimpleNamespace(Adam=optimizer), losses=SimpleNamespace(BinaryCrossentropy=loss), metrics=SimpleNamespace(BinaryAccuracy=accuracy))}

    exec(compile(ast.Module(body=[compile_call], type_ignores=[]), str(path), 'exec'), namespace)

    loss.assert_called_once_with(from_logits=True)
    accuracy.assert_called_once_with(name='acc', threshold=0.0)
    optimizer.assert_called_once_with(learning_rate=1e-3)
    metric = model.compile.call_args.kwargs['metrics'][0]
    actual = np.mean((np.asarray(logits) > metric.threshold) == truth)
    expected = np.mean((expit(logits) > 0.5) == truth)
    assert actual == expected
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'make_model')
    output_layer = next(node for node in ast.walk(definition) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'Dense')
    assert next(ast.literal_eval(keyword.value) for keyword in output_layer.keywords if keyword.arg == 'activation') is None
