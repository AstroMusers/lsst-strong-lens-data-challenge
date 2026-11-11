import pytest


@pytest.mark.parametrize('devices', [[], ['gpu0'], ['gpu0', 'gpu1']])
def test_training_initializes_gpu_and_dataset_before_plotting(devices):
    import ast
    import os
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    source_path = Path(__file__).resolve().parents[1] / 'train.py'
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    compile(tree, str(source_path), 'exec')
    assert any(isinstance(node, ast.ImportFrom) and node.module == 'model'
               and any(alias.name == 'make_model' for alias in node.names)
               for node in tree.body)
    assert not any(isinstance(node, ast.FunctionDef) and node.name == 'make_model'
                   for node in tree.body)
    start = next(index for index, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'gpus'
                         for target in node.targets))
    stop = next(index for index, node in enumerate(tree.body[start:], start)
                if isinstance(node, ast.For))
    archives = {name: np.zeros((index + 1, 41, 41, 5))
                for index, name in enumerate(('hsc_lens', 'hsc_nonlens',
                                              'slsim_lens', 'slsim_nonlens'))}
    loaded_paths = []
    gpu_calls = []

    def read_archive(archive_path):
        loaded_paths.append(archive_path)
        return archives

    namespace = {
        'tf': SimpleNamespace(config=SimpleNamespace(
            list_physical_devices=lambda kind: devices,
            set_visible_devices=lambda selected, kind: gpu_calls.append(selected),
            experimental=SimpleNamespace(
                set_memory_growth=lambda device, enabled: gpu_calls.append((device, enabled))))),
        'config': {'data_dir': 'example-data'}, 'os': os,
        'resolve_repository_path': lambda value: Path(value),
        'read_dataset_archive': read_archive,
        'plt': SimpleNamespace(subplots=lambda *args, **kwargs: ('figure', 'axes')),
        'print': lambda *args: None,
    }
    exec(compile(ast.Module(tree.body[start:stop], type_ignores=[]),
                 str(source_path), 'exec'), namespace)

    assert loaded_paths == [str(Path('example-data') / 'dataset.npz')]
    assert gpu_calls == ([['gpu0'], ('gpu0', True)] if devices else [])
    assert namespace['axes'] == 'axes'
    for images, label in namespace['datasets']:
        assert images is archives[label.replace('lenses', 'lens')]


@pytest.mark.parametrize(('num_classes', 'output_units'), [(2, 1), (5, 5)])
def test_make_model_returns_logits_with_expected_width(num_classes, output_units):
    pytest.importorskip('tensorflow')
    from model import make_model

    model = make_model((41, 41, 5), num_classes)

    assert model.output_shape == (None, output_units)
    assert model.layers[-1].activation.__name__ == 'linear'