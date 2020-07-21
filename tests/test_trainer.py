from types import SimpleNamespace
import subprocess
from trainer.train_seq2seq import init_model
from preprocess.dataset import load_datasets


def test_init_model():
    load_datasets(5, 'CPU')  # Need to run vocab.build
    args = SimpleNamespace(test_run=False, hid_dim=100, n_heads=2, n_layers=3)
    init_model(args)
    return


def test_run():
    result = subprocess.run(
        ['python', '../src/trainer/train_seq2seq.py', '--test_run'])
    assert result.returncode == 0
    return
