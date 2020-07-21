from types import SimpleNamespace

from numpy.lib.npyio import load
from trainer.train_seq2seq import init_model
from preprocess.dataset import load_datasets


def test_init_model():
    load_datasets(5, 'CPU')  # Need to run vocab.build
    args = SimpleNamespace(test_run=False, hid_dim=100, n_heads=2, n_layers=3)
    init_model(args)
    return
