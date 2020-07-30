from types import SimpleNamespace
import pytest
import trainer.train_seq2seq as target
from preprocess.dataset import load_datasets


@pytest.fixture
def mock_arg():
    return SimpleNamespace(test_run=False,
                           hid_dim=10,
                           n_heads=2,
                           n_layers=3,
                           dropout=0.1)


def test_init_model(mock_arg):
    load_datasets(5, 'cpu', test_mode=True)  # Need to run vocab.build
    target.init_model(mock_arg)
    return


def test_main(mock_arg, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_seq2seq.py", "--test_run"])
    assert target.main() == 0
