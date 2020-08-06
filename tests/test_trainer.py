from types import SimpleNamespace
import pytest
import trainer.train_seq2seq as target
from preprocess.dataset import load_data


@pytest.fixture
def mock_arg():
    return SimpleNamespace(test_run=False,
                           hid_dim=10,
                           n_heads=2,
                           n_layers=3,
                           dropout=0.1)


def test_init_model(mock_arg):
    data, SRC, TRG = load_data('mnli', 5, 'cpu', test_mode=True)
    target.init_model(mock_arg, SRC, TRG)
    return


def test_main(mock_arg, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_seq2seq.py", "--test_run"])
    assert target.main() == 0


def test_main_de2en(mock_arg, monkeypatch):
    monkeypatch.setattr("sys.argv",
                        ["train_seq2seq.py", "--test_run", "--de2en"])
    assert target.main() == 0
