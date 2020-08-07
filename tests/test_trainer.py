from types import SimpleNamespace
import pytest
import trainer.train_seq2seq as target
from preprocess.dataset import load_data


@pytest.fixture
def mock_arg():
    return SimpleNamespace(data='snli',model='lstm',test_run=False,
                           hid_dim=10,
                           n_heads=2,
                           n_layers=3,
                           dropout=0.1)


def test_init_model(mock_arg):
    data, SRC, TRG = load_data('mnli', 5, 'cpu', test_mode=True)
    target.init_model(mock_arg, SRC, TRG)
    return

@pytest.mark.parametrize('data',['snli','mnli'])
@pytest.mark.parametrize('model',['transformer','lstm'])
def test_main(data,model,mock_arg, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_seq2seq.py",data,model, "--test_run"])
    assert target.main() == 0

@pytest.mark.skip(reason='very slow')
def test_m30k(mock_arg, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_seq2seq.py",'m30k','lstm', "--test_run"])
    assert target.main() == 0


