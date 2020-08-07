import pytest
import torch
from model.transformer_seq2seq import Seq2Seq
from preprocess.dataset import load_data
from trainer.train_seq2seq import DEVICE
from utils import calculate_bleu, display_attention, tokenize_formal, translate_sentence


@pytest.fixture
def formula():
    return 'exists x0 . ( order ( x0 ) 	There is an order.'


@pytest.fixture
def dataset():
    data, SRC, TRG = load_data('mnli', 5, 'cpu', test_mode=True)
    return data, SRC, TRG


@pytest.fixture
def model(dataset):
    SRC = dataset[1]
    TRG = dataset[2]
    model = Seq2Seq(input_dim=len(SRC.vocab),
                    output_dim=len(TRG.vocab),
                    hid_dim=16,
                    n_heads=1,
                    n_layers=2,
                    device=torch.device('cpu'),
                    dropout=0.1)
    return model


def test_tokenize_formal(formula):
    result = tokenize_formal(formula)
    # Just check if it runs for now
    # sample = ' '.join(result[:20])
    # assert sample == ' exists x0 . ( order ( x0 ) & exists x1 . ( emission ( x1 ) & exists'


def test_translate_sentence(dataset, model, formula):
    SRC = dataset[1]
    TRG = dataset[2]
    result, attention = translate_sentence(formula, SRC, TRG, model, 'cpu')
    assert attention.size()[2] == len(result)
    assert attention.size()[3] == len(tokenize_formal(formula)) + 2
    return

@pytest.mark.skip(reason='cpu is not supported now')
def test_calculate_blue(dataset, model, formula):
    SRC = dataset[1]
    TRG = dataset[2]
    train, dev, test = dataset[0]
    result = calculate_bleu(test.dataset, SRC, TRG, model, DEVICE)
    assert isinstance(result, float)
    return


@pytest.mark.skip
def test_display_attention(dataset, model, formula):
    SRC = dataset[1]
    TRG = dataset[2]
    translation, attention = translate_sentence(formula, SRC, TRG, model,
                                                'cpu')
    display_attention(tokenize_formal(formula),
                      translation,
                      attention,
                      n_heads=1,
                      n_rows=1,
                      n_cols=1)
    return
