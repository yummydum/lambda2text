import pytest
from model.seq2seq import TransformerSeq2Seq
from preprocess.dataset import load_datasets, FORMAL, TEXT
from utils import display_attention, tokenize_formal, translate_sentence


@pytest.fixture
def formula():
    return 'exists x0 . ( order ( x0 ) & exists x1 . ( emission ( x1 ) & exists e0 . ( reduce ( e0 ) & ( Subj ( e0 ) = x0 ) & ( Acc ( e0 ) = x1 ) ) ) & exists x2 . ( technology ( x2 ) & control ( x2 ) & exists e1 . ( instal ( e1 ) & ( Subj ( e1 ) = x0 ) & ( Acc ( e1 ) = x2 ) ) ) & exists x3 . ( resources_DOT ( x3 ) & exists e2 . ( require ( e2 ) & ( Dat ( e2 ) = x0 ) & ( Acc ( e2 ) = x3 ) ) ) )	There is an upward leaping orchestral figure.'


@pytest.fixture
def dataset():
    return load_datasets(5, 'cpu',test_mode=True)


@pytest.fixture
def model():
    model = TransformerSeq2Seq(input_dim=len(FORMAL.vocab),
                               output_dim=len(TEXT.vocab),
                               hid_dim=16,
                               n_heads=1,
                               n_layers=2,
                               device='cpu',
                               dropout=0.1)
    return model


def test_tokenize_formal(formula):
    result = tokenize_formal(formula)
    # Just check if it runs for now
    # sample = ' '.join(result[:20])
    # assert sample == ' exists x0 . ( order ( x0 ) & exists x1 . ( emission ( x1 ) & exists'


def test_translate_sentence(dataset, model, formula):
    result, attention = translate_sentence(formula, FORMAL, TEXT, model, 'cpu')
    assert attention.size()[2] == len(result)
    assert attention.size()[3] == len(tokenize_formal(formula)) + 2
    return


@pytest.mark.skip
def test_display_attention(dataset, model, formula):
    translation, attention = translate_sentence(formula, FORMAL, TEXT, model,
                                                'cpu')
    display_attention(tokenize_formal(formula),
                      translation,
                      attention,
                      n_heads=1,
                      n_rows=1,
                      n_cols=1)
    return
