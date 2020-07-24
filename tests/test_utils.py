import pytest
from model.seq2seq import TransformerSeq2Seq
from preprocess.dataset import load_datasets, FORMAL, TEXT
from utils import display_attention, tokenize_formal, translate_sentence


@pytest.fixture
def formula():
    return '(exists x064341.(_patient(x064341) & _female(x064341) & _2(x064341) & exists x064342.(_room(x064342) & exists e064343.(_share(e064343) & (Subj(e064343) = x064341) & (Acc(e064343) = x064342))) & exists x064344.((_fever(x064344) | _symptom(x064344)) & exists e064345.(_have(e064345) & (Subj(e064345) = x064341) & (Acc(e064345) = x064344)))) & exists x064346.(_nurse(x064346) & _10(x064346) & exists x064347.((_fever(x064347) | _symptom(x064347)) & exists e064348.(_have(e064348) & (Subj(e064348) = x064346) & (Acc(e064348) = x064347)))) & exists x064349.(_about(x064349) & _people(x064349) & _100(x064349) & exists x064350.(_contact(x064350) & exists x064351.((x064351 = _them) & exists e064352.(_with(e064352,x064351) & (Subj(e064352) = x064350))) & exists e064353.(_have(e064353) & (Subj(e064353) = x064349) & (Acc(e064353) = x064350))) & exists e064354.(_quarantine(e064354) & (Acc(e064354) = x064349))))'


@pytest.fixture
def dataset():
    return load_datasets(5, 'cpu')


@pytest.fixture
def model():
    model = TransformerSeq2Seq(input_dim=len(FORMAL.vocab),
                               output_dim=len(TEXT.vocab),
                               hid_dim=16,
                               n_heads=1,
                               n_layers=2,
                               device='cpu')
    return model


# def test_tokenize_formal(formula):
#     result = tokenize_formal(formula)
#     sample = ' '.join(result[:20])
#     assert sample == '( exists x0 . ( patient ( x0 ) & female ( x0 ) & 2 ( x0 ) &'

# def test_translate_sentence(dataset, model, formula):
#     result, attention = translate_sentence(formula, FORMAL, TEXT, model, 'cpu')
#     assert attention.size()[2] == len(result)
#     assert attention.size()[3] == len(tokenize_formal(formula)) + 2
#     return

# @pytest.mark.skip
# def test_display_attention(dataset, model, formula):
#     translation, attention = translate_sentence(formula, FORMAL, TEXT, model,
#                                                 'cpu')
#     display_attention(tokenize_formal(formula),
#                       translation,
#                       attention,
#                       n_heads=1,
#                       n_rows=1,
#                       n_cols=1)
#     return
