import random
import numpy as np
import pytest
import torch
from torchtext.data import batch
from model.seq2seq import TransformerSeq2Seq, TransformerEncoder, TransformerDecoder
from preprocess.dataset import load_datasets, FORMAL, TEXT

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

INPUT_DIM = 10
OUTPUT_DIM = 19
N_HEADS = 7
HID_DIM = 14


@pytest.fixture(scope='module')
def data():
    data = load_dataset(5, 'cpu')
    return data


@pytest.fixture()
def encoder():
    model = TransformerEncoder(input_dim=INPUT_DIM,
                               hid_dim=HID_DIM,
                               n_layers=2,
                               n_heads=N_HEADS,
                               pf_dim=10,
                               dropout=0.5)
    return model


@pytest.fixture()
def decoder():
    model = TransformerDecoder(output_dim=OUTPUT_DIM,
                               hid_dim=HID_DIM,
                               n_layers=2,
                               n_heads=N_HEADS,
                               pf_dim=10,
                               dropout=0.5)
    return model


@pytest.fixture(scope='function')
def seq2seq():
    model = TransformerSeq2Seq(input_dim=INPUT_DIM,
                               output_dim=OUTPUT_DIM,
                               hid_dim=14,
                               n_heads=7,
                               n_layers=2)
    return model


def test_init_seq2seq(seq2seq):
    pass
    return


def test_init_encoder(encoder):
    pass
    return


def test_init_decoder(decoder):
    pass
    return


def test_tok_embed(encoder):
    batch_len = 2
    sentence_len = 4

    # 2 sentences
    input_ids = torch.LongTensor([[4, 3, 9, 3], [4, 3, 2, 9]])
    result = encoder.tok_embed(input_ids)
    assert result.size() == (batch_len, sentence_len, HID_DIM)
    return


def test_pos_embed(encoder):
    batch_len = 2
    sentence_len = 4
    # 2 sentences
    input_ids = torch.LongTensor([[4, 3, 9, 3], [4, 3, 2, 9]])
    result = encoder.pos_embed(encoder.tok_embed(input_ids))
    assert result.size() == (batch_len, sentence_len, HID_DIM)
    return


def test_src_mask_with_pad(seq2seq):
    # No pad
    input_ids = torch.LongTensor([[4, 3, 1, 1], [8, 8, 7, 1], [4, 3, 2, 9]])
    mask = seq2seq.make_src_mask(input_ids)
    batch_size = 3
    max_len = 4
    assert mask.size() == (batch_size, 1, 1, max_len)

    expected = torch.tensor([True, True, False, False])
    assert (mask[0][0][0] == expected).all().item()

    expected = torch.tensor([True, True, True, False])
    assert (mask[1][0][0] == expected).all().item()

    expected = torch.tensor([True, True, True, True])
    assert (mask[2][0][0] == expected).all().item()
    return


def test_trg_mask(seq2seq):
    input_ids = torch.LongTensor([[4, 3, 1, 1], [8, 8, 7, 1], [4, 3, 2, 9]])
    mask = seq2seq.make_trg_mask(input_ids)
    batch_size = 3
    max_len = 4
    assert mask.size() == (batch_size, 1, max_len, max_len)

    expected = torch.tensor([True, False, False, False])
    assert (mask[0][0][0] == expected).all().item()

    expected = torch.tensor([True, True, False, False])
    assert (mask[0][0][1] == expected).all().item()

    expected = torch.tensor([True, True, False, False])
    assert (mask[0][0][2] == expected).all().item()

    return


def test_forward_encoder(encoder, seq2seq):
    batch_size = 3
    sentence_len = 4
    src = torch.LongTensor([[4, 3, 1, 1], [8, 8, 7, 1], [4, 3, 2, 9]])
    src_mask = seq2seq.make_src_mask(src)
    result = encoder(src, src_mask)
    assert result.size() == (batch_size, sentence_len, HID_DIM)
    return


def test_forward_decoder(encoder, decoder, seq2seq):
    batch_size = 3
    src_sentence_len = 4
    trg_sentence_len = 2

    src = torch.LongTensor([[4, 3, 1, 1], [8, 8, 7, 1], [4, 3, 2, 9]])
    trg = torch.LongTensor([[2, 2], [3, 1], [4, 1]])

    src_mask = seq2seq.make_src_mask(src)
    trg_mask = seq2seq.make_trg_mask(trg)

    encoded = encoder(src, src_mask)
    result = decoder(trg, encoded, trg_mask, src_mask)
    assert result[0].size() == (batch_size, trg_sentence_len, OUTPUT_DIM)
    assert result[1].size() == (batch_size, N_HEADS, trg_sentence_len,
                                src_sentence_len)
    return


def test_forward_seq2seq(seq2seq):
    batch_size = 3
    src_sentence_len = 4
    trg_sentence_len = 5
    src = torch.LongTensor([[4, 3, 1, 1], [8, 8, 7, 1], [4, 3, 2, 9]])
    tgt = torch.LongTensor([[2, 2, 2, 1, 1], [3, 9, 1, 1, 1], [4, 1, 1, 1, 1]])
    result = seq2seq(src, tgt)
    assert result[0].size() == (batch_size, trg_sentence_len, OUTPUT_DIM)
    assert result[1].size() == (batch_size, N_HEADS, trg_sentence_len,
                                src_sentence_len)
    return
