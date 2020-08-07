import random
import numpy as np
import pytest
import torch
from model.lstm_seq2seq import Seq2Seq, Encoder, Decoder,Attention
from preprocess.dataset import load_data

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

INPUT_DIM = 10
OUTPUT_DIM = 19
EMBED_DIM = 8
HID_DIM = 14


@pytest.fixture(scope='module')
def data():
    data, SRC, TRG = load_data('mnli', 5, 'cpu', test_mode=True)
    return data


@pytest.fixture()
def encoder():
    model = Encoder(input_dim=INPUT_DIM,
                    emb_dim=EMBED_DIM,
                    enc_hid_dim=HID_DIM,
                    dec_hid_dim=HID_DIM,
                    dropout=0.5)
    return model


@pytest.fixture()
def decoder():
    model = Decoder(output_dim=OUTPUT_DIM,
                    emb_dim=EMBED_DIM,
                    enc_hid_dim=HID_DIM,
                    dec_hid_dim=HID_DIM,
                    dropout=0.5
                    )
    return model


@pytest.fixture(scope='function')
def seq2seq():
    model = Seq2Seq(input_dim=INPUT_DIM,
                    output_dim=OUTPUT_DIM,
                    hid_dim=14,
                    dropout=0.5,
                    device=torch.device('cpu'))
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
    result = encoder.embedding(input_ids)
    assert result.size() == (batch_len, sentence_len, EMBED_DIM)
    return

def test_forward_encoder(encoder, seq2seq):
    batch_size = 3
    sentence_len = 4
    src = torch.LongTensor([[4, 3, 1, 1], [8, 8, 7, 1], [4, 3, 2, 9]]).transpose(0,1)
    src_len = torch.LongTensor([2,3,4])
    encoder_outputs,hidden,cell = encoder(src,src_len)
    assert encoder_outputs.size() == (sentence_len, batch_size, HID_DIM)  # batch second
    return


def test_forward_decoder(encoder, decoder, seq2seq):
    batch_size = 3
    src_sentence_len = 4
    trg_sentence_len = 2

    src = torch.LongTensor([[4, 3, 1, 1], [8, 8, 7, 1], [4, 3, 2, 9]]).transpose(0,1)
    src_len = torch.LongTensor([2,3,4])
    encoder_outputs,hidden,cell = encoder(src, src_len)

    tgt = torch.LongTensor([2,2,2])
    mask = torch.LongTensor([[True, True, False, False], [True, True, True, False], [True, True, True, True]])
    result = decoder(tgt, hidden,cell, encoder_outputs,mask)
    assert result[0].size() == (batch_size, OUTPUT_DIM)
    assert result[1].size() == (batch_size, HID_DIM)
    return


def test_forward_seq2seq(seq2seq):
    batch_size = 3
    src_sentence_len = 4
    trg_sentence_len = 5
    src = torch.LongTensor([[4, 3, 1, 1], [8, 8, 7, 1], [4, 3, 2, 9]])
    src_len = torch.LongTensor([2,3,4])
    tgt = torch.LongTensor([[2, 2, 2, 1, 1], [3, 9, 1, 1, 1], [4, 1, 1, 1, 1]])
    result = seq2seq(src, src_len,tgt)
    assert result.size() == (5,batch_size,OUTPUT_DIM)
    return
