import pytest
from preprocess.dataset import load_data


@pytest.fixture(scope='module')
def data():
    data, SRC, TRG = load_data('mnli', 5, 'cpu', test_mode=True)
    return data, SRC, TRG


def test_splits(data):
    train, dev, test = data[0]

    # Number of batches
    assert (len(dev) * 8) - 10 <= len(train)
    assert len(train) <= len(dev) * 8
    assert len(dev) == len(test)

    # Iterable
    next(iter(train))
    next(iter(dev))
    next(iter(test))


def test_fields(data):

    SRC = data[1]
    TRG = data[2]

    # Should have been built
    assert hasattr(SRC, 'vocab')
    assert hasattr(TRG, 'vocab')

    # Formal
    assert SRC.vocab.itos[0] == '<unk>'
    assert SRC.vocab.itos[1] == '<pad>'
    assert SRC.vocab.itos[2] == '<sos>'
    assert SRC.vocab.itos[3] == '<eos>'

    assert '▁exists' in SRC.vocab.stoi.keys()

    # Text
    assert TRG.vocab.itos[0] == '<unk>'
    assert TRG.vocab.itos[1] == '<pad>'
    assert TRG.vocab.itos[2] == '<sos>'
    assert TRG.vocab.itos[3] == '<eos>'

    assert 'x043135' not in TRG.vocab.stoi

    return


def test_batch(data):
    batch = next(iter(data[0][0]))
    formal, formal_len = batch.src
    text, text_len = batch.trg

    assert formal.size() == (5, formal_len.max().item())
    assert text.size() == (5, text_len.max().item())
    assert len(formal_len) == 5
    assert len(text_len) == 5
    return


def test_padding(data):
    batch = next(iter(data[0][0]))
    formal, formal_len = batch.src
    text, text_len = batch.trg

    # Formal
    for i in range(5):
        length = formal_len[i]
        assert not (formal[i][:length] == 1).any().item()
        assert (formal[i][length:] == 1).all().item()

    # Text
    for i in range(5):
        length = text_len[i]
        assert not (text[i][:length] == 1).any().item()
        assert (text[i][length:] == 1).all().item()

    return


def test_vocab_num(data):
    SRC = data[1]
    TRG = data[2]
    assert len(SRC.vocab) <= 20000  # small freq filtered, thus less than 20000
    assert len(TRG.vocab) <= 20000
    return
