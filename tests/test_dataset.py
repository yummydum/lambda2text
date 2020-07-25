import pytest
from preprocess.dataset import load_datasets, FORMAL, TEXT


@pytest.fixture(scope='module')
def data():
    train, dev, test = load_datasets(5, 'cpu')
    return train, dev, test


def test_splits(data):
    train, dev, test = data

    # Number of batches
    assert (len(dev) * 8) - 10 <= len(train)
    assert len(train) <= len(dev) * 8
    assert len(dev) == len(test)

    # Iterable
    next(iter(train))
    next(iter(dev))
    next(iter(test))


def test_fields(data):

    # Should have been built
    assert hasattr(FORMAL, 'vocab')
    assert hasattr(TEXT, 'vocab')

    # Formal
    assert FORMAL.vocab.itos[0] == '<unk>'
    assert FORMAL.vocab.itos[1] == '<pad>'
    assert FORMAL.vocab.itos[2] == '<sos>'
    assert FORMAL.vocab.itos[3] == '<eos>'

    assert 'exists' in FORMAL.vocab.stoi

    # Text
    assert TEXT.vocab.itos[0] == '<unk>'
    assert TEXT.vocab.itos[1] == '<pad>'
    assert TEXT.vocab.itos[2] == '<sos>'
    assert TEXT.vocab.itos[3] == '<eos>'

    assert 'x043135' not in TEXT.vocab.stoi

    return


def test_batch(data):
    batch = next(iter(data[0]))
    formal, formal_len = batch.formal
    text, text_len = batch.text

    assert formal.size() == (5, formal_len.max().item())
    assert text.size() == (5, text_len.max().item())
    assert len(formal_len) == 5
    assert len(text_len) == 5
    return


def test_padding(data):
    batch = next(iter(data[0]))
    formal, formal_len = batch.formal
    text, text_len = batch.text

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

def test_vocab_num():
    assert len(FORMAL.vocab) <= 20000  # small freq filtered, thus less than 20000
    assert len(TEXT.vocab) <= 20000
    return 
