import random
import warnings
warnings.simplefilter('ignore')  # torchtext depreciation warning too verbose

import numpy as np
import torch
from torchtext import data
from torchtext.datasets import Multi30k
import spacy

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = data.Field(tokenize=tokenize_de,
                 init_token='<sos>',
                 eos_token='<eos>',
                 lower=True,
                 batch_first=True,
                 include_lengths=True)

TRG = data.Field(tokenize=tokenize_en,
                 init_token='<sos>',
                 eos_token='<eos>',
                 lower=True,
                 batch_first=True,
                 include_lengths=True)


def load_datasets(batch_size,
                  device,
                  test_mode=False):  # currently test mode does nothing
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(SRC, TRG))
    TRG.build_vocab(train_data, min_freq=2)
    SRC.build_vocab(train_data, min_freq=2)

    return data.BucketIterator.splits((train_data, valid_data, test_data),
                                      batch_size=batch_size,
                                      device=device)
