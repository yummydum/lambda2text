import warnings
warnings.simplefilter('ignore')  # torchtext depreciation warning too verbose

from torchtext import data
from torchtext.data.utils import get_tokenizer

from config import DATA_DIR
from utils import tokenize_formal, tokenize_text

SRC = data.Field(sequential=True,
                 use_vocab=True,
                 batch_first=True,
                 tokenize=tokenize_formal,
                 init_token='<sos>',
                 eos_token='<eos>',
                 include_lengths=True)
TRG = data.Field(sequential=True,
                 use_vocab=True,
                 batch_first=True,
                 tokenize=tokenize_text,
                 is_target=True,
                 init_token='<sos>',
                 eos_token='<eos>',
                 include_lengths=True)


def load_datasets(batch_size, device, test_mode=False):

    if test_mode:
        data_path = DATA_DIR / 'pairs/mnli_test.tsv'
    else:
        data_path = DATA_DIR / 'pairs/mnli.tsv'

    tsv_fld = {"src": ("src", SRC), "trg": ("trg", TRG)}
    dataset = data.TabularDataset(path=data_path, format='tsv', fields=tsv_fld)

    # Split
    train, dev, test = dataset.split(split_ratio=[0.8, 0.1, 0.1])

    # Build vocab
    SRC.build_vocab(train, min_freq=2)
    TRG.build_vocab(train, min_freq=2)

    # Return iterator
    return data.BucketIterator.splits((train, dev, test),
                                      batch_size=batch_size,
                                      shuffle=True,
                                      device=device,
                                      sort_key=lambda x: len(x.src))


def get_loader(is_formal):
    if is_formal:
        global load_datasets, SRC, TRG
        return load_datasets, SRC, TRG
    else:
        from preprocess.dataset_multi30k import load_datasets, SRC, TRG
        return load_datasets, SRC, TRG
