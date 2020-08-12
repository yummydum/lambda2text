import warnings
warnings.simplefilter('ignore')  # torchtext depreciation warning too verbose

from torchtext import data

from config import DATA_DIR
from utils import tokenize_formal, tokenize_text


def load_wrapper(data_name, batch_size, device, test_mode):

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

    if test_mode:
        data_path = DATA_DIR / f'pairs/{data_name}_test.tsv'
    else:
        data_path = DATA_DIR / f'pairs/{data_name}.tsv'

    tsv_fld = {"src": ("src", SRC), "trg": ("trg", TRG)}
    dataset = data.TabularDataset(path=data_path, format='tsv', fields=tsv_fld)

    # Split
    train, dev, test = dataset.split(split_ratio=[0.8, 0.1, 0.1])

    # Build vocab
    SRC.build_vocab(train, min_freq=2)
    TRG.build_vocab(train, min_freq=2)

    # Load data
    train, dev, test = data.BucketIterator.splits(
        (train, dev, test),
        batch_size=batch_size,
        shuffle=True,
        device=device,
        sort_key=lambda x: len(x.src))

    return (train, dev, test), SRC, TRG


def load_2018_data(which, batch_size, device, test_mode):
    """
    Data provided by author of https://www.aclweb.org/anthology/W18-6549/
    """

    SRC = data.Field(sequential=True,
                     use_vocab=True,
                     batch_first=True,
                     init_token='<sos>',
                     eos_token='<eos>',
                     include_lengths=True)
    TRG = data.Field(sequential=True,
                     use_vocab=True,
                     batch_first=True,
                     is_target=True,
                     init_token='<sos>',
                     eos_token='<eos>',
                     include_lengths=True)

    data_path = DATA_DIR / f'snli_datasets_cleansed/snli_{which}.tsv'

    tsv_fld = {"src": ("src", SRC), "trg": ("trg", TRG)}
    dataset = data.TabularDataset(path=data_path, format='tsv', fields=tsv_fld)

    # Split
    train, dev, test = dataset.split(split_ratio=[0.8, 0.1, 0.1])

    # Build vocab
    SRC.build_vocab(train, min_freq=2)
    TRG.build_vocab(train, min_freq=2)

    # Load data
    train, dev, test = data.BucketIterator.splits(
        (train, dev, test),
        batch_size=batch_size,
        shuffle=True,
        device=device,
        sort_key=lambda x: len(x.src))

    return (train, dev, test), SRC, TRG


def load_data(data, batch_size, device, test_mode):
    if data in {'mnli', 'snli'}:
        return load_wrapper(data, batch_size, device, test_mode)
    elif data == 'm30k':
        from preprocess.dataset_multi30k import load_datasets, SRC, TRG
        train, dev, test = load_datasets(batch_size, device, test_mode)
        return (train, dev, test), SRC, TRG
    elif data in {'2018_simple', '2018_formula', '2018_graph'}:
        which = data.split('_')[1]
        return load_2018_data(which, batch_size, device, test_mode)
    else:
        raise ValueError()
