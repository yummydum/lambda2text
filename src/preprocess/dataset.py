from torchtext import data
from torchtext.data.utils import get_tokenizer

from config import DATA_DIR
from utils import tokenize_formal

FORMAL = data.Field(sequential=True,
                    include_lengths=True,
                    use_vocab=True,
                    batch_first=True,
                    tokenize=tokenize_formal,
                    init_token='<sos>',
                    eos_token='<eos>')
TEXT = data.Field(sequential=True,
                  include_lengths=True,
                  use_vocab=True,
                  batch_first=True,
                  tokenize=get_tokenizer('basic_english'),
                  is_target=True,
                  init_token='<sos>',
                  eos_token='<eos>')


def load_datasets(batch_size, device):

    # Currently only use mrpc
    data_path = DATA_DIR / 'pairs/mrpc.tsv'
    tsv_fld = {"formal": ("formal", FORMAL), "text": ("text", TEXT)}
    dataset = data.TabularDataset(path=data_path, format='tsv', fields=tsv_fld)

    # Split
    train, dev, test = dataset.split(split_ratio=[0.8, 0.1, 0.1])

    # Build vocab
    FORMAL.build_vocab(train, min_freq=2)
    TEXT.build_vocab(train, min_freq=2)

    # Return iterator
    return data.BucketIterator.splits((train, dev, test),
                                      batch_size=batch_size,
                                      shuffle=True,
                                      device=device,
                                      sort_key=lambda x: len(x.formal))
