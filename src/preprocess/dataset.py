from torchtext import data
from torchtext.data.utils import get_tokenizer

from config import DATA_DIR
from utils import tokenize_formal,tokenize_text

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
                  tokenize=tokenize_text,
                  is_target=True,
                  init_token='<sos>',
                  eos_token='<eos>')


def load_datasets(batch_size, device,test_mode=False):

    if test_mode:
        data_path = DATA_DIR / 'pairs/mnli_test.tsv'
    else:
        data_path = DATA_DIR / 'pairs/mnli.tsv'

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
                                      device=device)
