from pathlib import Path
from torchtext import data

from config import DATA_DIR


def to_str(arr, foo):
    rows = len(arr)
    cols = len(arr[0])
    for i in range(rows):
        for j in range(cols):
            if arr[i][j] == '<pad>':
                arr[i][j] = -1
            else:
                arr[i][j] = int(arr[i][j])
    return arr


def read_pairs(batch_size, device):
    FORMAL = data.Field(sequential=True,
                        include_lengths=True,
                        use_vocab=True,
                        batch_first=True)
    ENCODED = data.Field(sequential=True,
                         include_lengths=True,
                         use_vocab=False,
                         batch_first=True,
                         is_target=True,
                         postprocessing=to_str)
    data_path = DATA_DIR / 'pairs/mrpc.tsv'
    tsv_fld = {"formal": ("formal", FORMAL), "encoded": ("encoded", ENCODED)}
    dataset = data.TabularDataset(path=data_path, format='tsv', fields=tsv_fld)
    FORMAL.build_vocab(dataset)
    return data.BucketIterator(dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               device=device)


if __name__ == "__main__":
    dataset = read_pairs(32, 'cpu')
    data = next(iter(dataset))
    from IPython import embed
    embed()