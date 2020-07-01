from pathlib import Path
from torch.utils.data import Dataset
from torchtext import data,datasets


def read_pairs(batch_size,device):
    ENCODED = data.Field(sequential=True,include_lengths=True,use_vocab=False,batch_first=True,is_target=True)
    FORMAL = data.Field(sequential=True,include_lengths=True,use_vocab=True,batch_first=True)
    data_path = Path('../data/pairs/mrpc.tsv')
    tsv_fld = {"formal":("formal",FORMAL),"encoded":("encoded",ENCODED)}
    dataset = data.TabularDataset(path=data_path,format='tsv',fields=tsv_fld)
    FORMAL.build_vocab(dataset)
    return data.BucketIterator(dataset, batch_size=batch_size, shuffle=True, device=device)