
from torch.utils.data import DataLoader

from dataset.CMUDataset import CMUData


class opt:
    cvNo = 1
    A_type = "comparE"
    V_type = "denseface"
    L_type = "bert_large"
    norm_method = 'trn'
    in_mem = False


def MosiDataLoader(dataset, batch_size, data_path):
    if dataset == 'mosi':
        data = {
            'train': CMUData(data_path, 'train'),
            'valid': CMUData(data_path, 'valid'),
            'test': CMUData(data_path, 'test'),
        }
        orig_dim = data['test'].get_dim()
        dataLoader = {
            ds: DataLoader(data[ds],
                           batch_size=batch_size,
                           num_workers=8)
            for ds in data.keys()
        }

    return dataLoader, orig_dim
