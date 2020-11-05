import os
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import from_scipy_sparse_matrix

class AirUSA(InMemoryDataset):
    r"""This dataset is the airport traffic network in the USA from the 
    `"Data Augmentation for Graph Neural Networks"
    <https://arxiv.org/pdf/2006.06830.pdf>`_ paper.
    Each node represents an airport and edge indicates the existence of 
    commercial flights between the airports. The node labels are generated 
    based on the label of activity measured by people and flights passed 
    the airports. The original graph does not have any features, one-hot degree 
    vectors are used as node features.
    
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    url = 'https://github.com/GAugAuthors/GAug/raw/master/data/graphs'

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['airport_adj.pkl', 'airport_features.pkl', 'airport_labels.pkl', 'airport_tvt_nids.pkl']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for file in self.raw_file_names:
            download_url(f'{self.url}/{file}', self.raw_dir)

    # noinspection PyTypeChecker
    def process(self):
        x = np.load(os.path.join(self.raw_dir, 'airport_features.pkl'), allow_pickle=True)
        y = np.load(os.path.join(self.raw_dir, 'airport_labels.pkl'), allow_pickle=True)
        adj = np.load(os.path.join(self.raw_dir, 'airport_adj.pkl'), allow_pickle=True)
        edge_index, _ = from_scipy_sparse_matrix(adj)

        train, val, test = np.load(os.path.join(self.raw_dir, 'airport_tvt_nids.pkl'), allow_pickle=True)
        train_mask = torch.zeros_like(y, dtype=torch.bool)
        val_mask = torch.zeros_like(y, dtype=torch.bool)
        test_mask = torch.zeros_like(y, dtype=torch.bool)

        train_mask[train] = True
        val_mask[val] = True
        test_mask[test] = True

        data = Data(
            x=x, edge_index=edge_index, y=y, num_nodes=len(y),
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return 'AirUSA()'
