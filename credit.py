import os
import ssl
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip


class Credit(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/chirag126/nifty/main/dataset/credit/'
    targets = ['NoDefaultNextMonth', 'Married', 'Single', 'Age', 'EducationLevel', 
               'MaxBillAmountOverLast6Months', 'MaxPaymentAmountOverLast6Months', 
               'MonthsWithZeroBalanceOverLast6Months', 'MonthsWithLowSpendingOverLast6Months', 
               'MonthsWithHighSpendingOverLast6Months', 'MostRecentBillAmount', 'MostRecentPaymentAmount', 
               'TotalOverdueCounts', 'TotalMonthsOverdue', 'HistoryOfOverduePayments']

    def __init__(self, root: str, target='NoDefaultNextMonth', transform=None, pre_transform=None):
        self.target = target
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self) -> list[str]:
        return ['credit.csv', 'credit_edges.txt.zip']

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> str:
        return f'data-{self.target}.pt'

    def download(self):
        context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context

        for filename in self.raw_file_names:
            path = download_url(os.path.join(self.url, filename), self.raw_dir)
            if filename.endswith('.zip'):
                extract_zip(path, self.raw_dir)
                # os.unlink(path)
        
        ssl._create_default_https_context = context

    def process(self):
        idx_features_labels = pd.read_csv(os.path.join(self.raw_dir, 'credit.csv'))
        header = list(idx_features_labels.columns)
        header.remove(self.target)
        header.remove('Single')

        edges_unordered = np.genfromtxt(os.path.join(self.raw_dir, 'credit_edges.txt')).astype('int')
        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[self.target].values
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        edge_index = from_scipy_sparse_matrix(adj)[0]

        x = torch.FloatTensor(np.array(features.todense()))
        y = torch.LongTensor(labels)

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'CreditDefaulter()'
