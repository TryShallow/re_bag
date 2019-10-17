import pickle
import torch
import os
import numpy as np
import scipy.sparse

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, file, data_patches=range(5), use_elmo=True, max_nodes=500, max_query_size=25, max_candidates=80, max_candidates_len=10,
                 use_edge=True):
        self.file = file
        self.use_elmo = use_elmo
        self.max_nodes = max_nodes
        self.max_query_size = max_query_size
        self.max_candidates = max_candidates
        self.max_candidates_len = max_candidates_len
        self.use_edge = use_edge
        self.data_gen_dir = 'processed_data'

        self.data_elmo = []
        self.data = []
        self.elmo_slice_len = 1000
        self.data_patches = data_patches
        self._init_data()

        self.idx = list(range(len(self.data)))
        self.counter = len(self.data)

    def __getitem__(self, index):
        data_mb = self.data[index]

        data_elmo_mb = None
        nodes_elmo_mb, query_elmo_mb = None, None
        if self.use_elmo:
            data_elmo_mb = self.data_elmo[index]
            nodes_elmo_mb, query_elmo_mb = self.build_elmo_data(data_elmo_mb)

        nodes_length = self.truncate_nodes_and_edges(data_mb, data_elmo_mb)
        query_length = self.truncate_query(data_mb, data_elmo_mb)
        adj = self.build_edge_data(data_mb)
        bmask_mb = np.pad(
                np.array(
                    [(i == np.array(data_mb['nodes_candidates_id'])).astype(np.uint8)
                     for i in range(len(data_mb['candidates']))]
                ), ((0, self.max_candidates - len(data_mb['candidates'])),
                    (0, self.max_nodes - len(data_mb['nodes_candidates_id']))), mode='constant'
            )
        return {
            'id_mb': index,
            'nodes_length_mb': nodes_length,
            'query_length_mb': query_length,
            'bmask_mb': bmask_mb,
            'adj_mb': adj,
            'answer_candidates_id_mb': data_mb['answer_candidate_id'],
            'nodes_elmo_mb': nodes_elmo_mb,
            'query_elmo_mb': query_elmo_mb,
        }

    @staticmethod
    def to(batch, device):
        batch['query_length_mb'] = batch['query_length_mb'].to(device)
        batch['nodes_elmo_mb'] = batch['nodes_elmo_mb'].to(device)
        batch['query_elmo_mb'] = batch['query_elmo_mb'].to(device)
        batch['nodes_length_mb'] = batch['nodes_length_mb'].type(torch.int32).to(device)
        batch['adj_mb'] = batch['adj_mb'].type(torch.float32).to(device)
        batch['bmask_mb'] = batch['bmask_mb'].type(torch.float32).to(device)
        return batch

    def __len__(self):
        return self.counter

    def _init_data(self):
        graph_file_name = '{}.preprocessed.pickle'.format(self.file)
        with open(graph_file_name, 'rb') as f:
            self.data = [d for d in pickle.load(f) if len(d['nodes_candidates_id']) > 0]
            self.data = self.data[:1000]
        if self.use_elmo:
            elmo_file_name = '{}.elmo.preprocessed.pickle'.format(self.file)
            with open(elmo_file_name, "rb") as f:
                self.data_elmo = [d for d in pickle.load(f) if len(d['nodes_elmo']) > 0]
                self.data_elmo = self.data_elmo[:1000]

    def _init_data1(self):
        graph_file_name = os.path.join(self.data_gen_dir,'{}.preprocessed.pickle'.format(self.file))
        with open(graph_file_name, 'rb') as f:
            all_data = pickle.load(f)
            for index in self.data_patches:
                self.data.extend([dg for dg in all_data[index * self.elmo_slice_len:
                                                      min(len(all_data), (index + 1) * self.elmo_slice_len)]
                                  if len(dg['nodes_candidates_id']) > 0])
            # self.data = [d for d in pickle.load(f) if len(d['nodes_candidates_id']) > 0]
        if self.use_elmo:
            elmo_file_name = os.path.join(self.data_gen_dir, '{}.elmo.preprocessed.pickle'.format(self.file))
            for index in self.data_patches:
                with open(elmo_file_name + '.' + str(index), 'rb') as f:
                    self.data_elmo.extend([de for de in pickle.load(f) if len(de['nodes_elmo']) > 0])

    def build_elmo_data(self, data_elmo_mb):
        filt = lambda x: np.array([x[:, 0].mean(0), x[0, 1], x[-1, 2]])
        data_elmo_mb['nodes_elmo'] = data_elmo_mb['nodes_elmo'][: self.max_nodes]
        nodes_elmo_mb = np.pad(np.array([filt(c) for c in data_elmo_mb['nodes_elmo']]),
                               ((0, self.max_nodes - len(data_elmo_mb['nodes_elmo'])),
                                (0, 0), (0, 0)), mode='constant').astype(np.float32)
        data_elmo_mb['query_elmo'] = data_elmo_mb['query_elmo'][: self.max_query_size]
        query_elmo_mb = np.pad(data_elmo_mb['query_elmo'],
                               ((0, self.max_query_size - data_elmo_mb['query_elmo'].shape[0]),
                                (0, 0), (0, 0)), mode='constant').astype(np.float32)
        return nodes_elmo_mb, query_elmo_mb

    def truncate_nodes_and_edges(self, data, data_elmo):
        nodes_length_mb = len(data['nodes_candidates_id'])
        exceed_nodes_th = nodes_length_mb > self.max_nodes
        if exceed_nodes_th:
            data['edges_in'] = self.truncate_edges(data['edges_in'])
            data['edges_out'] = self.truncate_edges(data['edges_out'])
            data['nodes_candidates_id'] = data['nodes_candidates_id'][: self.max_nodes]
            if self.use_elmo:
                data_elmo['nodes_elmo'] = data_elmo['nodes_elmo'][: self.max_nodes]
        data['nodes_candidates_id'] = data['nodes_candidates_id'][: self.max_nodes]

        return min(nodes_length_mb, self.max_nodes)

    def truncate_edges(self, edges):
        truncated_edges = []
        for edge_pair in edges:
            if edge_pair[0] >= self.max_nodes:
                continue
            if edge_pair[1] < self.max_nodes:
                truncated_edges.append(edge_pair)
        return truncated_edges

    def truncate_query(self, data, data_elmo):
        query_length_mb = len(data['query'])
        exceed_query_th = query_length_mb > self.max_query_size
        if exceed_query_th:
            if self.use_elmo:
                data_elmo['query_elmo'] = data_elmo['query_elmo'][: self.max_query_size]
        data['query'] = data['query'][: self.max_query_size]
        return min(query_length_mb, self.max_query_size)

    def build_edge_data(self, data_mb):
        len_nodes = len(data_mb['nodes_candidates_id'])
        if self.use_edge:
            adj_ = []
            if len(data_mb['edges_in']) == 0:
                adj_.append(np.zeros((self.max_nodes, self.max_nodes)))
            else:
                adj_.append(scipy.sparse.coo_matrix(
                    (np.ones(len(data_mb['edges_in'])), np.array(data_mb['edges_in']).T),
                    shape=(self.max_nodes, self.max_nodes)
                ).toarray())
            if len(data_mb['edges_out']) == 0:
                adj_.append(np.zeros((self.max_nodes, self.max_nodes)))
            else:
                adj_.append(scipy.sparse.coo_matrix(
                    (np.ones(len(data_mb['edges_out'])), np.array(data_mb['edges_out']).T),
                    shape=(self.max_nodes, self.max_nodes)
                ).toarray())
            adj = np.pad(
                np.ones((len_nodes, len_nodes)), ((0, self.max_nodes - len_nodes),
                                                  (0, self.max_nodes - len_nodes)), mode='constant'
            ) - adj_[0] - adj_[1] - np.pad(
                np.eye(len_nodes), ((0, self.max_nodes - len_nodes),
                                    (0, self.max_nodes - len_nodes)), mode='constant'
            )
            adj_.append(np.clip(adj, 0, 1))
            adj = np.stack(adj_, 0)
            d_ = adj.sum(-1)
            d_[np.nonzero(d_)] **= -1
            adj = adj * np.expand_dims(d_, -1)
        else:
            adj = np.pad(
                np.ones((len_nodes, len_nodes)), ((0, self.max_nodes - len_nodes),
                                                  (0, self.max_nodes - len_nodes)), mode='constant'
            ) - np.pad(
                np.eye(len_nodes), ((0, self.max_nodes - len_nodes),
                                    (0, self.max_nodes - len_nodes)), mode='constant'
            )
        return adj


if __name__ == '__main__':
    use_elmo = True
    add_query_node = False
    evaluation_mode = False
    max_nodes = 500
    max_query_size = 25
    max_candidates = 80
    max_candidates_len = 10

    dataset = MyDataset('dev.json')
    dataloader = DataLoader(dataset, 2)
    for i, d in enumerate(dataloader):
        print(i, d['adj_mb'].size())
        if i == 10:
            break

