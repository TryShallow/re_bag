import torch
import torch.nn as nn
import numpy as np

from dataset import MyDataset


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.use_elmo = True
        self.max_query_size = 25
        self.max_nodes = 500
        self.encoding_size = 512
        self.query_encoding_type = 'lstm'
        self.dropout = 0.8
        self.hops = 5

    def forward(self, x):
        query_length = x['query_length_mb']
        nodes_elmo = x['nodes_elmo_mb']
        query_elmo = x['query_elmo_mb']
        nodes_length = x['nodes_length_mb'].type(torch.int32)
        adj = x['adj_mb'].type(torch.float32)
        bmask = x['bmask_mb'].type(torch.float32)
        nodes_compress, query_compress = self.feature_layer(query_length, nodes_elmo, query_elmo)
        nodes_mask = torch.arange(self.max_nodes, dtype=torch.int32).unsqueeze(0)\
            .repeat((nodes_length.size(0), 1)) < nodes_length.unsqueeze(-1)
        nodes_mask = nodes_mask.type(torch.float32).unsqueeze(-1)
        nodes = nodes_compress * nodes_mask
        nodes = nn.Dropout(self.dropout)(nodes)
        last_hop = nodes
        for _ in range(self.hops):
            last_hop = self.gcn_layer(adj, last_hop, nodes_mask)
        bi_attention = self.bi_attention_layer(query_compress, nodes_compress, last_hop)
        return self.output_layer(bi_attention, bmask)

    def feature_layer(self, query_length, nodes_elmo, query_elmo):
        query_flat, nodes_flat = None, None
        if self.use_elmo:
            query_flat = torch.reshape(query_elmo, (-1, self.max_query_size, 3 * 1024))
            nodes_flat = torch.reshape(nodes_elmo, (-1, self.max_nodes, 3 * 1024))
        query_compress = None
        if self.query_encoding_type == 'lstm':
            lstm_size = self.encoding_size / 2
            query_compress, (h1, c1) = nn.LSTM(query_flat.size(-1), 256, 2, bidirectional=True, batch_first=True)(query_flat)
        nodes_compress = torch.tanh(nn.Linear(nodes_flat.size(-1), self.encoding_size)(nodes_flat))
        return nodes_compress, query_compress

    def gcn_layer(self, adj, hidden_tensor, hidden_mask):
        adjacency_tensor = adj
        hidden_tensors = torch.stack([
            nn.Linear(hidden_tensor.size(-1), hidden_tensor.size(-1))(hidden_tensor) for _ in range(adj.size(1))
        ], 1) * hidden_mask.unsqueeze(1)
        update = torch.sum(torch.matmul(adjacency_tensor, hidden_tensors), 1) +\
            nn.Linear(hidden_tensor.size(-1), hidden_tensor.size(-1))(hidden_tensor) * hidden_mask
        update_combined = torch.cat((update, hidden_tensor), -1)
        att = torch.sigmoid(nn.Linear(update_combined.size(-1), hidden_tensor.size(-1))(update_combined)) * hidden_mask
        return att * torch.tanh(update) + (1 - att) * hidden_tensor

    def bi_attention_layer(self, query_compress, nodes_compress, last_hop):
        expanded_query = query_compress.unsqueeze(-3).repeat((1, self.max_nodes, 1, 1))
        expanded_nodes = last_hop.unsqueeze(-2).repeat((1, 1, self.max_query_size, 1))
        context_query_similarity = expanded_nodes * expanded_query
        concat_attention_data = torch.cat((expanded_nodes, expanded_query, context_query_similarity), -1)
        similarity = torch.mean(nn.Linear(concat_attention_data.size(-1), 1, False)(concat_attention_data), -1)
        nodes2query = torch.matmul(nn.Softmax(-1)(similarity), query_compress)
        b = nn.Softmax(-1)(similarity.max(-1)[0])
        query2nodes = torch.matmul(b.unsqueeze(1), nodes_compress).repeat((1, self.max_nodes, 1))
        g = torch.cat((nodes_compress, nodes2query, nodes_compress * nodes2query, nodes_compress * query2nodes), -1)
        return g

    def output_layer(self, bi_attention, bmask):
        raw_predictions = torch.tanh(nn.Linear(bi_attention.size(-1), 128)(bi_attention))
        raw_predictions = nn.Linear(128, 1)(raw_predictions).squeeze(-1)
        predictions2 = bmask * raw_predictions.unsqueeze(1)
        predictions2 = torch.where(predictions2 == 0, torch.FloatTensor([-np.inf]).repeat(predictions2.size()),
                                   predictions2).max(-1)[0]
        return predictions2


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    use_elmo = True
    add_query_node = False
    evaluation_mode = False
    max_nodes = 500
    max_query_size = 25
    max_candidates = 80
    max_candidates_len = 10

    dataset = MyDataset('train.json')
    dataloader = DataLoader(dataset, 2)
    model = Model()
    for i, batch in enumerate(dataloader):
        outputs = model(batch)
        break