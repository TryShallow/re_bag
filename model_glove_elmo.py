import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self, device, use_elmo=True, use_glove=True):
        super(Model, self).__init__()

        self.use_elmo = use_elmo
        self.use_glove = use_glove
        self.max_query_size = 25
        self.max_nodes = 500
        self.encoding_size = 512
        self.query_encoding_type = 'linear'
        self.dropout = 0.8
        self.hops = 5
        self.device = device

        # feature layer
        if self.query_encoding_type == 'linear':
            self.query_linear = nn.Linear(3 * 1024 + 300, 512)
        elif self.query_encoding_type == 'lstm':
            self.query_lstm = nn.LSTM(3 * 1024 + 300, 256, 2, bidirectional=True, batch_first=True)
        self.nodes_linear = nn.Linear(3 * 1024 + 300, self.encoding_size)
        # gcn layer
        self.nodes_dropout = nn.Dropout(self.dropout)
        self.hidden_linears = nn.ModuleList([nn.Linear(512, 512)] * 4)
        self.combined_linear = nn.Linear(1024, 512)
        # bi_attention layer
        self.attention_linear = nn.Linear(512 * 3, 1, False)
        # output layer
        self.out_att1 = nn.Linear(2048, 128)
        self.out_att2 = nn.Linear(128, 1)

        # self.minf = torch.FloatTensor([-np.inf]).requires_grad_(False).to(self.device)

    # def _calc_param(self):
    #     param = dict()
    #     dim_feature = 0
    #     if self.use_glove:
    #         # dim_feature

    def forward(self, x):
        query_length = x['query_length_mb']
        nodes_elmo = x['nodes_elmo_mb']
        query_elmo = x['query_elmo_mb']
        nodes_glove = x['nodes_glove_mb']
        query_glove = x['query_glove_mb']
        nodes_length = x['nodes_length_mb']
        adj = x['adj_mb']
        bmask = x['bmask_mb']

        nodes_compress, query_compress = self.feature_layer(query_length, nodes_elmo, query_elmo,
                                                            nodes_glove, query_glove)
        nodes_mask = torch.arange(self.max_nodes, dtype=torch.int32).requires_grad_(False).to(self.device).unsqueeze(0)\
            .repeat((nodes_length.size(0), 1)) < nodes_length.unsqueeze(-1)
        nodes_mask = nodes_mask.type(torch.float32).unsqueeze(-1)
        nodes = nodes_compress * nodes_mask
        nodes = self.nodes_dropout(nodes)
        last_hop = nodes
        for _ in range(self.hops):
            last_hop = self.gcn_layer(adj, last_hop, nodes_mask)
        bi_attention = self.bi_attention_layer(query_compress, nodes_compress, last_hop)
        return self.output_layer(bi_attention, bmask)

    def feature_layer(self, query_length, nodes_elmo=None, query_elmo=None, nodes_glove=None,
                      query_glove=None):
        query_flat, nodes_flat = None, None
        if self.use_elmo:
            query_flat = torch.reshape(query_elmo, (-1, self.max_query_size, 3 * 1024))
            nodes_flat = torch.reshape(nodes_elmo, (-1, self.max_nodes, 3 * 1024))
        if self.use_glove:
            query_flat = torch.cat((query_flat, query_glove), -1)
            nodes_flat = torch.cat((nodes_flat, nodes_glove), -1)

        query_compress = None
        if self.query_encoding_type == 'lstm':
            query_compress, (h1, c1) = self.query_lstm(query_flat)
        elif self.query_encoding_type == 'linear':
            query_compress = self.query_linear(query_flat)
        # [batch_size, max_nodes, 512]
        nodes_compress = torch.tanh(self.nodes_linear(nodes_flat))
        return nodes_compress, query_compress

    def gcn_layer(self, adj, hidden_tensor, hidden_mask):
        adjacency_tensor = adj
        # [batch_size, 3, max_nodes, max_nodes]
        hidden_tensors = torch.stack([
            self.hidden_linears[i](hidden_tensor) for i in range(adj.size(1))
        ], 1) * hidden_mask.unsqueeze(1)
        # hidden_tensors = torch.stack([
        #     nn.Linear(hidden_tensor.size(-1), hidden_tensor.size(-1))(hidden_tensor) for _ in range(adj.size(1))
        # ], 1) * hidden_mask.unsqueeze(1)

        update = torch.sum(torch.matmul(adjacency_tensor, hidden_tensors), 1) +\
            self.hidden_linears[adj.size(1)](hidden_tensor) * hidden_mask
        update_combined = torch.cat((update, hidden_tensor), -1)
        att = torch.sigmoid(self.combined_linear(update_combined)) * hidden_mask
        return att * torch.tanh(update) + (1 - att) * hidden_tensor

    def bi_attention_layer(self, query_compress, nodes_compress, last_hop):
        expanded_query = query_compress.unsqueeze(-3).repeat((1, self.max_nodes, 1, 1))
        expanded_nodes = last_hop.unsqueeze(-2).repeat((1, 1, self.max_query_size, 1))
        context_query_similarity = expanded_nodes * expanded_query
        # [batch_size, max_nodes, max_query, d * 3]
        concat_attention_data = torch.cat((expanded_nodes, expanded_query, context_query_similarity), -1)
        # [batch_size, max_nodes, max_query]
        similarity = torch.mean(self.attention_linear(concat_attention_data), -1)
        # [batch_size, max_nodes, d]
        nodes2query = torch.matmul(nn.Softmax(-1)(similarity), query_compress)
        # [batch_size, max_nodes]
        b = nn.Softmax(-1)(similarity.max(-1)[0])
        # [batch_size, max_nodes, d]
        query2nodes = torch.matmul(b.unsqueeze(1), nodes_compress).repeat((1, self.max_nodes, 1))
        g = torch.cat((nodes_compress, nodes2query, nodes_compress * nodes2query, nodes_compress * query2nodes), -1)
        return g

    def output_layer(self, bi_attention, bmask):
        raw_predictions = torch.tanh(self.out_att1(bi_attention))
        raw_predictions = self.out_att2(raw_predictions).squeeze(-1)
        predictions2 = bmask * raw_predictions.unsqueeze(1)
        predictions2[predictions2 == 0] = - np.inf
        predictions2 = predictions2.max(-1)[0]
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

    # dataset = MyDataset('train.json', range(1))
    # dataloader = DataLoader(dataset, 2)
    # model = Model()
    # for i, batch in enumerate(dataloader):
    #     outputs = model(batch)
    #     break

    for p in Model().parameters():
        print(p)