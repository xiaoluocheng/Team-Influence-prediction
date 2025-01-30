import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.nn.pytorch import TypedLinear


class SimpleHGNConv(nn.Module):
    def __init__(self, edge_dim, in_dim, out_dim, num_heads, num_etypes, feat_drop=0.0,
                 negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super(SimpleHGNConv, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.edge_dim = edge_dim  # 确保 edge_dim 被正确定义
        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim)))
        self.W = nn.Parameter(torch.FloatTensor(in_dim, out_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)
        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))

        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)
        else:
            self.register_buffer("residual", None)

        self.beta = beta

    def forward(self, g, h, ntype, etype, presorted=False):
        emb = self.feat_drop(h)
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
        emb[torch.isnan(emb)] = 0.0

        edge_emb = self.W_r(self.edge_emb[etype], etype, presorted).view(-1, self.num_heads, self.edge_dim)
        row, col = g.edges()
        h_l = (self.a_l * emb).sum(dim=-1)[row]
        h_r = (self.a_r * emb).sum(dim=-1)[col]
        h_e = (self.a_e * edge_emb).sum(dim=-1)
        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        edge_attention = edge_softmax(g, edge_attention)

        if 'alpha' in g.edata.keys():
            res_attn = g.edata['alpha']
            edge_attention = edge_attention * (1 - self.beta) + res_attn * self.beta
        if self.num_heads == 1:
            edge_attention = edge_attention[:, 0].unsqueeze(1)

        with g.local_scope():
            emb = emb.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = edge_attention
            g.srcdata['emb'] = emb
            g.update_all(fn.u_mul_e('emb', 'alpha', 'm'), fn.sum('m', 'emb'))
            h_output = g.dstdata['emb'].view(-1, self.out_dim * self.num_heads)

        if self.residual:
            res = self.residual(h)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)

        return h_output

class SimpleHGN(nn.Module):
    def __init__(self, edge_dim, num_etypes, in_dim, hidden_dim, num_classes,
                 num_layers, heads, feat_drop, negative_slope, residual, beta, ntypes):
        super(SimpleHGN, self).__init__()
        self.ntypes = ntypes
        self.num_layers = num_layers
        self.hgn_layers = nn.ModuleList()
        self.activation = F.elu

        # Input layer
        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim, in_dim[0], hidden_dim, heads[0],
                num_etypes, feat_drop, negative_slope, False,
                self.activation, beta
            )
        )

        # Hidden layers
        for l in range(1, num_layers - 1):
            self.hgn_layers.append(
                SimpleHGNConv(
                    edge_dim, hidden_dim * heads[l - 1], hidden_dim, heads[l],
                    num_etypes, feat_drop, negative_slope, residual,
                    self.activation, beta
                )
            )

        # Output layer
        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim, hidden_dim * heads[-2], num_classes, heads[-1],
                num_etypes, feat_drop, negative_slope, residual,
                None, beta
            )
        )

    def forward(self, hg, h_dict):
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            g = dgl.to_homogeneous(hg, ndata='h')
            h = g.ndata['h']
            for l in range(self.num_layers):
                h = self.hgn_layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)
                h = h.flatten(1)

            # 将homogeneous图特征还原到heterogeneous图
            h_dict = {ntype: h[g.ndata['_TYPE'] == i] for i, ntype in enumerate(hg.ntypes)}

        return h_dict


# # 创建一个异构图
# g = graph1.create_heterograph()
# print(g)
# g = feature1.feat_test(g)
# print(g.nodes['author'].data['feat'].shape)  # 输出作者节点的特征形状
# print(g.nodes['paper'].data['feat'].shape)  # 输出作者节点的特征形状
# print(g.nodes['venue'].data['feat'].shape)  # 输出作者节点的特征形状
#
# # 设置特征字典
# features = {
#     'paper': g.nodes['paper'].data['feat'],
#     'author': g.nodes['author'].data['feat'],
#     'venue': g.nodes['venue'].data['feat']
# }
#
# # 模型参数
# edge_dim = 16
# num_etypes = len(g.etypes)
# in_dim = [features[ntype].shape[1] for ntype in g.ntypes]
# hidden_dim = 32
# num_classes = 4
# num_layers = 2
# heads = [4, 1]
# feat_drop = 0.1
# negative_slope = 0.2
# residual = True
# beta = 0.5
# ntypes = g.ntypes
#
# # 初始化模型
# model = SimpleHGN(edge_dim, num_etypes, in_dim, hidden_dim, num_classes,
#                   num_layers, heads, feat_drop, negative_slope, residual, beta, ntypes)
# # 前向传播
# with torch.no_grad():
#     outputs  = model(g, features)
# author_embeddings=outputs['author']
# print("Author Embeddings: ", author_embeddings.shape,author_embeddings)
# paper_embeddings=outputs['paper']
# print("Paper Embeddings: ", paper_embeddings.shape)