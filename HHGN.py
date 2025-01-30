import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv, GraphConv, HeteroGraphConv



# 定义一个包装器来处理 nn.Linear
# 定义一个包装器来处理 nn.Linear
class LinearWrapper(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(LinearWrapper, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, h):
        return self.linear(h)

class MultiRelationalConv(nn.Module):
    def __init__(self, in_feats, out_feats, rel_names):
        super(MultiRelationalConv, self).__init__()
        self.conv = HeteroGraphConv({
            rel: GraphConv(in_feats, out_feats) for rel in rel_names
        }, aggregate='sum')

    def forward(self, g, h):
        h_new = self.conv(g, h)
        # 保留未更新的节点类型的特征
        for ntype in h:
            if ntype not in h_new:
                h_new[ntype] = h[ntype]
        return h_new

class AdaptiveAttention(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, rel_names):
        super(AdaptiveAttention, self).__init__()
        self.num_heads = num_heads
        self.attn = HeteroGraphConv({
            rel: GATConv(in_feats, out_feats // num_heads, num_heads, allow_zero_in_degree=True) for rel in rel_names
        }, aggregate='sum')

    def forward(self, g, h):
        h_new = self.attn(g, h)
        # 展平多头注意力机制的输出
        for ntype in h_new:
            h_new[ntype] = h_new[ntype].flatten(1)
        # 保留未更新的节点类型的特征
        for ntype in h:
            if ntype not in h_new:
                h_new[ntype] = h[ntype]
        return h_new

class SimplifiedHeteroGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, rel_names):
        super(SimplifiedHeteroGraphConv, self).__init__()
        self.fc = nn.ModuleDict({
            rel: LinearWrapper(in_feats, out_feats) for rel in rel_names
        })

    def forward(self, g, h):
        with g.local_scope():
            h_new = {ntype: torch.zeros(g.number_of_nodes(ntype), h[ntype].shape[1], device=h[ntype].device) for ntype in g.ntypes}
            for rel in g.etypes:
                stype, etype, dtype = g.to_canonical_etype(rel)
                g.nodes[stype].data['h'] = h[stype]
                g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'h'), etype=rel)
                updated_feat = g.nodes[dtype].data['h']
                h_new[dtype] += self.fc[rel](updated_feat)
            for ntype in h:
                if ntype not in h_new:
                    h_new[ntype] = h[ntype]
            return h_new

class HHGN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads, rel_names):
        super(HHGN, self).__init__()
        self.multi_rel_conv = MultiRelationalConv(in_feats, hidden_feats, rel_names)
        self.adaptive_attn = AdaptiveAttention(hidden_feats, hidden_feats, num_heads, rel_names)
        self.simple_conv = SimplifiedHeteroGraphConv(hidden_feats, out_feats, rel_names)

    def forward(self, g, h):
        print("Before MultiRelationalConv:", {k: v.shape for k, v in h.items()})
        h = self.multi_rel_conv(g, h)
        print("After MultiRelationalConv:", {k: v.shape for k, v in h.items()})
        # h = self.adaptive_attn(g, h)
        h = self.multi_rel_conv(g, h)
        print("After AdaptiveAttention:", {k: v.shape for k, v in h.items()})
        # h = self.simple_conv(g, h)
        # print("After SimplifiedHeteroGraphConv:", {k: v.shape for k, v in h.items()})
        return h


#
# # 创建一个异构图
# g = graph1.create_heterograph()
# print(g)
# g = feature1.feat_test(g)
# print(g.nodes['author'].data['feat'].shape)  # 输出作者节点的特征形状
# print(g.nodes['paper'].data['feat'].shape)  # 输出作者节点的特征形状
# print(g.nodes['venue'].data['feat'].shape)  # 输出作者节点的特征形状
#
# in_feats = 64
# hidden_feats = 64
# out_feats = 16
# num_heads = 4
# rel_names = g.etypes
# features = {'paper': g.nodes['paper'].data['feat'], 'author': g.nodes['author'].data['feat'], 'venue': g.nodes['venue'].data['feat']}
#
# model = HHGN(in_feats, hidden_feats, out_feats, num_heads, rel_names)
# # 前向传播
# with torch.no_grad():
#     outputs  = model(g, features)
# author_embeddings=outputs['author']
# print("Author Embeddings: ", author_embeddings.shape)