
from DBLP import graph
import torch
import torch.nn as nn
from dgl.nn import GATConv, HeteroGraphConv
import dgl
import torch.nn.functional as F

class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, num_heads):
        super(HANLayer, self).__init__()
        self.meta_paths = meta_paths
        self.gat_layers = nn.ModuleDict()

        # 为每种元路径创建一个HeteroGraphConv层
        for meta_path in meta_paths:
            conv_dict = {}
            for srctype, etype, dsttype in meta_path:
                conv_dict[(srctype, etype, dsttype)] = GATConv(in_size, out_size, num_heads, allow_zero_in_degree=True)
            self.gat_layers[str(meta_path)] = HeteroGraphConv(conv_dict, aggregate='sum')
        print(meta_paths, "\n", len(meta_paths), out_size, num_heads)
        # 全连接层，用于将多头注意力机制的输出拼接并转换为最终的特征
        self.fc = nn.Linear(len(meta_paths) * out_size * num_heads, out_size * num_heads)

    def forward(self, g, h_dict):
        # 初始化字典，用于存储不同类型节点的语义嵌入
        semantic_embeddings = {ntype: [] for ntype in h_dict.keys()}

        for meta_path in self.meta_paths:
            for srctype, etype, dsttype in meta_path:
                # 使用edge_type_subgraph创建子图
                subgraph = dgl.edge_type_subgraph(g, [(srctype, etype, dsttype)])
                print("subgraph", subgraph)
                if srctype in h_dict and dsttype in h_dict:
                    h = {srctype: h_dict[srctype], dsttype: h_dict[dsttype]}
                    output = self.gat_layers[str(meta_path)](subgraph, h)[dsttype].flatten(1)
                    print("output", meta_path, output)
                    # 如果输出节点数量小于原始节点数量，则进行填充
                    if output.shape[0] < h_dict[dsttype].shape[0]:
                        padding = torch.zeros((h_dict[dsttype].shape[0] - output.shape[0], output.shape[1]))
                        output = torch.cat([output, padding], dim=0)

                    # 将输出添加到对应节点类型的语义嵌入列表中
                    semantic_embeddings[dsttype].append(output)
        for i, tensor in enumerate(semantic_embeddings['author']):
            print(f"第 {i + 1} 个张量的形状: {tensor.shape}")
            print(f"第 {i + 1} 个张量的形状: {tensor}")
        # 将不同元路径的输出拼接起来，并通过全连接层进行处理
        for ntype in semantic_embeddings.keys():
            if semantic_embeddings[ntype]:
                semantic_embeddings[ntype] = torch.cat(semantic_embeddings[ntype], dim=1)
            else:
                semantic_embeddings[ntype] = torch.zeros((h_dict[ntype].shape[0], self.fc.in_features))
        print("semantic_embeddings", semantic_embeddings['author'].shape)
        print("semantic_embeddings", self.fc(semantic_embeddings['author']))
        # 将语义嵌入通过全连接层进行变换，得到最终的节点特征
        return {ntype: self.fc(semantic_embeddings[ntype]) for ntype in semantic_embeddings.keys()}
class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads):
        super(HAN, self).__init__()
        self.layer1 = HANLayer(meta_paths, in_size, hidden_size, num_heads)
        self.layer2 = HANLayer(meta_paths, hidden_size * num_heads, out_size, num_heads)

    def forward(self, g, h_dict):
        h_dict = self.layer1(g, h_dict)
        for ntype in h_dict:
            h_dict[ntype] = F.elu(h_dict[ntype])
        print("h_dict",h_dict['author'].shape)
        h_dict = self.layer2(g, h_dict)
        return h_dict# 只返回作者的嵌入
# # 定义元路径
# meta_paths = [
#     [('paper', 'written_by', 'author')],
#     [('venue', 'hosts_author', 'author')],
#     [('author', 'collaborates_with', 'author')],
#
# ]
# # 创建异构图
# g = graph1.create_heterograph()
# g = feature1.feat_test(g)
#
# # 打印节点特征形状
# print(g.nodes['author'].data['feat'].shape)
# print(g.nodes['paper'].data['feat'].shape)
# print(g.nodes['venue'].data['feat'].shape)
#
# # 创建模型
# features = {'paper': g.nodes['paper'].data['feat'], 'author': g.nodes['author'].data['feat'],
#             'venue': g.nodes['venue'].data['feat']}
# # model = HANLayer(meta_paths, in_size=64, out_size=3, num_heads=2)
# # 创建模型
# model = HAN(meta_paths, in_size=64, hidden_size=10, out_size=5, num_heads=1)
# # 前向传播
# with torch.no_grad():
#     author_embeddings = model(g, features)
#     print("Author Embeddings:", author_embeddings.shape, author_embeddings)
