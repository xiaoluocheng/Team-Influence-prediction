import torch
import numpy as np
import random
import dgl
from DBLP import graph, label
from DBLP.HAN import HAN

from DBLP.RGCNmodel import RGCN
from DBLP.feature_load import feature_skipgram, read_paper_features, read_author_features, feature_fusion, feat_test
from DBLP.HHGN import HHGN
from DBLP.simpleHGN import SimpleHGN
from DBLP.team import get_team_vectors, read_teams, sort_team_by_h_index, calculate_h_index_averages, \
    average_pooling, concatenate_vectors, add_teamfeat, lstm_pooling

g = graph.create_heterograph()
# print(g.number_of_nodes('author'))
print(g)
# g=feature1.feat_test(g)
# 示例用法
g = feature_skipgram(g)  # 加载随机游走特征
author_features = read_author_features()
paper_features = read_paper_features()
g = feature_fusion(g, author_features, paper_features)
# g = feat_test(g)
print(g.nodes['author'].data['feat'].shape,g.nodes['author'].data['feat'].dtype)  # 输出作者节点的特征形状
print(g.nodes['paper'].data['feat'].shape,g.nodes['paper'].data['feat'].dtype)  # 输出作者节点的特征形状
print(g.nodes['venue'].data['feat'].shape,g.nodes['venue'].data['feat'].dtype)  # 输出作者节点的特征形状
# 将H-index分配给图中的作者节点
# g = label.add_author_h_index(g)
g = label.add_author_stats_to_graph(g)
print("Shape of H-index tensor:", g.nodes['author'].data['h_index'].shape)
# 准备输入特征
features = {
    'paper': g.nodes['paper'].data['feat'],
    'author': g.nodes['author'].data['feat'],
    'venue': g.nodes['venue'].data['feat']
}
def set_seed(seed):
    torch.manual_seed(seed)  # 设置 PyTorch CPU 生成随机数的种子
    torch.cuda.manual_seed_all(seed)  # 设置 PyTorch 所有 GPU 生成随机数的种子
    np.random.seed(seed)  # 设置 numpy 生成随机数的种子
    random.seed(seed)  # 设置 python 随机数生成器的种子
    dgl.random.seed(seed)  # 设置 DGL 的随机数种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 设置为 False 可以加速计算（如果输入数据维度或类型变化不大）

# 固定随机数种子
set_seed(1)
# RGCN
# model = RGCN(in_feats=64, h_feats=16, out_feats=3,g=g)

# HAN
# 定义元路径
# meta_paths = [
#     [('paper', 'written_by', 'author')],
#     [('venue', 'hosts_author', 'author')],
#     [('author', 'collaborates_with', 'author')],
# ]
# model = HAN(meta_paths, in_size=64, hidden_size=10, out_size=3, num_heads=1)

# model = HAN(num_metapath=3, in_size=10, hidden_size=64, out_size=5, num_heads=8, dropout=0.5)

# SimpleHGN

# edge_dim = 16
# num_etypes = len(g.etypes)
# in_dim = [features[ntype].shape[1] for ntype in g.ntypes]
# hidden_dim = 32
# num_classes = 3
# num_layers = 2
# heads = [4, 1]
# feat_drop = 0.1
# negative_slope = 0.2
# residual = True
# beta = 0.5
# ntypes = g.ntypes
# model = SimpleHGN(edge_dim, num_etypes, in_dim, hidden_dim, num_classes,
#                   num_layers, heads, feat_drop, negative_slope, residual, beta, ntypes)


# HHGN
#
in_feats = 64
hidden_feats = 4
out_feats = 3
num_heads = 4
rel_names = g.etypes

model = HHGN(in_feats, hidden_feats, out_feats, num_heads, rel_names)
#



# 前向传播
# g.nodes['author'].data['h']  = model(g, features)['author']
# 假设 h 是一个形状为 (num_nodes, 64) 的张量
h = g.nodes['author'].data['feat']
# 将特征分为 3 组，并对每组求均值
# group_size = [21, 21, 22]
g.nodes['author'].data['h'] = torch.stack([
    h[:, :21].mean(dim=1),  # 第一组均值
    h[:, 21:42].mean(dim=1),  # 第二组均值
    h[:, 42:].mean(dim=1)   # 第三组均值
], dim=1)
print("输出节点",g.nodes['author'].data['h'].shape)
# # 假设g是你的异构图
# with torch.no_grad():
#     model.eval()
#     outputs = model([g, g, g], features)


# 转发模型以提取特征，这里不进行任何训练
# with torch.no_grad():  # 确保不进行梯度计算
#     model.eval()  # 设置为评估模式，特别是如果模型有如dropout等会变化的层
#     outputs = model(g, features)
# 获取作者节点的特征向量
# g.nodes['author'].data['h'] = outputs['author']
# print("提取的作者节点特征向量:", g.nodes['author'].data['h'])

teams = read_teams()
team_vectors = get_team_vectors(g, teams, graph.author_id_mapping)
# for community_id, vectors in team_vectors.items():
#     print(f"Community {community_id} has vectors of shape: {vectors.shape}")
#     print("Vectors:", vectors)
# print("Vectors:", team_vectors)

# 假设 'teams' 是从之前的 'read_teams' 函数获得的团队数据
# sorted_team_vectors = sort_team_by_h_index(g, teams, graph.author_id_mapping)
# for community_id, vectors in sorted_team_vectors.items():
#    if vectors.dim()==3:
#      num = num + 1
#      print(f"Community {community_id} has vectors of shape: {vectors.shape}")
# print(num)


# pooled_team_vectors = average_pooling(team_vectors)
pooled_team_vectors = lstm_pooling(team_vectors, 3, 3)
print("Pooled team vectors shape:", pooled_team_vectors)
print("Pooled team vectors shape:", pooled_team_vectors.shape)
# 使用示例，假设最大成员数量为12
# max_members = 30
# concatenated_team_vectors = concatenate_vectors(team_vectors, max_members)
# print(concatenated_team_vectors.shape)
# print(team_vectors)
# num=0


team_h_indices_averages = calculate_h_index_averages(teams, g.nodes['author'].data['h_index'], graph.author_id_mapping)
teamfeat=add_teamfeat(teams, g.nodes['author'].data['publications'],  g.nodes['author'].data['citations'],graph.author_id_mapping)
# print("teamfeat",teamfeat.shape)
print("teamfeat",teamfeat)
# print(type(pooled_team_vectors),type(teamfeat))
# print(team_h_indices_averages.shape)

combined_tensor = torch.cat((pooled_team_vectors, teamfeat), dim=1)
print('team_vectors',pooled_team_vectors)
# print('pooled_team_vectors',pooled_team_vectors)
torch.save(combined_tensor, 'pooled_team_vectors.pth')

