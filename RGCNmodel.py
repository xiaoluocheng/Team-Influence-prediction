import torch
import torch.nn.functional as F
import dgl.nn as dglnn


class RGCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, g):
        super(RGCN, self).__init__()
        # 定义第一层 hetero-graph convolution
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, h_feats)
            for rel in g.etypes
        }, aggregate='sum')
        # 定义第二层 hetero-graph convolution
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(h_feats, out_feats)
            for rel in g.etypes
        }, aggregate='sum')

    def forward(self, g, inputs):
        # print("After inputs:", inputs['author'])  # 打印第一层卷积后的作者特征
        h = self.conv1(g, inputs)
        # print("After conv1:", h['author'])  # 打印第一层卷积后的作者特征
        # print("第一层后的输出节点类型：", h.keys())
        h = {k: F.relu(v) for k, v in h.items()}
        # print("第一层后的输出节点：", h)
        h = self.conv2(g, h)
        # print("After conv2:", h['author'])  # 打印第一层卷积后的作者特征
        # print("第二层后的输出节点类型：", h.keys())
        return h
#
# g = graph.create_heterograph()
# print(g)
# g=feature1.feat_test(g)
# # g = feature.assign_author_features(g)
# # g = feature.assign_paper_features(g)
# # g = feature.assign_venue_features(g)
# print(g.nodes['author'].data['feat'].shape)  # 输出作者节点的特征形状
# print(g.nodes['paper'].data['feat'].shape)  # 输出作者节点的特征形状
# print(g.nodes['venue'].data['feat'].shape)  # 输出作者节点的特征形状
#
# # print(g.nodes['author'].data['feat'])  # 输出作者节点的特征
# # print(g.nodes['paper'].data['feat'])  # 输出作者节点的特征
# # print(g.nodes['venue'].data['feat'])  # 输出作者节点的特征
# # # 将H-index分配给图中的作者节点
# # h_index_tensor = label.add_h_index_to_graph(g)
# # print("Shape of H-index tensor:", h_index_tensor.shape)
# # 准备输入特征
# features = {
#     'paper': g.nodes['paper'].data['feat'],
#     'author': g.nodes['author'].data['feat'],
#     'venue': g.nodes['venue'].data['feat']
# }
# model = RGCN(in_feats=64, h_feats=16, out_feats=10,g=g)
#
# g.nodes['paper'].data['h']  = model(g, features)['paper']
# print("提取的论文节点特征向量:", g.nodes['paper'].data['h'])
# g.nodes['author'].data['h']  = model(g, features)['author']
# # print("提取的作者节点特征向量:", g.nodes['author'].data['h'])
# print("前1000个作者节点的特征向量:", g.nodes['author'].data['h'][:100000])
# # g.nodes['venue'].data['h']  = model(g, features)['venue']
# # print("提取的发表地节点特征向量:", g.nodes['venue'].data['h'])
