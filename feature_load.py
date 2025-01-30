
import pickle
import torch
import torch.nn.functional as F
from DBLP.graph import create_heterograph, author_id_mapping, paper_id_mapping, venue_id_mapping


def feature_skipgram(g):
    with open('node_embeddings.pkl', 'rb') as f:
        node_embeddings = pickle.load(f)
    for ntype in node_embeddings:
        g.nodes[ntype].data['feat'] = node_embeddings[ntype].clone().detach()
        # g.nodes[ntype].data['feat']= g.nodes[ntype].data['feat'].astype(np.float64)
    g.nodes['paper'].data['feat']=g.nodes['paper'].data['feat'].to(torch.float32)
    return g

def read_author_features(target_dim=32):
    """
    从文件中读取作者特征并将其处理为指定维度（默认 32 维）。

    文件格式：每行包含 "author_id publications citations h_index num_collaborators num_venues"。
    注意：h_index 不会被读取。

    参数：
        file_path (str): 特征文件路径。
        target_dim (int): 目标特征维度，默认 32。

    返回：
        dict: 一个字典，键为 author_id，值为归一化后的特征向量 (Tensor)。
    """
    author_features = {}

    # 读取文件内容
    with open("data_process/author_stats.txt", 'r') as f:
        for line in f:
            elements = line.strip().split()
            author_id = elements[0]  # 作者ID
            # 提取特征（跳过 h_index 第3列）
            features = list(map(float, elements[1:3] + elements[4:]))
            # 转换为 Tensor
            features_tensor = torch.tensor(features, dtype=torch.float)
            # 映射到 target_dim（线性投影）
            projection_matrix = torch.randn(features_tensor.size(0), target_dim)
            projected_features = torch.matmul(features_tensor, projection_matrix)
            # 存储未归一化的特征
            author_features[author_id] = projected_features

    return author_features

def read_paper_features(target_dim=32):
    """
    从文件中读取论文特征并将其处理为指定维度（默认 32 维）。

    文件格式：每行包含 "paper_id citations num_references main_topic"。

    参数：
        file_path (str): 特征文件路径。
        target_dim (int): 目标特征维度，默认 32。

    返回：
        dict: 一个字典，键为 paper_id，值为处理后的特征向量 (Tensor)。
    """
    paper_features = {}

    # 读取文件内容
    with open("data_process/paper_stats.txt", 'r') as f:
        for line in f:
            elements = line.strip().split()
            paper_id = elements[0]  # 论文ID

            # 提取特征
            features = list(map(float, elements[1:]))

            # 转换为 Tensor
            features_tensor = torch.tensor(features, dtype=torch.float)

            # 映射到 target_dim（线性投影）
            projection_matrix = torch.randn(features_tensor.size(0), target_dim)
            projected_features = torch.matmul(features_tensor, projection_matrix)

            # 存储未归一化的特征
            paper_features[paper_id] = projected_features

    return paper_features

def feature_fusion(g, author_features, paper_features):
    """
    将随机游走生成的特征与初始节点特征进行融合，并归一化。

    参数：
        g (dgl.DGLGraph): 图对象，包含随机游走生成的特征。
        author_features (dict): 作者的初始特征字典，键为 author_id，值为特征张量。
        paper_features (dict): 论文的初始特征字典，键为 paper_id，值为特征张量。

    返回：
        dgl.DGLGraph: 融合特征后的图。
    """
    for ntype in g.ntypes:
        if ntype == 'author':
            # 从图中获取随机游走特征
            random_walk_feats = g.nodes[ntype].data['feat'][:, :32]

            # 获取初始特征
            initial_feats = []
            for node_id in g.nodes(ntype=ntype):
                author_id = str(node_id.item())
                if author_id in author_features:
                    initial_feats.append(author_features[author_id])
                else:
                    initial_feats.append(torch.zeros(32, dtype=torch.float32))
            initial_feats = torch.stack(initial_feats)

            # 特征拼接并归一化
            combined_feats = torch.cat([random_walk_feats, initial_feats], dim=1)
            g.nodes[ntype].data['feat'] = F.normalize(combined_feats, p=2, dim=1)

        elif ntype == 'paper':
            # 从图中获取随机游走特征
            random_walk_feats = g.nodes[ntype].data['feat'][:, :32]

            # 获取初始特征
            initial_feats = []
            for node_id in g.nodes(ntype=ntype):
                paper_id = str(node_id.item())
                if paper_id in paper_features:
                    initial_feats.append(paper_features[paper_id])
                else:
                    initial_feats.append(torch.zeros(32, dtype=torch.float32))
            initial_feats = torch.stack(initial_feats)

            # 特征拼接并归一化
            combined_feats = torch.cat([random_walk_feats, initial_feats], dim=1)
            g.nodes[ntype].data['feat'] = F.normalize(combined_feats, p=2, dim=1)

        elif ntype == 'venue':
            # 只有随机游走特征
            random_walk_feats = g.nodes[ntype].data['feat'][:, :32]

            # 将随机游走特征扩展为 64 维，填充零
            expanded_feats = torch.cat([random_walk_feats, torch.zeros(random_walk_feats.size(0), 32, dtype=torch.float32)], dim=1)

            # 归一化
            g.nodes[ntype].data['feat'] = F.normalize(expanded_feats, p=2, dim=1)

    return g
def feat_test(g):
    # 假设 'paper' 和 'venue' 节点的数量已经通过映射字典得到
    num_authors = len(author_id_mapping)
    num_papers = len(paper_id_mapping)
    num_venues = len(venue_id_mapping)
    feat_dim=64
    # 为 'paper' 和 'venue' 节点随机生成特征
    paper_features = torch.rand(num_papers, feat_dim)  # 生成随机数作为特征，特征维度为 10
    venue_features = torch.rand(num_venues, feat_dim)  # 同上
    author_features = torch.rand(num_authors, feat_dim)  # 同上

    # 将生成的特征赋值给图中对应的节点类型
    g.nodes['paper'].data['feat'] = paper_features
    g.nodes['venue'].data['feat'] = venue_features
    g.nodes['author'].data['feat'] = author_features

    return g
# g = create_heterograph()
# # 示例用法
# g = feature_skipgram(g)  # 加载随机游走特征
# author_features = read_author_features()
# paper_features = read_paper_features()
# g = feature_fusion(g, author_features, paper_features)
# print(g.nodes['author'].data['feat'].shape)
# print(g.nodes['paper'].data['feat'].shape)
# print(g.nodes['venue'].data['feat'].shape)

