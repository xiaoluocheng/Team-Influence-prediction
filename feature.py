import pickle

import dgl
import torch
import random
import numpy as np
from gensim.models import Word2Vec

from DBLP import graph


def create_heterograph():
    data_dict = {
        ('author', 'writes', 'paper'): (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
        ('paper', 'written_by', 'author'): (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
        ('venue', 'hosts_author', 'author'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
        ('author', 'published_at', 'venue'): (torch.tensor([0, 1]), torch.tensor([0, 1])),
        ('author', 'collaborates_with', 'author'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])),
        ('paper', 'cites', 'paper'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0]))
    }

    num_nodes_dict = {
        'author': 3,
        'paper': 4,
        'venue': 2
    }

    g = dgl.heterograph(data_dict, num_nodes_dict)
    return g
def find_neighbors_manually(g, current_node, current_ntype, next_ntype, edge_type):
    src, dst = g.edges(etype=(current_ntype, edge_type, next_ntype))
    neighbors = []
    for s, d in zip(src, dst):
        if s.item() == current_node:
            neighbors.append(d.item())
    return neighbors

def metapath_random_walk(g, start_node, start_ntype, metapaths, walk_length,metapath_index):
    path = [(start_node, start_ntype)]  # 初始化路径，包含起始节点和节点类型
    # print("metapath_index",metapath_index)
    while len(path) < walk_length:
        current_node, current_ntype = path[-1]
        # 获取当前的元路径
        metapath = metapaths[metapath_index % len(metapaths)]
        # print(len(path), walk_length,metapath)
        if metapath[0] != current_ntype:
            # 更新元路径索引
            metapath_index += 1
            continue # 如果当前节点类型不匹配元路径的起始节点类型，停止游走

        edge_type, next_type = metapath[1], metapath[2]  # 当前边类型和下一个节点类型
        # print(f"Current node: {current_node} (type {current_ntype}), edge: {edge_type}, next type: {next_type}")
        # neighbors = find_neighbors_manually(g, current_node, current_ntype, next_type, edge_type)
        # 使用DGL内置的successors函数查找邻居节点
        neighbors = g.successors(current_node, etype=(current_ntype, edge_type, next_type)).numpy().tolist()
        # print(f"Neighbors: {neighbors}",f"len: {len(neighbors)}")
        if len(neighbors) == 0:
            # print("break")
            # print(f"No neighbors found for node {current_node} of type {current_ntype} via edge {edge_type} to type {next_type}")
            break

        next_node = random.choice(neighbors)
        path.append((next_node, next_type))



    return [f"{ntype}_{node}" for node, ntype in path]  # 返回包含节点ID和类型的路径


def hin2vec_embedding(g, metapaths, walk_length=10, num_walks=10, vector_size=32, window=5, min_count=0, sg=1):
# def hin2vec_embedding(g, metapaths, walk_length=1, num_walks=1, vector_size=32, window=1, min_count=0, sg=1):
    walks = []
    processed_walks=0
    total_nodes = sum(g.num_nodes(ntype) for ntype in g.ntypes)
    total_walks = total_nodes * num_walks
    for ntype in g.ntypes:
        nodes = g.nodes(ntype=ntype)
        for node in nodes:
            node_id = node.item()  # 转换为整数
            for i in range(num_walks):
                walk = metapath_random_walk(g, node_id, ntype, metapaths, walk_length,metapath_index=i)
                # print("i", i, walk)
                if len(walk) > 1:  # 确保生成的walks长度大于1，表示有跳转
                    walks.append(walk)
            processed_walks += 1
            if processed_walks % 10000 == 0:
                print(f"Processed {processed_walks}/{total_walks} walks")

    # 打印部分walks以检查所有节点类型是否都有walks
    print("Sample walks:", walks[:5])

    # 使用 Word2Vec 训练嵌入，设置 sg=1 使用 Skip-gram 模型
    print("Training Word2Vec model...")
    model = Word2Vec(walks, vector_size=vector_size, window=window, min_count=min_count, sg=sg)

    # 检查Word2Vec模型的词汇表
    print("Word2Vec vocabulary size:", len(model.wv))
    print("Word2Vec sample keys:", list(model.wv.key_to_index.keys())[:10])

    # 为每个节点类型和节点 ID 创建嵌入字典，如果嵌入不存在，则使用零向量
    default_vector = [0] * vector_size
    node_embeddings = {}
    for ntype in g.ntypes:
        node_embeddings[ntype] = []
        for node in g.nodes(ntype=ntype):
            node_key = f"{ntype}_{node.item()}"
            if node_key in model.wv:
                node_embeddings[ntype].append(model.wv[node_key])
            else:
                node_embeddings[ntype].append(default_vector)
        node_embeddings[ntype] = torch.tensor(np.array(node_embeddings[ntype]))

    return node_embeddings


# 示例使用
g = graph.create_heterograph()
# g =create_heterograph()
print(g)  # 确认图的构建正确



# 定义元路径
metapaths = [
    ('author', 'collaborates_with', 'author'),
    ('paper', 'cites', 'paper'),
    ('author', 'writes', 'paper'),
    ('paper', 'written_by', 'author'),
    ('venue', 'hosts_author', 'author'),
    ('author', 'published_at', 'venue'),
]

# 获取节点嵌入
print("Starting node embedding generation...")
node_embeddings = hin2vec_embedding(g, metapaths, walk_length=10, num_walks=6)  # 使用较小的walk_length和num_walks进行测试

# 将嵌入特征存储到DGL图中
for ntype in node_embeddings:
    g.nodes[ntype].data['feat'] = node_embeddings[ntype]
# 保存节点嵌入特征到文件
with open('node_embeddings.pkl', 'wb') as f:
    pickle.dump(node_embeddings, f)
# 输出某个节点类型的嵌入
print(g.nodes['author'].data['feat'])
print(g.nodes['paper'].data['feat'])
print(g.nodes['venue'].data['feat'])
