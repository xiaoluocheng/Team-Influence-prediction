import dgl
import torch

from DBLP.set_seed import set_seed

# Step 1: 构建同构图
src = []
dst = []
set_seed(1)
# 读取作者之间的边信息
with open('data_process/author_author.txt', 'r') as file:
    for line in file:
        author1, author2 = line.strip().split("\t")
        src.append(author1)
        dst.append(author2)

# 提取所有的节点（作者）
nodes = list(set(src + dst))  # 获取唯一的 author_id 列表
# print(len(src),len(dst),len(nodes))
# 将作者的 author_id 映射为唯一的数字 ID
author_to_idx = {author_id: idx for idx, author_id in enumerate(nodes)}  # 映射 author_id 到索引

# 根据 author_to_idx 将 src 和 dst 转换为节点索引
src_idx = [author_to_idx[author] for author in src]
dst_idx = [author_to_idx[author] for author in dst]

# 使用 DGL 构建同构图
graph = dgl.graph((torch.tensor(src_idx), torch.tensor(dst_idx)))
graph = dgl.add_self_loop(graph)

# Step 2: 将 author_id 映射后的 ID 存储到 graph.ndata 中
graph.ndata['author_id'] = torch.tensor([author_to_idx[author] for author in nodes], dtype=torch.long)
# 保持原始的 author_id 列表作为一个普通的 Python 列表，方便后续操作
author_id_str_list = nodes

# Step 3: 读取特征文件并存入字典，同时提取 h_index 并存储
author_features = {}
h_index_list = []  # 用于存储 h_index 的列表
with open("data_process/author_stats.txt", 'r') as f:
    for line in f:
        elements = line.strip().split()
        author_id = elements[0]
        # 只提取 publications, citations, num_collaborators, num_venues
        publications = float(elements[1])
        citations = float(elements[2])
        h_index = float(elements[3])  # 提取 h_index
        num_collaborators = float(elements[4])
        num_venues = float(elements[5])

        # 将这些特征存储在字典中
        author_features[author_id] = [publications, citations, num_collaborators, num_venues]
        # author_features[author_id] = [publications, citations, h_index, h_index]
        h_index_list.append(h_index)  # 保存 h_index 到列表

# 通过 graph.ndata['author_id'] 获取映射的数字 ID
author_ids_in_graph = graph.ndata['author_id']

# 为每个节点赋值特征（根据 author_id 匹配特征）
node_features = []
h_index_tensor = []

for idx in author_ids_in_graph:
    # 通过数字索引获取对应的 author_id
    author_id = author_id_str_list[idx]

    # 如果该 author_id 存在于特征字典中
    if author_id in author_features:
        node_features.append(author_features[author_id])
        h_index_tensor.append(h_index_list[idx])  # 根据 idx 获取 hindex
    else:
        # 如果没有找到对应的特征，用零向量代替
        node_features.append([0.0] * 4)
        h_index_tensor.append(0.0)

# 将特征转换为 tensor 并存储在图的节点数据中
graph.ndata['feat'] = torch.tensor(node_features, dtype=torch.float)
graph.ndata['hindex'] = torch.tensor(h_index_tensor, dtype=torch.float)

# 检查图的信息
# print("Number of nodes:", graph.num_nodes())
# print("Number of edges:", graph.num_edges())
# print("feature:", graph.ndata['feat'])
# print("hindex:", graph.ndata['hindex'])

community_file = 'communities_2-10.txt'
subgraphs = []

with open(community_file, 'r') as f:
    lines = f.readlines()
    current_community = []

    for line in lines:
        line = line.strip()
        if line.startswith("Community"):
            # 遇到新的社区时，将之前的社区生成子图并保存
            if current_community:
                nodes = [author_to_idx[node_id] for node_id in current_community if node_id in author_to_idx]
                subg = graph.subgraph(nodes)

                # Step 1: 计算子图的 aveh
                hindex_sum = subg.ndata['hindex'].sum()  # 获取子图中所有节点的 hindex 和
                num_nodes = subg.num_nodes()  # 获取子图的节点数
                aveh = hindex_sum / num_nodes if num_nodes > 0 else 0.0  # 计算平均 hindex

                # Step 2: 将 aveh 作为新特征存储到子图的 ndata 中
                subg.ndata['aveh'] = torch.full((num_nodes,), aveh, dtype=torch.float)  # 为每个节点赋值 aveh
                # subg.ndata['feat'] = torch.rand((num_nodes, 4), dtype=torch.float)
                # subg.ndata['feat'] = torch.full((num_nodes, 4), aveh,dtype=torch.float)  # 设置 feat 为 [num_nodes, 4] 的张量，每一维均为 aveh
                subg = dgl.add_self_loop(subg)
                # 将子图添加到列表中
                subgraphs.append(subg)
                current_community = []
        else:
            # 添加作者ID到当前社区
            current_community.append(line)

    # 最后一组社区
    if current_community:
        nodes = [author_to_idx[node_id] for node_id in current_community if node_id in author_to_idx]
        subg = graph.subgraph(nodes)

        # Step 1: 计算子图的 aveh
        hindex_sum = subg.ndata['hindex'].sum()  # 获取子图中所有节点的 hindex 和
        num_nodes = subg.num_nodes()  # 获取子图的节点数
        aveh = hindex_sum / num_nodes if num_nodes > 0 else 0.0  # 计算平均 hindex

        # Step 2: 将 aveh 作为新特征存储到子图的 ndata 中
        subg.ndata['aveh'] = torch.full((num_nodes,), aveh, dtype=torch.float)  # 为每个节点赋值 aveh
        # 将子图添加到列表中
        # subg.ndata['feat'] = torch.rand((num_nodes, 4), dtype=torch.float) # 设置 feat 为 [num_nodes, 4] 的张量，每一维均为 aveh
        subgraphs.append(subg)

# # 输出每个子图的信息
# for i, subg in enumerate(subgraphs):
#     print(f"Subgraph {i}:")
#     print(f"  Number of nodes: {subg.num_nodes()}")
#     print(f"  Average h-index (aveh): {subg.ndata['aveh'][0].item()}")
#  aveh 在每个节点上都相同，取第一个节点输出即可

# Step 5: 对生成的子图进行处理或保存
# for i, subg in enumerate(subgraphs):
#     print(f"Subgraph {i}:")
#     # print(f"  Number of nodes: {subg.num_nodes()}")
#     # print(f"  Number of edges: {subg.num_edges()}")
#     print(f"  Node features:\n{subg.ndata['feat']}")