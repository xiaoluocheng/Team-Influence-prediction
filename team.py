# from HANmodel import HAN
# from HHGN import HHGN
# from simpleHGN import SimpleHGN


def read_teams():
    num_communities = 0
    teams = {}
    current_community = None
    with open("communities_2-10.txt", 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('Community'):
                num_communities += 1
                # 新的社区开始
                current_community = line.split(':')[0].split()[-1]
                teams[current_community] = []
            else:
                # 添加成员ID到当前社区
                if current_community:
                    teams[current_community].append(line)
    print(f'文件中的社区数量为: {num_communities} 个')
    return teams

# 使用该函数读取文件

# print(len(teams))
# 输出统计结果

def get_team_vectors(g, teams, author_id_mapping):
    team_vectors = {}
    for community, members in teams.items():
        # 获取团队成员在图中的索引，确保成员ID是字符串
        member_indices = [
            author_id_mapping[member]
            for member in members if member in author_id_mapping
        ]
        # 从图中提取节点向量
        if member_indices:  # 确保列表不为空
            team_vectors[community] = g.nodes['author'].data['h'][torch.tensor(member_indices)]
    return team_vectors


def sort_team_by_h_index(g, teams, author_id_mapping):
    sorted_team_vectors = {}
    h_index_tensor = g.nodes['author'].data['h_index']

    for community, members in teams.items():
        # 获取团队成员在图中的索引，确保成员ID是数字
        member_indices = [
            author_id_mapping[int(member)]
            for member in members if member.isdigit() and int(member) in author_id_mapping
        ]

        if member_indices:
            # 提取相应的H指数
            member_h_indices = h_index_tensor[torch.tensor(member_indices)]
            # 根据H指数排序成员（降序）
            _, sorted_indices = torch.sort(member_h_indices, descending=True)
            sorted_member_indices = torch.tensor(member_indices)[sorted_indices]

            # 根据排序的索引提取向量
            sorted_vectors = g.nodes['author'].data['feat'][sorted_member_indices]
            sorted_team_vectors[community] = sorted_vectors

    return sorted_team_vectors


import networkx as nx

# 从author_author.txt构建作者之间的引用关系图
def build_author_graph():
    G = nx.Graph()  # 无向图
    with open('data_process/author_author.txt', 'r') as f:
        for line in f:
            # 处理字符串ID
            author1, author2 = line.strip().split()
            G.add_edge(author1, author2)  # 添加边，表示作者之间的引用关系
    return G
# 主函数，添加团队特征
def add_teamfeat(teams, publications_tensor, citations_tensor, author_id_mapping):
    totals = []
    author_graph = build_author_graph()
    for members in teams.values():  # 假设teams是社区ID到成员列表的映射
        # 将成员ID映射到对应的索引
        member_indices = [author_id_mapping[member] for member in members if member in author_id_mapping]

        if member_indices:
            # 提取成员的发文量和引用量
            member_publications = publications_tensor[torch.tensor(member_indices)]
            member_citations = citations_tensor[torch.tensor(member_indices)]

            # 计算总发文量和总引用量
            total_publications = member_publications.sum().item()
            total_citations = member_citations.sum().item()

            # 计算团队规模
            team_size = len(member_indices)

            # 计算团队成员的互动频率（根据author_author图中成员之间的边数）
            team_subgraph = author_graph.subgraph(members)  # 提取团队成员子图
            interaction_frequency = team_subgraph.number_of_edges()

            # 计算团队成员的平均度中心性
            degrees = dict(author_graph.degree(members))  # 获取成员的度
            avg_degree_centrality = sum(degrees.values()) / team_size if team_size > 0 else 0

            # 将所有特征汇总
            totals.append([total_publications, total_citations, team_size, interaction_frequency, avg_degree_centrality])

    # 返回包含总发文量、总引用量、团队规模、互动频率和平均度中心性的张量，并保持二维形状以匹配模型输出
    return torch.tensor(totals)




# 你的其他相关数据，teams, publications_tensor, citations_tensor, author_id_mapping等需要在调用add_teamfeat函数时传入
# 示例调用:
# result = add_teamfeat(teams, publications_tensor, citations_tensor, author_id_mapping, author_graph)

def add_teamfeat1(teams, publications_tensor, citations_tensor, author_id_mapping):
    totals = []

    for members in teams.values():  # 假设teams是社区ID到成员列表的映射
        # 将成员ID映射到对应的索引
        member_indices = [author_id_mapping[member] for member in members if member in author_id_mapping]

        if member_indices:
            # 提取成员的发文量和引用量
            member_publications = publications_tensor[torch.tensor(member_indices)]
            member_citations = citations_tensor[torch.tensor(member_indices)]

            # 计算总发文量和总引用量
            total_publications = member_publications.sum().item()
            total_citations = member_citations.sum().item()

            totals.append([total_publications, total_citations])

    # 返回包含总发文量和总引用量的张量，并保持二维形状以匹配模型输出
    return torch.tensor(totals)



def calculate_h_index_averages(teams, h_index_tensor, author_id_mapping):
    h_indices_averages = []

    for members in teams.values():  # 假设teams是社区ID到成员列表的映射
        # 将成员ID映射到对应的索引
        member_indices = [author_id_mapping[member] for member in members if member in author_id_mapping]

        if member_indices:
            # 提取成员的H指数
            member_h_indices = h_index_tensor[torch.tensor(member_indices)]

            # 计算平均H指数
            average_h_index = member_h_indices.float().mean().item()
            h_indices_averages.append(average_h_index)

    # 返回包含平均H指数的张量，并保持二维形状以匹配模型输出
    return torch.tensor(h_indices_averages).unsqueeze(1)
    # return team_h_indices_averages

# 假设h_index_tensor是从图中得到的，其中存储了每个作者的H指数
# teams是前面得到的团队数据，author_id_mapping是作者ID到图索引的映射
# team_h_indices_averages = calculate_h_index_averages(teams, g.nodes['author'].data['h_index'], graph.author_id_mapping)
# print(team_h_indices_averages)
# 输出团队ID和对应的平均H指数
# for community_id, avg_h_index in team_h_indices_averages.items():
#     print(f"Community {community_id}: Average H-index = {avg_h_index}")

def average_pooling(team_vectors):
    pooled_vectors = {}
    for community, vectors in team_vectors.items():
        # 对每个团队的成员向量取平均值
        pooled_vectors[community] = torch.mean(vectors, dim=0)
    team_names = list(pooled_vectors.keys())
    pooled_vectors = torch.stack([pooled_vectors[name] for name in team_names])
    return pooled_vectors

def concatenate_vectors(team_vectors, max_members):
    concatenated_vectors = {}
    for community, vectors in team_vectors.items():
        # 如果团队成员数量少于 max_members，用零向量填充
        if vectors.shape[0] < max_members:
            padding = torch.zeros(max_members - vectors.shape[0], vectors.shape[1])
            vectors = torch.cat((vectors, padding), dim=0)
        # 截取或保留前 max_members 个成员向量
        concatenated_vectors[community] = vectors[:max_members].view(-1)
    return concatenated_vectors


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# 定义LSTM模型
class LSTMTeamFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMTeamFusion, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)  # 将输出维度变回3维

    def forward(self, vectors, lengths):
        # 打包序列，根据长度处理变长序列
        packed_input = pack_padded_sequence(vectors, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # 取每个序列最后一个有效时间步的隐藏状态
        last_outputs = hidden[-1]  # 取最后一层的输出
        return self.fc(last_outputs)  # 通过全连接层变为3维输出


# 示例函数，将team_vectors通过LSTM融合
def lstm_pooling(team_vectors, input_dim, hidden_dim):
    # 初始化LSTM模型
    lstm_model = LSTMTeamFusion(input_dim, hidden_dim)

    pooled_vectors = []
    lengths = []  # 用于记录每个团队的长度

    for community, vectors in team_vectors.items():
        lengths.append(vectors.size(0))  # 记录该团队的成员数量
        pooled_vectors.append(vectors)

    # 对team_vectors按长度排序（LSTM要求）
    sorted_lengths, sorted_idx = torch.sort(torch.tensor(lengths), descending=True)
    sorted_vectors = [pooled_vectors[i] for i in sorted_idx]

    # 将每个团队的成员向量打包为一个大的tensor
    padded_vectors = nn.utils.rnn.pad_sequence(sorted_vectors, batch_first=True)

    # 通过LSTM池化
    lstm_output = lstm_model(padded_vectors, sorted_lengths)

    # 根据排序索引恢复原顺序
    _, original_idx = torch.sort(sorted_idx)
    lstm_output = lstm_output[original_idx]

    return lstm_output


