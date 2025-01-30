import torch
import dgl

from DBLP import graph


# 假设g是你的异构图，graph.author_id_mapping是一个字典，映射author_id到索引
def add_author_stats_to_graph(g):
    # 读取author_stats.txt文件
    author_stats = {}
    with open('data_process/author_stats.txt', 'r') as f:
        for line in f:
            author_id, publications, citations, h_index,num_collaborators,num_venues = line.strip().split()
            author_stats[author_id] = {
                'publications': int(publications),
                'citations': int(citations),
                'h_index': int(h_index)
            }

    # 创建tensor来存储发文量、引用量和h指数
    num_authors = g.num_nodes('author')
    publications_tensor = torch.zeros((num_authors, 1))
    citations_tensor = torch.zeros((num_authors, 1))
    h_index_tensor = torch.zeros((num_authors, 1))

    # 将数据存储到tensor中
    for author_str_id, idx in graph.author_id_mapping.items():
        if author_str_id in author_stats:
            publications_tensor[idx] = author_stats[author_str_id]['publications']
            citations_tensor[idx] = author_stats[author_str_id]['citations']
            h_index_tensor[idx] = author_stats[author_str_id]['h_index']

    # 将数据存储到DGL图中
    g.nodes['author'].data['publications'] = publications_tensor
    g.nodes['author'].data['citations'] = citations_tensor
    g.nodes['author'].data['h_index'] = h_index_tensor

    return g

# 假设graph是你的图对象，并且文件路径是'data_process/author_stats.txt'

