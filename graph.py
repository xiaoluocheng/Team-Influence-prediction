import pandas as pd
import torch
import dgl

# 文件路径
citations_file = 'data_process/paper_paper.txt'
authors_file = 'data_process/paper_author.txt'
author_to_author_file = 'data_process/author_author.txt'
venue_to_author_file = 'data_process/venue_author.txt'

# 读取数据集
df_citations = pd.read_csv(citations_file, delim_whitespace=True, header=None, names=['src', 'dst'])
df_authors = pd.read_csv(authors_file, delim_whitespace=True, header=None, names=['src', 'dst'])
df_author_to_author = pd.read_csv(author_to_author_file, delim_whitespace=True, header=None, names=['src',  'dst'])
df_venue_to_author = pd.read_csv(venue_to_author_file, delim_whitespace=True, header=None, names=['src', 'dst'])

# 去除重复边
df_citations.drop_duplicates(inplace=True)
df_authors.drop_duplicates(inplace=True)
df_author_to_author.drop_duplicates(inplace=True)
df_venue_to_author.drop_duplicates(inplace=True)

# 为每种类型的节点创建映射
all_papers = pd.concat([df_citations['src'], df_citations['dst'], df_authors['src']])
all_authors = pd.concat([df_authors['dst'], df_author_to_author['src'], df_author_to_author['dst'], df_venue_to_author['dst']])
all_venues = df_venue_to_author['src']
paper_id_mapping = {pid: i for i, pid in enumerate(all_papers.unique())}
author_id_mapping = {aid: i for i, aid in enumerate(all_authors.unique())}
venue_id_mapping = {vid: i for i, vid in enumerate(all_venues.unique())}

def create_heterograph():
    # 应用映射
    df_citations['src'] = df_citations['src'].map(paper_id_mapping)
    df_citations['dst'] = df_citations['dst'].map(paper_id_mapping)
    df_authors['src'] = df_authors['src'].map(paper_id_mapping)
    df_authors['dst'] = df_authors['dst'].map(author_id_mapping)
    df_author_to_author['src'] = df_author_to_author['src'].map(author_id_mapping)
    df_author_to_author['dst'] = df_author_to_author['dst'].map(author_id_mapping)
    df_venue_to_author['src'] = df_venue_to_author['src'].map(venue_id_mapping)
    df_venue_to_author['dst'] = df_venue_to_author['dst'].map(author_id_mapping)

    # 定义边连接并创建异构图，包含反向边
    data_dict = {
        ('paper', 'cites', 'paper'): (torch.LongTensor(df_citations['src']), torch.LongTensor(df_citations['dst'])),
        ('paper', 'written_by', 'author'): (torch.LongTensor(df_authors['src']), torch.LongTensor(df_authors['dst'])),
        ('author', 'writes', 'paper'): (torch.LongTensor(df_authors['dst']), torch.LongTensor(df_authors['src'])),
        ('venue', 'hosts_author', 'author'): (torch.LongTensor(df_venue_to_author['src']), torch.LongTensor(df_venue_to_author['dst'])),
        ('author', 'published_at', 'venue'): (torch.LongTensor(df_venue_to_author['dst']), torch.LongTensor(df_venue_to_author['src'])),
        # ('author', 'collaborates_with', 'author'): (torch.LongTensor(df_author_to_author['dst']), torch.LongTensor(df_author_to_author['src'])),
        # ('paper', 'cites', 'paper'): (torch.LongTensor(df_citations['dst']), torch.LongTensor(df_citations['src'])),
        ('author', 'collaborates_with', 'author'): (torch.LongTensor(df_author_to_author['src']), torch.LongTensor(df_author_to_author['dst']))
    }

    # 创建异构图
    g = dgl.heterograph(data_dict, num_nodes_dict={'paper': len(paper_id_mapping), 'author': len(author_id_mapping), 'venue': len(venue_id_mapping)})
    return g

def get_author_id_mapping():
    return author_id_mapping

# # 创建异构图
# g = create_heterograph()
# # 显示图的信息
# print(g)
# def save_edge_relationships_to_file(hg, filename):
#     with open(filename, 'w') as f:
#         for canonical_etype in hg.canonical_etypes:
#             src_type, etype, dst_type = canonical_etype
#             u, v = hg.edges(etype=etype)
#             f.write(f"Edge type ('{src_type}', '{etype}', '{dst_type}'):\n")
#             for src, dst in zip(u.numpy(), v.numpy()):
#                 f.write(f"  {src_type} {src} -- {etype} --> {dst_type} {dst}\n")
#
# # 保存边关系到文件
# filename = 'edge_relationships.txt'
# save_edge_relationships_to_file(g, filename)
# print(f"Edge relationships saved to {filename}")