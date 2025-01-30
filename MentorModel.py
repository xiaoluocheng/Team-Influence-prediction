import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import networkx as nx
import dgl.function as fn

from DBLP.MentorModel_subgraph import subgraphs
from DBLP.set_seed import set_seed


class PGNNLayer(nn.Module):
    def __init__(self, in_feats, hidden_size, num_anchors):
        super(PGNNLayer, self).__init__()
        # Adjust the input dimension to match the concatenated feature dimension
        self.fc = nn.Linear(in_feats + num_anchors, hidden_size)

    def forward(self, g, features, anchor_distances):
        # Concatenate features with anchor distances
        combined_features = torch.cat([features, anchor_distances], dim=1)  # Combined features will have in_feats + num_anchors dimensions
        # print(f"Combined anchor_distances features shape: {anchor_distances.shape}")
        # print(f"Combined features shape: {combined_features.shape}")
        h = F.relu(self.fc(combined_features))
        g.ndata['h'] = h
        g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
        return g.ndata['h']


class MentorModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, num_anchors):
        super(MentorModel, self).__init__()

        # 通道模块
        self.gat1 = dglnn.GATConv(in_feats, hidden_size, num_heads=1)
        self.gin1 = dglnn.GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(hidden_size, hidden_size)))
        self.gin2 = dglnn.GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(hidden_size, hidden_size)))
        self.gin3 = dglnn.GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(hidden_size, hidden_size)))
        self.centrality_gin1 = dglnn.GINConv(nn.Sequential(nn.Linear(in_feats, hidden_size),
                                                           nn.ReLU(),
                                                           nn.Linear(hidden_size, hidden_size)))
        self.centrality_gin2 = dglnn.GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                           nn.ReLU(),
                                                           nn.Linear(hidden_size, hidden_size)))
        self.centrality_gin3 = dglnn.GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                           nn.ReLU(),
                                                           nn.Linear(hidden_size, hidden_size)))
        self.pgnn = PGNNLayer(in_feats, hidden_size, num_anchors)

        # 通道融合
        self.fc_combined = nn.Linear(3 * hidden_size, hidden_size)

        # 最终分类器
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, g, features, anchor_distances):
        # Topology Channel
        h = F.relu(self.gat1(g, features))
        h = h.mean(1)  # 如果 GAT 有多个 heads
        h = F.relu(self.gin1(g, h))
        h = F.relu(self.gin2(g, h))
        topo_out = self.gin3(g, h)

        # Centrality Channel
        h_c = F.relu(self.centrality_gin1(g, features))
        h_c = F.relu(self.centrality_gin2(g, h_c))
        cent_out = self.centrality_gin3(g, h_c)

        # Contextual Channel (P-GNN)
        ctx_out = self.pgnn(g, features, anchor_distances)
        ctx_out = torch.where(torch.isnan(ctx_out), torch.tensor(0.0), ctx_out)
        ctx_out = torch.where(torch.isinf(ctx_out), torch.tensor(0.0), ctx_out)

        # Channel Aggregation
        combined = torch.cat([topo_out, cent_out, ctx_out], dim=1)
        combined = F.relu(self.fc_combined(combined))
        g.ndata['combined'] = combined
        combined = dgl.mean_nodes(g, 'combined')

        # Final classification
        logits = self.fc(combined)
        return logits


def calculate_anchor_distances(g, num_anchors):
    G = g.to_networkx()
    anchors = torch.randperm(g.number_of_nodes())[:num_anchors].tolist()
    distances = []

    for node in G.nodes():
        node_distances = []
        for anchor in anchors:
            try:
                distance = nx.shortest_path_length(G, source=node, target=anchor)
            except nx.NetworkXNoPath:
                distance = float('inf')
            node_distances.append(distance)
        distances.append(node_distances)

    distances = torch.tensor(distances, dtype=torch.float32)
    return distances


# 初始化一个列表，用于收集所有子图的 logits
# all_logits = []

# for i, subg in enumerate(subgraphs):
#     # 计算 anchor distances (根据你的计算逻辑)
#     anchor_distances = calculate_anchor_distances(subg, num_anchors=1)  # Example anchor distance
#     # 初始化模型
#     model = MentorModel(in_feats=4, hidden_size=64, num_classes=3, num_anchors=1)
#     # 对每个子图进行前向传播，计算 logits
#     logits = model(subg, subg.ndata['feat'], anchor_distances)
#     # 将每个子图的 logits 结果添加到 all_logits 列表中
#     all_logits.append(logits)
#
# # 将所有 logits 合并为一个张量
# # 如果 logits 的维度是 [num_nodes, num_classes]，可以在第一个维度上进行拼接
# final_logits = torch.cat(all_logits, dim=0)
#
# # 输出整体 logits
# print(final_logits)


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
set_seed(1)
# 初始化模型、损失函数和优化器
model = MentorModel(in_feats=4, hidden_size=64, num_classes=3, num_anchors=1)
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005 )  # 使用 Adam 优化器

# 假设模型输出的 logits 是 [num_nodes, num_classes]，取最大值作为预测类别
subgraph_aveh = []

# Step 1: 提取所有子图的 aveh，并存储到 subgraph_aveh 中
for subg in subgraphs:
    # 提取每个子图的平均 hindex（aveh），假设所有节点的 aveh 是一样的，取第一个节点即可
    subgraph_aveh.append(subg.ndata['aveh'][0].item())

# Step 2: 对 aveh 值进行排序并划分为三类（标签仍与原子图相对应）
aveh_tensor = torch.tensor(subgraph_aveh)

# 计算每个区间的大小
num_subgraphs = len(aveh_tensor)
split_size = num_subgraphs // 3  # 将子图数量均分为三部分

# 设置阈值，将 aveh 分为三类
sorted_aveh, _ = torch.sort(aveh_tensor)
low_threshold = sorted_aveh[split_size - 1]  # 低区间的最大值
high_threshold = sorted_aveh[2 * split_size - 1]  # 中区间的最大值，高区间的最小值就是这个

# 根据 aveh 值的阈值划分为高、中、低（直接赋予原子图顺序）
subgraph_labels = torch.zeros_like(aveh_tensor)  # 初始化所有子图的分类标签
subgraph_labels[aveh_tensor > high_threshold] = 2  # 高区间为2
subgraph_labels[(aveh_tensor > low_threshold) & (aveh_tensor <= high_threshold)] = 1  # 中区间为1
subgraph_labels[aveh_tensor <= low_threshold] = 0  # 低区间为0

# Step 3: 划分训练集和测试集
train_idx, test_idx = train_test_split(range(num_subgraphs), test_size=0.2, random_state=42)

# 设置训练轮数
num_epochs = 50

# Step 4: 训练过程
for epoch in range(num_epochs):
    total_loss = 0.0
    all_logits = []
    all_labels = []

    # 遍历训练集中的每个子图进行训练
    for i in train_idx:
        subg = subgraphs[i]
        # 计算 anchor distances
        anchor_distances = calculate_anchor_distances(subg, num_anchors=1)

        # 将模型置于训练模式
        model.train()

        # Forward pass: 计算模型的 logits
        logits = model(subg, subg.ndata['feat'], anchor_distances)

        # 获取子图的真实标签
        label = subgraph_labels[i].unsqueeze(0).long()  # 标签需要是长整型，并增加维度

        # 计算损失
        loss = criterion(logits, label)
        total_loss += loss.item()

        # Backward pass: 计算梯度并优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 保存 logits 和真实标签
        all_logits.append(logits)
        all_labels.append(label)

    # 打印每轮的平均损失
    avg_loss = total_loss / len(train_idx)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Step 5: 在测试集上进行评估
model.eval()  # 进入评估模式
all_test_logits = []
all_test_labels = []

for i in test_idx:
    subg = subgraphs[i]

    # 计算 anchor distances
    anchor_distances = calculate_anchor_distances(subg, num_anchors=1)

    # Forward pass: 计算模型的 logits
    logits = model(subg, subg.ndata['feat'], anchor_distances)

    # 获取子图的真实标签
    label = subgraph_labels[i].unsqueeze(0).long()

    # 保存 logits 和真实标签
    all_test_logits.append(logits)
    all_test_labels.append(label)

import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score

# 拼接所有测试集的 logits 和 labels
final_logits = torch.cat(all_test_logits, dim=0)
final_labels = torch.cat(all_test_labels, dim=0)

# 计算预测类别
predicted_classes = torch.argmax(final_logits, dim=1)

# 转换 logits 为概率分数
predicted_probs = F.softmax(final_logits, dim=1)

# 确保去除梯度影响并转换为 NumPy
predicted_probs = predicted_probs.detach().cpu().numpy()
final_labels_np = final_labels.cpu().numpy()

# 计算各项指标
accuracy = accuracy_score(final_labels_np, predicted_classes.cpu().numpy())
recall = recall_score(final_labels_np, predicted_classes.cpu().numpy(), average='macro')
f1 = f1_score(final_labels_np, predicted_classes.cpu().numpy(), average='macro')
precision = precision_score(final_labels_np, predicted_classes.cpu().numpy(), average='macro', zero_division=0)

# 计算 ROC AUC，适用于多分类任务
roc_auc = roc_auc_score(final_labels_np, predicted_probs, multi_class='ovr', average='macro')

# 打印结果
print(f"Final accuracy on test set: {accuracy * 100:.2f}%")
print(f"Final recall on test set: {recall * 100:.2f}%")
print(f"Final F1-score on test set: {f1 * 100:.2f}%")
print(f"Final precision on test set: {precision * 100:.2f}%")
print(f"Final ROC AUC on test set: {roc_auc * 100:.2f}%")


