import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score, f1_score, accuracy_score
import torch.nn as nn
import random
import torch
import numpy as np
import torch.nn.functional as F

from DBLP.MentorModel_subgraph import subgraphs
from DBLP.set_seed import set_seed
set_seed(1)

class PaperDNNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(PaperDNNModel, self).__init__()

        # 定义两层全连接网络
        self.fc1 = nn.Linear(in_feats, hidden_size)  # 第一隐藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # 第二隐藏层
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)  # 输出层

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 第一隐藏层，使用 ReLU 激活函数
        x = self.fc1(x)
        x = self.relu(x)

        # 第二隐藏层，使用 Tanh 激活函数
        x = self.fc2(x)
        x = self.tanh(x)

        # 输出层，使用 Sigmoid 激活函数
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = torch.mean(x, dim=0, keepdim=True)  # 这一步将12*3变成1*3
        return x
# 使用按论文中的模型
model = PaperDNNModel(in_feats=4, hidden_size=64, num_classes=3)  # 使用论文中的模型

criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 使用 Adam 优化器

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

# print("Subgraph Labels:", subgraph_labels)

# 设置训练轮数
num_epochs = 10

# Step 3: 训练过程
for epoch in range(num_epochs):
    total_loss = 0.0
    all_logits = []
    all_labels = []

    # 遍历每个子图进行训练
    for i, subg in enumerate(subgraphs):
        # 计算 anchor distances

        # 将模型置于训练模式
        model.train()

        # Forward pass: 计算模型的 logits
        logits = model(subg.ndata['feat'])
        # print("logits：",logits)
        # 获取子图的真实标签
        label = subgraph_labels[i].unsqueeze(0).long()  # 标签需要是长整型，并增加维度
        # print("label：", label)
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
    avg_loss = total_loss / len(subgraphs)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Step 4: 在训练结束后进行评估
model.eval()  # 进入评估模式
final_logits = torch.cat(all_logits, dim=0)
final_labels = torch.cat(all_labels, dim=0)

# 计算预测类别
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# 计算预测类别
predicted_classes = torch.argmax(torch.abs(final_logits), dim=1)

# 打印调试信息
print("Final Labels:", final_labels)
print("Logits (Predicted Classes):", final_logits)
print("Predicted Classes:", predicted_classes)

# Step 5: 计算模型的准确率、Recall、F1-score、Precision 和 ROC AUC
accuracy = accuracy_score(final_labels.cpu(), predicted_classes.cpu())
recall = recall_score(final_labels.cpu(), predicted_classes.cpu(), average='macro')  # 使用宏平均
f1 = f1_score(final_labels.cpu(), predicted_classes.cpu(), average='macro')
precision = precision_score(final_labels.cpu(), predicted_classes.cpu(), average='macro')  # 精确率

# ROC AUC 计算需要概率分布，因此将 logits 转换为 softmax 概率
probabilities = torch.softmax(final_logits, dim=1).detach().cpu().numpy()

# 对标签进行 one-hot 编码以适配 ROC AUC 的多分类计算
final_labels_onehot = label_binarize(final_labels.cpu().numpy(), classes=list(range(probabilities.shape[1])))
roc_auc = roc_auc_score(final_labels_onehot, probabilities, average='macro', multi_class='ovr')  # 使用宏平均的 OVR 方法

# 打印所有指标
print(f"Final accuracy after training: {accuracy * 100:.2f}%")
print(f"Final recall after training: {recall * 100:.2f}%")
print(f"Final F1-score after training: {f1 * 100:.2f}%")
print(f"Final precision after training: {precision * 100:.2f}%")
print(f"Final ROC AUC after training: {roc_auc * 100:.2f}%")



