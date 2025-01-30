import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score
from DBLP.ModelRun import team_h_indices_averages
from DBLP.set_seed import set_seed

# 加载数据

# 固定随机数种子
set_seed(1)
pooled_team_vectors = torch.load('pooled_team_vectors.pth')
padded_vectors = pooled_team_vectors

print("Shape of padded vectors:", padded_vectors.shape)
print("Shape of H-index averages tensor:", team_h_indices_averages.shape)
# 将 team_h_indices_averages 分为三类
# 将 team_h_indices_averages 分为三类
quantiles = torch.quantile(team_h_indices_averages.float(), torch.tensor([0.33, 0.66]))

# 假设 team_h_indices_averages 是一维张量
team_h_indices_averages = team_h_indices_averages.view(-1)  # 确保是1D

# 创建一维标签张量
labels = torch.zeros(team_h_indices_averages.size(0), dtype=torch.long)

# 使用一维索引赋值
labels[team_h_indices_averages <= quantiles[0]] = 0  # 第一类
labels[(team_h_indices_averages > quantiles[0]) & (team_h_indices_averages <= quantiles[1])] = 1  # 第二类
labels[team_h_indices_averages > quantiles[1]] = 2  # 第三类

print("Shape of labels:", labels.shape)
print("Shape of labels:", labels)
# 确保两者数量一致
assert padded_vectors.size(0) == labels.size(0), "The number of teams and the number of labels must be the same."

# 设置模型、优化器和损失函数
input_dim = padded_vectors.shape[1]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = MLPClassifier(input_dim=input_dim, hidden_dim=64)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 分割数据为训练集和验证集
num_samples = padded_vectors.size(0)
indices = torch.randperm(num_samples)
train_indices = indices[:int(0.8 * num_samples)]
val_indices = indices[int(0.8 * num_samples):]

train_features = padded_vectors[train_indices]
train_targets = labels[train_indices]
val_features = padded_vectors[val_indices]
val_targets = labels[val_indices]
print("Shape of labels:", labels.shape)
print("Train targets shape:", train_targets.shape)
print("Val targets shape:", val_targets.shape)

# 检查目标张量的形状
assert train_targets.ndimension() == 1, "train_targets must be a 1D tensor"
assert val_targets.ndimension() == 1, "val_targets must be a 1D tensor"

from sklearn.metrics import precision_score, roc_auc_score

# 训练循环
for epoch in range(500):
    model.train()  # 切换到训练模式
    optimizer.zero_grad()  # 清空梯度

    # 前向传播
    train_outputs = model(train_features)
    train_loss = criterion(train_outputs, train_targets)  # 计算训练损失

    # 反向传播
    train_loss.backward(retain_graph=True)  # 仅执行一次反向传播
    optimizer.step()  # 更新参数

    # 评估模型
    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 在验证时禁用梯度计算
        val_outputs = model(val_features)  # 前向传播
        val_loss = criterion(val_outputs, val_targets)  # 计算验证损失

        # 获取预测的类别
        _, val_preds = torch.max(val_outputs, 1)

        # 计算评估指标
        accuracy = accuracy_score(val_targets.numpy(), val_preds.numpy())
        f1 = f1_score(val_targets.numpy(), val_preds.numpy(), average='macro')
        recall = recall_score(val_targets.numpy(), val_preds.numpy(), average='macro')
        precision = precision_score(val_targets.numpy(), val_preds.numpy(), average='macro', zero_division=1)

        # 对 ROC AUC 进行处理，需传入概率
        val_probs = F.softmax(val_outputs, dim=1).numpy()  # 计算每类的概率
        roc_auc = roc_auc_score(val_targets.numpy(), val_probs, multi_class='ovr')

    # 打印训练和验证信息
    print(f"Epoch {epoch}:")
    print(f"  Train Loss: {train_loss.item()}")
    print(f"  Val Loss: {val_loss.item()}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")

    # 指定输出预测值和真实值的轮次
    if epoch in [0, 49, 99]:
        print(f"  Predictions: {val_preds.numpy()}")
        print(f"  Actuals: {val_targets.numpy()}")

