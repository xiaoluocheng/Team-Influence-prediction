import random

import dgl
import torch
import numpy as np
def set_seed(seed):
    import os
    torch.manual_seed(seed)  # PyTorch随机种子
    torch.cuda.manual_seed_all(seed)  # 所有GPU的随机种子
    np.random.seed(seed)  # Numpy随机种子
    random.seed(seed)  # Python随机种子
    dgl.random.seed(seed)  # DGL随机种子
    torch.backends.cudnn.deterministic = True  # 确保卷积操作是确定的
    torch.backends.cudnn.benchmark = False  # 关闭自动优化
    os.environ['PYTHONHASHSEED'] = str(seed)  # 控制Python内部的哈希随机性
