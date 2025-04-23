import sys
sys.path.append('../')  # 将上一级目录添加到系统路径中
from models import GraphNet  # 你可以调整为你的实际文件名
import torch
from torch_geometric.data import Data
# 创建示例数据
x = torch.randn(4, 8)  # 4 nodes, 每个节点8维特征
edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])  # 边索引，共4条边
u = torch.randn(1, 16)  # 全局特征，1个全局向量，16维

# 创建模型
model = GraphNet(
    in_dims=(8, 4, 16),
    out_dims=(8, 4, 16),
    hidden_size=64,
    n_hidden=2
)

# 前向传播
new_x, new_edge_attr, new_u = model(x, edge_index, u=u)

# 打印输出的张量形状
print(new_x.shape)      # 应输出 torch.Size([4, 8])
print(new_edge_attr.shape)  # 应输出 torch.Size([4, 4])
print(new_u.shape)      # 应输出 torch.Size([1, 16])