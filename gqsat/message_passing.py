import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

# 定义节点类型常量，与 MiniSATEnv.py 中保持一致
NODE_TYPE_VAR = 1
NODE_TYPE_CLAUSE = 2
NODE_TYPE_COL = 0

class SATMessagePassing(nn.Module):
    """
    实现单轮V2C和C2V消息传递。
    遵循设计：注意力聚合 -> LayerNorm -> Dropout -> GRU残差更新。
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        # V2C (变量到子句) 消息传递层
        self.v2c_conv = GATv2Conv(
            in_channels, out_channels, heads=heads, concat=True, 
            negative_slope=negative_slope, dropout=dropout, add_self_loops=False
        )
        self.c_norm = nn.LayerNorm(out_channels * heads)
        self.c_update = nn.GRUCell(out_channels * heads, out_channels)

        # C2V (子句到变量) 消息传递层
        self.c2v_conv = GATv2Conv(
            (out_channels, in_channels), out_channels, heads=heads, concat=True,
            negative_slope=negative_slope, dropout=dropout, add_self_loops=False
        )
        self.v_norm = nn.LayerNorm(out_channels * heads)
        self.v_update = nn.GRUCell(out_channels * heads, out_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): 节点特征张量，形状 [num_nodes, in_channels]。
                        其中包含了初始节点特征和上一轮的隐藏状态。
            edge_index (LongTensor): 边索引张量，形状 [2, num_edges]。

        Returns:
            Tensor: 更新后的节点隐藏状态，形状 [num_nodes, out_channels]。
        """
        # 1. 准备工作：分离变量和子句的旧隐藏状态
        # 假设 x 的后 out_channels 维是上一轮的隐藏状态
        h_old = x[:, -self.out_channels:]
        
        var_mask = x[:, NODE_TYPE_COL] == NODE_TYPE_VAR
        clause_mask = x[:, NODE_TYPE_COL] == NODE_TYPE_CLAUSE
        num_vars = var_mask.sum().item()

        h_v_old = h_old[var_mask]
        h_c_old = h_old[clause_mask]

        # 2. V2C 消息传递
        # GATv2Conv 支持二部图，传入 (source_features, target_features) 元组
        # 这里，变量是源，子句是目标
        # 我们需要调整边索引以适应分离的节点集
        v2c_edge_index = edge_index[:, edge_index[0] >= num_vars]
        v2c_edge_index[0] -= num_vars # 将子句索引映射到 [0, num_clauses-1]

        m_c = self.v2c_conv((x[var_mask], x[clause_mask]), v2c_edge_index)
        m_c = self.dropout(self.c_norm(m_c))
        h_c_new = self.c_update(m_c, h_c_old)
        
        # 3. C2V 消息传递
        # 这里，子句是源，变量是目标
        # 使用更新后的子句隐藏状态 h_c_new 作为源特征
        c2v_edge_index = edge_index[:, edge_index[0] < num_vars]
        c2v_edge_index[1] -= num_vars # 将子句索引映射到 [0, num_clauses-1]

        m_v = self.c2v_conv((h_c_new, x[var_mask]), c2v_edge_index)
        m_v = self.dropout(self.v_norm(m_v))
        h_v_new = self.v_update(m_v, h_v_old)

        # 4. 组合最终结果
        x_new = torch.zeros(x.size(0), self.out_channels, device=x.device)
        x_new[var_mask] = h_v_new
        x_new[clause_mask] = h_c_new

        return x_new
