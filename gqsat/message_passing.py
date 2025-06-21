# Rewritten by Cascade to fix persistent IndentationError.
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
        # 1. 准备工作: 分离节点和创建索引映射
        node_types = x[:, NODE_TYPE_COL]
        var_mask = node_types == NODE_TYPE_VAR
        clause_mask = node_types == NODE_TYPE_CLAUSE

        # 为变量和子句节点创建从全局索引到局部索引的映射
        num_vars = var_mask.sum()
        num_clauses = clause_mask.sum()
        var_map = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
        clause_map = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
        var_map[var_mask] = torch.arange(num_vars, device=x.device)
        clause_map[clause_mask] = torch.arange(num_clauses, device=x.device)

        # 2. 严格分离和重映射边
        # 通过同时检查源和目标节点的类型来确保边的正确性
        src_types = node_types[edge_index[0]]
        dst_types = node_types[edge_index[1]]
        
        v2c_edge_mask = (src_types == NODE_TYPE_VAR) & (dst_types == NODE_TYPE_CLAUSE)
        c2v_edge_mask = (src_types == NODE_TYPE_CLAUSE) & (dst_types == NODE_TYPE_VAR)

        v2c_edges = edge_index[:, v2c_edge_mask]
        c2v_edges = edge_index[:, c2v_edge_mask]

        v2c_edges_remapped = torch.stack([
            var_map[v2c_edges[0]],
            clause_map[v2c_edges[1]]
        ])
        c2v_edges_remapped = torch.stack([
            clause_map[c2v_edges[0]],
            var_map[c2v_edges[1]]
        ])

        # 3. 分离特征和隐藏状态
        x_vars = x[var_mask]
        x_clauses = x[clause_mask]
        h_old = x[:, -self.out_channels:]
        h_v_old = h_old[var_mask]
        h_c_old = h_old[clause_mask]

        # 4. V2C 消息传递 (变量 -> 子句)
        m_c = self.v2c_conv((x_vars, x_clauses), v2c_edges_remapped)
        m_c = self.dropout(self.c_norm(m_c))
        h_c_new = self.c_update(m_c, h_c_old)

        # 5. C2V 消息传递 (子句 -> 变量)
        m_v = self.c2v_conv((h_c_new, x_vars), c2v_edges_remapped)
        m_v = self.dropout(self.v_norm(m_v))
        h_v_new = self.v_update(m_v, h_v_old)

        # 6. 组合最终结果
        x_new = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=h_v_new.dtype)
        x_new[var_mask] = h_v_new
        x_new[clause_mask] = h_c_new

        return x_new
