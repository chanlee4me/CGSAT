import numpy as np
import os
import sys
import torch # GRU 层需要 torch

# 调整路径以导入 gym_sat_Env
# 假设测试脚本位于 CGSAT/test/ 目录下
# CGSAT/minisat/minisat/gym/MiniSATEnv.py
# 从 CGSAT/test/ 到 CGSAT/ 是 '..'
# 然后从 CGSAT/ 到 minisat/minisat/gym/ 是 'minisat/minisat/gym'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from minisat.minisat.gym.MiniSATEnv import (
    gym_sat_Env,
    HANDCRAFTED_FEATURES_START_COL,
    NODE_TYPE_VAR,
    NODE_TYPE_CLAUSE,
    # NUM_HANDCRAFTED_VAR_FEATURES, # 这些在env内部使用，测试代码不需要直接导入
    # NUM_HANDCRAFTED_CLAUSE_FEATURES
)

# 用于参数的模拟类
class ArgsMock:
    def __init__(self):
        # 这个值主要用于初始化 MiniSATEnv 中的 action_space 和 observation_space
        # 对于 sample.cnf (3个变量)，设置为3是合适的
        self.nums_variable = 3 
        # 根据需要添加 MiniSATEnv 初始化或测试路径中严格要求的其他参数
        # 例如，如果模型参数被访问，可能需要添加：
        # self.n_hidden = 1
        # self.hidden_size = 64
        # ... 等

def test_initial_hidden_state():
    print("开始测试初始隐藏状态生成...")
    args = ArgsMock()
    
    # 指向包含 .cnf 文件的目录路径
    # 假设 'gqsat/test/' 目录相对于 CGSAT 根目录存在，并且包含 'sample.cnf'
    problems_dir = os.path.join(os.path.dirname(__file__), '..', 'gqsat', 'test')
    
    if not os.path.exists(problems_dir):
        print(f"错误: CNF 问题目录不存在: {problems_dir}")
        return
    sample_cnf_path = os.path.join(problems_dir, 'sample.cnf')
    if not os.path.exists(sample_cnf_path):
        print(f"错误: sample.cnf 文件不存在于: {sample_cnf_path}")
        # 为了使测试更健壮，如果 sample.cnf 不存在，可以尝试列出目录中的任何 .cnf 文件
        # 但为了简单起见，我们这里依赖 sample.cnf
        return

    # 为测试定义 GRU 输出维度
    test_var_gru_output_dim = 32
    test_clause_gru_output_dim = 24

    print(f"使用 CNF 目录: {problems_dir}")
    print(f"变量 GRU 输出维度: {test_var_gru_output_dim}, 子句 GRU 输出维度: {test_clause_gru_output_dim}")

    env = gym_sat_Env(
        problems_paths=problems_dir, # 指向包含 sample.cnf 的目录
        args=args,
        var_embedding_dim=16,    # 示例值
        clause_embedding_dim=16, # 示例值
        var_gru_output_dim=test_var_gru_output_dim,
        clause_gru_output_dim=test_clause_gru_output_dim
    )

    # 重置环境以加载问题并解析其状态
    # 这将调用 parse_state_as_graph()
    print("调用 env.reset()...")
    state = env.reset()
    vertex_data, edge_data, connectivity, global_features = state
    print("env.reset() 调用完成.")

    # --- 断言 ---

    # 1. 检查 vertex_in_size 是否正确配置
    # vertex_in_size 应该是 2 (类型, ID) + max(var_gru_output_dim, clause_gru_output_dim)
    expected_vertex_in_size = 2 + max(test_var_gru_output_dim, test_clause_gru_output_dim)
    assert env.vertex_in_size == expected_vertex_in_size, \
        f"预期的 vertex_in_size 为 {expected_vertex_in_size}, 实际为 {env.vertex_in_size}"
    print(f"Vertex_in_size 检查通过: {env.vertex_in_size}")

    # 2. 检查 vertex_data 的形状
    # 对于 sample.cnf (3个变量, 2个子句), 初始时所有变量都未赋值。
    # 图中的变量节点数应为3，子句节点数应为2。
    # 注意: GymSolver 内部的预处理（如单元传播）可能会改变活动子句/变量的数量。
    # 我们从实际生成的 vertex_data 中获取节点数量。
    
    num_graph_vars = np.sum(vertex_data[:, 0] == NODE_TYPE_VAR)
    num_graph_clauses = np.sum(vertex_data[:, 0] == NODE_TYPE_CLAUSE)
    actual_total_nodes = vertex_data.shape[0]

    # 确保解析的节点总数与变量和子句节点数之和一致
    assert actual_total_nodes == num_graph_vars + num_graph_clauses, \
        f"图中节点总数 ({actual_total_nodes}) 与变量 ({num_graph_vars}) 和子句 ({num_graph_clauses}) 节点数之和不匹配。"
    
    # 检查 sample.cnf 是否按预期加载 (3个变量，2个子句在初始图中)
    # 这是基于对 sample.cnf 的了解，并且假设没有变量在初始解析时被赋值消除
    # 如果 GymSolver 的初始化逻辑复杂，这个断言可能需要调整
    assert num_graph_vars == 3, f"对于 sample.cnf，预期有3个变量节点，实际有 {num_graph_vars}"
    assert num_graph_clauses == 2, f"对于 sample.cnf，预期有2个子句节点，实际有 {num_graph_clauses}"
    
    assert vertex_data.shape == (actual_total_nodes, expected_vertex_in_size), \
        f"预期的 vertex_data 形状为 ({actual_total_nodes}, {expected_vertex_in_size}), 实际为 {vertex_data.shape}"
    print(f"Vertex_data 形状检查通过: {vertex_data.shape}")

    # 3. 检查 GRU 输出部分是否已填充
    # 从 HANDCRAFTED_FEATURES_START_COL 开始的特征来自 GRU

    # 检查变量节点的 GRU 输出
    if num_graph_vars > 0:
        var_nodes_features = vertex_data[vertex_data[:, 0] == NODE_TYPE_VAR]
        var_gru_features_part = var_nodes_features[:, HANDCRAFTED_FEATURES_START_COL : HANDCRAFTED_FEATURES_START_COL + test_var_gru_output_dim]
        # 基本检查：假设手工特征和GRU权重/偏置不都为零，则输出不应全为零
        assert np.any(var_gru_features_part != 0), \
            "变量节点的 GRU 输出似乎全为零。请检查 GRU 层或手工特征。"
        print("变量节点 GRU 输出非全零检查通过。")

    # 检查子句节点的 GRU 输出
    if num_graph_clauses > 0:
        clause_nodes_features = vertex_data[vertex_data[:, 0] == NODE_TYPE_CLAUSE]
        clause_gru_features_part = clause_nodes_features[:, HANDCRAFTED_FEATURES_START_COL : HANDCRAFTED_FEATURES_START_COL + test_clause_gru_output_dim]
        # 基本检查
        assert np.any(clause_gru_features_part != 0), \
            "子句节点的 GRU 输出似乎全为零。请检查 GRU 层或手工特征。"
        print("子句节点 GRU 输出非全零检查通过。")

    # 4. 检查较短 GRU 输出的填充（如果维度不同）
    if test_var_gru_output_dim > test_clause_gru_output_dim:
        # 子句 GRU 输出较短，检查填充部分是否为零
        padding_start_col = HANDCRAFTED_FEATURES_START_COL + test_clause_gru_output_dim
        padding_end_col = HANDCRAFTED_FEATURES_START_COL + test_var_gru_output_dim # 等于 expected_vertex_in_size - 2
        if num_graph_clauses > 0:
            clause_nodes_all_features = vertex_data[vertex_data[:, 0] == NODE_TYPE_CLAUSE]
            padding_part = clause_nodes_all_features[:, padding_start_col:padding_end_col]
            assert np.all(padding_part == 0), \
                "较短的子句 GRU 输出的填充部分不全为零。"
            print("子句节点 GRU 输出填充检查通过。")
    elif test_clause_gru_output_dim > test_var_gru_output_dim:
        # 变量 GRU 输出较短，检查填充部分是否为零
        padding_start_col = HANDCRAFTED_FEATURES_START_COL + test_var_gru_output_dim
        padding_end_col = HANDCRAFTED_FEATURES_START_COL + test_clause_gru_output_dim # 等于 expected_vertex_in_size - 2
        if num_graph_vars > 0:
            var_nodes_all_features = vertex_data[vertex_data[:, 0] == NODE_TYPE_VAR]
            padding_part = var_nodes_all_features[:, padding_start_col:padding_end_col]
            assert np.all(padding_part == 0), \
                "较短的变量 GRU 输出的填充部分不全为零。"
            print("变量节点 GRU 输出填充检查通过。")
    else:
        print("变量和子句 GRU 输出维度相同，无需填充检查。")

    print("\n测试成功完成：初始隐藏状态生成符合预期。")
    if num_graph_vars > 0:
        print(f"变量节点 GRU 特征 (第一个节点的前5个特征): {var_gru_features_part[0, :5]}")
    if num_graph_clauses > 0:
        print(f"子句节点 GRU 特征 (第一个节点的前5个特征): {clause_gru_features_part[0, :5]}")

if __name__ == "__main__":
    test_initial_hidden_state()
