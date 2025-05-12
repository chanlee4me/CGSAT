# 实施图顶点特征修改的方案

本文档将指导您如何在您的 GraphSat 项目中实现对变量节点和子句节点特征的修改，以增强图神经网络对CNF公式的表征能力。

## 1. 核心任务与目标

修改变量节点和子句节点的特征向量定义，具体如下：

*   **变量节点特征（5维）**：
    1.  `pos_lit_degree`：变量的正文字出现频率。
    2.  `neg_lit_degree`：变量的负文字出现频率。
    3.  `lit_pos_neg_ratio`：(正文字出现次数) / (负文字出现次数 + 1)。
    4.  `horn_occurrence`：变量出现在 Horn 子句的次数 / 总子句数。
    5.  `clause_size_sum_inv`：所有含该变量的子句长度倒数之和。
*   **子句节点特征（15维 = 5标量 + 10向量）**：
    1.  `clause_degree`：子句长度 / 总变量数。
    2.  `is_binary`：是否为二元子句 (0/1)。
    3.  `is_ternary`：是否为三元子句 (0/1)。
    4.  `is_horn`：是否为 Horn 子句 (0/1)。
    5.  `clause_pos_neg_ratio`：子句内 (正文字数) / (负文字数 + 1)。
    6.  `clause_pe[0..9]`：10维位置编码，表示文字在子句内的相对顺序。

## 2. 准备工作与辅助函数

### 2.1. 定位相关代码文件
*   **特征提取逻辑**: 大概率在 `gqsat/utils.py` 或数据加载/预处理部分 (例如 `train.py` 或类似脚本)。
*   **GNN 模型定义**: 通常在 `gqsat/models.py`。
*   **CNF 数据结构**: 了解项目中如何表示CNF公式（变量列表、子句列表、文字表示等）。

### 2.2. 实现 `is_horn_clause` 辅助函数
在您的工具模块 (例如 `gqsat/utils.py`) 中添加此函数：

```python
# 示例: gqsat/utils.py

def is_horn_clause(clause_literals: list[int]) -> bool:
    """
    判断一个子句是否为 Horn 子句。
    Horn 子句是最多只有一个正文字的子句。
    Args:
        clause_literals: 子句中的文字列表 (例如 [1, -2, -3]，其中正数表示正文字，负数表示负文字的绝对值取反)。
    Returns:
        True 如果是 Horn 子句, 否则 False。
    """
    positive_literals_count = sum(1 for lit in clause_literals if lit > 0)
    return positive_literals_count <= 1
```

## 3. 修改特征提取逻辑

您需要修改或创建一个函数，该函数接收CNF公式的表示，并为每个变量和子句计算新的特征。

### 3.1. 变量节点特征提取 (输出5维向量)

对于CNF公式中的每个变量 `var_id` (假设从1到 `num_total_variables`)：

1.  **初始化**: `pos_lit_degree = 0`, `neg_lit_degree = 0`, `horn_clause_appearances = 0`, `sum_inverse_clause_lengths = 0.0`
2.  **获取全局信息**: `num_total_clauses` (公式中的总子句数), `all_clauses` (所有子句的列表，每个子句是其文字的列表)。
3.  **遍历所有子句 `clause` in `all_clauses`**：
    *   `literals_in_clause = clause.get_literals()` (根据您的数据结构调整)
    *   `clause_length = len(literals_in_clause)`
    *   **`pos_lit_degree` / `neg_lit_degree`**：
        *   If `var_id` in `literals_in_clause`: `pos_lit_degree += 1`
        *   If `-var_id` in `literals_in_clause`: `neg_lit_degree += 1`
    *   **`horn_occurrence`**：
        *   If `is_horn_clause(literals_in_clause)`:
            *   If `var_id` in `literals_in_clause` or `-var_id` in `literals_in_clause`:
                `horn_clause_appearances += 1`
    *   **`clause_size_sum_inv`**：
        *   If `var_id` in `literals_in_clause` or `-var_id` in `literals_in_clause`:
            *   If `clause_length > 0`: `sum_inverse_clause_lengths += 1.0 / clause_length`
4.  **计算最终特征**：
    *   `feat_pos_lit_degree = pos_lit_degree`
    *   `feat_neg_lit_degree = neg_lit_degree`
    *   `feat_lit_pos_neg_ratio = pos_lit_degree / (neg_lit_degree + 1.0)`
    *   `feat_horn_occurrence = horn_clause_appearances / num_total_clauses if num_total_clauses > 0 else 0.0`
    *   `feat_clause_size_sum_inv = sum_inverse_clause_lengths`
5.  **组合特征向量**: `[feat_pos_lit_degree, feat_neg_lit_degree, feat_lit_pos_neg_ratio, feat_horn_occurrence, feat_clause_size_sum_inv]`

### 3.2. 子句节点特征提取 (输出15维向量)

对于CNF公式中的每个子句 `clause_idx` (其文字列表为 `current_clause_literals`)：

1.  **获取全局/局部信息**: `num_total_variables`, `clause_length = len(current_clause_literals)`.
2.  **计算标量特征 (5维)**：
    *   `feat_clause_degree = clause_length / num_total_variables if num_total_variables > 0 else 0.0`
    *   `feat_is_binary = 1.0 if clause_length == 2 else 0.0`
    *   `feat_is_ternary = 1.0 if clause_length == 3 else 0.0`
    *   `feat_is_horn = 1.0 if is_horn_clause(current_clause_literals) else 0.0`
    *   `num_pos_lits = sum(1 for lit in current_clause_literals if lit > 0)`
    *   `num_neg_lits = sum(1 for lit in current_clause_literals if lit < 0)`
    *   `feat_clause_pos_neg_ratio = num_pos_lits / (num_neg_lits + 1.0)`
3.  **计算位置编码 `clause_pe` (10维)**：
    *   这部分推荐使用 `torch.nn.Embedding` (如果使用PyTorch)。
    *   **定义**: `MAX_LITERALS_IN_CLAUSE = K` (例如50, 根据数据集调整), `PE_DIM = 10`.
      `literal_pos_embedding = torch.nn.Embedding(K, PE_DIM)`
    *   **对于当前子句**：
        *   `positions = torch.arange(min(clause_length, K))`
        *   If `clause_length == 0`: `pe_vector = torch.zeros(PE_DIM)`
        *   Else: `embeddings = literal_pos_embedding(positions)`
              `pe_vector = torch.mean(embeddings, dim=0)` (或 `torch.sum`)
    *   `feat_clause_pe = pe_vector` (这是一个10维张量/向量)
4.  **组合特征向量**: 将5个标量特征与10维 `feat_clause_pe` 拼接起来。
    `[feat_clause_degree, feat_is_binary, feat_is_ternary, feat_is_horn, feat_clause_pos_neg_ratio] + list(feat_clause_pe.numpy())` (如果需要转为list of floats)

## 4. 更新 GNN 模型输入维度

在 `gqsat/models.py` (或等效文件) 中，修改GNN模型定义：

*   **变量节点处理层**: 输入维度从旧值改为 `5`。
*   **子句节点处理层**: 输入维度从旧值改为 `15`。

```python
# 伪代码示例 (PyTorch) in gqsat/models.py
import torch
import torch.nn as nn

class GraphSatModel(nn.Module):
    def __init__(self, hidden_dim, num_rounds, ...): # 其他参数
        super().__init__()
        # ... (其他层)

        # 假设之前有类似这样的层
        # self.var_input_mlp = nn.Linear(OLD_VAR_FEATURE_DIM, hidden_dim)
        # self.clause_input_mlp = nn.Linear(OLD_CLAUSE_FEATURE_DIM, hidden_dim)

        # 更新为新的维度
        self.var_input_mlp = nn.Linear(5, hidden_dim)
        self.clause_input_mlp = nn.Linear(15, hidden_dim)

        # 如果 clause_pe 的 Embedding 层在这里定义 (另一种方式)
        # self.MAX_LITERALS_IN_CLAUSE = 50 # 保持与特征提取一致
        # self.PE_DIM = 10
        # self.literal_pos_embedding = nn.Embedding(self.MAX_LITERALS_IN_CLAUSE, self.PE_DIM)
        # 在这种情况下，子句输入特征可能是 (5标量 + K*PE_DIM) 然后由模型处理，或者特征提取已完成PE聚合

        # ... (GNN消息传递层等)

    def forward(self, graph_data):
        # graph_data.var_features: [num_vars, 5]
        # graph_data.clause_features: [num_clauses, 15]

        var_node_embeddings = self.var_input_mlp(graph_data.var_features)
        clause_node_embeddings = self.clause_input_mlp(graph_data.clause_features)

        # ... (后续GNN传播)
        return ...
```

## 5. 集成与测试

1.  **数据加载与预处理**: 确保您的数据加载和预处理流程调用新的特征提取逻辑，并将生成的特征正确地组织到图数据结构中，供GNN模型使用。
2.  **维度检查**: 在运行前，仔细检查所有相关张量的维度，确保它们在整个模型中正确传递。
3.  **单元测试**: 为特征提取函数（特别是 `is_horn_clause` 和各个特征计算部分）编写单元测试。
4.  **端到端测试**: 使用一个小型的CNF实例运行整个流程（数据加载、特征提取、模型前向传播），以捕获集成错误。
5.  **性能评估**: 在修改完成后，重新训练模型并评估其在基准数据集上的性能，以验证新特征的有效性。

通过遵循这些步骤，您应该能够成功地在您的项目中实现所需的顶点特征修改。

<!-- 更新时间: 2025-05-12 -->
## 方案调整：所有特征均基于原始CNF静态计算

根据您的最新反馈，我们现在将所有为变量节点和子句节点设计的特征都视为**静态特征**。这意味着这些特征将仅根据**原始的CNF公式**在初始化时计算一次，并且在整个求解过程中保持不变。这样做旨在让GNN更充分地理解问题的初始内部结构，而不是跟踪求解过程中的动态变化。

这大大简化了实现，避免了之前讨论的关于如何从`miniSATEnv`获取动态信息以及更新特征的复杂性。

### 1. 最终确定的静态特征定义

所有特征均基于原始CNF公式计算。

*   **变量节点特征 (共5维)**:
    1.  **`pos_lit_degree`**: 变量在原始CNF公式中以正文字形式出现的频率。
    2.  **`neg_lit_degree`**: 变量在原始CNF公式中以负文字形式出现的频率。
    3.  **`lit_pos_neg_ratio`**: `pos_lit_degree / (neg_lit_degree + 1.0)`。
    4.  **`horn_occurrence`**: 变量（无论正负）出现在原始CNF公式的Horn子句中的次数，除以原始CNF公式中的总子句数。 (如果总子句数为0，则此特征为0)。
    5.  **`clause_size_sum_inv`**: 对于原始CNF公式中所有包含该变量（无论正负）的子句，其长度的倒数之和。

*   **子句节点特征 (共6个特征，其中位置编码为10维，总计15维)**:
    1.  **`clause_degree`**: 原始子句的长度 / CNF公式中的总变量数。 (如果总变量数为0，则此特征为0)。
    2.  **`is_binary`**: 如果原始子句长度为2，则为1，否则为0。
    3.  **`is_ternary`**: 如果原始子句长度为3，则为1，否则为0。
    4.  **`is_horn`**: 如果原始子句是Horn子句，则为1，否则为0。 (使用之前定义的 `is_horn_clause` 辅助函数)。
    5.  **`clause_pos_neg_ratio`**: 原始子句中正文字的数量 / (负文字的数量 + 1.0)。
    6.  **`clause_pe[0..9]`**: 基于原始子句内文字顺序的10维位置编码 (实现方式同前述，例如使用 `torch.nn.Embedding` 后对子句内各位置的嵌入向量进行聚合)。

### 2. 实现步骤 (简化版)

1.  **辅助函数 `is_horn_clause`**: (同前述方案，位于 `gqsat/utils.py` 或类似模块)
    ```python
    # 示例: gqsat/utils.py
    def is_horn_clause(clause_literals: list[int]) -> bool:
        positive_literals_count = sum(1 for lit in clause_literals if lit > 0)
        return positive_literals_count <= 1
    ```

2.  **特征提取逻辑 (在图初始化时执行一次)**:
    *   定位负责加载CNF文件、构建图数据结构的代码 (可能在 `gqsat/utils.py` 中的 `MiniSATEnv` 初始化部分，或者在 `dqn.py` / `evaluate.py` 中环境创建和重置的部分)。
    *   在该逻辑中，当原始CNF公式加载完毕后，遍历所有变量和子句，计算上述定义的静态特征。
    *   **变量节点特征计算**: (针对每个变量 `var_id`)
        *   需要访问原始CNF的所有子句 (`all_original_clauses`) 和总子句数 (`num_total_original_clauses`)。
        *   计算 `pos_lit_degree`, `neg_lit_degree`。
        *   计算 `lit_pos_neg_ratio`。
        *   计算 `horn_occurrence` (需要先识别出所有Horn子句)。
        *   计算 `clause_size_sum_inv`。
    *   **子句节点特征计算**: (针对每个原始子句 `clause_literals`)
        *   需要访问总变量数 (`num_total_variables`)。
        *   计算 `clause_degree`。
        *   计算 `is_binary`, `is_ternary`。
        *   计算 `is_horn`。
        *   计算 `clause_pos_neg_ratio`。
        *   计算 `clause_pe` (10维向量)。
    *   将计算得到的特征向量存储在图数据结构中，供GNN模型使用。

3.  **GNN 模型输入维度**: (`gqsat/models.py`)
    *   确保GNN模型的第一层（或处理节点输入特征的相应层）的维度与上述定义的特征维度一致：
        *   变量节点输入层：5维。
        *   子句节点输入层：15维。

4.  **移除动态更新逻辑**: 之前方案中关于在 `env.step()` 后更新特征的逻辑可以完全移除。

### 3. 优势

*   **实现简单**：代码逻辑更直接，减少了与环境交互获取状态的复杂性。
*   **性能更佳**：避免了在每个`step`中重新计算特征的开销。
*   **稳定性**：特征固定，可能使GNN的学习过程更稳定，因为它总是从问题的相同结构表示中学习。

### 4. 总结

这个调整后的方案更加符合您让GNN理解问题“原始内部结构”的目标。所有特征都将作为一次性的预处理步骤在加载CNF公式时计算。请确保在您的代码中相应地调整特征提取和GNN模型输入部分。

## 7. 补充实现：Q值预留位和变量正负区分

根据进一步的需求分析，我发现需要关注两个重要方面：Q值表示和变量正负区分。以下是补充实现的具体步骤：

### 7.1 为Q值预留特征位置

虽然Q值是由GNN模型动态计算的，但为了概念上的清晰，我在变量节点特征中预留了专门的位置：

1. 修改特征提取器中的变量特征维度：
```python
def extract_var_features(self):
    """为所有变量提取特征"""
    features = np.zeros((self.num_vars, 7), dtype=np.float32)
    
    # ... 计算5个CNF特征 ...
    
    # 特征6-7: 为Q值预留的位置 (初始值为0)
    # Q值将由DQN模型计算并输出，这里只是预留位置
    features[var_idx, 5:7] = 0.0
    
    return features
```

2. 更新MiniSATEnv中的特征维度描述：
```python
# 变量节点: 1(标识符) + 5(新特征) + 2(Q值预留位) = 8维
# 子句节点: 1(标识符) + 15(新特征) = 16维
self.vertex_in_size = 16  # 使用最大的特征维度
```

3. 相应地调整parse_state_as_graph和get_dummy_state方法中的特征赋值逻辑：
```python
# 在parse_state_as_graph中
vertex_data[i, 1:8] = var_features[var_idx]  # 包括5个CNF特征和2个Q值预留位

# 在get_dummy_state中
DUMMY_V[:, 1:8] = np.array([0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
```

### 7.2 确认变量正负区分方式

在分析代码后，我确认了当前系统已经使用边特征来区分变量的正负性：

1. 在MiniSATEnv.py的parse_state_as_graph方法中：
```python
# if positive, create a [0,1] edge from the var to the current clause, else [1,0]
edge_data[ec : ec + 2, int(l > 0)] = 1
```

2. 添加明确的注释，说明边特征的含义：
```python
# 边特征维度:
# [0,1] 表示正文字边
# [1,0] 表示负文字边
self.edge_in_size = 2
```

这种设计非常合理，因为：
- 同一变量可能在不同子句中以正负不同形式出现
- 边特征可以准确表达这种关系
- GNN的消息传递机制正好可以利用这一信息

### 7.3 与现有代码的兼容性分析

这些修改与现有代码完全兼容，原因如下：

1. **特征维度扩展**：我们只是扩展了变量特征的维度，而没有改变原有的5个CNF特征的计算和存储方式。

2. **Q值计算逻辑不变**：系统的Q值计算逻辑依然依赖于EncoderCoreDecoder模型的输出，我们只是在节点特征中预留了位置，但并未改变DQN的计算流程。

3. **边特征保持不变**：我们没有修改边特征的计算方式，只是添加了更明确的注释。

4. **前向兼容性**：由于我们维持了vertex_in_size = 16的设置，模型依然能够处理这些特征向量。GNN模型会学习如何利用有意义的特征并忽略填充的零值。

总的来说，这些补充实现增强了代码的可读性和概念清晰度，同时保持了与现有系统的完全兼容性。
