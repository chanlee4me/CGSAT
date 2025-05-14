"""
CNF特征提取模块 - 根据CNF公式计算节点特征
"""
import torch
import numpy as np


def is_horn_clause(clause_literals: list[int]) -> bool:
    """
    判断一个子句是否为 Horn 子句。
    Horn 子句是最多只有一个正文字的子句。
    
    Args:
        clause_literals: 子句中的文字列表 (例如 [1, -2, -3]，其中正数表示正文字，负数表示负文字)
    Returns:
        True 如果是 Horn 子句, 否则 False
    """
    positive_literals_count = sum(1 for lit in clause_literals if lit > 0)
    return positive_literals_count <= 1


class CNFFeatureExtractor:
    """
    从CNF公式中提取变量节点和子句节点特征的类
    """
    
    def __init__(self, clauses, num_vars):
        """
        初始化CNF特征提取器
        
        Args:
            clauses: 子句列表，每个子句是一个整数列表，表示文字 (正数为正文字，负数为负文字)
            num_vars: CNF公式中的变量总数
        """
        self.clauses = clauses
        self.num_vars = num_vars
        self.num_clauses = len(clauses)
        
        # 预计算一些常用的统计信息
        self.var_pos_occurrences = self._count_var_occurrences(positive=True)
        self.var_neg_occurrences = self._count_var_occurrences(positive=False)
        self.horn_clauses = [is_horn_clause(clause) for clause in self.clauses]
        self.num_horn_clauses = sum(self.horn_clauses)
        
        # 预计算变量在Horn子句中的出现情况
        self.var_in_horn = self._count_var_in_horn_clauses()
        
        # 为位置编码设置参数
        self.max_literals_in_clause = 50  # 可以根据实际数据调整
        self.pe_dim = 10
        
        # 初始化位置嵌入层 - 注意：根据您的要求，位置编码将置0，但保留定义以备将来使用
        # self.literal_pos_embedding = torch.nn.Embedding(self.max_literals_in_clause, self.pe_dim)

        # 一次性计算并存储所有原始特征
        self._precompute_all_features()

    def _precompute_all_features(self):
        """
        预计算并存储所有变量和子句的特征。
        """
        # 预计算变量特征
        var_features_list = []
        for var_idx in range(self.num_vars):
            features_var = np.zeros(7, dtype=np.float32) # 保持7维以兼容原有代码，Q值预留位在外部处理
            features_var[0] = self.var_pos_occurrences[var_idx]
            features_var[1] = self.var_neg_occurrences[var_idx]
            features_var[2] = self.var_pos_occurrences[var_idx] / (self.var_neg_occurrences[var_idx] + 1.0)
            if self.num_clauses > 0:
                features_var[3] = self.var_in_horn[var_idx] / self.num_clauses
            else:
                features_var[3] = 0.0
            
            sum_inverse_clause_lengths = 0.0
            for clause in self.clauses: # 仅使用原始子句计算
                clause_length = len(clause)
                if clause_length == 0:
                    continue
                if any(abs(lit) - 1 == var_idx for lit in clause):
                    sum_inverse_clause_lengths += 1.0 / clause_length
            features_var[4] = sum_inverse_clause_lengths
            features_var[5:7] = 0.0 # Q值预留位
            var_features_list.append(features_var)
        self.precomputed_var_features = np.array(var_features_list, dtype=np.float32)

        # 预计算子句特征
        clause_features_list = []
        for clause_idx, clause in enumerate(self.clauses):
            clause_length = len(clause)
            scalar_features = np.zeros(5, dtype=np.float32)
            position_features = np.zeros(self.pe_dim, dtype=np.float32) # 位置编码置0

            if self.num_vars > 0:
                scalar_features[0] = clause_length / self.num_vars
            else:
                scalar_features[0] = 0.0
            
            scalar_features[1] = 1.0 if clause_length == 2 else 0.0
            scalar_features[2] = 1.0 if clause_length == 3 else 0.0
            scalar_features[3] = 1.0 if self.horn_clauses[clause_idx] else 0.0
            
            num_pos_lits = sum(1 for lit in clause if lit > 0)
            num_neg_lits = sum(1 for lit in clause if lit < 0)
            scalar_features[4] = num_pos_lits / (num_neg_lits + 1.0)
            
            # 位置编码特征直接置0
            # if clause_length > 0 and hasattr(self, 'literal_pos_embedding'):
            #     positions = torch.arange(min(clause_length, self.max_literals_in_clause))
            #     embeddings = self.literal_pos_embedding(positions)
            #     pe_vector = torch.mean(embeddings, dim=0).detach().numpy()
            #     position_features = pe_vector
                
            clause_features_list.append(np.concatenate([scalar_features, position_features]))
        self.precomputed_clause_features = np.array(clause_features_list, dtype=np.float32)

    def _count_var_occurrences(self, positive=True):
        """
        计算每个变量在子句中以正文字/负文字形式出现的次数
        
        Args:
            positive: 如果为True，统计正文字出现次数；否则统计负文字
        
        Returns:
            包含每个变量出现次数的列表 (索引从0开始)
        """
        occurrences = [0] * self.num_vars
        
        for clause in self.clauses:
            for lit in clause:
                var_idx = abs(lit) - 1  # 变量索引从0开始
                if (lit > 0) == positive:
                    occurrences[var_idx] += 1
                    
        return occurrences
    
    def _count_var_in_horn_clauses(self):
        """
        计算每个变量在Horn子句中出现的次数
        
        Returns:
            包含每个变量在Horn子句中出现次数的列表
        """
        var_in_horn = [0] * self.num_vars
        
        for i, clause in enumerate(self.clauses):
            if not self.horn_clauses[i]:
                continue
                
            for lit in clause:
                var_idx = abs(lit) - 1
                var_in_horn[var_idx] += 1
                
        return var_in_horn
    
    def extract_var_features(self):
        """
        为所有变量提取特征 (返回预计算的特征)
        
        Returns:
            变量节点特征矩阵，形状为 [num_vars, 7]
        """
        return self.precomputed_var_features
    
    def extract_clause_features(self):
        """
        为所有原始子句提取特征 (返回预计算的特征)
        
        Returns:
            子句节点特征矩阵，形状为 [num_clauses, 15] (5个标量特征 + 10维位置编码)
        """
        return self.precomputed_clause_features

    def extract_features_for_new_clause(self, new_clause: list[int]):
        """
        为新生成的学习子句计算特征。
        位置编码特征将置为0。
        对于变量相关的统计特征（如 pos/neg occurrences, horn occurrences），
        我们不能简单地更新全局统计量，因为这会影响原始问题的特征。
        因此，对于新子句，某些基于全局统计的特征可能需要特殊处理或近似。
        这里我们尝试计算其基本特征。

        Args:
            new_clause: 新学习到的子句，一个整数列表。

        Returns:
            新子句的特征向量，形状为 [15]
        """
        clause_length = len(new_clause)
        scalar_features = np.zeros(5, dtype=np.float32)
        position_features = np.zeros(self.pe_dim, dtype=np.float32) # 位置编码置0

        # 特征1: clause_degree - 子句长度 / 变量总数
        if self.num_vars > 0:
            scalar_features[0] = clause_length / self.num_vars
        else:
            scalar_features[0] = 0.0
            
        # 特征2: is_binary - 二元子句标志 (0/1)
        scalar_features[1] = 1.0 if clause_length == 2 else 0.0
        
        # 特征3: is_ternary - 三元子句标志 (0/1)
        scalar_features[2] = 1.0 if clause_length == 3 else 0.0
        
        # 特征4: is_horn - Horn子句标志 (0/1)
        scalar_features[3] = 1.0 if is_horn_clause(new_clause) else 0.0
        
        # 特征5: clause_pos_neg_ratio - 子句内部正／负字面比例
        num_pos_lits = sum(1 for lit in new_clause if lit > 0)
        num_neg_lits = sum(1 for lit in new_clause if lit < 0)
        scalar_features[4] = num_pos_lits / (num_neg_lits + 1.0)
        
        # 特征6: clause_pe[0..9] - 10维位置编码 (置0)
        # position_features 已经初始化为0

        return np.concatenate([scalar_features, position_features])
