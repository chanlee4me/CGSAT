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
        
        # 初始化位置嵌入层
        self.literal_pos_embedding = torch.nn.Embedding(self.max_literals_in_clause, self.pe_dim)
        
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
        为所有变量提取特征
        
        Returns:
            变量节点特征矩阵，形状为 [num_vars, 5]
        """
        features = np.zeros((self.num_vars, 7), dtype=np.float32)
        
        for var_idx in range(self.num_vars):
            # 特征1: pos_lit_degree - 变量的正文字出现频率
            features[var_idx, 0] = self.var_pos_occurrences[var_idx]
            
            # 特征2: neg_lit_degree - 变量的负文字出现频率
            features[var_idx, 1] = self.var_neg_occurrences[var_idx]
            
            # 特征3: lit_pos_neg_ratio - (正文字出现次数) / (负文字出现次数 + 1)
            features[var_idx, 2] = self.var_pos_occurrences[var_idx] / (self.var_neg_occurrences[var_idx] + 1.0)
            
            # 特征4: horn_occurrence - 变量出现在Horn子句的次数 / 总子句数
            if self.num_clauses > 0:
                features[var_idx, 3] = self.var_in_horn[var_idx] / self.num_clauses
            else:
                features[var_idx, 3] = 0.0
                
            # 特征5: clause_size_sum_inv - 所有含该变量的子句长度倒数之和
            sum_inverse_clause_lengths = 0.0
            for clause in self.clauses:
                clause_length = len(clause)
                if clause_length == 0:
                    continue
                    
                # 检查变量是否出现在子句中(正文字或负文字)
                if any(abs(lit) - 1 == var_idx for lit in clause):
                    sum_inverse_clause_lengths += 1.0 / clause_length
                    
            features[var_idx, 4] = sum_inverse_clause_lengths
            
            # 特征6-7: 为Q值预留的位置 (初始值为0)
            # Q值将由DQN模型计算并输出，这里只是预留位置
            features[var_idx, 5:7] = 0.0
            
        return features
    
    def extract_clause_features(self):
        """
        为所有子句提取特征
        
        Returns:
            子句节点特征矩阵，形状为 [num_clauses, 15] (5个标量特征 + 10维位置编码)
        """
        # 5个标量特征
        scalar_features = np.zeros((self.num_clauses, 5), dtype=np.float32)
        # 10维位置编码
        position_features = np.zeros((self.num_clauses, 10), dtype=np.float32)
        
        for clause_idx, clause in enumerate(self.clauses):
            clause_length = len(clause)
            
            # 特征1: clause_degree - 子句长度 / 变量总数
            if self.num_vars > 0:
                scalar_features[clause_idx, 0] = clause_length / self.num_vars
            else:
                scalar_features[clause_idx, 0] = 0.0
                
            # 特征2: is_binary - 二元子句标志 (0/1)
            scalar_features[clause_idx, 1] = 1.0 if clause_length == 2 else 0.0
            
            # 特征3: is_ternary - 三元子句标志 (0/1)
            scalar_features[clause_idx, 2] = 1.0 if clause_length == 3 else 0.0
            
            # 特征4: is_horn - Horn子句标志 (0/1)
            scalar_features[clause_idx, 3] = 1.0 if self.horn_clauses[clause_idx] else 0.0
            
            # 特征5: clause_pos_neg_ratio - 子句内部正／负字面比例
            num_pos_lits = sum(1 for lit in clause if lit > 0)
            num_neg_lits = sum(1 for lit in clause if lit < 0)
            scalar_features[clause_idx, 4] = num_pos_lits / (num_neg_lits + 1.0)
            
            # 特征6: clause_pe[0..9] - 10维位置编码
            if clause_length > 0:
                # 对于子句中的每个位置，获取其嵌入
                positions = torch.arange(min(clause_length, self.max_literals_in_clause))
                embeddings = self.literal_pos_embedding(positions)
                # 计算平均池化得到最终的位置编码
                pe_vector = torch.mean(embeddings, dim=0).detach().numpy()
                position_features[clause_idx] = pe_vector
                
        # 合并标量特征和位置编码
        features = np.concatenate([scalar_features, position_features], axis=1)
        return features
