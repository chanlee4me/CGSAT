\
import unittest
import numpy as np
import sys
import os

# 将父目录添加到sys.path以允许导入gqsat模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cnf_features import CNFFeatureExtractor, is_horn_clause

def parse_cnf_file(file_path: str) -> tuple[list[list[int]], int]:
    """
    解析CNF文件并提取子句和变量数。
    忽略注释行和问题行。
    """
    clauses = []
    num_vars = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('c'):
                continue
            elif line.startswith('p cnf'):
                parts = line.split()
                num_vars = int(parts[2])
                # num_clauses_in_file = int(parts[3]) # 可以选择性使用
            elif line: # 非空行，且不是注释或问题行
                # 处理可能跨越多行的子句定义，或者一行内有多个子句（尽管不常见）
                literals_str = line.split()
                current_clause = []
                for lit_str in literals_str:
                    lit = int(lit_str)
                    if lit == 0:
                        if current_clause: # 避免添加空子句
                            clauses.append(current_clause)
                        current_clause = []
                    else:
                        current_clause.append(lit)
                if current_clause: # 处理行尾没有0但有文字的情况（不规范但可能存在）
                    clauses.append(current_clause)
    return clauses, num_vars

class TestCNFFeatures(unittest.TestCase):

    def setUp(self):
        self.clauses = [
            [1, -2],       # Horn, Binary
            [-1, 2, -3],   # Horn, Ternary
            [1, 2, 3]      # Non-Horn, Ternary
        ]
        self.num_vars = 3
        self.extractor = CNFFeatureExtractor(self.clauses, self.num_vars)

    def test_is_horn_clause(self):
        self.assertTrue(is_horn_clause([1, -2, -3]))
        self.assertTrue(is_horn_clause([-1, -2, -3]))
        self.assertTrue(is_horn_clause([1]))
        self.assertFalse(is_horn_clause([1, 2, -3]))
        self.assertFalse(is_horn_clause([1, 2, 3]))

    def test_extractor_initialization_and_precomputation(self):
        # Test variable features
        self.assertEqual(self.extractor.precomputed_var_features.shape, (self.num_vars, 5))
        
        # Var 1 (index 0)
        # pos_lit_degree: 2 (in clause 0, 2)
        # neg_lit_degree: 1 (in clause 1)
        # lit_pos_neg_ratio: 2 / (1 + 1) = 1.0
        # horn_occurrence: 2 (clauses 0, 1 are Horn and contain var 1) / 3 (total clauses) = 2/3
        # clause_size_sum_inv: (1/2 for clause 0) + (1/3 for clause 1) + (1/3 for clause 2) = 0.5 + 0.333... + 0.333... = 1.166...
        expected_var0_feat = np.array([2, 1, 1.0, 2/3, 1/2 + 1/3 + 1/3], dtype=np.float32)
        for i in range(5):
            self.assertAlmostEqual(self.extractor.precomputed_var_features[0, i], expected_var0_feat[i], places=5)

        # Var 2 (index 1)
        # pos_lit_degree: 2 (in clause 1, 2)
        # neg_lit_degree: 1 (in clause 0)
        # lit_pos_neg_ratio: 2 / (1 + 1) = 1.0
        # horn_occurrence: 2 (clauses 0, 1 are Horn and contain var 2) / 3 = 2/3
        # clause_size_sum_inv: (1/2 for clause 0) + (1/3 for clause 1) + (1/3 for clause 2) = 1.166...
        expected_var1_feat = np.array([2, 1, 1.0, 2/3, 1/2 + 1/3 + 1/3], dtype=np.float32)
        for i in range(5):
            self.assertAlmostEqual(self.extractor.precomputed_var_features[1, i], expected_var1_feat[i], places=5)

        # Var 3 (index 2)
        # pos_lit_degree: 1 (in clause 2)
        # neg_lit_degree: 1 (in clause 1)
        # lit_pos_neg_ratio: 1 / (1 + 1) = 0.5
        # horn_occurrence: 1 (clause 1 is Horn and contains var 3) / 3 = 1/3
        # clause_size_sum_inv: (1/3 for clause 1) + (1/3 for clause 2) = 0.666...
        expected_var2_feat = np.array([1, 1, 0.5, 1/3, 1/3 + 1/3], dtype=np.float32)
        for i in range(5):
            self.assertAlmostEqual(self.extractor.precomputed_var_features[2, i], expected_var2_feat[i], places=5)

        # Test clause features
        self.assertEqual(self.extractor.precomputed_clause_features.shape, (len(self.clauses), 5 + self.extractor.pe_dim))
        
        # Clause 0 ([1, -2])
        # clause_degree: 2 / 3
        # is_binary: 1.0
        # is_ternary: 0.0
        # is_horn: 1.0
        # clause_pos_neg_ratio: 1 / (1 + 1) = 0.5
        expected_clause0_scalar_feat = np.array([2/3, 1.0, 0.0, 1.0, 0.5], dtype=np.float32)
        for i in range(5):
            self.assertAlmostEqual(self.extractor.precomputed_clause_features[0, i], expected_clause0_scalar_feat[i], places=5)
        self.assertTrue(np.all(self.extractor.precomputed_clause_features[0, 5:] == 0)) # PE should be zeros

        # Clause 1 ([-1, 2, -3])
        # clause_degree: 3 / 3 = 1.0
        # is_binary: 0.0
        # is_ternary: 1.0
        # is_horn: 1.0
        # clause_pos_neg_ratio: 1 / (2 + 1) = 1/3
        expected_clause1_scalar_feat = np.array([1.0, 0.0, 1.0, 1.0, 1/3], dtype=np.float32)
        for i in range(5):
            self.assertAlmostEqual(self.extractor.precomputed_clause_features[1, i], expected_clause1_scalar_feat[i], places=5)
        self.assertTrue(np.all(self.extractor.precomputed_clause_features[1, 5:] == 0))

        # Clause 2 ([1, 2, 3])
        # clause_degree: 3 / 3 = 1.0
        # is_binary: 0.0
        # is_ternary: 1.0
        # is_horn: 0.0
        # clause_pos_neg_ratio: 3 / (0 + 1) = 3.0
        expected_clause2_scalar_feat = np.array([1.0, 0.0, 1.0, 0.0, 3.0], dtype=np.float32)
        for i in range(5):
            self.assertAlmostEqual(self.extractor.precomputed_clause_features[2, i], expected_clause2_scalar_feat[i], places=5)
        self.assertTrue(np.all(self.extractor.precomputed_clause_features[2, 5:] == 0))

    def test_extract_var_features(self):
        var_features = self.extractor.extract_var_features()
        self.assertEqual(var_features.shape, (self.num_vars, 5))
        self.assertTrue(np.array_equal(var_features, self.extractor.precomputed_var_features))

    def test_extract_clause_features(self):
        clause_features = self.extractor.extract_clause_features()
        self.assertEqual(clause_features.shape, (len(self.clauses), 5 + self.extractor.pe_dim))
        self.assertTrue(np.array_equal(clause_features, self.extractor.precomputed_clause_features))

    def test_extract_features_for_new_clause(self):
        new_clause = [-2, 3, 4] # Assume var 4 is a new variable, num_vars for extractor is 3
                               # For new clauses, num_vars from original problem is used for clause_degree
        num_vars_original = self.extractor.num_vars # Should be 3
        
        features = self.extractor.extract_features_for_new_clause(new_clause)
        self.assertEqual(features.shape, (5 + self.extractor.pe_dim,))
        
        # Expected scalar features for new_clause = [-2, 3, 4]
        # clause_length = 3
        # clause_degree: 3 / num_vars_original (3) = 1.0
        # is_binary: 0.0
        # is_ternary: 1.0
        # is_horn: True (lit 3 is positive, -2 and 4 (if treated as neg) or if 4 is pos, still Horn)
        # For is_horn_clause, it only counts positive literals. So [-2, 3, 4] has 2 positive (3, 4), so it's NOT Horn.
        # Let's re-evaluate: new_clause = [-2, 3, 4]. Positive lits: 3, 4. Count = 2. Not Horn.
        # is_horn: 0.0
        # clause_pos_neg_ratio: 2 (pos: 3, 4) / (1 (neg: -2) + 1) = 2 / 2 = 1.0

        expected_scalar = np.array([
            len(new_clause) / num_vars_original, # clause_degree
            0.0,                                 # is_binary
            1.0,                                 # is_ternary
            0.0,                                 # is_horn (re-evaluated)
            1.0                                  # clause_pos_neg_ratio (re-evaluated)
        ], dtype=np.float32)
        
        for i in range(5):
            self.assertAlmostEqual(features[i], expected_scalar[i], places=5)
        self.assertTrue(np.all(features[5:] == 0)) # PE should be zeros

class TestCNFFeaturesWithFile(unittest.TestCase):
    def setUp(self):
        # 使用我们创建的sample.cnf文件
        self.cnf_file_path = os.path.join(os.path.dirname(__file__), 'sample.cnf')
        self.clauses, self.num_vars = parse_cnf_file(self.cnf_file_path)
        self.extractor = CNFFeatureExtractor(self.clauses, self.num_vars)

    def test_parsed_clauses_and_vars(self):
        expected_clauses = [
            [1, -2],
            [-1, 2, -3],
            [1, 2, 3]
        ]
        expected_num_vars = 3
        self.assertEqual(self.num_vars, expected_num_vars)
        self.assertEqual(len(self.clauses), len(expected_clauses))
        for i, clause in enumerate(self.clauses):
            self.assertListEqual(clause, expected_clauses[i])

    def test_features_from_cnf_file(self):
        # 这个测试与之前的 test_extractor_initialization_and_precomputation 类似
        # 但数据源是解析的CNF文件
        self.assertEqual(self.extractor.precomputed_var_features.shape, (self.num_vars, 5))
        
        # Var 1 (index 0)
        expected_var0_feat = np.array([2, 1, 1.0, 2/3, 1/2 + 1/3 + 1/3], dtype=np.float32)
        for i in range(5):
            self.assertAlmostEqual(self.extractor.precomputed_var_features[0, i], expected_var0_feat[i], places=5)

        # Var 2 (index 1)
        expected_var1_feat = np.array([2, 1, 1.0, 2/3, 1/2 + 1/3 + 1/3], dtype=np.float32)
        for i in range(5):
            self.assertAlmostEqual(self.extractor.precomputed_var_features[1, i], expected_var1_feat[i], places=5)

        # Var 3 (index 2)
        expected_var2_feat = np.array([1, 1, 0.5, 1/3, 1/3 + 1/3], dtype=np.float32)
        for i in range(5):
            self.assertAlmostEqual(self.extractor.precomputed_var_features[2, i], expected_var2_feat[i], places=5)

        self.assertEqual(self.extractor.precomputed_clause_features.shape, (len(self.clauses), 5 + self.extractor.pe_dim))
        
        # Clause 0 ([1, -2])
        expected_clause0_scalar_feat = np.array([2/3, 1.0, 0.0, 1.0, 0.5], dtype=np.float32)
        for i in range(5):
            self.assertAlmostEqual(self.extractor.precomputed_clause_features[0, i], expected_clause0_scalar_feat[i], places=5)
        self.assertTrue(np.all(self.extractor.precomputed_clause_features[0, 5:] == 0))

        # Clause 1 ([-1, 2, -3])
        expected_clause1_scalar_feat = np.array([1.0, 0.0, 1.0, 1.0, 1/3], dtype=np.float32)
        for i in range(5):
            self.assertAlmostEqual(self.extractor.precomputed_clause_features[1, i], expected_clause1_scalar_feat[i], places=5)
        self.assertTrue(np.all(self.extractor.precomputed_clause_features[1, 5:] == 0))

        # Clause 2 ([1, 2, 3])
        expected_clause2_scalar_feat = np.array([1.0, 0.0, 1.0, 0.0, 3.0], dtype=np.float32)
        for i in range(5):
            self.assertAlmostEqual(self.extractor.precomputed_clause_features[2, i], expected_clause2_scalar_feat[i], places=5)
        self.assertTrue(np.all(self.extractor.precomputed_clause_features[2, 5:] == 0))

if __name__ == '__main__':
    unittest.main()
