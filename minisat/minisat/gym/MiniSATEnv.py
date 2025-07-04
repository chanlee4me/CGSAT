#################################################################################################################################
# All the source files in `minisat` folder were initially copied and later modified from https://github.com/feiwang3311/minisat #
# (which was taken from the MiniSat source at https://github.com/niklasso/minisat). The MiniSAT license is below.               #
#################################################################################################################################
# MiniSat -- Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
#            Copyright (c) 2007-2010  Niklas Sorensson
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 

# Graph-Q-SAT-UPD. This file is heavly changed and supports variable-sized SAT problems, multiple datasets
# and generates graph-state representations for Graph-Q-SAT.

import numpy as np
import gym
import random
from os import listdir
from os.path import join, realpath, split
from .GymSolver import GymSolver
from gym import spaces
import sys
# 导入特征提取器
import torch
import torch.nn as nn # 添加 torch.nn 导入
sys.path.append(realpath(join(split(realpath(__file__))[0], '../../../../')))
from gqsat.cnf_features import CNFFeatureExtractor, is_horn_clause

MINISAT_DECISION_CONSTANT = 32767
# 特征列索引常量
NODE_TYPE_COL = 0  # 节点类型所在列：1表示变量，2表示子句
NODE_ID_COL = 1    # 节点原始ID所在列
HANDCRAFTED_FEATURES_START_COL = 2 # 手工计算特征开始的列

# 手工特征维度常量
NUM_HANDCRAFTED_VAR_FEATURES = 5
NUM_HANDCRAFTED_CLAUSE_FEATURES = 15

# 节点类型常量
NODE_TYPE_VAR = 1
NODE_TYPE_CLAUSE = 2

# VAR_ID_IDX 保留以便兼容旧的引用，但建议后续逐步替换为 NODE_TYPE_COL
VAR_ID_IDX = NODE_TYPE_COL # put 1 at the position of this index to indicate that the node is a variable (for type identification)
import logging # 添加 logging 模块导入
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    filename='edge&node-feature.log',  # 指定日志文件名
    filemode='w'  # 每次运行时覆盖日志文件
)
class gym_sat_Env(gym.Env):
    def __init__(
        self,
        problems_paths,
        args,
        test_mode=False,
        max_cap_fill_buffer=True,
        penalty_size=None,
        with_restarts=None, #注意，是否重启要和minisat保持一致
        compare_with_restarts=None,
        max_data_limit_per_set=None,
        # added by cl≠
        max_decisions_cap=None,  # 新增初始化参数
        var_embedding_dim=16,    # 变量嵌入特征的维度 
        clause_embedding_dim=16, # 子句嵌入特征的维度
    ):
        super(gym_sat_Env, self).__init__()

        self.problems_paths = [realpath(el) for el in problems_paths.split(":")]
        self.args = args
        self.test_mode = test_mode

        self.action_space = spaces.Discrete(2 * self.args.nums_variable)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.args.nums_variable,), dtype=np.float32)

        self.max_data_limit_per_set = max_data_limit_per_set
        pre_test_files = [
            [join(dir, f) for f in listdir(dir) if f.endswith(".cnf")]
            for dir in self.problems_paths
        ]
        if self.max_data_limit_per_set is not None:
            pre_test_files = [
                np.random.choice(el, size=max_data_limit_per_set, replace=False)
                for el in pre_test_files
            ]
        self.test_files = [sl for el in pre_test_files for sl in el]

        self.metadata = {}
        self.max_decisions_cap = max_decisions_cap if max_decisions_cap is not None else 2147483647
        self.max_cap_fill_buffer = max_cap_fill_buffer
        self.penalty_size = penalty_size if penalty_size is not None else 0.0001
        self.with_restarts = True if with_restarts is None else with_restarts
        self.compare_with_restarts = (
            False if compare_with_restarts is None else compare_with_restarts
        )

        try:
            for dir in self.problems_paths:
                self.metadata[dir] = {}
                with open(join(dir, "METADATA")) as f:
                    for l in f:
                        k, rscore, msscore = l.split(",")
                        self.metadata[dir][k] = [int(rscore), int(msscore)]
        except Exception as e:
            print(e)
            print("No metadata available, that is fine for metadata generator.")
            self.metadata = None

        self.test_file_num = len(self.test_files)
        self.test_to = 0
        self.step_ctr = 0
        self.curr_problem = None

        # --- 架构重构：移除模型逻辑，定义原始特征维度 ---
        self.global_in_size = 1
        self.var_embedding_dim = var_embedding_dim
        self.clause_embedding_dim = clause_embedding_dim

        # 定义原始特征的维度 (手工特征 + 可学习嵌入)
        self.var_feature_dim = NUM_HANDCRAFTED_VAR_FEATURES + self.var_embedding_dim
        self.clause_feature_dim = NUM_HANDCRAFTED_CLAUSE_FEATURES + self.clause_embedding_dim

        # 节点特征的总维度 = 2 (类型, ID) + 填充到最大值的特征维度
        # 模型将需要处理这种异构和填充
        self.vertex_in_size = 2 + max(self.var_feature_dim, self.clause_feature_dim)
        
        # 边特征维度保持不变
        # [1,0] -> 正边, [0,1] -> 负边 表示正文字边 (变量 -> 子句)
        # [0,1] 表示负文字边 (变量 -> 子句)
        # 注意：原始代码中 edge_data[ec : ec + 2, int(l > 0)] = 1 的逻辑，
        # 如果 l > 0 (正文字), int(l > 0) 是 1, 那么 edge_data[:, 1] = 1, 即 [0,1]
        # 如果 l < 0 (负文字), int(l > 0) 是 0, 那么 edge_data[:, 0] = 1, 即 [1,0]
        # 这与注释中的 [0,1]正、[1,0]负 是对应的。
        self.edge_in_size = 2  

    def parse_state_as_graph(self):

        # if S is already Done, should return a dummy state to store in the buffer.
        if self.S.getDone():
            # to not mess with the c++ code, let's build a dummy graph which will not be used in the q updates anyways
            # since we multiply (1-dones)
            empty_state = self.get_dummy_state()
            self.decision_to_var_mapping = {
                el: el
                for sl in range(empty_state[0].shape[0])
                for el in (2 * sl, 2 * sl + 1)
            }
            return empty_state, True

        # S is not yet Done, parse and return real state
        # 获取 MiniSAT 当前元数据：总变量数、深度、初始子句数、重启次数等
        (
            total_var,
            _,
            current_depth,
            n_init_clauses,
            num_restarts,
            _,
        ) = self.S.getMetadata()
        # 从 MiniSAT 中获取每个变量的赋值状态（2 表示未赋值）
        var_assignments = self.S.getAssignments()
        
        num_var = sum([1 for el in var_assignments if el == 2])

        # 构造所有合法的决策（对于每个未赋值变量，有两个决策：正/负）
        valid_decisions = [
            el
            for i in range(len(var_assignments))
            for el in (2 * i, 2 * i + 1)
            if var_assignments[i] == 2
        ]
        # 收集未赋值的变量索引
        valid_vars = [
            idx for idx in range(len(var_assignments)) if var_assignments[idx] == 2
        ]
        # we need remapping since we keep only unassigned vars in the observations,
        # however, the environment does know about this, it expects proper indices of the variables
        # （由于只保留未赋值变量）将原始变量下标映射到紧凑的 [0..num_unassigned-1]
        vars_remapping = {el: i for i, el in enumerate(valid_vars)}
        self.decision_to_var_mapping = {
            i: val_decision for i, val_decision in enumerate(valid_decisions)
        }

        # we should return the vertex/edge numpy objects from the c++ code to make this faster
        clauses = self.S.getClauses()

        clause_counter = 0
        clauses_lens = [len(cl) for cl in clauses]
        # Handle case where solver simplifies the formula to have no clauses.
        self.max_clause_len = max(clauses_lens) if clauses else 0
        edge_data = np.zeros((sum(clauses_lens) * 2, 2), dtype=np.float32)
        connectivity = np.zeros((2, edge_data.shape[0]), dtype=int)
        ec = 0
        for cl in clauses:
            for l in cl:
                # if positive, create a [0,1] edge from the var to the current clause, else [1,0]
                # data = [1, 0] if l==True else [0, 1]

                # this is not a typo, we want two edge here
                edge_data[ec : ec + 2, int(l > 0)] = 1

                remapped_l = vars_remapping[abs(l) - 1]
                # from var to clause
                connectivity[0, ec] = remapped_l
                connectivity[1, ec] = num_var + clause_counter
                # from clause to var
                connectivity[0, ec + 1] = num_var + clause_counter
                connectivity[1, ec + 1] = remapped_l

                ec += 2
            clause_counter += 1

        vertex_data = np.zeros(
            (num_var + clause_counter, self.vertex_in_size), dtype=np.float32
        )  

        # 1. 填充节点类型 (第 NODE_TYPE_COL 列)
        vertex_data[:num_var, NODE_TYPE_COL] = NODE_TYPE_VAR  
        vertex_data[num_var:, NODE_TYPE_COL] = NODE_TYPE_CLAUSE 

        # 2. 填充节点原始ID (第 NODE_ID_COL 列)
        #    对于变量节点，存储其在原始问题中的索引 (注意：valid_vars 存储的是原始索引)
        for i in range(num_var):
            vertex_data[i, NODE_ID_COL] = valid_vars[i]
        #    对于子句节点，存储其在当前子句列表中的索引 (0 到 clause_counter-1)
        for i in range(clause_counter):
            vertex_data[num_var + i, NODE_ID_COL] = i
        
        # --- 架构重构：生成原始特征，不再进行GRU计算 ---
        # 1. 准备变量特征 (手工 + 嵌入)
        var_ids = valid_vars
        var_features = np.zeros((num_var, self.var_feature_dim), dtype=np.float32)
        # 提取变量的手工特征 (NUM_HANDCRAFTED_VAR_FEATURES 个)
        var_handcrafted_features = CNFFeatureExtractor(clauses, total_var).extract_var_features() 
        for i in range(num_var):
            original_var_idx = var_ids[i]
            var_features[i, :NUM_HANDCRAFTED_VAR_FEATURES] = var_handcrafted_features[original_var_idx]
            if self.var_embedding_dim > 0:
                var_features[i, NUM_HANDCRAFTED_VAR_FEATURES:] = np.zeros(self.var_embedding_dim)

        # 2. 准备子句特征 (手工 + 嵌入)
        clause_ids = list(range(clause_counter))
        clause_features = np.zeros((clause_counter, self.clause_feature_dim), dtype=np.float32)
        # 提取子句的手工特征 (NUM_HANDCRAFTED_CLAUSE_FEATURES 个)
        clause_handcrafted_features = CNFFeatureExtractor(clauses, total_var).extract_clause_features() 
        clause_features[:, :NUM_HANDCRAFTED_CLAUSE_FEATURES] = clause_handcrafted_features
        if self.clause_embedding_dim > 0:
            for i in range(clause_counter):
                original_clause_idx = clause_ids[i]
                clause_features[i, NUM_HANDCRAFTED_CLAUSE_FEATURES:] = np.zeros(self.clause_embedding_dim)

        # 3. 创建统一的、填充过的节点特征矩阵 v_data
        max_feature_dim = max(self.var_feature_dim, self.clause_feature_dim)

        # 填充变量节点数据 (并进行右侧零填充)
        for i in range(num_var):
            vertex_data[i, HANDCRAFTED_FEATURES_START_COL : HANDCRAFTED_FEATURES_START_COL + self.var_feature_dim] = var_features[i]

        # 填充子句节点数据 (并进行右侧零填充)
        for i in range(clause_counter):
            clause_node_actual_idx = i + num_var
            vertex_data[clause_node_actual_idx, HANDCRAFTED_FEATURES_START_COL : HANDCRAFTED_FEATURES_START_COL + self.clause_feature_dim] = clause_features[i]

        # 日志输出 (vertex_data 现在包含原始特征)
        logging.debug("--- Graph Data for Current State (after feature preparation) ---")
        logging.debug(f"Total Vertex data shape: {vertex_data.shape}")
        
        # 分别记录变量节点数据
        variable_nodes_data = vertex_data[:num_var, :]
        logging.debug(f"Variable Node Data shape: {variable_nodes_data.shape}")
        logging.debug(f"Variable Node Data (first 5 rows if available):\n{variable_nodes_data[:5]}")
        if variable_nodes_data.shape[0] > 5:
            logging.debug(f"Variable Node Data (last 5 rows if available):\n{variable_nodes_data[-5:]}")

        # 分别记录子句节点数据
        clause_nodes_data = vertex_data[num_var:, :]
        logging.debug(f"Clause Node Data shape: {clause_nodes_data.shape}")
        logging.debug(f"Clause Node Data (first 5 rows if available):\n{clause_nodes_data[:5]}")
        if clause_nodes_data.shape[0] > 5:
            logging.debug(f"Clause Node Data (last 5 rows if available):\n{clause_nodes_data[-5:]}")
        
        # 完整数据仍然可以按需记录，但可能非常大，默认注释掉或部分记录
        # logging.debug(f"Full Vertex data:\n{vertex_data}") 

        logging.debug(f"Edge data shape: {edge_data.shape}")
        logging.debug(f"Edge data (first 10 rows if available):\n{edge_data[:10]}")
        logging.debug(f"Connectivity shape: {connectivity.shape}")
        logging.debug(f"Connectivity data (first 10 columns if available):\n{connectivity[:, :10]}")
        # sys.exit(0) # DEBUG: This was preventing the test from completing.
        return (
            (
                vertex_data,
                edge_data,
                connectivity,
                np.zeros((1, self.global_in_size), dtype=np.float32),
            ),
            bool(False),
        )

    def random_pick_satProb(self):
        if self.test_mode:  # in the test mode, just iterate all test files in order
            filename = self.test_files[self.test_to]
            self.test_to += 1
            if self.test_to >= self.test_file_num:
                self.test_to = 0
            return filename
        else:  # not in test mode, return a random file in "uf20-91" folder.
            return self.test_files[random.randint(0, self.test_file_num - 1)]

    def reset(self):
        self.step_ctr = 0
        # addde by cl 由于在环境的构造函数中直接传入了max_decisions_cap,所以在这里就不用参数传入了，直接调用
        # if max_decisions_cap is None:
        #     max_decisions_cap = sys.maxsize
        # self.max_decisions_cap = max_decisions_cap
        self.curr_problem = self.random_pick_satProb() #这里是随机选择一个问题，所以在一个batch中，可能会有重复问题
        self.S = GymSolver(self.curr_problem, self.with_restarts, self.max_decisions_cap)
        self.max_clause_len = 0

        self.curr_state, self.isSolved = self.parse_state_as_graph()
        #added by cl 在这里为observation_space重新赋值（由于在初始化时observation_space的值是为了符合API规范我随意取的，数据结构不一致）
        self.observation_space = self.curr_state
        return self.curr_state

    def step(self, decision, dummy=False):
        # now when we drop variables, we store the mapping
        # convert dropped var decision to the original decision id
        if decision >= 0:
            decision = self.decision_to_var_mapping[decision]
        self.step_ctr += 1

        if dummy:
            self.S.step(MINISAT_DECISION_CONSTANT)
            (
                num_var,
                _,
                current_depth,
                n_init_clauses,
                num_restarts,
                _,
            ) = self.S.getMetadata()
            return (
                None,
                None,
                self.S.getDone(),
                {
                    "curr_problem": self.curr_problem,
                    "num_restarts": num_restarts,
                    "max_clause_len": self.max_clause_len,
                },
            )

        if self.step_ctr > self.max_decisions_cap:
            while not self.S.getDone():
                self.S.step(MINISAT_DECISION_CONSTANT)
                if self.max_cap_fill_buffer:
                    # return every next state when param is true
                    break
                self.step_ctr += 1
            else:
                # if we are here, we are not filling the buffer and we need to reduce the counter by one to
                # correct for the increment for the last state
                self.step_ctr -= 1
        else:
            # TODO for debugging purposes, we need to add all the checks
            # I removed this action_set checks for performance optimisation

            # var_values = self.curr_state[0][:, 2]
            # var_values = self.S.getAssignments()
            # action_set = [
            #     a
            #     for v_idx, v in enumerate(var_values)
            #     for a in (v_idx * 2, v_idx * 2 + 1)
            #     if v == 2
            # ]

            if decision < 0:  # this is to say that let minisat pick the decision
                decision = MINISAT_DECISION_CONSTANT
            elif (
                decision % 2 == 0
            ):  # this is to say that pick decision and assign positive value
                decision = int(decision / 2 + 1)
            else:  # this is to say that pick decision and assign negative value
                decision = 0 - int(decision / 2 + 1)

            # if (decision == MINISAT_DECISION_CONSTANT) or orig_decision in action_set:
            old_conflicts = self.S.getNumConflicts()
            self.S.step(decision)
            new_conflicts = self.S.getNumConflicts()
            # else:
            #    raise ValueError("Illegal action")

        self.curr_state, self.isSolved = self.parse_state_as_graph()
        (
            num_var,
            _,
            current_depth,
            n_init_clauses,
            num_restarts,
            _,
        ) = self.S.getMetadata()

        # if we fill the buffer, the rewards are the same as GQSAT was making decisions
        if self.step_ctr > self.max_decisions_cap and not self.max_cap_fill_buffer:
            # if we do not fill the buffer, but play till the end, we still need to penalize
            # since GQSAT hasn't solved the problem
            step_reward = -self.penalty_size
        else:
            # 基于冲突增量的密集奖励 (新逻辑：冲突越多奖励越高)
            conflict_increase = new_conflicts - old_conflicts
            # 如果未解决，除了密集奖励外，再加一个小的惩罚项
            step_reward = conflict_increase if self.isSolved else conflict_increase - self.penalty_size
        return (
            #added by cl
            #TODO 看下curr_state是什么样的，然后修改初始的observation_space
            self.curr_state,    
            step_reward,
            self.isSolved,
            {
                "curr_problem": self.curr_problem,
                "num_restarts": num_restarts,
                "max_clause_len": self.max_clause_len,
            },
        )


    def normalized_score(self, steps, problem):
        pdir, pname = split(problem)
#         NoneType' object is not subscriptable
#   File "/4T/chenli/CGSAT/minisat/minisat/gym/MiniSATEnv.py", line 352, in normalized_score
#     no_restart_steps, restart_steps = self.metadata[pdir][pname]
#                                       ~~~~~~~~~~~~~^^^^^^
#   File "/4T/chenli/CGSAT/gqsat/utils.py", line 510, in evaluate
#     ns = eval_env.normalized_score(sctr, eval_env.curr_problem)
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/4T/chenli/CGSAT/dqn.py", line 276, in <module>
#     agent, args, include_train_set=False

#                     )
#                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# TypeError: 'NoneType' object is not subscriptable
        # TODO：下面这行语句报错如上 
        no_restart_steps, restart_steps = self.metadata[pdir][pname]
        if self.compare_with_restarts:
            return restart_steps / steps
        else:
            return no_restart_steps / steps

    def get_dummy_state(self):
        DUMMY_V = np.zeros((2, self.vertex_in_size), dtype=np.float32) # 创建包含两个虚拟节点的特征矩阵
        
        # 填充节点类型 (第 NODE_TYPE_COL 列)
        DUMMY_V[:, NODE_TYPE_COL] = NODE_TYPE_VAR  # 将两个虚拟节点都设置为变量类型
        
        # 填充节点ID (第 NODE_ID_COL 列) - 可以使用虚拟ID
        DUMMY_V[0, NODE_ID_COL] = 0 # 虚拟变量节点0的ID
        DUMMY_V[1, NODE_ID_COL] = 1 # 虚拟变量节点1的ID
        
        # 填充手工特征 (从 HANDCRAFTED_FEATURES_START_COL 列开始)
        # 为虚拟变量节点填充 NUM_HANDCRAFTED_VAR_FEATURES 个手工特征
        dummy_handcrafted_var_features = np.array([0.5, 0.5, 1.0, 0.0, 0.0], dtype=np.float32) # 示例手工特征
        if len(dummy_handcrafted_var_features) != NUM_HANDCRAFTED_VAR_FEATURES:
            # 如果示例特征数量不匹配，则用0填充或截断 (确保维度正确)
            dummy_handcrafted_var_features = np.zeros(NUM_HANDCRAFTED_VAR_FEATURES, dtype=np.float32)
            # 或者根据需要调整示例特征

        start_col = HANDCRAFTED_FEATURES_START_COL
        end_col = start_col + NUM_HANDCRAFTED_VAR_FEATURES
        DUMMY_V[:, start_col:end_col] = dummy_handcrafted_var_features
        
        # 嵌入特征部分将保持为0，由 np.zeros 初始化得到
        
        DUMMY_STATE = (
            DUMMY_V,
            np.zeros((2, self.edge_in_size), dtype=np.float32),
            np.eye(2, dtype=np.long),
            np.zeros((1, self.global_in_size), dtype=np.float32),
        )
        return (
            DUMMY_STATE[0],
            DUMMY_STATE[1],
            DUMMY_STATE[2],
            np.zeros((1, self.global_in_size), dtype=np.float32),
        )
