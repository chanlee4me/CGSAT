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

MINISAT_DECISION_CONSTANT = 32767
VAR_ID_IDX = (
    0  # put 1 at the position of this index to indicate that the node is a variable
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
        max_decisions_cap=None  # 新增初始化参数
    ):

        self.problems_paths = [realpath(el) for el in problems_paths.split(":")]
        self.args = args
        self.test_mode = test_mode

        # added by cl
        # 实际中这两个定义不会对内部逻辑产生影响，仅作为对外接口说明
        # 定义动作空间：对于每个变量有两个动作（正/负赋值），因此总动作数为 2 * num_vars
        self.action_space = spaces.Discrete(2 * self.args.nums_variable)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.args.nums_variable,), dtype=np.float32)

        self.max_data_limit_per_set = max_data_limit_per_set
        # 收集多个目录下的 .cnf 文件路径，组织成二维列表
        pre_test_files = [
            [join(dir, f) for f in listdir(dir) if f.endswith(".cnf")]
            for dir in self.problems_paths
        ]
        if self.max_data_limit_per_set is not None:
            pre_test_files = [
                np.random.choice(el, size=max_data_limit_per_set, replace=False)
                for el in pre_test_files
            ]
        # 将上方得到的二维列表 pre_test_files 展平为 一维列表 self.test_files
        self.test_files = [sl for el in pre_test_files for sl in el]

        self.metadata = {}
        # added by cl
        self.max_decisions_cap = max_decisions_cap if max_decisions_cap is not None else sys.maxsize
        # self.max_decisions_cap = float("inf")
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
        #初始化一些后续要用到的参数
        self.test_file_num = len(self.test_files) # 计算并存储了 test_files 列表的长度
        self.test_to = 0 # 用作一个索引，用于遍历&指向当前将要处理的测试文件。

        self.step_ctr = 0 # 追踪程序执行的步数或迭代次数
        self.curr_problem = None # 程序运行中会被赋值为具体的问题对象或数据

        self.global_in_size = 1 # 全局数据的维度
        self.vertex_in_size = 2 # 将顶点输入大小设置为 2，代表图结构中每个顶点的特征数量或输入维度。
        self.edge_in_size = 2  # 将边的输入大小设置为 2，说明在图结构中，每条边附带的特征有两个维度或数据项
        self.max_clause_len = 0 # 记录某个约束或表达式中最长子句的长度

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
        # 获取每个变量的赋值状态（2 表示未赋值）
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
        # 由于只保留未赋值变量，需要把原始变量索引映射到紧凑的 [0..num_var-1]
        vars_remapping = {el: i for i, el in enumerate(valid_vars)}
        self.decision_to_var_mapping = {
            i: val_decision for i, val_decision in enumerate(valid_decisions)
        }

        # we should return the vertex/edge numpy objects from the c++ code to make this faster
        clauses = self.S.getClauses()

        if len(clauses) == 0:
            # this is to avoid feeding empty data structures to our model
            # when the MiniSAT environment returns an empty graph
            # it might return an empty graph since we do not construct it when
            # step > max_cap and max_cap can be zero (all decisions are made by MiniSAT's VSIDS).
            empty_state = self.get_dummy_state()
            self.decision_to_var_mapping = {
                el: el
                for sl in range(empty_state[0].shape[0])
                for el in (2 * sl, 2 * sl + 1)
            }
            return empty_state, False

        clause_counter = 0
        clauses_lens = [len(cl) for cl in clauses]
        self.max_clause_len = max(clauses_lens)
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
        )  # both vars and clauses are vertex in the graph
        vertex_data[:num_var, VAR_ID_IDX] = 1
        vertex_data[num_var:, VAR_ID_IDX + 1] = 1

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
            self.S.step(decision)
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
            step_reward = 0 if self.isSolved else -self.penalty_size
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
#   File "/4T/chenli/GraphSat_cli/minisat/minisat/gym/MiniSATEnv.py", line 352, in normalized_score
#     no_restart_steps, restart_steps = self.metadata[pdir][pname]
#                                       ~~~~~~~~~~~~~^^^^^^
#   File "/4T/chenli/GraphSat_cli/gqsat/utils.py", line 510, in evaluate
#     ns = eval_env.normalized_score(sctr, eval_env.curr_problem)
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/4T/chenli/GraphSat_cli/dqn.py", line 276, in <module>
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
        DUMMY_V = np.zeros((2, self.vertex_in_size), dtype=np.float32)
        DUMMY_V[:, VAR_ID_IDX] = 1
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
