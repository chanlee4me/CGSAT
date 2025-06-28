# Copyright 2019-2020 Nvidia Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX, NODE_TYPE_COL, NODE_TYPE_VAR
from collections import deque


# 抽象基类，定义了智能体应该实现的接口
class Agent(object):
    # 在给定环境状态下，选择一个动作（这里的 state 代表环境状态）
    def act(self, state):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


# 该类继承自 Agent 类，表示一个使用MiniSAT求解器的智能体
class MiniSATAgent(Agent):
    """Use MiniSAT agent to solve the problem"""

    #在执行 act()方法时，会返回 -1，触发底层环境使用MiniSAT内置的VSIDS来做决策
    def act(self, observation):
        return -1  # this will make GymSolver use VSIDS to make a decision

    # 返回该智能体的字符串描述 "<MiniSAT Agent>"
    def __str__(self):
        return "<MiniSAT Agent>"


# 继承自 Agent 类，它的决策机制是随机的，从可行动作空间中均匀采样动作。
class RandomAgent(Agent):
    """Uniformly sample the action space"""

    # 构造函数接收一个 action_space 参数，该参数定义了智能体的动作空间。
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    # 该方法会随机从动作空间中选择一个动作
    def act(self, observation):
        return self.action_space.sample()

    # 返回该智能体的字符串描述 "<Random Agent>"。
    def __str__(self):
        return "<Random Agent>"


# 基于一个神经网络（net）来做决策，并且能够使用epsilon-greedy策略进行动作选择
class GraphAgent:
    
    def __init__(self, net, args):# 接收一个神经网络模型net和一个参数字典 args作为输入

        self.net = net
        self.device = args.device   # args.device 表示设备（CPU 或 GPU）
        self.debug = args.debug     # args.debug 表示是否开启调试模式
        self.qs_buffer = []
        # action pool settings
        self.pool_enabled = hasattr(args, 'action_pool_size') and args.action_pool_size > 0
        self.pool_size = args.action_pool_size if self.pool_enabled else 0
        self.action_pool = deque()

    # 将历史数据（hist_buffer）传入神经网络进行前向计算，输出每个节点的 Q 值
    def forward(self, hist_buffer):
        self.net.eval()
        # 使用 PyTorch 将数据转移到指定的设备（self.device），并通过神经网络计算输出。
        with torch.no_grad():
            vdata, edata, conn, udata = hist_buffer[0]
            vdata = torch.tensor(vdata, device=self.device)
            edata = torch.tensor(edata, device=self.device)
            udata = torch.tensor(udata, device=self.device)
            conn = torch.tensor(conn, device=self.device)
            vout, eout, _ = self.net(x=vdata, edge_index=conn, edge_attr=edata, u=udata)
            # 选择变量节点的输出（节点类型为 NODE_TYPE_VAR）
            res = vout[vdata[:, NODE_TYPE_COL] == NODE_TYPE_VAR]

            if self.debug:
                self.qs_buffer.append(res.flatten().cpu().numpy())
            return res

    # 根据 epsilon-greedy 策略决定智能体采取的动作
    def act(self, hist_buffer, eps):
        # 评估模式且动作池启用时，使用动作池策略
        if self.pool_enabled and not self.net.training:
            if not self.action_pool:
                qs = self.forward(hist_buffer)
                if qs.numel() > 0:
                    flat_qs = qs.flatten()
                    topk_k = min(self.pool_size, flat_qs.numel())
                    topk_indices = torch.topk(flat_qs, topk_k).indices.tolist()
                    self.action_pool.extend(topk_indices)
            # 弹出动作并检查其有效性
            while self.action_pool:
                candidate = self.action_pool.popleft()
                if self._check_valid(hist_buffer[-1], candidate):
                    return candidate

        # 训练模式或动作池被禁用/耗尽时，使用标准的 epsilon-greedy 策略
        if np.random.random() < eps:
            # 随机选择一个动作
            vfeat = hist_buffer[-1][0]
            # 找到所有变量节点（节点类型为 NODE_TYPE_VAR）
            unassigned_vars_mask = vfeat[:, NODE_TYPE_COL] == NODE_TYPE_VAR
            unassigned_var_indices = np.where(unassigned_vars_mask)[0]

            if len(unassigned_var_indices) == 0:
                return -1 # 没有可选择的动作，让 MiniSat 决策

            # 从所有未赋值变量中随机选择一个，然后随机选择极性
            rand_var_idx = np.random.randint(0, len(unassigned_var_indices))  # 在变量列表中的索引
            random_polarity = np.random.randint(0, 2)  # 0 或 1 (负极性或正极性)
            action = rand_var_idx * 2 + random_polarity  # 编码为动作
            return int(action)
        else:
            # 贪心选择：计算 Q 值并选择最优动作
            qs = self.forward(hist_buffer)
            if qs.numel() == 0:
                return -1 # 没有可选择的动作
            return self.choose_actions(qs)

    def _check_valid(self, obs, action):
        # 检查动作是否对应一个当前未赋值的变量
        vfeat = obs[0]  # vertex features
        
        # 找到所有变量节点（节点类型为 NODE_TYPE_VAR）
        var_mask = vfeat[:, NODE_TYPE_COL] == NODE_TYPE_VAR
        num_vars_in_graph = var_mask.sum()
        
        # 简化的检查：动作索引直接对应图中的变量节点
        # action // 2 应该对应图中第几个变量节点
        action_var_idx = action // 2
        
        # 检查动作变量索引是否在有效范围内
        return action_var_idx < num_vars_in_graph

    def choose_actions(self, qs):
        return qs.flatten().argmax().item()
