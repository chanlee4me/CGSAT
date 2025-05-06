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
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX
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
            res = vout[vdata[:, VAR_ID_IDX] == 1]# vout 是节点的输出，最后选取符合条件的节点

            if self.debug:
                self.qs_buffer.append(res.flatten().cpu().numpy())
            return res

    # 根据 epsilon-greedy 策略决定智能体采取的动作
    def act(self, hist_buffer, eps):
        # If action pool enabled (only in evaluation), try pool
        if self.pool_enabled:
            # refill pool if empty
            if not self.action_pool:
                qs = self.forward(hist_buffer)
                flat_qs = qs.flatten()
                # get top-K actions
                topk = torch.topk(flat_qs, min(self.pool_size, flat_qs.numel())).indices.tolist()
                self.action_pool = deque(topk)
            # pop and return first valid action
            while self.action_pool:
                action = self.action_pool.popleft()
                if self._check_valid(hist_buffer[-1], action):
                    return int(action)
            # if pool exhausted or no valid actions, fall through to default
        # epsilon-greedy fallback
        if np.random.random() < eps:
            vars_to_decide = np.where(hist_buffer[-1][0][:, VAR_ID_IDX] == 1)[0]
            acts = [a for v in vars_to_decide for a in (v * 2, v * 2 + 1)]
            return int(np.random.choice(acts))
        else:
            qs = self.forward(hist_buffer)
            return self.choose_actions(qs)

    def _check_valid(self, obs, action):
        # Map flat action to variable index and check if variable still unassigned
        # obs[0] is vertex features array; VAR_ID_IDX marks variable nodes
        vfeat = obs[0]
        # determine current variable nodes
        var_mask = vfeat[:, VAR_ID_IDX] == 1
        # number of variables available
        num_vars = int(np.sum(var_mask))
        var_idx = action // 2
        return 0 <= var_idx < num_vars

    def choose_actions(self, qs):
        return qs.flatten().argmax().item()
