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
from gqsat.utils import batch_graphs
# 该类的作用是实现一个图结构的 Replay Buffer，用于存储智能体的状态转移（观察、动作、奖励和是否完成）。
# 通过 add_transition 方法将每次交互的数据存储进缓存
# 使用 sample 方法从中随机抽取一批数据进行训练
# batch 方法将观察数据转换为适合神经网络处理的图格式。
# 这个缓存机制常用于强化学习中的经验回放，以提高训练的效率和稳定性。

class ReplayGraphBuffer:
    # 构造函数接收两个参数：args和size。args是一个参数字典，size是缓冲区的大小。
    def __init__(self, args, size):

        self.ctr = 0        #计数器，用来记录当前存储位置。初始值为 0。
        self.full = False   #标志位，用来表示缓冲区是否已满。初始值为 False。
        self.size = size    #缓冲区的大小。size 为缓冲区的大小。
        self.device = args.device   #用于存储数据的设备（CPU 或 GPU）
        self.dones = torch.ones(size) # 存储每个经验的“完成”标志。使用torch.ones初始化，每个元素代表是否已经结束。
        self.rewards = torch.zeros(size)#存储每个经验的奖励值。使用 torch.zeros初始化，表示奖励的初始值为 0。
        self.actions = torch.zeros(size, dtype=torch.long) #存储每个经验的动作。使用 torch.zeros初始化，表示动作初始值为 0。
        # dtype=object allows to store references to objects of arbitrary size
        self.observations = np.zeros((size, 4), dtype=object)#存储每个经验的观测值（即状态信息）。4是观察数据的维度，dtype=object允许存储不同大小的对象。

    # 用于将一个新的经验（状态、动作、奖励、是否终止）添加到缓存中。
    def add_transition(self, obs, a, r_next, done_next):
        #obs: 当前的观察状态，通常包含多个元素，例如节点数据、边数据等。
        # a: 执行的动作。
        #r_next: 执行该动作后获得的奖励。
        #done_next: 是否到达终止状态（例如在游戏中是否结束）。
        
        # self.ctr 为当前存储位置。将动作、奖励、是否终止信息存储到对应的位置。
        self.dones[self.ctr] = int(done_next)
        self.rewards[self.ctr] = r_next
        self.actions[self.ctr] = a

        # obs 为当前的观察状态，通常包含多个元素，例如节点数据、边数据等。
        # should be vertex_data, edge_data, connectivity, global
        for el_idx, el in enumerate(obs):
            self.observations[self.ctr][el_idx] = el

        if (self.ctr + 1) % self.size == 0:
            self.ctr = 0
            self.full = True
        else:
            self.ctr += 1

    # 从缓存中随机选择一批数据，供模型进行训练
    def sample(self, batch_size):
        # to be able to grab the next, we use -1
        curr_size = self.ctr - 1 if not self.full else self.size - 1
        idx = np.random.choice(range(0, curr_size), batch_size)
        return (
            self.batch(self.observations[idx]),
            self.actions[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.batch(self.observations[idx + 1]),
            1.0 - self.dones[idx].to(self.device),
        )

    # 将一批观察数据转换为图结构数据，并将其转换为 PyTorch 张量
    def batch(self, obs):
        return batch_graphs(
            [[torch.tensor(i, device=self.device) for i in el] for el in obs],
            self.device,
        )
