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

from torch import nn
from torch_scatter import scatter_max
import torch
from torch.optim.lr_scheduler import StepLR
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX

# 通过GNN进行强化学习中的 Q 值更新。
# 它使用经验回放来采样状态转移，并通过优化器和目标网络计算目标 Q 值
# 每次 step 调用都会执行一次训练步骤，其中包括计算 Q 值、损失、反向传播和网络更新
# 目标网络每隔一定步数更新一次，以确保稳定性


# 实现了基于图神经网络（GNN）的深度Q学习（DQN）算法
# 用于训练一个强化学习（RL）代理（Agent），使其能够解决图结构问题（如 SAT 问题）
class GraphLearner:
    #net: 当前的图神经网络模型，即当前智能体的网络
    #target: 目标网络，用于DQN中的目标网络更新
    #buffer: 经验回放缓冲区，用于存储历史的状态转移数据
    #args: 参数字典，包含了训练过程中的超参数
    def __init__(self, net, target, buffer, args):
        self.net = net          # 主网络：负责预测当前状态的 Q 值
        self.target = target    # 目标网络：延迟更新的网络，用于计算目标 Q 值，减少训练波动
        self.target.eval()      # 目标网络设为评估模式（不计算梯度）
        # 优化器与学习率调度器
        #使用 Adam 优化器来更新神经网络的参数，学习率 lr 由 args.lr 指定。
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        #学习率调度器，每隔一定步数（由 args.lr_scheduler_frequency 指定）调整学习率
        #args.lr_scheduler_gamma 是学习率的衰减因子
        self.lr_scheduler = StepLR(
            self.optimizer, args.lr_scheduler_frequency, args.lr_scheduler_gamma
        )
        # 损失函数（MSE 或 Huber Loss）
        if args.loss == "mse":
            self.loss = nn.MSELoss()
        elif args.loss == "huber":
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError("Unknown Loss function.")
        # 参数与缓冲区
        self.bsize = args.bsize     # 批次大小
        self.gamma = args.gamma     # 折扣因子（用于计算未来奖励）
        self.buffer = buffer        # 经验回放缓冲区：存储历史经验（状态、动作、奖励等），支持随机采样以打破数据相关性。
        self.target_update_freq = args.target_update_freq  # 目标网络更新频率
        self.step_ctr = 0           # 训练步数计数器
        self.grad_clip = args.grad_clip # 梯度裁剪阈值（防止梯度爆炸）
        self.grad_clip_norm_type = args.grad_clip_norm_type # 梯度裁剪类型
        self.device = args.device   # 计算设备（CPU/GPU）
    # 根据主网络预测当前状态的 Q 值
    def get_qs(self, states):
        vout, eout, _ = self.net(
            x=states[0],          # 顶点特征（形状：[num_nodes, node_feature_dim]）
            edge_index=states[2],  # 边连接索引（形状：[2, num_edges]）
            edge_attr=states[1],   # 边特征（形状：[num_edges, edge_feature_dim]）
            v_indices=states[4],   # 顶点索引（用于恢复批次结构）
            e_indices=states[5],   # 边索引（用于恢复批次结构）
            u=states[6]           # 全局特征（形状：[batch_size, global_feature_dim]）
        )
        return vout[states[0][:, VAR_ID_IDX] == 1], states[3] # 返回可行动作对应的 Q 值
    # 根据目标网络计算目标 Q 值
    def get_target_qs(self, states):
        # 目标网络前向传播：结构与主网络相同，但输出需分离梯度（detach）
        vout, eout, _ = self.target(
            x=states[0],
            edge_index=states[2],
            edge_attr=states[1],
            v_indices=states[4],
            e_indices=states[5],
            u=states[6],
        )
        return vout[states[0][:, VAR_ID_IDX] == 1].detach(), states[3] # 返回可行动作对应的 Q 值
    # 训练步骤
    def step(self):
        # 1、从缓冲区采样一个批次大小的数据
        s, a, r, s_next, nonterminals = self.buffer.sample(self.bsize)
        # calculate the targets first to optimize the GPU memory

        # 2、计算目标 Q 值（禁止梯度传播）
        with torch.no_grad():
            target_qs, target_vertex_sizes = self.get_target_qs(s_next)
            idx_for_scatter = [
                [i] * el.item() * 2 for i, el in enumerate(target_vertex_sizes)# 每个图的变量节点数扩展为双倍（正负文字）
            ]
            idx_for_scatter = torch.tensor(
                [el for subl in idx_for_scatter for el in subl],
                dtype=torch.long,
                device=self.device,
            ).flatten()
            # 使用scatter_max找到每个图的最大Q值
            target_qs = scatter_max(target_qs.flatten(), idx_for_scatter, dim=0)[0]
            # 计算目标值：当前奖励 + γ * 下一状态的最大Q值（若状态非终止）
            targets = r + nonterminals * self.gamma * target_qs

        # 3、计算当前 Q 值并计算损失
        self.net.train()     # 设置主网络为训练模式
        qs, var_vertex_sizes = self.get_qs(s)   
        # qs.shape[1] values per node (same num of actions per node)

            # 构建索引以选择实际执行的动作对应的Q值
        gather_idx = (var_vertex_sizes * qs.shape[1]).cumsum(0).roll(1)
        gather_idx[0] = 0 # 第一个索引置零

            # 从扁平化的Q值中选择对应动作的Q值
        qs = qs.flatten()[gather_idx + a] # 选择实际执行动作的 Q 值

            # 计算损失（预测Q值与目标Q值的差距）
        loss = self.loss(qs, targets)

        # 4. 反向传播与参数更新
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()             # 反向计算梯度
            # 梯度裁剪（防止梯度爆炸）
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), self.grad_clip, norm_type=self.grad_clip_norm_type
        )
        self.optimizer.step() # 更新主网络参数

        # 5. 定期同步目标网络参数
        if not self.step_ctr % self.target_update_freq:
            self.target.load_state_dict(self.net.state_dict())
        # 6. 更新步数计数器与学习率
        self.step_ctr += 1

        # I do not know a better solution for getting the lr from the scheduler.
        # This will fail for different lrs for different layers.
        lr_for_the_update = self.lr_scheduler.get_last_lr()[0]   # 获取当前学习率

        self.lr_scheduler.step()        # 更新学习率（按StepLR策略）
        return {
            "loss": loss.item(),    # 损失值
            "grad_norm": grad_norm, # 梯度范数（裁剪后的）
            "lr": lr_for_the_update,# 当前学习率
            "average_q": qs.mean(), # 当前批次Q值的平均值
        }
