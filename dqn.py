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

import numpy as np
import torch

import os
from collections import deque
import pickle
import copy
import yaml

from gqsat.utils import build_argparser, evaluate, make_env
from gqsat.models import EncoderCoreDecoder, SatModel
from gqsat.agents import GraphAgent
from gqsat.learners import GraphLearner
from gqsat.buffer import ReplayGraphBuffer

from tensorboardX import SummaryWriter
from collections import defaultdict
from datetime import datetime
import logging

# 这段代码实现了一个强化学习的训练过程，使用图神经网络来解决SAT类问题。包括了以下几个重要步骤：
    # 参数解析和训练恢复。
    # 智能体（GraphAgent）与环境交互。
    # 使用 GraphLearner 执行模型训练。
    # 定期保存模型、优化器状态和训练进度。
    # 定期评估模型的性能，并记录到 TensorBoard 中

# 保存训练的状态，包括模型、优化器、学习率调度器、经验回放缓冲区以及训练进度等信息
def save_training_state(
    model,
    learner,
    episodes_done,
    transitions_seen,
    best_eval_so_far,
    best_checkpoint,   #added by cl 传入最佳检查点路径字典
    args,
    in_eval_mode=False,
):
    # save the model
    model_path = os.path.join(args.logdir, f"model_{learner.step_ctr}.chkp")
    torch.save(model.state_dict(), model_path)

    # save the experience replay
    buffer_path = os.path.join(args.logdir, "buffer.pkl")

    with open(buffer_path, "wb") as f:
        pickle.dump(learner.buffer, f)

    # save important parameters
    train_status = {
        "step_ctr": learner.step_ctr,
        "latest_model_name": model_path,
        "buffer_path": buffer_path,
        "args": args,
        "episodes_done": episodes_done,
        "logdir": args.logdir,
        "transitions_seen": transitions_seen,
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "optimizer_class": type(learner.optimizer),
        # "best_eval_so_far": best_eval_so_far,
        "best_eval_so_far": {k: float(v) for k, v in best_eval_so_far.items()}, # added by cl转换为 Python float
        "best_checkpoint": dict(best_checkpoint),  #added by cl记录最佳检查点路径
        "scheduler_class": type(learner.lr_scheduler),
        "scheduler_state_dict": learner.lr_scheduler.state_dict(),
        "in_eval_mode": in_eval_mode,
    }
    status_path = os.path.join(args.logdir, "status.yaml")

    with open(status_path, "w") as f:
        yaml.dump(train_status, f, default_flow_style=False)

    return status_path

# 计算epsilon-greedy策略中的 epsilon 值
def get_annealed_eps(n_trans, args):
    if n_trans < args.init_exploration_steps:
        return args.eps_init
    if n_trans > args.eps_decay_steps:
        return args.eps_final
    else:
        assert n_trans - args.init_exploration_steps >= 0
        return (args.eps_init - args.eps_final) * (
            1 - (n_trans - args.init_exploration_steps) / args.eps_decay_steps
        ) + args.eps_final

# 将字符串表示的激活函数转换为 PyTorch 中的激活函数类
def arg2activation(activ_str):
    if activ_str == "relu":
        return torch.nn.ReLU
    elif activ_str == "tanh":
        return torch.nn.Tanh
    elif activ_str == "leaky_relu":
        return torch.nn.LeakyReLU
    else:
        raise ValueError("Unknown activation function")

import random
#added by cl
def set_seed(seed):
    # Python随机模块
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保卷积结果确定性（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 主函数，用于训练图神经网络模型
if __name__ == "__main__":
    # 命令行参数解析
    parser = build_argparser()#构造命令行参数解析器，获取超参数、路径、配置信息等
    args = parser.parse_args() #解析命令行参数。将结果放在 args，后续代码通过args来配置  
    set_seed(args.seed)
    # 根据是否有可用的GPU，选择计算设备
    args.device = (
        torch.device("cpu")
        if args.no_cuda or not torch.cuda.is_available()
        else torch.device("cuda")
    )
    # 创建 SummaryWriter 以记录训练日志，设置日志记录器。
    # 该日志保存的位置在../runs/xx下（即本项目的上一级目录下的runs/xx中）
    # tips：命令行输入：tensorboard --logdir=log 可实现训练可视化
        # 创建新的日志子目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.logdir, f"seed{args.seed}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    args.logdir = log_dir

    # 初始化SummaryWriter和日志记录
    writer = SummaryWriter(args.logdir)
    log_file_path = os.path.join(args.logdir, "training.log")
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if args.status_dict_path:
        # 如果提供了 status_dict_path，则要从之前保存的状态恢复训练

        # 从指定路径读取保存的训练状态（利用yaml反序列化）
        with open(args.status_dict_path, "r") as f:
            train_status = yaml.load(f, Loader=yaml.Loader)

        # 提取训练状态中的评估模式标志 (in_eval_mode)
        eval_resume_signal = train_status["in_eval_mode"]

        # 恢复 args 配置
        args = train_status["args"]
        args.logdir = log_dir # 确保使用新的日志目录
        # 加载模型结构,并加载保存的权重
        net = SatModel.load_from_yaml(os.path.join(args.logdir, "model.yaml")).to(
            args.device
        )
        net.load_state_dict(torch.load(train_status["latest_model_name"]))
        # 创建一个 target_net，是一个深拷贝模型，用于目标网络。
        target_net = SatModel.load_from_yaml(
            os.path.join(args.logdir, "model.yaml")
        ).to(args.device)
        target_net.load_state_dict(net.state_dict())

        # 加载之前保存的缓冲区（ReplayBuffer），用于存储经验和回放
        with open(train_status["buffer_path"], "rb") as f:
            buffer = pickle.load(f)

        # 创建强化学习学习者 learner，并恢复之前的步数计数器
        learner = GraphLearner(net, target_net, buffer, args)
        learner.step_ctr = train_status["step_ctr"]

        # 恢复优化器并加载其状态字典
        learner.optimizer = train_status["optimizer_class"](
            net.parameters(), lr=args.lr
        )
        learner.optimizer.load_state_dict(train_status["optimizer_state_dict"])
        
        # 恢复学习率调度器并加载其状态字典
        learner.lr_scheduler = train_status["scheduler_class"](
            learner.optimizer, args.lr_scheduler_frequency, args.lr_scheduler_gamma
        )
        learner.lr_scheduler.load_state_dict(train_status["scheduler_state_dict"])

        # load misc training status params
        # 恢复已处理的过渡数量 n_trans 和已完成的训练轮次 ep
        n_trans = train_status["transitions_seen"]
        ep = train_status["episodes_done"]

        # 创建一个训练环境（SAT问题求解环境），并开始训练
        env = make_env(args.train_problems_paths, args, test_mode=False)
        
        # 创建一个智能体（agent），它基于当前的模型（net）来做出决策
        agent = GraphAgent(net, args)
        
        # 恢复评估过程中的最佳评估结果
        best_eval_so_far = train_status["best_eval_so_far"]

    else:
        # 没有恢复训练，则从头开始训练
        # training mode, learning from scratch or continuing learning from some previously trained model
        

        # 设置模型保存路径
        model_save_path = os.path.join(args.logdir, "model.yaml")

        # 初始化一个字典（存储的是键值对），用于记录每个数据集的最佳评估成绩，初始值为-1
        # best_eval_so_far = (
        #     {args.eval_problems_paths: -1}
        #     if not args.eval_separately_on_each
        #     else {k: -1 for k in args.eval_problems_paths.split(":")}
        # )
        #added by cl
        best_eval_so_far = defaultdict(lambda: -1)
        best_checkpoint = defaultdict(str)  # 新增：记录每个 sc_key 对应的最佳检查点路径
        # 创建训练环境
        # TODO 这句话返回时会报错没有元数据文件
        env = make_env(args.train_problems_paths, args, test_mode=False)
        
        if args.model_dir is not None:
            # 如果提供了已有模型的路径，则加载该模型进行继续训练
            net = SatModel.load_from_yaml(
                os.path.join(args.model_dir, "model.yaml")
            ).to(args.device)
            net.load_state_dict(
                torch.load(os.path.join(args.model_dir, args.model_checkpoint))
            )
        else:
            # 如果没有提供已有模型路径，则从头开始创建一个新的模型
            net = EncoderCoreDecoder(
                (env.vertex_in_size, env.edge_in_size, env.global_in_size),
                core_out_dims=(
                    args.core_v_out_size,
                    args.core_e_out_size,
                    args.core_e_out_size,
                ),
                out_dims=(2, None, None),
                core_steps=args.core_steps,
                dec_out_dims=(
                    args.decoder_v_out_size,
                    args.decoder_e_out_size,
                    args.decoder_e_out_size,
                ),
                encoder_out_dims=(
                    args.encoder_v_out_size,
                    args.encoder_e_out_size,
                    args.encoder_e_out_size,
                ),
                save_name=model_save_path,
                e2v_agg=args.e2v_aggregator,
                n_hidden=args.n_hidden,
                hidden_size=args.hidden_size,
                activation=arg2activation(args.activation),
                independent_block_layers=args.independent_block_layers,
            ).to(args.device)
        
        # 打印模型结构，并创建目标网络 target_net
        print(str(net))

        target_net = copy.deepcopy(net)
        # 创建回放缓冲区和智能体
        buffer = ReplayGraphBuffer(args, args.buffer_size)
        agent = GraphAgent(net, args)
        # 初始化过渡次数、训练轮次、学习者和评估信号
        n_trans = 0
        ep = 0
        learner = GraphLearner(net, target_net, buffer, args)
        eval_resume_signal = False
    print(args.__str__())
    loss = None
    
    # 主训练循环
    while learner.step_ctr < args.batch_updates:
        # 开始训练过程，直到达到预定的批量更新次数

        # 初始化回报 ret，并重置环境，检查环境是否已解决
        ret = 0
            # added by cl
            # obs包括了4样东西： 顶点特征.边特征.边连接关系.全局特征s
        obs = env.reset()
            # prev: obs = env.reset(args.train_time_max_decisions_allowed)
        done = env.isSolved

        if args.history_len > 1:
            raise NotImplementedError(
                "History len greater than one is not implemented for graph nets."
            )
        # 初始化历史缓冲区，用于存储过去的观察
        hist_buffer = deque(maxlen=args.history_len)
        for _ in range(args.history_len):
            hist_buffer.append(obs)

        # 初始化步数和保存标志
        ep_step = 0
        save_flag = False

        while not done:
            # 在问题未解决的情况下，智能体基于历史信息选择动作
            annealed_eps = get_annealed_eps(n_trans, args)
            action = agent.act(hist_buffer, eps=annealed_eps)

            # 执行选定的动作，并将新的过渡添加到缓冲区
                # added by cl
            next_obs, r, done, _ = env.step(action)
            buffer.add_transition(obs, action, r, done)

            # 更新当前观察，并将其添加到历史缓冲区，同时增加回报
            obs = next_obs
            hist_buffer.append(obs)
            ret += r
            
            # 定期执行学习步骤
            if (not n_trans % args.step_freq) and (
                buffer.ctr > max(args.init_exploration_steps, args.bsize + 1)
                or buffer.full
            ):
                step_info = learner.step() #在执行learner.step()时 step_ctr会自增1
                if annealed_eps is not None:
                    step_info["annealed_eps"] = annealed_eps

                # we increment the step_ctr in the learner.step(), that's why we need to do -1 in tensorboarding
                # we do not need to do -1 in checking for frequency since 0 has already passed

                # 每隔一定步骤保存训练状态
                if not learner.step_ctr % args.save_freq:
                    # save the exact model you evaluated and make another save after the episode ends
                    # to have proper transitions in the replay buffer to pickle
                    status_path = save_training_state(
                        net,
                        learner,
                        ep - 1,
                        n_trans,
                        best_eval_so_far,
                        best_checkpoint,  # added by cl新增参数
                        args,
                        in_eval_mode=eval_resume_signal,
                    )
                    save_flag = True
                # 在每个训练周期结束后清理缓存
                if learner.step_ctr % 100 == 0:
                    torch.cuda.empty_cache()
                # 定期评估模型性能
                #评估条件： 训练步数是评估频率的整数倍 or 收到评估恢复信号
                if (
                    args.env_name == "sat-v0" and not learner.step_ctr % args.eval_freq
                ) or eval_resume_signal:
                    scores, _, eval_resume_signal = evaluate(
                        agent, args, include_train_set=False
                    )
                    # 处理评估结果， 更新最佳评估分数记录
                    for sc_key, sc_val in scores.items():
                        # list can be empty if we hit the time limit for eval
                        if len(sc_val) > 0:
                            res_vals = [el for el in sc_val.values() if not np.isnan(el)]  # 过滤 NaN
                            # res_vals = [el for el in sc_val.values()]
                            if len(res_vals) == 0:
                                continue  # 跳过全 NaN
                            median_score = np.nanmedian(res_vals)
                            # added by cl 清理 sc_key 中的非法字符（关键步骤！）
                            safe_sc_key = sc_key.lstrip('/').replace('/', '_').replace(' ', '_').replace(':', '_')
                            # def sanitize_key(key):
                            #     # 替换所有非字母数字字符为下划线
                            #     return re.sub(r'[^a-zA-Z0-9_]', '_', key).lstrip('_')

                            # safe_sc_key = sanitize_key(sc_key)  # 使用统一清理函数
                            # added by cl
                            if (
                                best_eval_so_far[safe_sc_key] < median_score
                                or best_eval_so_far[safe_sc_key] == -1
                            ):
                                best_eval_so_far[safe_sc_key] = float(median_score)  # 转换为 Python float
                                best_checkpoint[safe_sc_key] = f"model_{learner.step_ctr}.chkp"  # 记录检查点路径
                                #added by cl 输出日志信息
                                logging.info(f"[best_checkpoint] {safe_sc_key}", best_checkpoint[safe_sc_key])
                            writer.add_scalar(
                                f"data/median relative score: {safe_sc_key}",
                                np.nanmedian(res_vals),
                                learner.step_ctr - 1,
                            )
                            writer.add_scalar(
                                f"data/mean relative score: {safe_sc_key}",
                                np.nanmean(res_vals),
                                learner.step_ctr - 1,
                            )
                            writer.add_scalar(
                                f"data/max relative score: {safe_sc_key}",
                                np.nanmax(res_vals),
                                learner.step_ctr - 1,
                            )
                    for k, v in best_eval_so_far.items():
                        writer.add_scalar(k, v, learner.step_ctr - 1)
                # 记录学习步骤信息到 TensorBoard。
                for k, v in step_info.items():
                    writer.add_scalar(k, v, learner.step_ctr - 1)

                writer.add_scalar("data/num_episodes", ep, learner.step_ctr - 1)
            
            # 更新过渡次数和训练步骤
            n_trans += 1 #每执行一步（决策），过渡次数+1
            ep_step += 1

        # 记录每个回合的回报、步数和最后奖励，并打印信息
        writer.add_scalar("data/ep_return", ret, learner.step_ctr - 1)
        writer.add_scalar("data/ep_steps", env.step_ctr, learner.step_ctr - 1)
        writer.add_scalar("data/ep_last_reward", r, learner.step_ctr - 1)
        print(f"Episode {ep + 1}: Return {ret}.")
        ep += 1
        # 如果训练过程中需要保存状态，执行保存操作
        if save_flag:
            status_path = save_training_state(
                net,
                learner,
                ep - 1,
                n_trans,
                best_eval_so_far,
                best_checkpoint,
                args,
                in_eval_mode=eval_resume_signal,
            )
            save_flag = False
