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
import pickle
import yaml

from gqsat.utils import build_eval_argparser, evaluate
from gqsat.models import SatModel
from gqsat.agents import GraphAgent

import os
import time
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

# 添加函数：同时打印到控制台和日志文件
def log_print(log_file, message):
    print(message)
    log_file.write(message + "\n")

if __name__ == "__main__":
    parser = build_eval_argparser()
    eval_args = parser.parse_args()
    set_seed(eval_args.seed)
    
    # 创建日志目录
    if not os.path.exists(eval_args.logdir):
        os.makedirs(eval_args.logdir)
    
    # 创建日志文件
    log_filename = os.path.join(eval_args.logdir, f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    log_file = open(log_filename, "w")
    
    with open(os.path.join(eval_args.model_dir, "status.yaml"), "r") as f:
        train_status = yaml.load(f, Loader=yaml.Loader)
        args = train_status["args"]

    # use same args used for training and overwrite them with those asked for eval
    for k, v in vars(eval_args).items():
        setattr(args, k, v)

    args.device = (
        torch.device("cpu")
        if args.no_cuda or not torch.cuda.is_available()
        else torch.device("cuda")
    )
    net = SatModel.load_from_yaml(os.path.join(args.model_dir, "model.yaml")).to(
        args.device
    )

    # modify core steps for the eval as requested
    if args.core_steps != -1:
        # -1 if use the same as for training
        net.steps = args.core_steps

    net.load_state_dict(
        torch.load(os.path.join(args.model_dir, args.model_checkpoint)), strict=False
    )

    agent = GraphAgent(net, args)

    st_time = time.time()
    scores, eval_metadata, _ = evaluate(agent, args)
    end_time = time.time()

    # 保存评估结果到pickle文件
    pickle_path = os.path.join(eval_args.logdir, "eval_results.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(scores, f)
        
    # 记录模型信息
    log_print(log_file, f"Model directory: {args.model_dir}")
    log_print(log_file, f"Model checkpoint: {args.model_checkpoint}")
    log_print(log_file, f"Evaluation is over. It took {end_time - st_time:.2f} seconds for the whole procedure")
    
    highest_relative_score = -float('inf')

    # 记录详细结果
    for pset, pset_res in scores.items():
        res_list = [el for el in pset_res.values()]
        log_print(log_file, f"Results for {pset}")
        log_print(log_file, f"median_relative_score: {np.nanmedian(res_list)}, mean_relative_score: {np.mean(res_list)}")
        # 更新最高relative_score
        highest_relative_score = max(highest_relative_score, np.nanmax(res_list))
    
    # 输出最高的relative_score
    log_print(log_file, f"Highest relative_score during evaluation: {highest_relative_score}")
    
    # 记录评估参数
    log_print(log_file, "\nEvaluation parameters:")
    for k, v in vars(args).items():
        if k in ["eval_problems_paths", "test_time_max_decisions_allowed", "core_steps", "eps_final", "eval_time_limit", "action_pool_size"]:
            log_print(log_file, f"  {k}: {v}")
    
    # 关闭日志文件
    log_file.close()
    
    print(f"评估结果已保存到: {log_filename}")