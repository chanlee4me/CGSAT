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

import os
import gym, minisat  # you need the latter to run __init__.py and register the environment.
from collections import defaultdict
from gqsat.utils import build_argparser
from gqsat.agents import MiniSATAgent

DEBUG_ROLLOUTS = 10  # if --debug flag is present, run this many of rollouts, not the whole problem folder

# 以下代码是调用原始 MiniSAT 求解器进行问题求解，记录其性能数据，并写入每个问题目录下的 METADATA 文件
# METADATA 文件格式：每行记录一个问题的无重启决策步数和带重启决策步数。

#阶段1：测试阶段
def main():
    #解析命令行参数
    parser = build_argparser()
    args = parser.parse_args()
    # key is the name of the problem file, value is a list with two values [minisat_steps_no_restarts, minisat_steps_with_restarts]
    results = defaultdict(list) ## 结果存储：{问题路径: [无重启步数, 带重启步数]}
    # 创建 sat-v0 强化学习环境，分别测试两种模式（minisat带重启和不带重启）
    for with_restarts in [False, True]:
        env = gym.make(
            "sat-v0",
            args=args,
            problems_paths=args.eval_problems_paths,
            test_mode=True,
            with_restarts=with_restarts,
            max_decisions_cap=args.test_time_max_decisions_allowed
        )
        # 使用 MiniSATAgent 作为求解代理（封装 MiniSAT 算法）
        agent = MiniSATAgent()
        print(f"Testing agent {agent}... with_restarts is set to {with_restarts}")
        pr = 0 # Problem Rollout 计数器
        while env.test_to != 0 or pr == 0: # 遍历所有问题
            observation = env.reset() # 重置环境，加载新问题
            done = False
            while not done: # 求解当前问题
                action = agent.act(observation) # MiniSAT 内部选择动作
                # added by cl
                observation, reward, done, info = env.step(action)
                # observation, reward, done, info = env.step(action, dummy=True)
            # 记录结果
            print(
                f'Rollout {pr+1}, steps {env.step_ctr}, num_restarts {info["num_restarts"]}.'
            )
            results[env.curr_problem].append(env.step_ctr)
            pr += 1
            if args.debug and pr >= DEBUG_ROLLOUTS: # 调试模式下提前终止
                break
        env.close() # 关闭环境释放资源
    return results, args


from os import path

if __name__ == "__main__":
    results, args = main() # 运行测试，获取结果和参数
    for pdir in args.eval_problems_paths.split(":"):# 遍历所有评估路径（按冒号分隔）
        with open(os.path.join(pdir, "METADATA"), "w") as f:
            for el in sorted(results.keys()):
                cur_dir, pname = path.split(el) # 分离目录和文件名
                if path.realpath(pdir) == path.realpath(cur_dir): # 仅处理属于当前目录的问题
                    # no restarts/with restarts
                    # METADATA 文件格式：每行记录一个问题的无重启步数和带重启步数。
                    f.write(f"{pname},{results[el][0]},{results[el][1]}\n")
