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
import gym, minisat
import time
import glob
import math
import multiprocessing
from collections import defaultdict
from os import path

from gqsat.utils import build_argparser
from gqsat.agents import MiniSATAgent

TIMEOUT_SECONDS = 3600

def solve_one_problem(task):
    problem_path, with_restarts, args = task
    # Each process needs to create its own environment and agent
    try:
        env = gym.make(
            "sat-v0",
            args=args,
            # Pass the directory of the problem file
            problems_paths=path.dirname(problem_path),
            test_mode=True,
            with_restarts=with_restarts,
            max_decisions_cap=args.test_time_max_decisions_allowed
        )
        # Force the environment to load only the specific problem for this task
        env.all_problems = [problem_path]
        env.test_to = 1

        agent = MiniSATAgent()
        
        observation = env.reset()
        start_time = time.perf_counter()
        done = False
        
        while not done:
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > TIMEOUT_SECONDS:
                print(f"Timeout (> {TIMEOUT_SECONDS}s) on problem {path.basename(problem_path)} with_restarts={with_restarts}")
                return (problem_path, with_restarts, -1, TIMEOUT_SECONDS)

        elapsed_time = time.perf_counter() - start_time
        print(
            f'Finished {path.basename(problem_path)} with_restarts={with_restarts}, steps {env.step_ctr}, time {elapsed_time:.4f}s'
        )
        return (problem_path, with_restarts, env.step_ctr, elapsed_time)

    except Exception as e:
        print(f"Error processing {problem_path}: {e}")
        return (problem_path, with_restarts, -2, -2) # Indicate error
    finally:
        if 'env' in locals():
            env.close()

def main():
    parser = build_argparser()
    args = parser.parse_args()

    # 1. Find all problem files
    all_problems = []
    for pdir in args.eval_problems_paths.split(':'):
        if path.isdir(pdir):
            # Use glob to find all .cnf files recursively
            all_problems.extend(glob.glob(path.join(pdir, '**', '*.cnf'), recursive=True))
    
    if not all_problems:
        print("No .cnf files found in the specified paths.")
        return

    # Remove duplicates that might arise from overlapping paths
    all_problems = sorted(list(set(all_problems)))
    print(f"Found {len(all_problems)} unique problems to solve.")

    # 2. Create tasks for the multiprocessing pool
    tasks = []
    for problem in all_problems:
        tasks.append((problem, False, args)) # Task for no-restarts
        tasks.append((problem, True, args))  # Task for with-restarts

    # 3. Run tasks in parallel
    # Use 2/3 of available CPU cores, with a minimum of 1
    num_workers = math.ceil(os.cpu_count() * 2 / 3)
    if num_workers == 0:
        num_workers = 1
    
    print(f"Starting parallel solving with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        raw_results = pool.map(solve_one_problem, tasks)

    # 4. Process and aggregate results
    # a value of [steps_no_restarts, time_no_restarts, steps_with_restarts, time_with_restarts]
    results = defaultdict(lambda: [-3, -3, -3, -3]) # Default for missing data
    for res in raw_results:
        if res:
            p_path, wr, steps, time = res
            if wr: # with restarts
                results[p_path][2] = steps
                results[p_path][3] = time
            else: # no restarts
                results[p_path][0] = steps
                results[p_path][1] = time

    return results, args

if __name__ == "__main__":
    results, args = main()
    if not results:
        print("No results to write.")
    else:
        # Write results to a single METADATA file in the first specified directory
        output_dir = args.eval_problems_paths.split(':')[0]
        output_file = os.path.join(output_dir, "METADATA_WITH_TIME")
        print(f"Writing results to {output_file}")
        
        with open(output_file, "w") as f:
            # Write header
            f.write("pname,steps_no_restarts,time_no_restarts,steps_with_restarts,time_with_restarts\n")
            for el in sorted(results.keys()):
                pname = path.basename(el)
                res_list = results[el]
                f.write(f"{pname},{res_list[0]},{res_list[1]},{res_list[2]},{res_list[3]}\n")
        print("Done.")
