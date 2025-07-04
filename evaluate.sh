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

python3 evaluate.py \
  --logdir /4T/chenli/data/log_test \
  --env-name sat-v0 \
  --core-steps -1 \
  --eps-final 0.0 \
  --eval-time-limit 100000000000000 \
  --no_restarts \
  --test_time_max_decisions_allowed 500 \
  --eval-problems-paths /4T/chenli/data/uf100-430/test \
  --model-dir /4T/chenli/data/log/seed11_2025-06-26_19-47-19\
  --model-checkpoint model_1000.chkp \
  --action-pool-size 5
