#!/usr/bin/env cs_python
# Copyright 2024 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test compile output dir')
parser.add_argument('--cmaddr', help='IP:port for CS system')
args = parser.parse_args()

# Get matrix dimensions from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

N = int(compile_data['params']['N'])
d_k = int(compile_data['params']['d_k'])

# Construct Q, K, V
Q = np.random.rand(N * d_k).astype(np.float32)
K = np.random.rand(N * d_k).astype(np.float32)
V = np.random.rand(N * d_k).astype(np.float32)

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbols for Q, K, V, result on device
Q_symbol = runner.get_id('Q')
K_symbol = runner.get_id('K')
V_symbol = runner.get_id('V')
result_symbol = runner.get_id('result')

# Load and run the program
runner.load()
runner.run()

# Copy Q, K, V to device
runner.memcpy_h2d(
    Q_symbol,
    Q,
    0,
    0,
    1,
    1,
    N * d_k,
    streaming=False,
    order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT,
    nonblock=False,
)
runner.memcpy_h2d(
    K_symbol,
    K,
    0,
    0,
    1,
    1,
    N * d_k,
    streaming=False,
    order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT,
    nonblock=False,
)
runner.memcpy_h2d(
    V_symbol,
    V,
    0,
    0,
    1,
    1,
    N * d_k,
    streaming=False,
    order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT,
    nonblock=False,
)

# Launch the compute function on device
runner.launch('compute', nonblock=False)

# Copy result back from device
result = np.zeros([N * d_k], dtype=np.float32)
runner.memcpy_d2h(
    result,
    result_symbol,
    0,
    0,
    1,
    1,
    N * d_k,
    streaming=False,
    order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT,
    nonblock=False,
)

# Stop the program
runner.stop()

print("SUCCESS!")