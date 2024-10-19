# Our custom modules.
from Kernel import Kernel
from agent.flamingo.SA_ClientAgent import SA_ClientAgent as ClientAgent
from agent.flamingo.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from model.LatencyModel import LatencyModel
from util import util
from util import param

# Standard modules.
from datetime import timedelta
from math import floor
from nacl.encoding import Base64Encoder
from nacl.signing import SigningKey
import numpy as np
from os.path import exists
import pandas as pd
from sys import exit
from time import time

# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
# 一些配置文件需要额外的命令行参数，以便在粗略并行化过程中轻松控制代理或模拟超参数。
import argparse

parser = argparse.ArgumentParser(description='Detailed options for PPFL config.')
parser.add_argument('-a', '--clear_learning', action='store_true',
                    help='Learning in the clear (vs SMP protocol)')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-i', '--num_iterations', type=int, default=5,
                    help='Number of iterations for the secure multiparty protocol)')
parser.add_argument('-k', '--skip_log', action='store_true',
                    help='Skip writing agent logs to disk')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-n', '--num_clients', type=int, default=5,
                    help='Number of clients for the secure multiparty protocol)')
parser.add_argument('-o', '--neighborhood_size', type=int, default=1,
                    help='Number of neighbors a client has (should only enter the multiplication factor of log(n))')
parser.add_argument('--round_time', type=int, default=10,
                    help='Fixed time the server waits for one round')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('-p', '--parallel_mode', type=bool, default=True,
                    help='turn on parallel mode at server side')
parser.add_argument('-d', '--debug_mode', type=bool, default=False,
                    help='print debug info')
parser.add_argument('--config_help', action='store_true',
                    help='Print argument options for this config file')

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    exit()

# Historical date to simulate.  Required even if not relevant.
# 要模拟的历史日期。即使不相关，也是必需的。
historical_date = pd.to_datetime('2023-01-01')

# Requested log directory.
log_dir = args.log_dir
skip_log = args.skip_log

# Random seed specification on the command line.  Default: None (by clock).
# If none, we select one via a specific random method and pass it to seed()
# so we can record it for future use.  (You cannot reasonably obtain the
# automatically generated seed when seed() is called without a parameter.)
# 命令行上的随机种子规范。默认值：无（按时钟）。
# 如果没有，我们通过特定的随机方法选择一个，并将其传递给seed（），这样我们就可以记录下来以供将来使用。
# （当在没有参数的情况下调用seed（）时，您无法合理地获得自动生成的种子。）

# Note that this seed is used to (1) make any random decisions within this
# config file itself and (2) to generate random number seeds for the
# (separate) Random objects given to each agent.  This ensure that when
# the agent population is appended, prior agents will continue to behave
# in the same manner save for influences by the new agents.  (i.e. all prior
# agents still have their own separate PRNG sequence, and it is the same as
# before)
# 请注意，此种子用于（1）在此配置文件本身中做出任何随机决策，以及（2）为给定给每个代理的（单独的）random对象生成随机数种子。
# 这确保了当添加代理群体时，除了新代理的影响外，先前的代理将继续以相同的方式行事。
# （即所有先前的试剂仍然有自己单独的PRNG序列，与以前相同）

seed = args.seed
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

# Config parameter that causes util.util.print to suppress most output.
# 导致util.util.print抑制大部分输出的配置参数。
util.silent_mode = not args.verbose
num_clients = args.num_clients
# 客户端的邻居数量（只应输入log（n）的乘法因子）
neighborhood_size = args.neighborhood_size
round_time = args.round_time
num_iterations = args.num_iterations
parallel_mode = args.parallel_mode
debug_mode = args.debug_mode

if not param.assert_power_of_two(num_clients):
    raise ValueError("Number of clients must be power of 2")

print("Silent mode: {}".format(util.silent_mode))
print("Configuration seed: {}\n".format(seed))

# Since the simulator often pulls historical data, we use a real-world
# nanosecond timestamp (pandas.Timestamp) for our discrete time "steps",
# which are considered to be nanoseconds.  For other (or abstract) time
# units, one can either configure the Timestamp interval, or simply
# interpret the nanoseconds as something else.

# 由于模拟器经常提取历史数据，我们使用真实世界的纳秒时间戳（pandas.timestamp）来表示离散的时间“步长”，这些步长被认为是纳秒。
# 对于其他（或抽象）时间单位，可以配置时间戳间隔，也可以简单地将纳秒解释为其他时间单位。

# What is the earliest available time for an agent to act during the
# simulation?
# 代理在模拟过程中最早可以采取行动的时间是什么？
midnight = historical_date
kernelStartTime = midnight

# When should the Kernel shut down?
# 内核应该什么时候关闭？
kernelStopTime = midnight + pd.to_timedelta('2000:00:00')

# This will configure the kernel with a default computation delay
# (time penalty) for each agent's wakeup and recvMsg.  An agent
# can change this at any time for itself.  (nanoseconds)

# 这将为每个代理的唤醒和recvMsg配置默认的计算延迟（时间惩罚）。
# 代理人可以随时为自己改变这一点。（纳秒）
defaultComputationDelay = 1000000000 * 0.1  # ns

# IMPORTANT NOTE CONCERNING AGENT IDS: the id passed to each agent must:
# 关于代理id的重要提示：传递给每个代理的id必须：
#    1. be unique
#    2. equal its index in the agents list
#    2. 等于代理列表中的索引
# This is to avoid having to call an extra getAgentListIndexByID()
# 　这是为了避免调用额外的getAgentListIndexByID（）
# in the kernel every single time an agent must be referenced.
# 在内核中，每次必须引用代理时。


### Configure the Kernel.
### 配置内核。
kernel = Kernel("Base Kernel",
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))

### Obtain random state for whatever latency model will be used.
### 获取将使用的任何延迟模型的随机状态。
latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))

### Configure the agents.  When conducting "agent of change" experiments, the
### new agents should be added at the END only.
### 配置代理。在进行“变革剂”实验时，应仅在终点添加新的试剂。
agent_count = 0
agents = []
agent_types = []

### What accuracy multiplier will be used?
### 将使用什么精度倍数？
accy_multiplier = 100000

### What will be the scale of the shared secret?
### 共享秘密的规模有多大？
secret_scale = 1000000

### FOR MACHINE LEARNING APPLICATIONS: LOAD DATA HERE
### 对于机器学习应用程序：在此处加载数据
#
#   The data should be loaded only once (for speed).  Data should usually be
#   shuffled, split into training and test data, and passed to the client
#   parties.
#   数据应该只加载一次（为了速度）。数据通常应该被洗牌，分为训练和测试数据，并传递给客户端。
#
#   X_data should be a numpy array with column-wise features and row-wise
#   examples.  y_data should contain the same number of rows (examples)
#   and a single column representing the label.
#   X_data应该是一个具有列特征和行示例的numpy数组。
#   y_data应包含相同数量的行（示例）和表示标签的单个列。
#
#   Usually this will be passed through a function to shuffle and split
#   the data into the structures expected by the PPFL clients.  For example:
#   X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state = shuffle_seed)
#   通常，这将通过一个函数传递，以将数据洗牌并拆分为PPFL客户端所期望的结构。
#   例如：X_train，X_test，y_train，y_test=train_test_split（X_data，y_data，test_size=0.25，random_state=shuffle_sed）
#

# For the template, we generate some random numbers that could serve as example data.
# 对于模板，我们生成了一些可以作为示例数据的随机数。
# Note that this data is unlikely to be useful for learning.  It just fills the need
# for data of a specific format.
# 请注意，这些数据不太可能对学习有用。它只是满足了对特定格式数据的需求

# Randomly shuffle and split the data for training and testing.
# 随机洗牌和拆分数据以进行训练和测试。
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25)

#
#
### END OF LOAD DATA SECTION
### 负载数据段结束


### Configure a population of cooperating learning client agents.
### 配置一组协作学习客户端代理。
a, b = agent_count, agent_count + num_clients

client_init_start = time()

# Iterate over all client IDs.
# Client index number starts from 1.
for i in range(a, b):
    agents.append(ClientAgent(id=i,
                              name="PPFL Client Agent {}".format(i),
                              type="ClientAgent",
                              iterations=num_iterations,
                              num_clients=num_clients,
                              neighborhood_size=neighborhood_size,
                              # multiplier = accy_multiplier, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                              # split_size = split_size, secret_scale = secret_scale,
                              debug_mode=debug_mode,
                              random_state=np.random.RandomState(
                                  seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))))

agent_types.extend(["ClientAgent" for i in range(a, b)])

client_init_end = time()
init_seconds = client_init_end - client_init_start
td_init = timedelta(seconds=init_seconds)
print(f"Client init took {td_init}")

### Configure a service agent.
### 配置服务代理。

agents.extend([ServiceAgent(
    id=b,  # set id to be a number after all clients
    name="PPFL Service Agent",
    type="ServiceAgent",
    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
    msg_fwd_delay=0,
    users=[*range(a, b)],
    iterations=num_iterations,
    round_time=pd.Timedelta(f"{round_time}s"),
    num_clients=num_clients,
    neighborhood_size=neighborhood_size,
    parallel_mode=parallel_mode,
    debug_mode=debug_mode,
)])

agent_types.extend(["ServiceAgent"])

### Configure a latency model for the agents.
### 为代理配置延迟模型。

# Get a new-style cubic LatencyModel from the networking literature.
# 从网络文献中获得一种新型的立方体潜在模型。
pairwise = (len(agent_types), len(agent_types))

model_args = {'connected'  : True,

              # All in NYC.
              # Only matters for evaluating "real world" protocol duration,
              # not for accuracy, collusion, or reconstruction.
              # 都在纽约。只对评估“现实世界”协议持续时间很重要，对准确性、合谋或重建不重要。

              'min_latency': np.random.uniform(low=10000000, high=100000000, size=pairwise),
              'jitter'     : 0.3,
              'jitter_clip': 0.05,
              'jitter_unit': 5,
              }

latency_model = LatencyModel(latency_model='cubic',
                             random_state=latency_rstate,
                             kwargs=model_args)

# Start the kernel running.
results = kernel.runner(agents=agents,
                        startTime=kernelStartTime,
                        stopTime=kernelStopTime,
                        agentLatencyModel=latency_model,
                        defaultComputationDelay=defaultComputationDelay,
                        skip_log=skip_log,
                        log_dir=log_dir)

# Print parameter summary and elapsed times by category for this experimental trial.
print()
print(f"######## Microbenchmarks ########")
print(f"Protocol Iterations: {num_iterations}, Clients: {num_clients}, ")

print()
print("Service Agent mean time per iteration (except setup)...")
print(f"    Report step:         {results['srv_report']}")
print(f"    Crosscheck step:     {results['srv_crosscheck']}")
print(f"    Reconstruction step: {results['srv_reconstruction']}")
print()
print("Client Agent mean time per iteration (except setup)...")
print(f"    Report step:         {results['clt_report'] / num_clients}")
print(f"    Crosscheck step:     {results['clt_crosscheck'] / param.committee_size}")
print(f"    Reconstruction step: {results['clt_reconstruction'] / param.committee_size}")
print()
