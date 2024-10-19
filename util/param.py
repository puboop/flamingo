from Cryptodome.Cipher import AES, ChaCha20
# chacha20算法的基本思想：加密时，将明文数据与用户之间约定的某些数据进行异或操作，得到密文数据；
# 由异或操作的特点可知，在解密时，只需要将密文数据与用户之间约定的那些数据再次进行异或操作，就得到了明文数据。
from Cryptodome.Random import get_random_bytes
import numpy as np
import pandas as pd
import math

# System parameters
vector_len = 16000
vector_type = 'uint32'
committee_size = 60
fraction = 1 / 3
fixed_key = b"abcd"

# Waiting time
# Set according to a target dropout rate (e.g., 1%) 
# and message lantecy (see model/LatencyModel.py)
wt_flamingo_report = pd.Timedelta('10s')
wt_flamingo_crosscheck = pd.Timedelta('3s')
wt_flamingo_reconstruction = pd.Timedelta('3s')

wt_google_adkey = pd.Timedelta('10s')
wt_google_graph = pd.Timedelta('10s')
wt_google_share = pd.Timedelta('30s')  # ensure all user_choice received messages
wt_google_collection = pd.Timedelta('10s')
wt_google_crosscheck = pd.Timedelta('3s')
wt_google_recontruction = pd.Timedelta('2s')

# WARNING: 
# this should be a random seed from beacon service;
# we use a fixed one for simplicity
# 获取一个指定长度的bytes对象, 它实际上是在获取不同操作系统特定提供的随机源,
# 它可以被用来做随机加密的key使用
root_seed = get_random_bytes(32)
nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00'


def assert_power_of_two(x):
    return (math.ceil(math.log2(x)) == math.floor(math.log2(x)))


# choose committee members
def choose_committee(root_seed, committee_size, num_clients):
    # 用于生成独特的加密流，以保证加密的安全性。
    # 即使使用相同的密钥，如果 Nonce 不同，加密输出（密文）也会不同
    prg_committee_holder = ChaCha20.new(key=root_seed, nonce=nonce)

    data = b"secr" * committee_size * 128
    prg_committee_bytes = prg_committee_holder.encrypt(data)
    committee_numbers = np.frombuffer(prg_committee_bytes, dtype=vector_type)

    user_committee = set()
    cnt = 0
    while (len(user_committee) < committee_size):
        sampled_id = committee_numbers[cnt] % num_clients
        (user_committee).add(sampled_id)
        cnt += 1

    return user_committee


# choose neighbors
# 选择邻居
def findNeighbors(root_seed, current_iteration, num_clients, id, neighborhood_size):
    neighbors_list = set()  # a set, instead of a list

    # compute PRF(root, iter_num), output a seed. can use AES
    # 计算PRF（根，iter_num），输出种子。可以使用AES
    prf = ChaCha20.new(key=root_seed, nonce=nonce)
    current_seed = prf.encrypt(current_iteration.to_bytes(32, 'big'))

    # compute PRG(seed), a binary string
    # 计算PRG（种子），一个二进制字符串
    prg = ChaCha20.new(key=current_seed, nonce=nonce)

    # compute number of bytes we need for a graph
    # 计算图所需的字节数
    num_choose = math.ceil(math.log2(num_clients))  # number of neighbors I choose
    num_choose = num_choose * neighborhood_size

    bytes_per_client = math.ceil(math.log2(num_clients) / 8)
    segment_len = num_choose * bytes_per_client
    num_rand_bytes = segment_len * num_clients
    data = b"a" * num_rand_bytes
    graph_string = prg.encrypt(data)

    # find the segment for myself
    # 为自己找到这个片段
    my_segment = graph_string[id * segment_len: (id + 1) * segment_len]

    # define the number of bits within bytes_per_client that can be convert to int (neighbor's ID)
    # 定义bytes_per_client中可以转换为int（邻居ID）的位数
    bits_per_client = math.ceil(math.log2(num_clients))
    # default number of clients is power of two
    for i in range(num_choose):
        tmp = my_segment[i * bytes_per_client: (i + 1) * bytes_per_client]
        # 'big'：使用大端序，数据按照从最高有效字节（MSB）到最低有效字节（LSB）的顺序排列。
        # 'little'：使用小端序，数据按照从最低有效字节（LSB）到最高有效字节（MSB）的顺序排列。
        tmp_neighbor = int.from_bytes(tmp, 'big') & ((1 << bits_per_client) - 1)
        # 随机邻居选择恰好是它自己，跳过
        if tmp_neighbor == id:  # random neighbor choice happened to be itself, skip
            continue
        if tmp_neighbor in neighbors_list:  # client already chose tmp_neighbor, skip
            continue
        neighbors_list.add(tmp_neighbor)

    # now we have a list for who I chose
    # find my ID in the rest, see which segment I am in. add to neighbors_list
    # 在其余部分找到我的ID，看看我在哪个部分。添加到neighbors_list
    for i in range(num_clients):
        if i == id:
            continue
        seg = graph_string[i * segment_len: (i + 1) * segment_len]
        ls = parse_segment_to_list(seg, num_choose, bits_per_client, bytes_per_client)
        if id in ls:
            # add current segment owner into neighbors_list
            # 将当前段所有者添加到邻居列表中
            neighbors_list.add(i)

    return neighbors_list


def parse_segment_to_list(segment, num_choose, bits_per_client, bytes_per_client):
    cur_ls = set()
    # take a segment (byte string), parse it to a list
    for i in range(num_choose):
        cur_bytes = segment[i * bytes_per_client: (i + 1) * bytes_per_client]
        cur_no = int.from_bytes(cur_bytes, 'big') & ((1 << bits_per_client) - 1)
        cur_ls.add(cur_no)

    return cur_ls
