from agent.Agent import Agent
from message.Message import Message
from message.Message import MessageType

import multiprocessing
import dill
import json
import time
import logging

import math
import numpy as np
import pandas as pd
import random

from Cryptodome.PublicKey import ECC
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Random import get_random_bytes
from Cryptodome.Hash import SHA256
from Cryptodome.Signature import DSS

from util import param
from util import util
from util.crypto import ecchash
from util.crypto.secretsharing import secret_int_to_points, points_to_secret_int

from message.new_msg import  ReqMsg
def parallel_mult(vec, coeff):
    """Scalar multiplication for EC points in parallel."""
    points = vec.apply(lambda row: ECC.EccPoint(row[0], row[1]), axis=1)
    points = points * coeff
    points = pd.DataFrame([(p.x, p.y) for p in points])
    points = points.applymap(lambda x: int(x))

    return points


# PPFL_ServiceAgent class inherits from the base Agent class.
class SA_ServiceAgent(Agent):

    def __str__(self):
        return "[server]"

    def __init__(self, id, name, type,
                 random_state=None,
                 msg_fwd_delay=1000000,
                 round_time=pd.Timedelta("10s"),
                 iterations=4,
                 key_length=32,
                 num_clients=10,
                 neighborhood_size=1,
                 parallel_mode=1,
                 debug_mode=0,
                 users={}):

        # Base class init. 
        super().__init__(id, name, type, random_state)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if debug_mode:
            logging.basicConfig()

        # System parameters.
        # 系统参数。
        # 转发对等客户端中继消息的时间
        self.msg_fwd_delay = msg_fwd_delay  # time to forward a peer-to-peer client relay message
        # 每轮默认等待时间
        self.round_time = round_time  # default waiting time per round
        # 迭代次数
        self.no_of_iterations = iterations  # number of iterations
        # parallel
        self.parallel_mode = parallel_mode  # parallel

        # Input parameters.
        # 输入参数。
        # 每轮培训的用户数
        self.num_clients = num_clients  # number of users per training round
        # 用户ID列表
        self.users = users  # the list of user IDs
        self.vector_len = param.vector_len
        self.vector_dtype = param.vector_type
        self.vec_sum_partial = np.zeros(self.vector_len, dtype=self.vector_dtype)
        self.final_sum = np.zeros(self.vector_len, dtype=self.vector_dtype)

        # Security parameeters.
        # 安全参数。
        self.prime = ecchash.n
        self.key_length = key_length
        self.neighborhood_size = neighborhood_size
        self.committee_threshold = 0

        # Read keys.
        # 读取密钥。
        self.server_key = util.read_key("pki_files/server_key.pem")
        self.system_sk = util.read_sk("pki_files/system_pk.pem")

        # agent accumulation of elapsed times by category of tasks
        # 按任务类别计算的代理运行时间累积
        self.elapsed_time = {'REPORT'        : pd.Timedelta(0),
                             'CROSSCHECK'    : pd.Timedelta(0),
                             'RECONSTRUCTION': pd.Timedelta(0),
                             }

        # Initialize pools.
        self.user_vectors = {}
        self.pairwise_cipher = {}
        self.mi_cipher = {}
        self.recon_index = {}

        self.recv_user_vectors = {}
        self.recv_pairwise_cipher = {}
        self.recv_mi_cipher = {}
        self.recv_recon_index = {}

        self.user_committee = {}
        self.committee_shares_pairwise = {}
        self.committee_shares_mi = {}
        self.committee_sigs = {}

        self.recv_committee_shares_pairwise = {}
        self.recv_committee_shares_mi = {}
        self.recv_committee_sigs = {}

        self.dec_target_pairwise = {}
        self.dec_target_mi = {}

        # Track the current iteration and round of the protocol.
        # 跟踪协议的当前迭代和轮次。
        self.current_iteration = 1
        self.current_round = 0

        # 映射消息处理功能
        # Map the message processing functions
        self.aggProcessingMap = {
            0: self.initialize,
            1: self.report,
            2: self.forward_signatures,
            3: self.reconstruction,
        }

        self.namedict = {
            0: "initialize",
            1: "report",
            2: "forward_signatures",
            3: "reconstruction",
        }

    # Simulation lifecycle messages.
    # 仿真生命周期消息。
    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()

        # Initialize custom state properties into which we will accumulate results later.
        # 初始化自定义状态属性，稍后我们将在其中累积结果。
        self.kernel.custom_state['srv_report'] = pd.Timedelta(0)
        self.kernel.custom_state['srv_crosscheck'] = pd.Timedelta(0)
        self.kernel.custom_state['srv_reconstruction'] = pd.Timedelta(0)

        # This agent should have negligible (or no) computation delay until otherwise specified.
        # 除非另有规定，否则此代理的计算延迟应可忽略不计（或没有）。
        self.setComputationDelay(0)

        # Request a wake-up call as in the base Agent.
        # 请求唤醒电话，就像在基本代理中一样。
        super().kernelStarting(startTime)

    def kernelStopping(self):
        # Add the server time components to the custom state in the Kernel, for output to the config.
        # Note that times which should be reported in the mean per iteration are already so computed.
        self.kernel.custom_state['srv_report'] += (
                self.elapsed_time['REPORT'] / self.no_of_iterations)
        self.kernel.custom_state['srv_crosscheck'] += (
                self.elapsed_time['CROSSCHECK'] / self.no_of_iterations)
        self.kernel.custom_state['srv_reconstruction'] += (
                self.elapsed_time['RECONSTRUCTION'] / self.no_of_iterations)

        # Allow the base class to perform stopping activities.
        super().kernelStopping()

    def wakeup(self, currentTime):
        """
        The service agent wakes up at the end of each round.

        More specifically, it:
        1. Stores the received messages.
        2. When timing out occurs or 
           when it collects a sufficient number of messages
           from the clients it is waiting for, 
           initiates processing and replying to messages 

        Args:
        - currentTime: The (absolute) current time when the agent wakes up.
                       Note that currentTime is the 'start' of the function.
        """
        super().wakeup(currentTime)
        self.agent_print(
            f"wakeup in iteration {self.current_iteration} at function {self.namedict[self.current_round]}; current time is {currentTime}")

        # In the k-th iteration
        # 在第k次迭代中
        self.aggProcessingMap[self.current_round](currentTime)

    def receiveMessage(self, currentTime, msg):
        """Collect messages from clients.
        
        Three types: 
        - VECTOR message meant for report step, 
        - SIGN message meant for crosscheck step,
        - SHARED_RESULT message meant for reconstruction step.
        """

        # Allow the base Agent to do whatever it needs to.
        # 允许基础代理做任何它需要做的事情。
        super().receiveMessage(currentTime, msg)

        # Get the sender's id (should be client id)
        # 获取发件人的id（应该是客户端id）
        sender_id = msg.body['sender']
        if __debug__: self.logger.info(f"received vector from client {sender_id} at {currentTime}")

        # Collect masked vectors from clients
        # 从客户端收集掩码向量
        if msg.body['msg'] == "VECTOR":

            if msg.body['iteration'] == self.current_iteration:

                # Store the vectors
                # 存储矢量
                self.recv_user_vectors[sender_id] = msg.body['vector']

                # parse cipher for shares of mi
                # mi共享的解析密码
                self.recv_mi_cipher[sender_id] = util.deserialize_tuples_bytes(msg.body['enc_mi_shares'])

                # parse the cipher for pairwise mask
                # 解析成对掩码的密码
                cur_clt_pairwise_cipher = util.deserialize_dim1_elgamal(msg.body['enc_pairwise'])

                # update pairwise cipher
                # 更新成对密码
                for d in (self.recv_pairwise_cipher, cur_clt_pairwise_cipher):
                    self.recv_pairwise_cipher.update(d)

            else:
                if __debug__:
                    self.logger.info(
                        f"LATE MSG: Server receives VECTORS from iteration {msg.body['iteration']} client {msg.body['sender']}")

        # Collect signed labels from decryptors
        # 从解密器收集签名标签
        elif msg.body['msg'] == "SIGN":

            if msg.body['iteration'] == self.current_iteration:
                # 将签名转发给所有解密器
                # forward the signatures to all decryptors
                self.recv_committee_sigs[sender_id] = msg.body['signed_labels']

            else:
                if __debug__: self.logger.info(
                    f"LATE MSG: Server receives signed labels from iteration {msg.body['iteration']} client {msg.body['sender']}")

        # Collect partial decryption results from decryptors
        # 从解密器收集部分解密结果
        elif msg.body['msg'] == "SHARED_RESULT":

            if msg.body['iteration'] == self.current_iteration:

                self.recv_committee_shares_pairwise[sender_id] = util.deserialize_dim1_ecp(
                    msg.body['shared_result_pairwise'])
                self.recv_committee_shares_mi[sender_id] = util.deserialize_dim1_list(msg.body['shared_result_mi'])
                self.recv_recon_index[sender_id] = msg.body['committee_member_idx']

                # self.logger.info(f"communication for received decryption shares: {comm}") 
            else:
                if __debug__:
                    self.logger.info(
                        f"LATE MSG: Server receives SHARED_RESULT from iteration {msg.body['iteration']} client {msg.body['sender']}")

    # Processing and replying the messages.
    # 处理和回复消息。
    def initialize(self, currentTime):
        dt_protocol_start = pd.Timestamp('now')

        # Setup committee (decryptors).
        self.user_committee = param.choose_committee(param.root_seed,
                                                     param.committee_size,
                                                     self.num_clients)
        self.committee_threshold = int(param.fraction * len(self.user_committee))

        # Simulate the Shamir share of SK at each decryptor
        sk_shares = secret_int_to_points(secret_int=self.system_sk,
                                         point_threshold=self.committee_threshold,
                                         num_points=len(self.user_committee),
                                         prime=self.prime)

        # Send shared sk to committee members
        if __debug__: self.logger.info(f"Server sends to committee members:, {self.user_committee}")

        cnt = 0
        for id in self.user_committee:
            self.sendMessage(id,
                             Message({"msg"                 : "COMMITTEE_SHARED_SK",
                                      "committee_member_idx": cnt + 1,  # the share evaluation x-point starts at 1
                                      "sk_share"            : sk_shares[cnt],
                                      }),
                             tag="comm_dec_server")
            cnt += 1

        self.current_round = 1

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        self.setWakeup(currentTime + server_comp_delay + pd.Timedelta('2s'))

    def report(self, currentTime):
        """Process masked vectors.
            Server at this point should receive:
            vectors, encryption of mi shares, encryption of h_ijt
        """

        dt_protocol_start = pd.Timestamp('now')

        self.report_read_from_pool()
        self.report_process()
        self.report_clear_pool()
        self.report_send_message()

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        self.agent_print("run time for report step:", server_comp_delay)

        # Accumulate into time log.
        self.recordTime(dt_protocol_start, "REPORT")

        self.current_round = 2

        self.setWakeup(currentTime + server_comp_delay + param.wt_flamingo_crosscheck)

    def report_read_from_pool(self):
        # assign user vectors to a new var. empty user vectors immediately.
        self.user_vectors = self.recv_user_vectors
        self.recv_user_vectors = {}

        # for each client, a list of encrypted mi shares (#shares = #commmittee members)
        self.mi_cipher = self.recv_mi_cipher
        self.recv_mi_cipher = {}

        # for each client, a list of encrypted pairwise secrets 
        self.pairwise_cipher = self.recv_pairwise_cipher
        self.recv_pairwise_cipher = {}

    def report_clear_pool(self):
        # Empty the pool for those upcoming messages before server sends requests
        self.recv_committee_shares_mi = {}
        self.recv_committee_shares_pairwise = {}
        self.recv_recon_index = {}
        self.recv_committee_sigs = {}

    def report_process(self):
        self.agent_print("number of collected vectors:", len(self.user_vectors))

        # a list for committee member to know 
        # which pairwise key is used to decrypt which entry
        self.client_id_list = list(self.mi_cipher.keys())

        # each row of df_mi_shares is the shares of an mi from a client
        self.df_mi_cipher = pd.DataFrame((self.mi_cipher).values())

        # compute which pairwise secrets are in dec target:
        # only edge between an online client and an offline client
        online_set = set(self.user_vectors.keys())
        offline_set = set(self.users) - set(online_set)
        if __debug__: self.logger.info(f"online clients: {len(online_set)}")

        # Compute partial sum.
        self.vec_sum_partial = np.zeros(self.vector_len, dtype=self.vector_dtype)
        for id in self.user_vectors:
            if len(self.user_vectors[id]) != self.vector_len:
                raise RuntimeError("Client sends vector of incorrect length.")
            self.vec_sum_partial += self.user_vectors[id]
        if __debug__: self.logger.info(f"partial sum = {self.vec_sum_partial}")

        # assemble ciphertexts from self.pairwise_cipher, send to committee for decryption
        self.dec_target_pairwise = {}

        # used for server later in reconstruction phase to know whether + or -
        self.recon_symbol = {}

        # iterate over offline clients
        for id in offline_set:
            # TODO OPTMIZATION: store neighbors to reduce time
            # find neighbors for client id
            # client id is from 0
            clt_neighbors_list = param.findNeighbors(param.root_seed,
                                                     self.current_iteration,
                                                     self.num_clients,
                                                     id,
                                                     self.neighborhood_size)

            for nb in clt_neighbors_list:  # for all neighbors of this client
                if nb in online_set:  # if this client id's neighbor nb is online
                    if (nb, id) not in list(
                            self.pairwise_cipher.keys()):  # find tuples (nb, id) in self.pairwise_cipher
                        raise RuntimeError("Message lost:", (
                        nb, id))  # the first component nb is online client, the second component id is offlne client
                    self.dec_target_pairwise[(nb, id)] = self.pairwise_cipher[(nb, id)]
                    if nb > id:
                        self.recon_symbol[(nb, id)] = 1
                    elif nb < id:
                        self.recon_symbol[(nb, id)] = -1
                    else:  # id == nb
                        raise RuntimeError("id should not be its own neighbor.")

        msg_to_sign = dill.dumps(offline_set)
        hash_container = SHA256.new(msg_to_sign)
        signer = DSS.new(self.server_key, 'fips-186-3')
        signature = signer.sign(hash_container)
        self.labels_and_sig = (msg_to_sign, signature)

    def report_send_message(self):
        # Should send only the c1 component of the ciphertext to the committee
        cnt = 0
        for id in self.user_committee:
            self.sendMessage(id,
                             Message({"msg"                : "SIGN",
                                      "iteration"          : self.current_iteration,
                                      "dec_target_pairwise": util.serialize_dim1_elgamal(self.dec_target_pairwise),
                                      "dec_target_mi"      : util.serialize_tuples_bytes(self.df_mi_cipher[cnt]),
                                      "client_id_list"     : self.client_id_list,
                                      "labels"             : self.labels_and_sig,
                                      }),
                             tag="comm_dec_server")
            cnt += 1

    def forward_signatures(self, currentTime):
        """Forward cross check information for decryptors."""

        dt_protocol_start = pd.Timestamp('now')

        self.forward_signatures_read_from_pool()
        self.forward_signatures_clear_pool()
        self.forward_signatures_send_message()

        self.current_round = 3

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        self.agent_print("run time for crosscheck step:", server_comp_delay)
        self.setWakeup(currentTime + server_comp_delay + param.wt_flamingo_reconstruction)

        # Accumulate into time log.
        self.recordTime(dt_protocol_start, "CROSSCHECK")

    def forward_signatures_read_from_pool(self):
        self.committee_sigs = self.recv_committee_sigs
        self.recv_committee_shares_mi = {}

    def forward_signatures_clear_pool(self):
        # Empty the pool for those upcoming messages before server sends requests
        self.recv_committee_shares_pairwise = {}
        self.recv_recon_index = {}

    def forward_signatures_send_message(self):
        for id in self.user_committee:
            self.sendMessage(id,
                             Message({"msg"      : "DEC",
                                      "iteration": self.current_iteration,
                                      "labels"   : self.committee_sigs,
                                      }),
                             tag="comm_sign_server")

    def reconstruction(self, currentTime):
        """Reconstruct sum."""

        dt_protocol_start = pd.Timestamp('now')

        self.reconstruction_read_from_pool()
        self.reconstruction_process()
        self.reconstruction_clear_pool()
        self.reconstruction_send_message()

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        self.agent_print("run time for reconstruction step:", server_comp_delay)

        # Accumulate into time log.
        self.recordTime(dt_protocol_start, "RECONSTRUCTION")

        print()
        print("######## Iteration completion ########")
        print(f"[Server] finished iteration {self.current_iteration} at {currentTime + server_comp_delay}")
        print()

        self.current_round = 1

        # End of the iteration
        self.current_iteration += 1
        if (self.current_iteration > self.no_of_iterations):
            return

        self.setWakeup(currentTime + server_comp_delay + param.wt_flamingo_report)

    def reconstruction_read_from_pool(self):
        # if not enough shares received, wait for 0.1 sec
        if len(self.recv_committee_shares_pairwise) < self.committee_threshold:
            time.sleep(0.1)

        # TEST
        # _, json_string = self.serialize_dim2_ecp(self.recv_committee_shares_pairwise)
        # self.recv_committee_shares_pairwise = self.deserialize_dim2_ecp(json_string)

        self.committee_shares_pairwise = self.recv_committee_shares_pairwise
        self.recv_committee_shares_pairwise = {}

        self.committee_shares_mi = self.recv_committee_shares_mi
        self.recv_committee_shares_mi = {}

        self.recon_index = self.recv_recon_index
        self.recv_recon_index = {}

    def reconstruction_clear_pool(self):
        self.user_vectors = {}
        self.committee_shares_pairwise = {}
        self.committee_shares_mi = {}
        self.recon_index = {}

        # Empty the pool for those upcoming messages before server sends requests
        self.user_masked_input = {}
        self.recv_pairwise_cipher = {}
        self.recv_mi_cipher = {}
        self.recv_user_vectors = {}

    def reconstruction_process(self):
        self.agent_print("number of collected shares from decryptors:", len(self.committee_shares_pairwise))
        if len(self.committee_shares_pairwise) < self.committee_threshold:
            raise RuntimeError("No enough shares for decryption received.")

        # TODO OPTIMIZATION: only extract the shares of first 20 committees

        """Recover mi."""
        df_mi_shares = pd.DataFrame(self.committee_shares_mi)
        df_mi_shares = df_mi_shares.iloc[:, :self.committee_threshold]
        primary_points = []  # the shares of mi of the first online users
        for id in df_mi_shares:
            primary_points.append((self.recon_index[id], df_mi_shares[id][0]))

        primary_recon_secret, interpolate_coefficients = points_to_secret_int(
            points=primary_points, prime=self.prime, isecc=0)

        """Compute mi from shares."""
        cnt = 0
        for id in df_mi_shares:
            # each df_mi[id] is a vector of EC points
            # interpolation_coefficients is a list, each component is a number
            df_mi_shares[id] = (df_mi_shares[id] * interpolate_coefficients[cnt]) % self.prime
            cnt += 1

        sum_df = pd.DataFrame(np.sum(df_mi_shares.values, axis=1) % self.prime)
        sum_df[0] = sum_df[0].apply(lambda var: var.to_bytes(self.key_length, 'big'))

        """Compute mi mask vectors."""
        prg_mi = {}
        mi_vec = np.zeros(self.vector_len, dtype=self.vector_dtype)
        for i in range(len(sum_df)):
            prg_mi_holder = ChaCha20.new(key=sum_df[0][i], nonce=param.nonce)
            data = param.fixed_key * self.vector_len
            prg_mi[i] = prg_mi_holder.encrypt(data)
            mi_vec = mi_vec - np.frombuffer(prg_mi[i], dtype=self.vector_dtype)

        if not self.dec_target_pairwise:
            self.agent_print("no client dropped out.")
            self.final_sum = self.vec_sum_partial + mi_vec
        else:
            if not self.parallel_mode:
                df_pairwise = pd.DataFrame(self.committee_shares_pairwise)
                df_pairwise = df_pairwise.iloc[:, :self.committee_threshold]
                cnt = 0
                for id in df_pairwise:
                    # multiply interpolate coefficients, 
                    # might be slow since it is EC scalar mult
                    df_pairwise[id] = df_pairwise[id] * interpolate_coefficients[cnt]
                    cnt += 1
                sum_df = -pd.DataFrame(list(df_pairwise.sum(axis=1)))  # compute c_0^{-s}
            else:
                df_pairwise = self.committee_shares_pairwise
                cnt = 0
                for k in df_pairwise.keys():
                    if cnt == self.committee_threshold:
                        break
                    df_pairwise[k] = pd.DataFrame([(p.x, p.y) for p in df_pairwise[k]])
                    df_pairwise[k] = df_pairwise[k].applymap(lambda x: int(x))
                    cnt += 1

                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                prods = pool.starmap(parallel_mult, zip(df_pairwise.values(), interpolate_coefficients))

                pool.close()
                pool.terminate()
                pool.join()

                prods = [p.apply(lambda row: ECC.EccPoint(row[0], row[1]), axis=1) for p in prods]
                prods = pd.DataFrame(prods)
                prods = prods.transpose()
                sum_df = -pd.DataFrame(list(prods.sum(axis=1)))  # compute c_0^{-s}

            # compute c1 column
            tmp_list = list(self.dec_target_pairwise.values())
            dec_list = list(zip(*tmp_list))[1]
            dec_df = pd.DataFrame(dec_list)
            if len(dec_df) != len(sum_df):
                raise RuntimeError("length error.")

            # the decryption result is stored in sum_df
            sum_df = sum_df + dec_df
            sum_df[0] = sum_df[0].apply(lambda var:
                                        SHA256.new(int(var.x).to_bytes(self.key_length, 'big')
                                                   + int(var.y).to_bytes(self.key_length, 'big')).digest()[
                                        0: self.key_length])

            # compute pairwise mask vectors
            prg_pairwise = {}
            cancel_vec = np.zeros(self.vector_len, dtype=self.vector_dtype)

            if len(sum_df) != len(self.recon_symbol):
                raise RuntimeError("The decrypted length is wrong.")
            recon_symbol_list = list(self.recon_symbol.values())

            for i in range(len(sum_df)):
                prg_pairwise_holder = ChaCha20.new(key=sum_df[0][i], nonce=param.nonce)
                data = param.fixed_key * self.vector_len
                prg_pairwise[i] = prg_pairwise_holder.encrypt(data)

                if recon_symbol_list[i] == 1:
                    cancel_vec = cancel_vec + np.frombuffer(prg_pairwise[i], dtype=self.vector_dtype)
                elif recon_symbol_list[i] == -1:
                    cancel_vec = cancel_vec - np.frombuffer(prg_pairwise[i], dtype=self.vector_dtype)

            self.final_sum = self.vec_sum_partial + cancel_vec + mi_vec

        self.agent_print("final sum:", self.final_sum)

    def reconstruction_send_message(self):
        # Send the result back to each client.
        for id in self.users:
            self.sendMessage(id,
                             Message({"msg"   : "REQ",
                                      "sender": 0,
                                      "output": 1,  # For machine learning, should be current model weights
                                      }),
                             tag="comm_output_server")

    # ======================== UTIL ========================

    def recordTime(self, startTime, categoryName):
        # Accumulate into time log.
        dt_protocol_end = pd.Timestamp('now')
        self.elapsed_time[categoryName] += dt_protocol_end - startTime

    def agent_print(*args, **kwargs):
        """
        Custom print function that adds a [Server] header before printing.

        Args:
            *args: Any positional arguments that the built-in print function accepts.
            **kwargs: Any keyword arguments that the built-in print function accepts.
        """
        print(*args, **kwargs)

    def send_request_prove(self):
        for client in self.kernel.agents:
            if client.id == self.id:
                continue
            self.kernel.prove_queue.put((
                MessageType.PROVE,
                ReqMsg(client_id=client.id,
                       client_obj=client)
            ))
