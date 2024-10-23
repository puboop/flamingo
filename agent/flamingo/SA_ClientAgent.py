from agent.Agent import Agent
from agent.flamingo.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from agent.flamingo.SA_Manage import SA_Manage as Manage
from message.Message import Message, MessageType
from message.new_msg import ReqMsg

import dill
import time
import logging

import math
import libnum
import numpy as np
import pandas as pd
import random

# pycryptodomex library functions
from Cryptodome.PublicKey import ECC
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Random import get_random_bytes
from Cryptodome.Hash import SHA256
from Cryptodome.Signature import DSS

# other user-level crypto functions
import hashlib
from util import param
from util import util
from util.DiffieHellman import DHKeyExchange, mod_args
from util.crypto import ecchash
from util.crypto.secretsharing import secret_int_to_points, points_to_secret_int

from util.AesCrypto import aes_encrypt, aes_decrypt


# The PPFL_TemplateClientAgent class inherits from the base Agent class.
class SA_ClientAgent(Agent):

    def __str__(self):
        return "[client]"

    # Default param:
    # num of iterations = 4
    # key length = 32 bytes
    # neighbors ~ 2 * log(num per iter) 
    def __init__(self, id, name, type,
                 iterations=4,
                 key_length=32,
                 num_clients=128,
                 neighborhood_size=1,
                 debug_mode=0,
                 random_state=None):

        # Base class init
        super().__init__(id, name, type, random_state)

        # Set logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if debug_mode:
            logging.basicConfig()

        """Read keys."""
        # Read system-wide pk
        self.system_pk = util.read_pk(f"pki_files/system_pk.pem")

        # sk is used to establish pairwise secret with neighbors' public keys
        self.key = util.read_key(f"pki_files/client{self.id}.pem")
        self.secret_key = self.key.d
        self.private_key = self.key.export_key(format='PEM')  # 以PEM形式存私钥
        self.public_key = self.key.public_key().export_key(format='PEM')  # 以PEM形式存公钥

        # 创建一个用于 Diffie-Hellman 密钥交换的对象。
        self.dh_key_obj = DHKeyExchange(mod_args.q, mod_args.g, self.private_key)

        """Set parameters."""
        self.num_clients = num_clients
        # 邻域大小
        self.neighborhood_size = neighborhood_size
        self.vector_len = param.vector_len
        self.vector_dtype = param.vector_type
        self.prime = ecchash.n
        self.key_length = key_length
        self.neighbors_list = set()  # neighbors
        self.cipher_stored = None  # Store cipher from server across steps

        """Select committee."""
        self.user_committee = param.choose_committee(param.root_seed,
                                                     param.committee_size,
                                                     self.num_clients)
        self.committee_shared_sk = None
        self.committee_member_idx = None

        # If it is in the committee:
        # read pubkeys of every other client and precompute pairwise keys
        # 读取其他所有客户端的pubkeys并预计算成对密钥
        self.symmetric_keys = {}
        if self.id in self.user_committee:
            for i in range(num_clients):
                pk = util.read_pk(f"pki_files/client{i}.pem")
                self.symmetric_keys[i] = pk * self.secret_key  # group 
                self.symmetric_keys[i] = (int(self.symmetric_keys[i].x) & ((1 << 128) - 1)).to_bytes(16, 'big')  # bytes

        # Accumulate this client's run time information by step.
        self.elapsed_time = {'REPORT'        : pd.Timedelta(0),
                             'CROSSCHECK'    : pd.Timedelta(0),
                             'RECONSTRUCTION': pd.Timedelta(0),
                             }

        # Iteration counter
        self.no_of_iterations = iterations
        self.current_iteration = 1
        self.current_base = 0

        # State flag
        self.setup_complete = False
        # 是否完成了对Kn,m的聚合操作,生成Kn
        self.agg_finish = False
        # 是否完成对alpha m的聚合操作,生成alpha
        self.agg_alpha_finish=False
        # 是否完成对Km的聚合操作，生成K
        self.agg_Km_finish = False

        # 初始化一个字典，用于存储与管理者（manage）相关的 Diffie-Hellman 密钥交换信息
        self.manages_dh_key = dict()
        # Kn
        self.agg_manages_keys = ""
        # alpha
        self.alpha = ""
        # K
        self.K = ""
        # 客户端的PRO_n
        self.Client_PRO = ""
        #疑问：都设置为”“，都是字符串类型吗？如果不是，那这样会有问题吗？

    # Simulation lifecycle messages.
    def kernelStarting(self, startTime):

        # Initialize custom state properties into which we will later accumulate results.
        # To avoid redundancy, we allow only the first client to handle initialization.

        # 初始化自定义状态属性，稍后我们将在其中累积结果。为了避免冗余，我们只允许第一个客户端处理初始化。
        if self.id == 0:
            self.kernel.custom_state['clt_report'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_crosscheck'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_reconstruction'] = pd.Timedelta(0)

        # Find the PPFL service agent, so messages can be directed there.
        # 找到PPFL服务代理，以便将消息定向到那里。
        self.serviceAgentID = self.kernel.findAgentByType(ServiceAgent)

        self.setComputationDelay(0)

        # Request a wake-up call as in the base Agent.  Noise is kept small because
        # the overall protocol duration is so short right now.  (up to one microsecond)
        super().kernelStarting(startTime +
                               pd.Timedelta(self.random_state.randint(low=0, high=1000), unit='ns'))

    def kernelStopping(self):

        # Accumulate into the Kernel's "custom state" this client's elapsed times per category.
        # Note that times which should be reported in the mean per iteration are already so computed.
        # These will be output to the config (experiment) file at the end of the simulation.

        self.kernel.custom_state['clt_report'] += (
                self.elapsed_time['REPORT'] / self.no_of_iterations)
        self.kernel.custom_state['clt_crosscheck'] += (
                self.elapsed_time['CROSSCHECK'] / self.no_of_iterations)
        self.kernel.custom_state['clt_reconstruction'] += (
                self.elapsed_time['RECONSTRUCTION'] / self.no_of_iterations)

        super().kernelStopping()

    # Simulation participation messages.
    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        dt_wake_start = pd.Timestamp('now')
        self.sendVectors(currentTime)

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

        # with signatures of other clients from the server
        # 具有来自服务器的其他客户端的签名
        if msg.body['msg'] == "COMMITTEE_SHARED_SK":  # 委员会共享SK
            self.committee_shared_sk = msg.body['sk_share']
            self.committee_member_idx = msg.body['committee_member_idx']

        elif msg.body['msg'] == "SIGN":  # 签名
            if msg.body['iteration'] == self.current_iteration:
                dt_protocol_start = pd.Timestamp('now')
                self.cipher_stored = msg
                self.signSendLabels(currentTime, msg.body['labels'])
                self.recordTime(dt_protocol_start, 'CROSSCHECK')

        elif msg.body['msg'] == "DEC":
            if msg.body['iteration'] == self.current_iteration:
                dt_protocol_start = pd.Timestamp('now')

                if self.cipher_stored == None:
                    if __debug__: self.logger.info("did not recv sign")
                else:
                    if self.cipher_stored.body['iteration'] == self.current_iteration:
                        self.decryptSendShares(
                            util.deserialize_dim1_elgamal(self.cipher_stored.body['dec_target_pairwise']),
                            util.deserialize_tuples_bytes(self.cipher_stored.body['dec_target_mi']),
                            self.cipher_stored.body['client_id_list'])

                self.cipher_stored = None
                self.recordTime(dt_protocol_start, 'RECONSTRUCTION')

        # End of the protocol / start the next iteration
        # Receiving the output from the server
        elif msg.body['msg'] == "REQ" and self.current_iteration != 0:
            # End of the iteration
            # Reset temp variables for each iteration

            # Enter next iteration
            self.current_iteration += 1
            if self.current_iteration > self.no_of_iterations:
                return

            dt_protocol_start = pd.Timestamp('now')
            self.sendVectors(currentTime)
            self.recordTime(dt_protocol_start, "REPORT")

    ###################################
    # Round logics
    ###################################
    def sendVectors(self, currentTime):

        dt_protocol_start = pd.Timestamp('now')

        # Find this client's neighbors: parse graph from PRG(PRF(iter, root_seed))
        # 查找此客户端的邻居：从PRG解析图（PRF（iter，root_seed））
        self.neighbors_list = param.findNeighbors(param.root_seed,
                                                  self.current_iteration,
                                                  self.num_clients,
                                                  self.id,
                                                  self.neighborhood_size)
        if __debug__:
            self.logger.info("client indices in neighbors list starts from 0")
            self.logger.info(f"client {self.id} neighbors list: {self.neighbors_list}")

        # Download public keys of neighbors from PKI file
        # 从PKI文件下载邻居的公钥
        # Client index starting frrom 0
        # 客户端索引从0开始
        neighbor_pubkeys = {}
        for id in self.neighbors_list:
            neighbor_pubkeys[id] = util.read_pk(f"pki_files/client{id}.pem")

        # send symmetric encryption of shares of mi  
        mi_bytes = get_random_bytes(self.key_length)
        mi_number = int.from_bytes(mi_bytes, 'big')

        mi_shares = secret_int_to_points(secret_int=mi_number,
                                         point_threshold=int(param.fraction * len(self.user_committee)),
                                         num_points=len(self.user_committee),
                                         prime=self.prime)

        committee_pubkeys = {}
        for id in self.user_committee:
            committee_pubkeys[id] = util.read_pk(f"pki_files/client{id}.pem")

        # separately encrypt each share
        # 分别加密每个共享
        enc_mi_shares = []
        # id is the x-axis
        cnt = 0
        for id in self.user_committee:
            per_share_bytes = (mi_shares[cnt][1]).to_bytes(self.key_length, 'big')

            # can be pre-computed
            key_with_committee_group = self.secret_key * committee_pubkeys[id]
            key_with_committee_bytes = (int(key_with_committee_group.x) & ((1 << 128) - 1)).to_bytes(16, 'big')

            per_share_encryptor = AES.new(key_with_committee_bytes, AES.MODE_GCM)
            # nouce should be sent with ciphertext
            nonce = per_share_encryptor.nonce

            # 加密并校验数据的完整性
            tmp, _ = per_share_encryptor.encrypt_and_digest(per_share_bytes)
            enc_mi_shares.append((tmp, nonce))
            cnt += 1

        # Compute mask, compute masked vector
        # 计算掩码，计算掩码向量
        # PRG individual mask
        prg_mi_holder = ChaCha20.new(key=mi_bytes, nonce=param.nonce)
        data = param.fixed_key * self.vector_len
        prg_mi = prg_mi_holder.encrypt(data)

        # compute pairwise masks r_ij
        neighbor_pairwise_secret_group = {}  # g^{a_i a_j} = r_ij in group
        neighbor_pairwise_secret_bytes = {}

        for id in self.neighbors_list:
            neighbor_pairwise_secret_group[id] = self.secret_key * neighbor_pubkeys[id]
            # hash the g^{ai aj} to 256 bits (16 bytes)
            px = (int(neighbor_pairwise_secret_group[id].x)).to_bytes(self.key_length, 'big')
            py = (int(neighbor_pairwise_secret_group[id].y)).to_bytes(self.key_length, 'big')

            hash_object = SHA256.new(data=(px + py))
            neighbor_pairwise_secret_bytes[id] = hash_object.digest()[0:self.key_length]

        neighbor_pairwise_mask_seed_group = {}
        neighbor_pairwise_mask_seed_bytes = {}

        """Mapping group elements to bytes.
            compute h_{i, j, t} to be PRF(r_ij, t)
            map h (a binary string) to a EC group element
            encrypt the group element
            map the group element to binary string (hash the x, y coordinate)
        """
        for id in self.neighbors_list:
            round_number_bytes = self.current_iteration.to_bytes(16, 'big')

            h_ijt = ChaCha20.new(key=neighbor_pairwise_secret_bytes[id],
                                 nonce=param.nonce).encrypt(round_number_bytes)
            h_ijt = str(int.from_bytes(h_ijt[0:4], 'big') & 0xFFFF)

            # map h_ijt to a group element
            dst = ecchash.test_dst("P256_XMD:SHA-256_SSWU_RO_")
            neighbor_pairwise_mask_seed_group[id] = ecchash \
                .hash_str_to_curve(msg=h_ijt,
                                   count=2,
                                   modulus=self.prime,
                                   degree=ecchash.m,
                                   blen=ecchash.L,
                                   expander=ecchash.XMDExpander(dst, hashlib.sha256, ecchash.k))

            px = (int(neighbor_pairwise_mask_seed_group[id].x)).to_bytes(self.key_length, 'big')
            py = (int(neighbor_pairwise_mask_seed_group[id].y)).to_bytes(self.key_length, 'big')

            hash_object = SHA256.new(data=(px + py))
            neighbor_pairwise_mask_seed_bytes[id] = hash_object.digest()[0:self.key_length]

        prg_pairwise = {}
        for id in self.neighbors_list:
            prg_pairwise_holder = ChaCha20.new(key=neighbor_pairwise_mask_seed_bytes[id], nonce=param.nonce)
            data = param.fixed_key * self.vector_len
            prg_pairwise[id] = prg_pairwise_holder.encrypt(data)

        """Client inputs.
            For machine learning, replace it with model weights.
            For testing, set to unit vector.
        """
        vec = np.ones(self.vector_len, dtype=self.vector_dtype)
        # 客户端本地梯度，后面计算客户端的PRO_n要用
        self.vec_n = np.ones(self.vector_len, dtype=self.vector_dtype)

        # vectorize bytes: 32 bit integer, 4 bytes per component
        vec_prg_mi = np.frombuffer(prg_mi, dtype=self.vector_dtype)
        if len(vec_prg_mi) != self.vector_len:
            raise RuntimeError("vector length error")

        vec += vec_prg_mi
        vec_prg_pairwise = {}

        for id in self.neighbors_list:
            vec_prg_pairwise[id] = np.frombuffer(prg_pairwise[id], dtype=self.vector_dtype)

            if len(vec_prg_pairwise[id]) != self.vector_len:
                raise RuntimeError("vector length error")
            if self.id < id:
                vec = vec + vec_prg_pairwise[id]
            elif self.id > id:
                vec = vec - vec_prg_pairwise[id]
            else:
                raise RuntimeError("id itself appears in its neighbor list")

        # compute encryption of H(t)^{r_ij} (already a group element), only for < relation
        cipher_msg = {}

        for id in self.neighbors_list:
            # the set sent to the server is indexed from 0
            cipher_msg[(self.id, id)] = self.elgamal_enc_group(self.system_pk, neighbor_pairwise_mask_seed_group[id])

        if __debug__:
            client_comp_delay = pd.Timestamp('now') - dt_protocol_start
            self.logger.info(f"client {self.id} computation delay for vector: {client_comp_delay}")
            self.logger.info(f"client {self.id} sends vector at {currentTime + client_comp_delay}")

        # Send the vector to the server
        self.sendMessage(self.serviceAgentID,
                         Message({"msg"          : "VECTOR",
                                  "iteration"    : self.current_iteration,
                                  "sender"       : self.id,
                                  "vector"       : vec,
                                  "enc_mi_shares": util.serialize_tuples_bytes(enc_mi_shares),
                                  "enc_pairwise" : util.serialize_dim1_elgamal(cipher_msg)}),
                         tag="comm_key_generation")

    def signSendLabels(self, currentTime, msg_to_sign):

        msg_to_sign = dill.dumps(msg_to_sign)
        hash_container = SHA256.new(msg_to_sign)
        signer = DSS.new(self.key, 'fips-186-3')
        signature = signer.sign(hash_container)
        client_signed_labels = (msg_to_sign, signature)

        self.sendMessage(self.serviceAgentID,
                         Message({"msg"                 : "SIGN",
                                  "iteration"           : self.current_iteration,
                                  "sender"              : self.id,
                                  "signed_labels"       : client_signed_labels,
                                  "committee_member_idx": self.committee_member_idx,
                                  "signed_labels"       : client_signed_labels,
                                  }),
                         tag="comm_sign_client")

    def decryptSendShares(self, dec_target_pairwise, dec_target_mi, client_id_list):

        dt_protocol_start = pd.Timestamp('now')

        if self.committee_shared_sk == None:
            if __debug__:
                self.logger.info(
                    f"Decryptor {self.committee_member_idx} is asked to decrypt, but does not have sk share.")
            self.sendMessage(self.serviceAgentID,
                             Message({"msg"                 : "NO_SK_SHARE",
                                      "iteration"           : self.current_iteration,
                                      "sender"              : self.id,
                                      "shared_result"       : None,
                                      "committee_member_idx": None,
                                      }),
                             tag="no_sk_share")
            return

            # CHECK SIGNATURES

        """Compute decryption of pairwise secrets. 计算成对秘密的解密。
            dec_target is a matrix dec_target是一个矩阵
            just need to mult sk with each of the entry 只需在每个条目中添加多个sk
            needs elliptic curve ops需要椭圆曲线运算
        """
        dec_shares_pairwise = []
        dec_target_list_pairwise = list(dec_target_pairwise.values())

        for i in range(len(dec_target_list_pairwise)):
            c0 = dec_target_list_pairwise[i][0]
            dec_shares_pairwise.append(self.committee_shared_sk[1] * c0)

        """Compute decryption for mi shares. 计算mi共享的解密。
            dec_target_mi is a list of AES ciphertext (with nonce) dec_target_mi是AES密文（带随机数）的列表
            decrypt each entry of dec_target_mi 解密dec_target_mi的每个条目
        """
        dec_shares_mi = []
        cnt = 0
        for id in client_id_list:
            sym_key = self.symmetric_keys[id]
            dec_entry = dec_target_mi[cnt]
            nonce = dec_entry[1]
            cipher_holder = AES.new(sym_key, AES.MODE_GCM, nonce=nonce)
            plaintext = cipher_holder.decrypt(dec_entry[0])
            plaintext = int.from_bytes(plaintext, 'big')
            dec_shares_mi.append(plaintext)
            cnt += 1

        clt_comp_delay = pd.Timestamp('now') - dt_protocol_start

        if __debug__:
            self.logger.info(f"[Decryptor] run time for reconstruction step: {clt_comp_delay}")

        self.sendMessage(self.serviceAgentID,
                         Message({"msg"                   : "SHARED_RESULT",
                                  "iteration"             : self.current_iteration,
                                  "sender"                : self.id,
                                  "shared_result_pairwise": util.serialize_dim1_ecp(dec_shares_pairwise),
                                  "shared_result_mi"      : util.serialize_dim1_list(dec_shares_mi),
                                  "committee_member_idx"  : self.committee_member_idx,
                                  }),
                         tag="comm_secret_sharing")

    def elgamal_enc_group(self, system_pk, ptxt_point):
        # the order of secp256r1
        n = ecchash.n

        # ptxt is in ECC group
        enc_randomness_bytes = get_random_bytes(32)
        enc_randomness = (int.from_bytes(enc_randomness_bytes, 'big')) % n

        # base point in secp256r1
        base_point = ECC.EccPoint(ecchash.Gx, ecchash.Gy)

        c0 = enc_randomness * base_point
        c1 = ptxt_point + (system_pk * enc_randomness)
        return (c0, c1)

    # ======================== UTIL ========================

    def recordTime(self, startTime, categoryName):
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

    def send_dh_public_key(self):
        for manage in self.kernel.all_manager:  # 遍历所有找到的管理者对象，然后将自己的 Diffie-Hellman 公钥（self.dh_key_obj.public_key）通过消息队列发送给每个管理者。
            self.kernel.prove_queue.put((
                MessageType.CLIENT_SWITCH_PUBLIC,
                ReqMsg(id=self.id,  # 客户端的 ID。
                       dh_public_key=self.dh_key_obj.public_key,  # 客户端的 Diffie-Hellman 公钥。
                       manage_id=manage.id)  # 接收公钥的管理者的 ID。
            ))  # 这里是客户端将其 Diffie-Hellman 公钥发送给管理者

    #收到管理者m的公钥，生成共享密钥kn,m
    def receive_manage_public_key(self, manage_id, manage_public_key):
        shared_key = self.dh_key_obj.compute_shared_secret(self.private_key, manage_public_key, mod_args.q)
        self.manages_dh_key[manage_id] = {
            "shared_key": shared_key,
            "manage"    : manage_public_key,
        }

    # 聚合Kn,m，生成Kn
    def aggregation_manages_public_key(self):
        if len(self.manages_dh_key) == len(self.kernel.all_manager):
            print(__file__, "\t", self.id, "\t开始聚合所有管理端key！")
            for key in self.manages_dh_key.values():
                self.agg_manages_keys += key["manage"]
            self.agg_finish = True

    #对称解密ct，得到Km和manage_alpha
    def receive_cipher_text(self, data: tuple):
        manage_id, agg_clients_keys, manage_alpha = data
        # 用对称密钥解密，但没体现对称密钥，如何引入(因为要和id一一对应)
        agg_clients_keys = aes_decrypt(agg_clients_keys)
        manage_alpha = aes_decrypt(manage_alpha)

        self.manages_dh_key[manage_id]["agg_clients_keys"] = agg_clients_keys
        self.manages_dh_key[manage_id]["manage_alpha"] = manage_alpha

    # 聚合alpha
    def aggregation_manages_alpha(self):
        if len(self.manages_dh_key) == len(self.kernel.all_manager):
            print(__file__, "\t", self.id, "\t开始聚合所有管理者的alpha！")
            for key in self.manages_dh_key.values():
                self.alpha += key["manage_alpha"]
            self.agg_alpha_finish = True

    # 聚合K
    def aggregation_manages_Km(self):
        if len(self.manages_dh_key) == len(self.kernel.all_manager):
            print(__file__, "\t", self.id, "\t开始聚合所有管理者的Km！")
            for key in self.manages_dh_key.values():
                self.K += key["agg_clients_keys"]
            self.agg_Km_finish = True

    # 客户端生成CLIENT_PRO并发送至服务器
    def send_Client_PRO(self):
        self.Client_PRO= self.agg_manages_keys + self.alpha * self.vec_n
        self.kernel.Client_PRO_queue.put((
            MessageType.CLIENT_PRO,
            ReqMsg(id=self.id,  # 客户端的 ID。
                   PRO_n=self.Client_PRO,  # 客户端的 PRO。
                   Service_ID=self.agents[service_id]# 传给服务器，服务器的 ID。这个怎么写
                   )
        ))

    # 客户端先接收PRO和Zt，然后用PRO验证Zt
    def verify_result(self,result,tuple):
        PRO, Zt= result
        K = self.K
        alpha = self.alpha
        if PRO - K- alpha * Zt ==0 :
            print("The server's aggregation result is correct")
        else:
            print("The server's aggregation result is wrong")
            #还得写一个结束进程的指令0