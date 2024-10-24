# **************************************
# --*-- coding: utf-8 --*--
# @Time    : 2024-10-21
# @Author  : white
# @FileName: SA_Manage.py
# @Software: PyCharm
# **************************************
import random

from util import util
from util.DiffieHellman import DHKeyExchange, mod_args
from message.Message import Message, MessageType
from message.new_msg import ReqMsg
# from .SA_ClientAgent import SA_ClientAgent as ClientAgent
from util.AesCrypto import aes_encrypt, aes_decrypt


class SA_Manage:
    def __init__(self, id, name, type, key_length=32):
        self.id = id
        self.key = util.read_key(f"pki_files/manager{self.id}.pem")
        # 把对应的私钥赋值给self.secret_key
        self.secret_key = self.key.d

        self.private_key = self.key.export_key(format='PEM')  # 以 PEM 格式导出私钥。
        self.public_key = self.key.public_key().export_key(format='PEM')  # 以 PEM 格式导出公钥。
        # 创建一个 DHKeyExchange 对象，用于 Diffie-Hellman 密钥交换，使用的参数为 mod_args.q 和 mod_args.g，以及实例的私钥。
        self.dh_key_obj = DHKeyExchange(mod_args.q, mod_args.g, self.private_key)
        # 一个字典，用于存储与客户端的 Diffie-Hellman 密钥交换相关的信息。
        self.clients_dh_key = dict()

        # 是否完成了对km,n的聚合操作，生成Km,初始值为 False。
        self.agg_finish = False
        # k1,即Km
        self.agg_clients_keys = ""
        # 随机数manage_alpha
        self.manage_alpha = random.randint(10, 100)

    # 发送同态加密后的密文ct。ct包括Km和manage_alpha
    def send_cipher_text(self):
        # 空字典，存放每个ct
        encrypt_data = {}
        # 遍历系统中的所有客户端
        for client in self.kernel.all_clients:
            share_key = self.clients_dh_key[client.id]["shared_key"]
            encrypt_data[client] = (
                # 管理者id
                self.id,
                # 对Km进行aes加密 这里应该是有问题的，
                # agg_clients_keys是要被加密的明文，是用共享密钥加密的，但这里没体现共享密钥
                aes_encrypt(self.agg_clients_keys, share_key),
                # 对manage_alpha进行aes加密，问题同上。
                aes_encrypt(self.manage_alpha, share_key),
                # 还有一点，应该是manage_alpha和agg_clients_keys先拼接，再加密
                # 客户端那边的话就是先解密，再分割了。
            )
        return encrypt_data

    # 聚合Km,n,生成Km
    def aggregation_clients_public_key(self):
        # 条件检查 self.clients_dh_key 中的公钥数量是否与 self.kernel.all_clients 中的客户端数量相等。
        # self.clients_dh_key就是管理者根据客户端公钥生成的共享密钥
        if len(self.clients_dh_key) == len(self.kernel.all_clients):
            # 打印日志信息，表示开始聚合客户端的公钥。文件路径、谁在执行操作、日志信息
            print(__file__, "\t", self.id, "\t开始聚合所有客户端key！")
            for key in self.clients_dh_key.values():
                self.agg_clients_keys += key["client_key"]
            self.agg_finish = True

    def send_dh_public_key(self, client_id, client_dh_key):
        # 管理者在 self.clients_dh_key 中保存收到的客户端公钥，键为 client_id，值为客户端的 Diffie-Hellman 公钥。
        # 计算管理者与客户端之间的共享密钥。
        shared_key = self.dh_key_obj.compute_shared_secret(self.private_key, client_dh_key, mod_args.q)
        # 管理者将计算出的共享密钥保存到 self.dh_key_obj.shared_key 中。
        self.clients_dh_key[client_id] = {
            "client_key": client_dh_key,
            "shared_key": shared_key,
        }

        # 将一轮训练里产生的每一个共享密钥都保存起来。
        # self.dh_key_obj.shared_keys = getattr(self.dh_key_obj, 'shared_keys', []) + [shared_key]

        # 管理者对共享密钥求和生成K管
        # self.Manage_sum_key = getattr(self, 'Manage_dh_key', 0) + shared_key
        self.kernel.prove_queue.put((
            MessageType.MANAGE_SWITCH_PUBLIC,
            # 管理者接收到客户端的公钥后，计算双方的共享密钥，并将自己的公钥发送回客户端。
            ReqMsg(id=self.id,  # 管理者的 ID。
                   dh_public_key=self.dh_key_obj.public_key,  # 管理者的 Diffie-Hellman 公钥。
                   client_id=client_id)  # 客户端的 ID。
        ))

    def kernelInitializing(self, kernel):
        self.kernel = kernel

    def kernelStarting(self, time):
        pass
