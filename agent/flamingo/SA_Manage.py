# **************************************
# --*-- coding: utf-8 --*--
# @Time    : 2024-10-21
# @Author  : white
# @FileName: SA_Manage.py
# @Software: PyCharm
# **************************************
from util import util
from util.DiffieHellman import DHKeyExchange, mod_args
from message.Message import Message, MessageType
from message.new_msg import ReqMsg


class SA_Manage:
    def __init__(self, id, name, type, key_length=32):
        self.id = id
        self.key = util.read_key(f"pki_files/client{self.id}.pem")
        self.secret_key = self.key.d

        self.private_key = self.key.export_key(format='PEM')
        self.public_key = self.key.public_key().export_key(format='PEM')

        self.dh_key_obj = DHKeyExchange(mod_args.q, mod_args.g, self.private_key)

        self.clients_dh_key = dict()

    def send_dh_public_key(self, client_id, client_dh_key):
        self.clients_dh_key[client_id] = client_dh_key
        shared_key = self.dh_key_obj.compute_shared_secret(self.private_key, client_dh_key, mod_args.q)
        self.dh_key_obj.shared_key = shared_key
        self.kernel.prove_queue.put((
            MessageType.MANAGE_SWITCH_PUBLIC,
            ReqMsg(id=self.id,
                   dh_public_key=self.dh_key_obj.public_key,
                   client_id=client_id)
        ))


    def kernelInitializing(self, kernel):
        self.kernel = kernel

    def kernelStarting(self, time):
        pass
