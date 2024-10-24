# **************************************
# --*-- coding: utf-8 --*--
# @Time    : 2024-10-21
# @Author  : white
# @FileName: DiffieHellman.py
# @Software: PyCharm
# **************************************
import random
import math

q = 23  # 实际中应当使用足够大的素数
g = 5  # 基数（生成元），需确保g与q-1互质


class ModArgs:
    def __init__(self, q=None, g=None):
        self.low = 10000
        self.high = 100000
        if q is not None and g is not None:
            self.q = q
            self.g = g
        else:
            self.q = self.get_prime()
            self.g = self.get_base()

    def get_base(self):
        while True:
            prime_int = random.randint(self.low, self.high)
            if self.is_prime(prime_int) and prime_int != self.q:
                if self.q > prime_int and (self.q % prime_int) != 0:
                    return prime_int
                if self.q < prime_int and (prime_int % self.q) != 0:
                    return prime_int

    def get_prime(self):
        while True:
            prime_int = random.randint(self.low, self.high)
            if self.is_prime(prime_int):
                return prime_int

    def is_prime(self, random_int):
        # 处理小于或等于 1 的情况
        if random_int <= 1:
            return False
        # 2 和 3 是素数
        if random_int == 2 or random_int == 3:
            return True
        # 排除偶数
        if random_int % 2 == 0:
            return False
        # 排除 3 的倍数
        if random_int % 3 == 0:
            return False
        # 检查从 5 到 sqrt(n) 之间的所有数
        for i in range(5, int(math.sqrt(random_int)) + 1, 6):
            if random_int % i == 0 or random_int % (i + 2) == 0:
                return False
        return True


mod_args = ModArgs()


class DHKeyExchange:
    def __init__(self, q, g, private_key):
        """
        初始化DH密钥交换类，p为素数，g为生成元
        """
        self.q = q
        self.g = g
        self.public_key = self.compute_public_key(private_key)

    def generate_private_key(self, private):
        """
        生成私钥
        私钥应该是一个随机数，这里简化处理，随机生成一个小于p的数
        """
        return private

    def compute_public_key(self, private_key):
        """
        根据私钥计算公钥
        """
        return pow(self.g, private_key, self.q)

    @staticmethod
    def compute_shared_secret(private_key, other_public_key, q):
        """
        根据本地私钥和对方公钥计算共享密钥
        """
        return pow(other_public_key, private_key, q)
