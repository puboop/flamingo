# **************************************
# --*-- coding: utf-8 --*--
# @Time    : 2024-10-22
# @Author  : white
# @FileName: AesCrypto.py
# @Software: PyCharm
# **************************************
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64

KEY = 'B4274920A7021C423E55AD15AB3C79B4'  # 密钥长度必须为16、24或32字节


def aes_encrypt(plain_text, _key=None):
    key = _key if _key else KEY
    # 生成16字节的随机密钥向量
    iv = get_random_bytes(16)

    # 使用CBC模式，创建AES加密对象
    cipher = AES.new(key.encode(), AES.MODE_CBC, iv)

    # 进行填充，保证待加密数据长度为16的倍数
    padded_text = _pad(plain_text)

    # 加密数据
    encrypted_text = cipher.encrypt(padded_text.encode('utf-8'))

    # 返回iv和加密后的数据
    return base64.b64encode(iv + encrypted_text).decode()


def aes_decrypt(encrypted_text, _key=None):
    key = _key if _key else KEY
    # 解码加密后的数据
    encrypted_text = base64.b64decode(encrypted_text)

    # 获取iv和加密后的数据
    iv = encrypted_text[:16]
    encrypted_text = encrypted_text[16:]

    # 使用CBC模式，创建AES解密对象
    cipher = AES.new(key.encode(), AES.MODE_CBC, iv)

    # 解密数据
    decrypted_text = cipher.decrypt(encrypted_text)

    # 去除填充的数据
    unpadded_text = _unpad(decrypted_text)

    # 返回解密后的数据
    return unpadded_text.decode('utf-8')


def _pad(s):
    block_size = AES.block_size
    padding = block_size - (len(s) % block_size)
    return s + padding * chr(padding)


def _unpad(s):
    return s[:-s[-1]]
