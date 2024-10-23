from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64

# 密钥长度必须为16、24或32字节
KEY = 'B4274920A7021C423E55AD15AB3C79B4'  # 示例32字节密钥


def _pad(text, block_size=16):
    """填充函数，保证文本长度是block_size的倍数"""
    pad_len = block_size - len(text) % block_size
    return text + chr(pad_len) * pad_len


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


def encrypt_two_texts(text1, text2, _key=None):
    # 将两个明文拼接在一起
    combined_text = text1 + text2

    # 调用 aes_encrypt 对拼接后的明文进行加密
    encrypted_text = aes_encrypt(combined_text, _key)

    return encrypted_text


# 主测试函数
if __name__ == "__main__":
    text1 = "Hello"
    text2 = "World"
    encrypted_output = encrypt_two_texts(text1, text2)
    print("Encrypted Text:", encrypted_output)
