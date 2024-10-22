# **************************************
# --*-- coding: utf-8 --*--
# @Time    : 2024-10-21
# @Author  : white
# @FileName: new_msg.py
# @Software: PyCharm
# **************************************


class ReqMsg:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
