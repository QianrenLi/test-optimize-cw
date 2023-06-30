import threading

import numpy as np


class test_a():
    def __init__(self):
        self.list_a = []

    def append_v(self, a):
        self.list_a.append(a)
        return self
    
    def clear(self):
        self.list_a.clear()
        return self

    def get_a(self):
        print(self.list_a)
        return self


class test_b():
    def __init__(self, list_b):
        self.list_b = list_b

    def get_a(self):
        print(self.list_b)
        return self


test_list = []

test_A = test_a()
test_B = test_b(test_A.list_a)

test_A.append_v(50).get_a()
test_B.get_a()
test_A.clear().get_a()
test_B.get_a()