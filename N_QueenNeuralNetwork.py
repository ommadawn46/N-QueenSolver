from numpy import random
import numpy as np

class N_QueenNeuralNetwork:
    def __init__(self, n):
        self.n = n
        self.u = random.random((n, n))
        self.v = np.zeros((n, n))
        for i in range(n):
            self.v[i, np.argmax(self.u[i])] = 1
        self.forget_threshold = -100
        self.a = 1
        self.b = 1
        self.c = 1
        self.d = 2

    def train(self):
        v, n, u = self.v, self.n, self.u
        for x in range(n):
            u[x] += self.delta(x)
            if np.min(u[x]) < self.forget_threshold:
                u[x] = random.random((n,))
            v[x] = np.zeros(n)
            v[x, np.argmax(u[x])] = 1

    def delta(self, x):
        v, n = self.v, self.n
        col_s = np.sum(v, axis=0)
        lr_s, ur_s = [], []
        for y in range(n):
            lr_r = np.arange(-min(x, y), min(n-x, n-y))
            ur_r = np.arange(-min(x, n-y-1), min(n-x, y+1))
            lr_s.append(np.sum(v[x+lr_r, y+lr_r]) - v[x, y])
            ur_s.append(np.sum(v[x+ur_r, y-ur_r]) - v[x, y])
        lr_s, ur_s = np.array(lr_s), np.array(ur_s)
        col_d = -self.a * (col_s - 1)
        lr_d = -self.b * lr_s
        ur_d = -self.b * ur_s
        col_h = self.c * (col_s == 0)
        cross_h = self.d * ((col_s == 0) & (lr_s == 0) & (ur_s == 0))
        return col_d + ur_d + lr_d + col_h + cross_h

    def check(self):
        v, n = self.v, self.n
        if not (np.sum(v, axis=0) == 1).all():
            return False
        for bx in range(n):
            p = np.arange(bx+1)
            if np.sum(v[-p+bx, p]) > 1 or np.sum(v[-p+bx, -p+n-1]) > 1:
                return False
            p = np.arange(n-bx)
            if np.sum(v[p+bx, p]) > 1 or np.sum(v[p+bx, -p+n-1]) > 1:
                return False
        return True

    def get_queen_points(self):
        return set((x, y) for x, y in np.argwhere(self.v == 1))
