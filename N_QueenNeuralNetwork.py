from numpy import random
import numpy as np

class N_QueenNeuralNetwork:
    def __init__(self, n):
        self.n = n
        self.u = np.zeros((self.n, self.n))
        self.v = random.randint(0, 2, (n, n))
        self.forget_threshold = -100
        self.a = 1
        self.b = 1
        self.c = 1

    def train(self):
        v, n, u = self.v, self.n, self.u
        for x in range(n):
            for y in range(n):
                u[x, y] += self.delta(x, y)
                if u[x, y] < self.forget_threshold:
                    u[x] = np.zeros(n)
                vx = np.zeros(n)
                vx[np.argmax(u[x])] = 1
                v[x] = vx

    def delta(self, x, y):
        v, n = self.v, self.n
        row_s = np.sum(v[x])
        col_s = np.sum(v.T[y])
        lr_s = 0
        for k in range(-min(x, y), min(n-x, n-y)):
            if k != 0:
                lr_s += v[x+k, y+k]
        ur_s = 0
        for k in range(-min(x, n-y-1), min(n-x, y+1)):
            if k != 0:
                ur_s += v[x+k, y-k]
        row_d = -self.a * (row_s - 1)
        col_d = -self.a * (col_s - 1)
        lr_d = -self.b * lr_s
        ur_d = -self.b * ur_s
        row_h = self.c * (row_s == 0)
        col_h = self.c * (col_s == 0)
        return row_d + col_d + ur_d + lr_d + row_h + col_h

    def check(self):
        v, n = self.v, self.n
        if len(v[v]) != n:
            return False
        if not (np.sum(v, axis=0) == 1).all():
            return False
        if not (np.sum(v, axis=1) == 1).all():
            return False
        for bx in range(n):
            sm1 = sm2 = 0
            for p in range(bx+1):
                sm1 += v[bx-p, p]
                sm2 += v[bx-p, n-p-1]
            if sm1 > 1 or sm2 > 1:
                return False
            sm1 = sm2 = 0
            for p in range(n-bx):
                sm1 += v[bx+p, p]
                sm2 += v[bx+p, n-p-1]
            if sm1 > 1 or sm2 > 1:
                return False
        return True

    def get_queen_points(self):
        points = set()
        for x in range(self.n):
            for y in range(self.n):
                if(self.v[x, y]):
                    points.add((x, y))
        return points
