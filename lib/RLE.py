import numpy as np
import random


class generator:
    '''Generator class

    Input:
    ------
    M: the number of modes

    N: the number of particles

    Q: the guessed compression

    maxiter: the maximal number of trials

    Ouput:
    ------
    multiple functions that generates the generator matrix
    '''

    def __init__(self, M, N, Q, maxiter, bias, bound):
        self.M = M
        self.N = N
        self.Q = Q
        self.hd = 2
        self.bias = bias
        self.bound = bound
        # 
        if len(bias) == 0:
            def generate_binary_vectors(M, Q, N, excite, sample):
                '''
                Generate Training data
                '''
                binary_vectors = set()
                vector = []
                for i in range(len(excite)):
                    for j in range(sample[i]):
                        vector = ['0'] * M
                        for idx in random.sample(list(range(Q, M)), excite[i]):
                            vector[idx] = '1'
                        for idx in random.sample(list(range(Q)), N - excite[i]):
                            vector[idx] = '1'
                        binary_vectors.add(''.join(vector))

                return list(binary_vectors)

            s = generate_binary_vectors(self.M, self.Q, self.N, [1, 2], [500, 500])
        else:
            s = bias

        lc = int(np.ceil(Q / 2))
        if lc % 2 == 1:
            lc -= 1
        self.s = s
        self.lc = lc

        self.maxiter = maxiter

    def modtest2(self):
        '''generator matrix with substantially simpler check
        '''
        #    s10=np.array(to10(s))
        aa = 0
        for _ in range(self.maxiter):
            B, a = self.Bg()
            ps = []
            for i in self.s:
                v = self.to10(list(i))
                v = B.dot(v) % 2
                vs = ''.join('0' if j == 0 else '1' for j in v)
                ps.append(vs)
            aa += len(self.s)

            g = len(set(ps))
            f = len(self.s)
            print('Injectivity: ', g, f, g / f)
            if g / f > self.bound:
                return B

    def Bg(self):
        B = list(np.eye(self.Q, dtype=np.int32))
        a = []

        while len(B) < self.M:
            tmp = np.zeros(self.Q, dtype=np.int32)
            ls = random.sample(list(range(int(self.Q))), self.lc)
            tmp[ls] = 1
            for _ in range(len(tmp)):
                atmp = "".join(map(str, tmp))
            c = [self.hd]
            c.extend(
                sum(np.array(list(a[i])) != np.array(list(atmp)))
                for i in range(len(a))
            )
            while min(c) < self.hd:
                tmp = np.zeros(self.Q, dtype=np.int32)
                ls = random.sample(list(range(int(self.Q))), self.lc)
                tmp[ls] = 1
                for i in range(len(tmp)):
                    atmp = "".join(map(str, tmp))
                c = [self.hd]
                c.extend(
                    sum(np.array(list(a[i])) != np.array(list(atmp)))
                    for i in range(len(a))
                )
            B.append(tmp)
            a.append(atmp)

        B = np.array(B).T
        return B, a

    def to10(self, s):
        a = []
        a.extend(int(str(i), 2) for i in s)
        return np.array(a)


def encoder(orb, e, Q, maxiter=5000, bias=[], bound=0):
    '''Compute the compression matrix with the molecule information.

    Input:
    ------
    mol: molecule object.

    Q: the guessed qubit number.

    freeze: freezing core orbitals.

    RHF: restricted Hartree-Fock.

    Output:
    -------
    A: a (Q, M) matrix
    '''
    g = generator(orb, e, Q, maxiter, bias, bound)
    return g.modtest2()
