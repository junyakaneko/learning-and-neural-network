# This file is part of "Junya's self learning project about Neural Network."
#
# "Junya's self learning project about Neural Network"
# is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# "Junya's self learning project about Neural Network"
# is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#
# (c) Junya Kaneko <jyuneko@hotmail.com>


import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from chapter3.problem1 import sigmoid


__author__ = 'Junya Kaneko <jyuneko@hotmail.com>'


class NeuralNetwork:
    def __init__(self, shape):
        self.W = [np.random.choice((-1.0, 1.0), (shape[i + 1], shape[i] + (1 if i < len(shape) - 1 else 0))) for i in range(len(shape) - 1)]
        self.dEdS = [np.zeros(len(layer)) for layer in self.W]
        self.Y = []

    def error(self, t):
        assert len(self.Y[-1]) == len(t)
        return np.power(self.Y[-1] - t, 2).sum()

    def f(self, s):
        return sigmoid(s)

    def dedy(self, y, t):
        assert isinstance(y, np.ndarray) and isinstance(t, (np.ndarray, list))
        return 2.0 * (y - t)

    def dyds(self, l):
        return (np.ones(self.Y[l].shape) - self.Y[l]) * self.Y[l]

    def activate(self, x):
        assert len(x) == len(self.W[0][0]) - 1
        self.Y = [np.append(x, [1.0, ]), ]
        for l in range(len(self.W)):
            self.Y.append(np.ones(len(self.W[l]) + (1 if l < len(self.W[l]) - 1 else 0)))
            for i, w in enumerate(self.W[l]):
                self.Y[-1][i] = sigmoid(w.dot(self.Y[-2]))
        return self.Y[-1]

    def propagate_backward(self, t):
        for l in [-1 - _l for _l in range(len(self.W))]:
            for j in range(len(self.W[l])):
                for i in range(len(self.W[l][j])):
                    if l == -1:
                        self.dEdS[l][j] = self.dedy(self.Y[-1], t)[j] * self.dyds(-1)[j]
                    else:
                        for k in range(len(self.W[l + 1])):
                            self.dEdS[l][j] += self.dEdS[l + 1][k] * self.W[l + 1][k][j]
                        self.dEdS[l][j] *= self.dyds(l)[j]

    def update_coefs(self, epsilon):
        for l in [-1 - _l for _l in range(len(self.W))]:
            for j in range(len(self.W[l])):
                for i in range(len(self.W[l][j])):
                    self.W[l][j][i] -= epsilon * self.dEdS[l][j] * self.Y[l - 1][i]

    def learn(self, xs, ts, epsilon=0.05, threshold=0.01, nloops=10000):
        for n in range(nloops):
            errors = []
            for xi, x in enumerate(xs):
                self.activate(x)
                errors.append(self.error(ts[xi]))
            if max(errors) < threshold:
                return n
            for xi, x in enumerate(xs):
                self.activate(x)
                self.propagate_backward(ts[xi])
                self.update_coefs(epsilon)
        return nloops


if __name__ == '__main__':
    networks = [NeuralNetwork((3, 5, 1)) for i in range(5)]

    xs = [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    ts = [[0.0, ], [0.0, ], [1.0, ], [0.0, ], [1.0, ], [1.0, ]]

    print('===============================')
    print('Learnings')
    print('===============================')
    for network in networks:
        print('Number of loops', network.learn(xs, ts))

    xs += [[0.0, 1.0, 0.0], [0.0, 1.1, 1.1]]
    ts += [[1.0, ], [0.0, ]]

    print()
    print('===============================')
    print('Answers')
    print('===============================')
    for network in networks:
        print('=============1=============')
        for xi, x in enumerate(xs):
            print('Answer:', ts[xi][0], 'Output:', network.activate(x)[0])
