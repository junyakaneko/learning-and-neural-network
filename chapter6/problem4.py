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
        self._Ws = [np.random.uniform(-1.0, 1.0, (shape[si], shape[si - 1] + 1)) for si in range(1, len(shape))]
        self._Ys = []

    @property
    def nlayers(self):
        return len(self._Ws)

    def _get_nrows(self, layer_index):
        return len(self._Ws[layer_index])

    def _get_ncols(self, layer_index):
        return len(self._Ws[layer_index][0])

    def _do_back_propagation(self, x, t):
        y = self.get_values(x)
        dss = [2.0 * (y - t) * (np.ones(y.shape) - y) * y, ]
        for li in [-1 - _li for _li in range(self.nlayers - 1)]:
            dss = [np.zeros(self._get_ncols(li)), ] + dss
            for j in range(self._get_ncols(li)):
                for k in range(self._get_nrows(li)):
                    dss[0][j] += dss[1][k] * self._Ws[li][k][j]
                if j < self._get_ncols(li) - 1:
                    dss[0][j] *= (1 - self._Ys[li - 1][j]) * self._Ys[li - 1][j]
        return dss

    def _update_coefs(self, dss, epsilon):
        for li in range(self.nlayers):
            for j in range(self._get_nrows(li)):
                for i in range(self._get_ncols(li)):
                    self._Ws[li][j][i] -= epsilon * dss[li][j] * self._Ys[li][i]

    def learn(self, xs, ts, epsilon, threshold, nloops):
        for s in range(nloops):
            errors = []
            for xi, x in enumerate(xs):
                self._update_coefs(self._do_back_propagation(x, ts[xi]), epsilon)
                errors.append(np.linalg.norm(self.get_values(x) - ts[xi]))
            if np.max(errors) <= threshold:
                return s + 1, np.max(errors)
        return nloops, np.max(errors)

    def get_values(self, x):
        self._Ys = [np.append(x, [1.0, ]), ]
        for li in range(self.nlayers):
            self._Ys.append(np.array([sigmoid(s) for s in np.matmul(self._Ws[li], self._Ys[li])] + ([1.0, ] if li < self.nlayers - 1 else [])))
        return self._Ys[-1]

    def get_binaries(self, x, threshold=0.5):
        return np.array([0.0 if y < threshold else 1.0 for y in self.get_values(x)])


if __name__ == '__main__':
    networks = []
    for i in range(10):
        networks.append(NeuralNetwork((3, 5, 1)))

    xs = [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    ts = [[0.0, ], [0.0, ], [1.0, ], [0.0, ], [1.0, ], [1.0, ]]

    print('===============================')
    print('Learnings')
    print('===============================')
    for network in networks:
        print(network.learn(xs, ts, 0.05, 0.05, 10000))

    xs += [[0.0, 1.0, 0.0], [0.0, 1.1, 1.1]]
    ts += [[1.0, ], [0.0, ]]

    print()
    print('===============================')
    print('Answers')
    print('===============================')
    for network in networks:
        print('=============1=============')
        for xi, x in enumerate(xs):
            print('Answer:', ts[xi][0], 'Output:', network.get_binaries(x)[0])