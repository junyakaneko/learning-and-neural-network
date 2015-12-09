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


class Element:
    def __init__(self, w, epsilon=0.5, nloops=100):
        self._w = np.append([np.random.uniform(-5, 5), ], w)
        self._epsilon = epsilon
        self._nloops = nloops

    def get_value(self, x):
        return sigmoid(self._w.dot(np.append([1.0, ], x)))

    def update(self, datum):
        x = datum[:-1]
        y = datum[-1]
        val = self.get_value(x)
        self._w -= self._epsilon * 2.0 * (val - y) * val * (1.0 - val) * np.append([1.0, ], x)

    def learn(self, data):
        for i in range(self._nloops):
            for datum in data:
                self.update(datum)

if __name__ == '__main__':
    data = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 3.0, 1.0], [2.0, 1.0, 1.0], [1.5, 2.0, 1.0]])

    elements = [
        Element([np.random.uniform(-5, 5), np.random.uniform(-5, 5)], nloops=10),
        Element([np.random.uniform(-5, 5), np.random.uniform(-5, 5)], nloops=100),
        Element([np.random.uniform(-5, 5), np.random.uniform(-5, 5)], nloops=1000),
    ]

    for element in elements:
        element.learn(data)

    x0s, x1s, ys = np.hsplit(data, 3)

    fig = pyplot.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(x0s, x1s, ys, label='Actual')
    for element in elements:
        ax.plot(x0s, x1s, [element.get_value(datum[:-1]) for datum in data], label='%s' % element._nloops)
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.title('Actual data and approximations')
    pyplot.legend(loc="upper left")

    pyplot.subplot(212)
    for i, element in enumerate(elements):
        pyplot.bar([i, ],
                   np.sum([np.power(datum[-1] - element.get_value(datum[:-1]), 2) for datum in data]),
                   align='center', width=0.4)
    pyplot.xticks(range(len(elements)), [element._nloops for element in elements])
    pyplot.xlabel('Number of learning loops')
    pyplot.ylabel('Square root error')
    pyplot.title('Square root errors')

    pyplot.show()
