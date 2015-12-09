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


__author__ = 'Junya Kaneko <jyuneko@hotmail.com>'


class XY:
    def __init__(self, coef):
        self._coef = coef
        self._v = 0
        self._w = 0

    def set_input(self, v, w):
        self._v = v
        self._w = w

    def update(self, value):
        pass

    @property
    def value(self):
        return None


class X(XY):
    def update(self, value):
        self._coef += 0.01 * value * 2 * self.value

    @property
    def value(self):
        return self._v + self._coef * self._w


class Y(XY):
    def update(self, value):
        self._coef += 0.01 * value * np.power(self._v, 2)

    @property
    def value(self):
        return self._coef * np.power(self._v, 2) * self._w


class Z:
    def __init__(self, c):
        self._c = c
        self._x = None
        self._y = None
        self._w = 0

    def set_input(self, w, x, y):
        self._w = w
        self._x = x
        self._y = y

    def update(self, true_output):
        d = -2 * self._w * (self.value - true_output)
        self._c += 0.01 * d
        return d

    @property
    def value(self):
        return np.power(self._x.value, 2) + self._y.value - self._c * self._w


class Network:
    def __init__(self, a, b, c):
        self._v = 0
        self._w = 0
        self._x = X(a)
        self._y = Y(b)
        self._z = Z(c)

    def set_input(self, v, w):
        self._v = v
        self._w = w
        self._x.set_input(v, w)
        self._y.set_input(v, w)
        self._z.set_input(w, self._x, self._y)

    def update(self, true_value):
        d = self._z.update(true_value)
        self._x.update(d)
        self._y.update(d)

    @property
    def value(self):
        return self._z.value


if __name__ == '__main__':
    network = Network(-1, 3, 2)

    network.set_input(1, -2)

    for l in range(30):
        network.update(2)
        print(l, network.value)
