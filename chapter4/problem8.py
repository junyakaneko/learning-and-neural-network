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


__author__ = 'JunyaKaneko <jyuneko@hotmail.com>'


def is_in_domain(x, rdom):
    for i, d in enumerate(rdom):
        if not (d[0] <= x[i] <= d[1]):
            return False
    return True


def gradient_method(f, rdom, ix, dx, epsilon, nstep=10000):
    x = ix if isinstance(ix, np.ndarray) else np.array(ix, np.float)
    nx = ix if isinstance(ix, np.ndarray) else np.array(ix, np.float)
    value = f(x)
    if not is_in_domain(x + dx, rdom):
        return None
    for s in range(0, nstep):
        for i in range(len(dx)):
            dxi = np.zeros(len(dx), np.float)
            dxi[i] = dx[i]
            for j in range(len(dxi)):
                if j != i:
                    assert dxi[j] == 0.0
                else:
                    assert dxi[j] == dx[i]
            nx[i] = x[i] - epsilon * (f(x + dxi) - f(x))/dxi[i]
        nvalue = f(nx)
        if not is_in_domain(nx, rdom) or not is_in_domain(nx + dx, rdom):
            return None
        elif nvalue >= value:
            return x
        else:
            x = nx
            value = nvalue
    return None


if __name__ == '__main__':
    def f(x):
        return np.power(x[0]-x[2], 2) + np.power(x[0] + x[1], 2) + np.power(x[1] - 1, 2) + \
               np.power(2.0 * x[3] + x[0] - x[1], 2) + np.power(x[4] + x[0] - x[2], 2)

    rdom = np.array([-5.0, 5.0] * 5)
    rdom.shape = (5, 2)
    ix = [0.0, 0.0, 0.0, 0.0, 0.0]
    dx = [0.01, 0.01, 0.01, 0.01, 0.01]
    epsilon = 0.01

    x = gradient_method(f, rdom, ix, dx, epsilon)
    print(np.round(x, 1))