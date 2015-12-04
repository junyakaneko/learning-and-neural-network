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

__author__ = 'JunyaKaneko <jyuneko@hotmail.com>'


def gradient_method1d(f, rdom, ix, dx, epsilon, nstep=10000):
    x = ix
    value = f(x)

    if x + dx > rdom[1]:
        return None

    for s in range(0, nstep):
        grad = (f(x + dx) - f(x))/dx
        nx = x - epsilon * grad
        nvalue = f(nx)

        if not (rdom[0] <= nx <= rdom[1] - dx):
            break
        elif nvalue >= value:
            return x
        else:
            x = nx
            value = nvalue
    return None


if __name__ == '__main__':
    def f(x):
        return -2 * np.power(x, 2) + 5 * x + 2

    domain = [x/100 for x in range(-500, 500)]
    values = [f(x) for x in domain]
    x = gradient_method1d(lambda _x: -f(_x), [domain[0], domain[-1]], 0, 0.01, 0.01)

    pyplot.plot(domain, values)
    if x:
        pyplot.scatter(x, f(x), marker='o')
    pyplot.show()
