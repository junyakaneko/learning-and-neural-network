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


def hill_climbing(values, initial_index, nsteps=100):
    i = initial_index
    for s in range(0, nsteps):
        if i - 1 < 0 or i + 1 >= len(values):
            break
        if values[i - 1] <= values[i] and values[i + 1] <= values[i]:
            return i
        elif values[i - 1] > values[i]:
            i -= 1
        else:
            i += 1
    return None


if __name__ == '__main__':
    def f(x):
        return 2 * np.power(x, 2) - 3 * x + 5

    domain = np.array([i for i in range(-5, 6)])
    values = np.array([-f(i) for i in domain])

    i = hill_climbing(values, int(len(domain)/2))

    pyplot.plot(domain, -values)
    pyplot.scatter([domain[i], ], [-values[i], ], marker='o')
    pyplot.xlabel('x')
    pyplot.ylabel('f(x)')
    pyplot.show()
