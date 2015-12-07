import numpy as np
from matplotlib import pyplot
from time import time


class Learner:
    def __init__(self, a, da=0.01, epsilon=0.01, nsteps=1000):
        self._a = a
        self._da = da
        self._epsilon = epsilon
        self._nsteps = nsteps
        self._learning_time = 0

    @property
    def learning_time(self):
        return self._learning_time

    def _get_error(self, data, a):
        return np.array([np.power(x[1] - a * x[0], 2) for x in data]).sum()

    def learn(self, data):
        prev_e = np.infty
        start = time()
        for s in range(self._nsteps):
            de = (self._get_error(data, self._a) - self._get_error(data, self._a + self._da)) / self._da
            a = self._a + self._epsilon * de
            e = self._get_error(data, a)
            if e >= prev_e:
                self._learning_time = time() - start
                return True
            else:
                self._a = a
                prev_e = e
        return False

    def get_value(self, x):
        return self._a * x


if __name__ == '__main__':
    data = np.array([[1, 1], [2, 2.5], [3, 2.5], [4, 4.5], [5, 4.5]])

    b_learner = Learner(1)
    s_learner = Learner(1)

    b_success = b_learner.learn(data)
    s_success = False

    s_learning_times = []
    for datum in data:
        s_success = s_learner.learn([datum, ])
        if s_success:
            s_learning_times.append(s_learner.learning_time)
        else:
            s_success = False
            break

    if b_success and s_success:
        x1, x2 = np.hsplit(data, 2)

        pyplot.subplot(2, 1, 1)
        pyplot.scatter(x1, x2, label='Actual data')
        pyplot.plot(x1, [b_learner.get_value(x) for x in x1], label='Batch')
        pyplot.plot(x1, [s_learner.get_value(x) for x in x1], label='Sequential')
        pyplot.title('Actual data v.s. approx line')
        pyplot.xlabel('x1')
        pyplot.ylabel('x2')
        pyplot.legend(loc="upper left")

        pyplot.subplot(2, 1, 2)
        pyplot.bar([0, 1], [b_learner.learning_time, np.array(s_learning_times).sum()], align="center", width=0.4)
        pyplot.title("Batch's v.s. sequential's learning time")
        pyplot.xticks([0, 1], ['Batch', 'Sequential'])

        pyplot.tight_layout()
        pyplot.show()
