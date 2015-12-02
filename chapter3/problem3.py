from chapter3.problem1 import sigmoid
import numpy as np
import pandas as pd
from matplotlib import pyplot


class TimeVariantElement:
    def __init__(self, name, w, initial_state, alpha=1.0, theta=0.0, time_delta=0.001):
        self._name = name
        self._w = w if isinstance(w, np.ndarray) else np.array(w)
        self._alpha = alpha
        self._theta = theta
        self._time_delta = time_delta
        self._states = [initial_state, ]
        self._values = [sigmoid(self._states[-1], self._alpha, self._theta), ]

    def update(self, input):
        _input = input if isinstance(input, np.ndarray) else np.array(input)
        self._states.append(self._states[-1] + self._time_delta * (-self._states[-1] + self._w.dot(_input)))
        self._values.append(sigmoid(self._states[-1], alpha=self._alpha, theta=self._theta))

    @property
    def state(self):
        return self._states[-1]

    @property
    def value(self):
        return self._values[-1]

    @property
    def domain(self):
        return np.array([i for i in range(0, len(self._states))], np.float) * self._time_delta

    @property
    def data_frame(self):
        bin_values = []
        for state in self._states:
            bin_values.append(0 if sigmoid(np.round(state, 4)) < 0.5 else 1)
        return pd.DataFrame(data={'states': self._states, 'values': self._values, 'binaries': bin_values}, index=self.domain)

    @property
    def name(self):
        return self._name


if __name__ == '__main__':
    """
    alpha の値を大きくするにつれて、x1, x2, x3 の内部状態は時間とともに 0 に負の方向から限りなく近づく様子が見られる。
    しかし、近づくのに限界があるため、有効数字を設定してやると良い。
    """
    x1 = TimeVariantElement('x1', [0.0, -2.0, -2.0], 0.0, alpha=1000000.0)
    x2 = TimeVariantElement('x2', [-1.0, -2.0, 0.0], 0.0, alpha=1000000.0)
    x3 = TimeVariantElement('x3', [0.0, -3.0, -1.0], 0.0, alpha=1000000.0)

    for i in range(0, 5000):
        x = [x1.value, x2.value, x3.value]
        x1.update(x)
        x2.update(x)
        x3.update(x)

    fig, axes = pyplot.subplots(nrows=3, ncols=3)

    for i, elem in enumerate((x1, x2, x3)):
        elem.data_frame['states'].plot(ax=axes[0, i], color='blue')
        elem.data_frame['values'].plot(ax=axes[1, i], color='green')
        elem.data_frame['binaries'].plot(ax=axes[2, i], color='red')
        for j in (0, 1, 2):
            axes[j, i].set_title(elem.name)
            axes[j, i].set_xlabel('t')
        axes[0, i].set_ylabel('state')
        axes[1, i].set_ylabel('output')
        axes[2, i].set_ylabel('binary_output')
    pyplot.tight_layout()
    pyplot.show()
