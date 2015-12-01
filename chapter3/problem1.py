import numpy as np
import logging


def sigmoid(z, alpha=1, theta=0):
    return 1 / (1 + np.exp(-alpha * (z - theta)))


class Element:
    def __init__(self, w, alpha, theta):
        self._w = w if isinstance(w, np.ndarray) else np.array(w)
        self._alpha = alpha
        self._theta = theta

    def out(self, input, alpha=None, theta=None):
        _input = input if isinstance(input, np.ndarray) else np.array(input)
        _alpha = alpha if alpha else self._alpha
        _theta = theta if theta else self._theta
        logging.debug('PARAMETERS (alpha, theta): (%s, %s)' % (_alpha, _theta))
        logging.debug('INPUT: (%s), SIGMOID: %s' % (','.join([str(i) for i in _input]), sigmoid(self._w.dot(_input), _alpha, _theta)))
        if sigmoid(self._w.dot(_input), _alpha, _theta) > 0.5:
            logging.debug('OUTPUT: %s' % 1)
            return 1
        else:
            logging.debug('OUTPUT: %s' % 0)
            return 0

    def __call__(self, input, alpha=None, theta=None):
        return self.out(input, alpha, theta)


class AndElement(Element):
    def __init__(self):
        super().__init__(w=[1, 1], alpha=1, theta=1.5)


class OrElement(Element):
    def __init__(self):
        super().__init__(w=[1, 1], alpha=1, theta=0.5)


class XorElement:

    def out(self, input):
        or_element = OrElement()

        i1 = np.array([input[0], -input[1]])
        i2 = np.array([-input[0], input[1]])

        o1 = or_element(i1)
        o2 = or_element(i2)
        return or_element([o1, o2])

    def __call__(self, input):
        return self.out(input)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    and_element = AndElement()
    or_element = OrElement()
    xor_element = XorElement()

    assert and_element([1, 1])
    assert not and_element([1, 0])
    assert not and_element([0, 1])
    assert not and_element([0, 0])

    assert or_element([1, 1])
    assert or_element([1, 0])
    assert or_element([0, 1])
    assert not or_element([0, 0])

    assert not xor_element([1, 1])
    assert xor_element([1, 0])
    assert xor_element([0, 1])
    assert not xor_element([0, 0])
