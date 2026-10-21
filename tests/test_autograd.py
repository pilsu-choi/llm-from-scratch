import unittest

import numpy as np

from scratchlm.autograd import Tensor, cross_entropy


def gradcheck(f, *arrays, eps=1e-6):
    """Compare autograd grads with central finite differences, element by element."""
    ts = [Tensor(a) for a in arrays]
    f(*ts).backward()
    for t, a in zip(ts, arrays):
        num = np.zeros_like(a)
        for i in np.ndindex(a.shape):
            hi, lo = a.copy(), a.copy()
            hi[i] += eps
            lo[i] -= eps
            fx = lambda arr: f(*[Tensor(arr if b is a else b) for b in arrays]).data
            num[i] = (fx(hi) - fx(lo)) / (2 * eps)
        np.testing.assert_allclose(t.grad, num, rtol=1e-5, atol=1e-7)


class TestAutograd(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def r(self, *shape):
        return self.rng.normal(size=shape)

    def test_elementwise_and_broadcast(self):
        gradcheck(lambda a, b: ((a * b + a**2 - b / a).tanh() + b).sum(), self.r(3, 4) + 3, self.r(1, 4))
        gradcheck(lambda a, b: ((a.exp() + 1).log() * b.relu() - 2 * a).mean(), self.r(2, 3), self.r(3))

    def test_matmul_reduce_reshape(self):
        gradcheck(lambda a, b: (a @ b).tanh().sum(axis=1).mean(), self.r(4, 3), self.r(3, 5))
        gradcheck(lambda a, b: (a @ b).reshape(2, 12).sum(axis=0, keepdims=True).sum(), self.r(2, 3, 4), self.r(4, 4))

    def test_embedding_lookup_accumulates(self):
        ids = np.array([[0, 2], [2, 2]])  # id 2 appears three times
        gradcheck(lambda E: (E[ids] ** 2).sum(), self.r(4, 3))

    def test_cross_entropy(self):
        y = np.array([1, 0, 3])
        gradcheck(lambda z: cross_entropy(z, y), self.r(3, 4))
        z = np.array([[1.0, 2.0, 3.0]])
        ref = -np.log(np.exp(2.0) / np.exp(z).sum())
        self.assertAlmostEqual(cross_entropy(Tensor(z), np.array([1])).data, ref)

    def test_mlp_learns_xor(self):
        x, y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float), np.array([0, 1, 1, 0])
        W1, b1, W2 = Tensor(self.r(2, 8)), Tensor(np.zeros(8)), Tensor(self.r(8, 2))
        for _ in range(500):
            loss = cross_entropy((Tensor(x) @ W1 + b1).tanh() @ W2, y)
            loss.backward()
            for p in (W1, b1, W2):
                p.data -= 0.5 * p.grad
        self.assertLess(loss.data, 0.05)


if __name__ == "__main__":
    unittest.main()
