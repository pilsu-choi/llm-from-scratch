import importlib.util
import os
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import numpy as np
    import torch

    from scratchlm.data import batches
    from scratchlm.optim import AdamW, clip_grad_norm, cosine_lr


@unittest.skipUnless(HAS_TORCH, "torch not installed (run on PGX: scripts/pgx.sh python -m unittest)")
class TestOptim(unittest.TestCase):
    def test_adamw_matches_torch(self):
        torch.manual_seed(0)
        ours = [torch.randn(5, 3, requires_grad=True), torch.randn(3, requires_grad=True)]
        ref = [p.detach().clone().requires_grad_() for p in ours]
        opt = AdamW(ours, betas=(0.9, 0.95), weight_decay=0.1)
        topt = torch.optim.AdamW(  # reference only (tests may use torch.optim)
            [{"params": [ref[0]], "weight_decay": 0.1}, {"params": [ref[1]], "weight_decay": 0.0}], lr=1e-2, betas=(0.9, 0.95), eps=1e-8
        )
        for step in range(20):
            lr = 1e-2 * (0.5 if step > 10 else 1.0)
            for g in topt.param_groups:
                g["lr"] = lr
            for ps in (ours, ref):
                (ps[0].sin().sum() * ps[1].sum() + ps[1].square().sum()).backward()
            opt.step(lr)
            opt.zero_grad()
            topt.step()
            topt.zero_grad()
        for a, b in zip(ours, ref):
            torch.testing.assert_close(a, b)

    def test_state_dict_roundtrip(self):
        p = torch.randn(4, 4, requires_grad=True)
        opt = AdamW([p])
        p.sum().backward()
        opt.step(1e-3)
        opt2 = AdamW([p])
        opt2.load_state_dict(opt.state_dict())
        self.assertEqual(opt2.t, 1)
        torch.testing.assert_close(opt2.v[0], opt.v[0])

    def test_cosine_lr(self):
        self.assertAlmostEqual(cosine_lr(0, 1.0, 10, 100), 0.1)
        self.assertAlmostEqual(cosine_lr(9, 1.0, 10, 100), 1.0)
        self.assertAlmostEqual(cosine_lr(55, 1.0, 10, 100), 0.55)  # halfway through the cosine: (1 + 0.1) / 2
        self.assertAlmostEqual(cosine_lr(100, 1.0, 10, 100), 0.1)

    def test_clip_grad_norm(self):
        ps = [torch.zeros(3, requires_grad=True), torch.zeros(4, requires_grad=True)]
        ps[0].grad, ps[1].grad = torch.full((3,), 3.0), torch.full((4,), 2.0)  # norm sqrt(27 + 16)
        norm = clip_grad_norm(ps, 1.0)
        self.assertAlmostEqual(norm.item(), 43**0.5, places=5)
        self.assertAlmostEqual(torch.cat([p.grad for p in ps]).norm().item(), 1.0, places=5)

    def test_batches_are_shifted_windows(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.bin")
            np.arange(1000, dtype=np.uint16).tofile(path)
            x, y = next(batches(path, 4, 16, "cpu"))
            self.assertEqual(x.shape, (4, 16))
            torch.testing.assert_close(y, x + 1)  # target = input shifted by one token


if __name__ == "__main__":
    unittest.main()
