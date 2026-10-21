import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as F

    from scratchlm.model import GPT, Config, RMSNorm, apply_rope, cross_entropy, rope_tables

    DEV = "cuda" if torch.cuda.is_available() else "cpu"

TINY = dict(vocab_size=97, ctx_len=32, n_layer=2, n_head=4, d_model=32)


@unittest.skipUnless(HAS_TORCH, "torch not installed (run on PGX: scripts/pgx.sh python -m unittest)")
class TestModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32 comparisons
        self.ids = torch.randint(0, 97, (2, 20), device=DEV)

    def model(self, **kw):
        return GPT(Config(**{**TINY, **kw})).to(DEV)

    def test_param_count_formula(self):
        for tie in (True, False):
            c = Config(**TINY, tie_embeddings=tie)
            d, V, L = c.d_model, c.vocab_size, c.n_layer
            per_layer = 4 * d * d + 3 * d * c.d_ff + 2 * d  # qkv+proj, gate/up/down, two norms
            expected = V * d * (1 if tie else 2) + L * per_layer + d
            self.assertEqual(sum(p.numel() for p in GPT(c).parameters()), expected)

    def test_manual_attention_matches_sdpa(self):
        m = self.model(sdpa=False)
        ref = m(self.ids)
        for b in m.blocks:
            b.attn.sdpa = True
        torch.testing.assert_close(m(self.ids), ref, atol=1e-5, rtol=1e-5)

    def test_causal(self):
        m = self.model()
        ids2 = self.ids.clone()
        ids2[:, 10:] = (ids2[:, 10:] + 1) % 97  # change only the future
        torch.testing.assert_close(m(ids2)[:, :10], m(self.ids)[:, :10])
        self.assertFalse(torch.allclose(m(ids2)[:, 10:], m(self.ids)[:, 10:]))

    def test_kv_cache_matches_full_forward(self):
        for sdpa in (False, True):
            m = self.model(sdpa=sdpa)
            full = m(self.ids)
            caches = [{} for _ in m.blocks]
            parts = [m(self.ids[:, :7], caches=caches)]  # prefill 7
            parts.append(m(self.ids[:, 7:12], caches=caches))  # a 5-token chunk on a non-empty cache (explicit mask path)
            parts += [m(self.ids[:, t : t + 1], caches=caches) for t in range(12, 20)]  # then token by token
            torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-5, rtol=1e-5)

    def test_rope_depends_only_on_relative_position(self):
        cos, sin = rope_tables(16, 64, 10000.0)
        q, k = torch.randn(16), torch.randn(16)
        dot = lambda m, n: apply_rope(q, cos[m], sin[m]) @ apply_rope(k, cos[n], sin[n])
        torch.testing.assert_close(dot(3, 10), dot(40, 47))
        torch.testing.assert_close(apply_rope(q, cos[5], sin[5]).norm(), q.norm())  # rotation keeps length

    def test_rmsnorm_and_cross_entropy_match_torch(self):
        x = torch.randn(4, 32)
        n = RMSNorm(32)
        torch.testing.assert_close(n(x), F.rms_norm(x, (32,), n.weight, n.eps))
        z, t = torch.randn(3, 5, 11), torch.randint(0, 11, (3, 5))
        t[0, :2] = -1
        torch.testing.assert_close(cross_entropy(z, t), F.cross_entropy(z.flatten(0, 1), t.flatten(), ignore_index=-1))
        self.assertEqual(cross_entropy(z, torch.full_like(t, -1)).item(), 0.0)

    def test_initial_loss_is_near_uniform(self):
        _, loss = self.model()(self.ids[:, :-1], self.ids[:, 1:])
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(97.0)).item(), delta=0.1)

    def test_overfits_one_sequence(self):
        m = self.model()
        for _ in range(150):
            _, loss = m(self.ids[:, :-1], self.ids[:, 1:])
            loss.backward()
            with torch.no_grad():
                for p in m.parameters():
                    p -= 0.5 * p.grad  # plain SGD; the real optimizer (AdamW) is stage 3
                    p.grad = None
        self.assertLess(loss.item(), 0.5)

    def test_generate_greedy_matches_argmax(self):
        m = self.model()
        out = m.generate(self.ids[:, :5], max_new=6, temperature=0)
        self.assertEqual(out.shape, (2, 11))
        torch.testing.assert_close(out[:, 5], m(self.ids[:, :5])[:, -1].argmax(-1))
        torch.testing.assert_close(out[:, 10], m(out[:, :10])[:, -1].argmax(-1))


if __name__ == "__main__":
    unittest.main()
