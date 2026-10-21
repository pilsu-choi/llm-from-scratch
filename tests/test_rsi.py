import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestRSI(unittest.TestCase):
    def test_pose_splits_frontier_and_replay(self):
        import numpy as np

        from scratchlm.rsi import pose
        from scratchlm.tokenizer import Tokenizer

        groups = pose(np.random.default_rng(0), 1000, 5)
        n_front = sum(len(v) for k, v in groups.items() if max(k) == 5)
        self.assertEqual(sum(map(len, groups.values())), 1000)
        self.assertEqual(n_front, 500)
        self.assertTrue(all(max(k) <= 5 for k in groups))
        tok = Tokenizer({}, {})
        for probs in groups.values():  # one operand-length pair -> one prompt length -> one batch
            self.assertEqual(len({len(tok.encode(p)) for p in probs}), 1)

    def test_only_verified_answers_are_kept(self):
        import numpy as np
        import torch

        from scratchlm.model import GPT, Config
        from scratchlm.rsi import pose, sample_and_filter
        from scratchlm.tasks import check_addition, chat_tokenizer
        from scratchlm.tokenizer import Tokenizer

        tok = chat_tokenizer(Tokenizer({}, {"<|endoftext|>": 256}))
        torch.manual_seed(0)
        model = GPT(Config(vocab_size=tok.vocab_size, ctx_len=64, n_layer=1, n_head=2, d_model=16)).eval()
        groups = pose(np.random.default_rng(0), 40, 2)
        kept, stats = sample_and_filter(model, tok, groups, 2, k=4, temperature=1.0)
        self.assertTrue(all(check_addition(p, a) for p, a in kept))  # a random model's lucky hits only
        self.assertEqual(stats["frontier"]["n"] + stats["replay"]["n"], 40)
        self.assertEqual(stats["frontier"]["kept"] + stats["replay"]["kept"], len(kept))


if __name__ == "__main__":
    unittest.main()
