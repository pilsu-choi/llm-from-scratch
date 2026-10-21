import importlib.util
import os
import tempfile
import unittest

import numpy as np

from scratchlm.tasks import ASSISTANT, USER, addition, chat_tokenizer, check_addition, encode_chat, parse_instruct, words_used
from scratchlm.tokenizer import Tokenizer

HAS_TORCH = importlib.util.find_spec("torch") is not None

DOC = """Features: Dialogue
Words: ball, happy, run
Story:
Tom had a red ball. "Run!" he said. He was happy.
Summary: Tom plays with his ball.
"""


class TestTasks(unittest.TestCase):
    def test_addition_and_verifier(self):
        q = "What is 0000000000000025 + 0000000000000017?"
        self.assertTrue(check_addition(q, "24000000000000000"))  # 42 padded to 17 digits, ones digit first
        self.assertFalse(check_addition(q, "24"))
        self.assertFalse(check_addition(q, "00000000000000042"))
        rng = np.random.default_rng(0)
        for da, db in ((1, 1), (3, 2), (6, 4)):
            p, a = addition(rng, da, db)
            self.assertTrue(check_addition(p, a))
            self.assertEqual([len(str(int(x))) for x in p[8:-1].split(" + ")], [da, db])  # significant digits
            self.assertEqual([len(x) for x in p[8:-1].split(" + ")], [16, 16])  # fixed width
            p, a = addition(rng, da, db, shift=True)
            self.assertTrue(check_addition(p, a))
            self.assertEqual([len(str(int(x)).rstrip("0") or "0") <= d for x, d in zip(p[8:-1].split(" + "), (da, db))], [True, True])

    def test_parse_instruct_fields_before_and_after_story(self):
        prompt, story = parse_instruct(DOC)
        self.assertEqual(story, 'Tom had a red ball. "Run!" he said. He was happy.')
        self.assertIn("Words: ball, happy, run", prompt)
        self.assertIn("Summary: Tom plays with his ball.", prompt)  # came after the story
        self.assertEqual(words_used(prompt, story), 1.0)
        self.assertAlmostEqual(words_used(prompt, "a ball"), 1 / 3)

    def test_chat_template_masks_prompt(self):
        base = Tokenizer.train(["What is 1 + 2? 3"] * 5, vocab_size=300, specials=["<|endoftext|>"])
        tok = chat_tokenizer(base)
        self.assertEqual(tok.vocab_size, base.vocab_size + 2)
        ids, mask = encode_chat(tok, "What is 1 + 2?", "3")
        start = ids.index(tok.specials[ASSISTANT]) + 1
        self.assertEqual(ids[0], tok.specials[USER])
        self.assertEqual(mask, [0] * start + [1] * (len(ids) - start))  # answer + <eot> only
        self.assertEqual(tok.decode(ids[start:]), "3<|endoftext|>")


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestSFTPlumbing(unittest.TestCase):
    def test_masked_batches(self):
        from scratchlm.data import batches

        with tempfile.TemporaryDirectory() as d:
            np.arange(100, dtype=np.uint16).tofile(os.path.join(d, "s.bin"))
            (np.arange(100) % 2).astype(np.uint8).tofile(os.path.join(d, "s.mask"))  # odd positions trained
            x, y = next(batches(os.path.join(d, "s.bin"), 8, 10, "cpu"))
            self.assertTrue(((y == -1) == (x % 2 == 1)).all())  # target x+1 is ignored when x+1 is even

    def test_load_checkpoint_grows_vocab(self):
        import dataclasses

        import torch

        from scratchlm.model import GPT, Config, load_checkpoint

        c = Config(vocab_size=50, ctx_len=16, n_layer=1, n_head=2, d_model=16)
        m = GPT(c)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ck.pt")
            torch.save({"model": m.state_dict(), "model_config": dataclasses.asdict(c)}, p)
            m2 = load_checkpoint(p, "cpu", vocab_size=52)
        self.assertEqual(m2.embed.weight.shape, (52, 16))
        torch.testing.assert_close(m2.embed.weight[:50], m.embed.weight)
        ids = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(m2(ids)[..., :50], m(ids))  # old tokens' logits unchanged


if __name__ == "__main__":
    unittest.main()
