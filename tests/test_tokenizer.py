import os
import tempfile
import unittest

from scratchlm.tokenizer import PAT, Tokenizer

TEXTS = [
    "Once upon a time, there was a little dog named Max. He didn't like_rain!!\n\n",
    "옛날 옛적에 작은 강아지가 살았어요. 123456 ²½ 🐶 \t tabs\r\n",
]


class TestTokenizer(unittest.TestCase):
    def test_pretokenize_is_lossless(self):
        for t in TEXTS:
            self.assertEqual("".join(PAT.findall(t)), t)

    def test_roundtrip_and_compression(self):
        tok = Tokenizer.train(TEXTS * 20, vocab_size=400, specials=["<|endoftext|>"])
        for t in TEXTS + ["unseen text 처음 보는 문장 ✨"]:
            self.assertEqual(tok.decode(tok.encode(t)), t)
        self.assertLess(len(tok.encode(TEXTS[0])), len(TEXTS[0].encode()))

    def test_digits_are_single_tokens(self):
        tok = Tokenizer.train(["12 123 1234 100+200=300"] * 50, vocab_size=400)
        self.assertEqual([tok.decode([i]) for i in tok.encode("3051+47")], list("3051+47"))

    def test_specials_and_save_load(self):
        tok = Tokenizer.train(TEXTS, vocab_size=300, specials=["<|endoftext|>"])
        eot = tok.specials["<|endoftext|>"]
        self.assertEqual(tok.encode("hi<|endoftext|>bye").count(eot), 1)
        self.assertEqual(eot, tok.vocab_size - 1)
        with tempfile.TemporaryDirectory() as d:
            tok.save(os.path.join(d, "t.json"))
            self.assertEqual(Tokenizer.load(os.path.join(d, "t.json")).encode(TEXTS[1]), tok.encode(TEXTS[1]))


if __name__ == "__main__":
    unittest.main()
