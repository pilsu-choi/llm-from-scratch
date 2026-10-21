from scratchlm.tokenizer import PAT, Tokenizer

TEXTS = [
    "Once upon a time, there was a little dog named Max. He didn't like_rain!!\n\n",
    "옛날 옛적에 작은 강아지가 살았어요. 123456 ²½ 🐶 \t tabs\r\n",
]


def test_pretokenize_is_lossless():
    for t in TEXTS:
        assert "".join(PAT.findall(t)) == t


def test_roundtrip_and_compression():
    tok = Tokenizer.train(TEXTS * 20, vocab_size=400, specials=["<|endoftext|>"])
    for t in TEXTS + ["unseen text 처음 보는 문장 ✨"]:
        assert tok.decode(tok.encode(t)) == t
    assert len(tok.encode(TEXTS[0])) < len(TEXTS[0].encode())


def test_digits_are_single_tokens():
    tok = Tokenizer.train(["12 123 1234 100+200=300"] * 50, vocab_size=400)
    assert [tok.decode([i]) for i in tok.encode("3051+47")] == list("3051+47")


def test_specials_and_save_load(tmp_path):
    tok = Tokenizer.train(TEXTS, vocab_size=300, specials=["<|endoftext|>"])
    eot = tok.specials["<|endoftext|>"]
    ids = tok.encode("hi<|endoftext|>bye")
    assert ids.count(eot) == 1 and eot == tok.vocab_size - 1
    tok.save(tmp_path / "t.json")
    tok2 = Tokenizer.load(tmp_path / "t.json")
    assert tok2.encode(TEXTS[1]) == tok.encode(TEXTS[1])
