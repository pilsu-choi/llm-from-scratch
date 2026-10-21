import importlib.util
import unittest

import numpy as np

from scratchlm.tasks import COLORS, IMG, question, render, scene

HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestScenes(unittest.TestCase):
    def test_render_draws_each_object_in_its_cell_and_color(self):
        rng = np.random.default_rng(0)
        objs = scene(rng, 6)
        img = render(objs, rng)
        self.assertEqual(img.shape, (IMG, IMG, 3))
        for shape, color, r, c in objs:
            cell = img[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16].reshape(-1, 3)
            self.assertGreater((cell == COLORS[color]).all(1).sum(), 30)  # a filled shape, not a stray pixel

    def test_answers_match_the_scene(self):
        rng = np.random.default_rng(1)
        objs = [("circle", "red", 0, 0), ("circle", "blue", 1, 1), ("square", "red", 2, 2)]
        for _ in range(200):
            q, a = question(rng, objs)
            if q == "How many shapes are there?":
                self.assertEqual(a, "3")
            elif q == "How many red shapes are there?":
                self.assertEqual(a, "2")
            elif q == "How many circles are there?":
                self.assertEqual(a, "2")
            elif q == "What color is the square?":
                self.assertEqual(a, "red")
            elif q == "Is there a blue circle?":
                self.assertEqual(a, "yes")
            elif q == "Is there a blue square?":
                self.assertEqual(a, "no")


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestVLM(unittest.TestCase):
    def test_image_features_replace_placeholders(self):
        import torch

        from scratchlm.model import GPT, Config
        from scratchlm.vision import N_PATCHES, VLM, ViT

        torch.manual_seed(0)
        lm = GPT(Config(vocab_size=20, ctx_len=96, n_layer=1, n_head=2, d_model=32))
        vit = ViT(Config(vocab_size=1, ctx_len=N_PATCHES, n_layer=1, n_head=2, d_model=16, causal=False))
        vlm = VLM(lm, vit, image_id=19)
        ids = torch.tensor([[1] + [19] * N_PATCHES + [2, 3]])
        imgs = torch.rand(1, IMG, IMG, 3) - 0.5
        e = vlm.embed(ids, imgs)
        torch.testing.assert_close(e[0, 0], lm.embed.weight[1])  # text tokens untouched
        self.assertFalse(torch.allclose(e[0, 1], lm.embed.weight[19]))  # placeholders replaced
        other = vlm.embed(ids, imgs.flip(1))
        self.assertFalse(torch.allclose(e[0, 1 : 1 + N_PATCHES], other[0, 1 : 1 + N_PATCHES]))  # depends on the image
        out = vlm.generate(ids, imgs, 3)
        self.assertEqual(out.shape, (1, ids.shape[1] + 3))

    def test_vit_is_bidirectional(self):
        import torch

        from scratchlm.model import Config
        from scratchlm.vision import ViT

        torch.manual_seed(0)
        vit = ViT(Config(vocab_size=1, ctx_len=64, n_layer=1, n_head=2, d_model=16, causal=False))
        a = torch.rand(1, IMG, IMG, 3)
        b = a.clone()
        b[0, -8:, -8:] += 1  # change only the last patch
        self.assertFalse(torch.allclose(vit(a)[0, 0], vit(b)[0, 0]))  # the first patch token still sees it


if __name__ == "__main__":
    unittest.main()
