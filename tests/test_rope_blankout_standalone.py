"""CPU tests independent of Transformers downloads or trained model weights."""

from types import SimpleNamespace
import unittest

import torch

from src.blankout import _blank_positions, blankout_profile
from src.measure_rope_blankout import probability_gradient, w1


class ToyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(18)
        self.embed = torch.nn.Embedding(16,4)
        self.out = torch.nn.Linear(4,16)

    def get_input_embeddings(self):
        return self.embed

    def forward(self,inputs_embeds,attention_mask=None,use_cache=False):
        return SimpleNamespace(logits=self.out(inputs_embeds.cumsum(1)))


class BlankoutTests(unittest.TestCase):
    def test_one_zeroed_row_and_no_mutation(self):
        base = torch.ones(1,7,4)
        changed = _blank_positions(base,torch.tensor([0,3,6]))
        self.assertTrue(torch.equal((changed==0).all(-1).sum(-1),torch.ones(3,dtype=torch.long)))
        self.assertTrue(bool((base==1).all()))

    def test_batching_fixed_target_and_manual_scores(self):
        model = ToyCausalLM().eval()
        ids = torch.arange(7)[None]
        mask = torch.ones_like(ids)
        small,target,z0 = blankout_profile(model,ids,mask,1)
        batched,t2,z2 = blankout_profile(model,ids,mask,3)
        torch.testing.assert_close(small,batched,atol=2e-6,rtol=2e-6)
        self.assertEqual(target,t2)
        self.assertEqual(z0,z2)
        for j in range(7):
            emb = model.embed(ids).detach()
            emb[:,j] = 0
            expected = abs(z0-float(model(emb).logits[0,-1,target].detach()))
            self.assertAlmostEqual(float(small[j]),expected,places=5)
        gradient = probability_gradient(model,ids,mask,target)
        self.assertEqual(tuple(gradient.shape),(7,))
        self.assertGreater(float(gradient.sum()),0)

    def test_w1(self):
        self.assertEqual(w1([1,0,0],[0,0,1]),1)
        self.assertEqual(w1([0,1,0],[0,2,0]),0)


if __name__ == "__main__":
    unittest.main()
