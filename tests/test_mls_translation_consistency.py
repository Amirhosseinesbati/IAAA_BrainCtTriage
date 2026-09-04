import unittest
import numpy as np
import torch
from src.strategies.mls_heatmap.translation_consistency import (
    overlap_slices, translate_image, translated_targets, consistency_js, combine_losses,
)
from src.strategies.mls_heatmap.utils import generate_gaussian_heatmap


class PairedTranslationTest(unittest.TestCase):
    def test_all_directions_no_wrap(self):
        x = torch.arange(64).reshape(1, 1, 8, 8)
        for dx, dy in ((2,0),(-2,0),(0,2),(0,-2),(0,0)):
            source, target = overlap_slices(8, 8, dx, dy)
            moved = translate_image(x, dx, dy)
            self.assertTrue(torch.equal(x[..., source[0], source[1]], moved[..., target[0], target[1]]))
            self.assertEqual(int(moved.sum()), int(x[..., source[0], source[1]].sum()))

    def test_regenerated_target_matches_existing_convention(self):
        coords = torch.tensor([[[100.5,120.25],[180.5,300.25],[135.5,240.25]]])
        for dx, dy in ((8,0),(-8,0),(0,8),(0,-8)):
            moved, target, eligible, valid = translated_targets(coords, torch.ones(1,3), torch.ones(1), dx, dy, 512, 3.)
            expected, _ = generate_gaussian_heatmap(moved[0].tolist(),512,128,3.)
            self.assertTrue(torch.equal(target[0], expected))
            self.assertTrue(bool(eligible.all() & valid.all()))

    def test_invalid_positive_excluded_negative_retained(self):
        coords = torch.tensor([[[509.,100.]]*3, [[-1.,-1.]]*3])
        _, target, eligible, valid = translated_targets(coords,torch.ones(2,3),torch.tensor([1.,0.]),8,0,512,3.)
        self.assertEqual(eligible.tolist(),[False,True])
        self.assertFalse(bool(valid.any()))
        self.assertEqual(float(target.sum()),0.)

    def test_equivariant_distribution_zero(self):
        torch.manual_seed(42)
        x = torch.randn(2,3,16,16)
        for dx,dy in ((8,0),(-8,0),(0,8),(0,-8)):
            y = translate_image(x,dx//4,dy//4)
            self.assertLess(abs(float(consistency_js(x,y,torch.ones(2,dtype=torch.bool),dx,dy))),1e-7)

    def test_gradients_both_views_and_no_positive(self):
        x=torch.randn(2,3,16,16,requires_grad=True)
        y=torch.randn(2,3,16,16,requires_grad=True)
        loss=consistency_js(x,y,torch.tensor([True,False]),8,0)
        gx,gy=torch.autograd.grad(loss,(x,y))
        for grad in (gx,gy):
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad[0].abs().sum()),0)
            self.assertEqual(float(grad[1].abs().sum()),0)
        self.assertEqual(float(consistency_js(x,y,torch.zeros(2,dtype=torch.bool),8,0)),0)

    def test_zero_weight_exact(self):
        a=torch.tensor(2.,requires_grad=True)
        b=torch.tensor(4.,requires_grad=True)
        c=torch.tensor(float('nan'),requires_grad=True)
        loss=combine_losses(a,b,c,0.)
        self.assertEqual(float(loss),3.)
        self.assertEqual(torch.autograd.grad(loss,(a,b,c),allow_unused=True),(torch.tensor(.5),torch.tensor(.5),None))


if __name__=='__main__':unittest.main()
