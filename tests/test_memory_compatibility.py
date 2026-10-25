"""Compare the reimplementation against an explicitly supplied original model.

MEM3R_REFERENCE_MODEL=/path/to/original/src/dust3r/model.py python -m unittest discover -s tests -v
Only selected memory definitions are loaded; the full inference stack is unnecessary.
"""
import ast
import math
import os
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []  # Verify arithmetic without compiler/backend variability.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def scope():
    return dict(torch=torch, nn=nn, F=F, math=math, rearrange=rearrange)


@unittest.skipUnless(os.environ.get('MEM3R_REFERENCE_MODEL'), 'Supply an original model for equivalence checks')
class MemoryEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = scope()
        definitions(os.environ['MEM3R_REFERENCE_MODEL'],
                    {'RMSNorm', 'inv_softplus', 'silu_backprop', 'TTTMemory', 'ChannelWiseStateGate'}, cls.original)
        cls.rewritten = scope()
        directory = ROOT / 'src/dust3r/memory'
        definitions(directory / 'numerics.py', {'RMSNorm', 'inv_softplus', 'silu_backprop'}, cls.rewritten)
        definitions(directory / 'fast_weights.py', {'TTTMemory'}, cls.rewritten)
        definitions(directory / 'state_gates.py', {'ChannelWiseStateGate'}, cls.rewritten)

    def compare_memory(self, nonlinear, objective, clip):
        torch.manual_seed(29)
        options = dict(size=8, k_dim=7, v_dim=12, num_heads=3,
                       use_swiglu=nonlinear, loss_fn=objective)
        original = self.original['TTTMemory'](**options)
        rewritten = self.rewritten['TTTMemory'](**options)
        rewritten.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(list(original.state_dict()), list(rewritten.state_dict()))
        summary = torch.randn(2, 1, 7, requires_grad=True)
        target = (torch.randn(2, 1, 12) * 20).requires_grad_()
        summary_copy = summary.detach().clone().requires_grad_()
        target_copy = target.detach().clone().requires_grad_()
        state = original.mem.expand(2, -1, -1)
        state_copy = rewritten.mem.expand(2, -1, -1)
        for _ in range(3):
            torch.testing.assert_close(original.inquire(summary, state), rewritten.inquire(summary_copy, state_copy))
            state = original.update_mem(state, summary, target, grad_clip_norm=clip)
            state_copy = rewritten.update_mem(state_copy, summary_copy, target_copy, grad_clip_norm=clip)
            torch.testing.assert_close(state, state_copy)
        output = original.inquire(summary, state)
        output_copy = rewritten.inquire(summary_copy, state_copy)
        torch.testing.assert_close(output, output_copy)
        output.square().sum().backward()
        output_copy.square().sum().backward()
        torch.testing.assert_close(summary.grad, summary_copy.grad)
        torch.testing.assert_close(target.grad, target_copy.grad)
        for (name, param), (other_name, other) in zip(original.named_parameters(), rewritten.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(param.grad, other.grad)

    def test_alignment_with_clipping(self):
        self.compare_memory(True, 'inner', 0.001)

    def test_alignment_without_clipping(self):
        self.compare_memory(True, 'inner', 0.0)

    def test_swiglu_l2(self):
        self.compare_memory(True, 'l2', 1.0)

    def test_linear_l2(self):
        self.compare_memory(False, 'l2', 1.0)

    def test_state_gate(self):
        torch.manual_seed(7)
        original = self.original['ChannelWiseStateGate'](12, 7)
        rewritten = self.rewritten['ChannelWiseStateGate'](12, 7)
        rewritten.load_state_dict(original.state_dict(), strict=True)
        state = torch.randn(2, 8, 12, requires_grad=True)
        summary = torch.randn(2, 1, 7, requires_grad=True)
        reference = original(state, summary)
        actual = rewritten(state, summary)
        torch.testing.assert_close(reference, actual)
        reference_grad = torch.autograd.grad(reference.square().sum(), (state, summary))
        actual_grad = torch.autograd.grad(actual.square().sum(), (state, summary))
        for left, right in zip(reference_grad, actual_grad):
            torch.testing.assert_close(left, right)


if __name__ == '__main__':
    unittest.main()
