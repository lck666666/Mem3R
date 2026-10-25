"""Dependency-free checks for renamed options and legacy checkpoint metadata."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'src/dust3r/model.py').read_text())

class BaseConfig:
    def __init__(self):
        pass

SCOPE = {'PretrainedConfig': BaseConfig}
NAMES = {'normalize_memory_option', 'ARCroco3DStereoConfig', 'load_model'}
exec(compile(ast.Module(body=[n for n in TREE.body if getattr(n, 'name', '') in NAMES],
                        type_ignores=[]), 'option_definitions', 'exec'), SCOPE)

class OptionCompatibility(unittest.TestCase):
    def test_config_accepts_old_and_new_values(self):
        for old, new in [('cut3r_learnable_state', 'mem3r'),
                         ('cut3r_learnable_state_ttt3r', 'mem3r_ttt3r'),
                         ('cut3r_learnable_state_ttsa3r', 'mem3r_ttsa3r')]:
            for value in (old, new):
                config = SCOPE['ARCroco3DStereoConfig'](model_update_type=value,
                                                     pose_retriever_type='ttt_linear')
                self.assertEqual(config.model_update_type, new)
                self.assertEqual(config.pose_retriever_type, 'ttt')

    def test_checkpoint_loader_normalizes_metadata_and_overrides(self):
        class StubModel:
            def __init__(self, config):
                self.config = config
                self.pose_retriever_type = config.pose_retriever_type
                self.pose_head_flag = True
            def _init_pose_retriever(self):
                assert self.pose_retriever_type == 'ttt'
            def load_state_dict(self, weights, strict):
                assert strict
            def to(self, device):
                return self
        expression = "StubModel(ARCroco3DStereoConfig(model_update_type='cut3r_learnable_state_ttt3r', pose_retriever_type='ttt_linear', landscape_only=True))"
        SCOPE.update(StubModel=StubModel, torch=SimpleNamespace(load=lambda *a, **kw: {
            'args': SimpleNamespace(model=expression), 'model': {}}))
        model = SCOPE['load_model']('placeholder.pth', 'cpu', verbose=False,
                                   model_update_type='cut3r_learnable_state_ttsa3r',
                                   pose_retriever_type='ttt_linear')
        self.assertEqual(model.config.model_update_type, 'mem3r_ttsa3r')
        self.assertEqual(model.config.pose_retriever_type, 'ttt')
        self.assertEqual(model.pose_retriever_type, 'ttt')

    def test_every_entry_point_uses_independent_options(self):
        import argparse
        import sys
        from unittest.mock import patch
        from memory_options import add_memory_arguments
        for path in ['demo.py', 'eval/relpose/launch.py', 'eval/video_depth/launch.py',
                     'eval/mv_recon/launch.py']:
            tree = ast.parse((ROOT / path).read_text())
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name in ('parse_args', 'get_args_parser'))
            scope = dict(argparse=argparse, add_memory_arguments=add_memory_arguments,
                         dataset_metadata={'sintel': {}, '7scenes': {}})
            exec(compile(ast.Module(body=[fn], type_ignores=[]), path, 'exec'), scope)
            with patch.object(sys, 'argv', [path, '--seq_path', 'frames']):
                parsed = scope[fn.name]()
            args = parsed.parse_args([]) if isinstance(parsed, argparse.ArgumentParser) else parsed
            self.assertEqual((args.pose_memory, args.state_gate, args.state_update),
                             ('ttt', 'channel', 'direct'), path)

    def test_all_independent_combinations(self):
        import argparse
        from memory_options import add_memory_arguments, model_memory_kwargs, configure_memory
        parser = add_memory_arguments(argparse.ArgumentParser())
        expected = {
            ('channel', 'direct'): 'mem3r', ('channel', 'ttt3r'): 'mem3r_ttt3r',
            ('channel', 'ttsa3r'): 'mem3r_ttsa3r', ('none', 'direct'): 'cut3r',
            ('none', 'ttt3r'): 'ttt3r', ('none', 'ttsa3r'): 'ttsa3r',
        }
        for pose in ('ttt', 'local'):
            for (gate, update), mode in expected.items():
                args = parser.parse_args(['--pose_memory', pose, '--state_gate', gate,
                                          '--state_update', update])
                options = model_memory_kwargs(args)
                self.assertEqual(options, dict(pose_retriever_type=pose, model_update_type=mode))
                model = SimpleNamespace(config=SimpleNamespace())
                configure_memory(model, args)
                self.assertEqual(model.config.model_update_type, mode)
                self.assertEqual(model.config.state_gate, gate)
                self.assertEqual(model.config.state_update, update)
                self.assertEqual(model.config.pose_memory, pose)

    def test_both_model_update_paths_cover_all_variants(self):
        comparisons = [n for n in ast.walk(TREE) if isinstance(n, ast.Compare)]
        for value in ['mem3r', 'mem3r_ttt3r', 'mem3r_ttsa3r']:
            count = sum(any(isinstance(x, ast.Constant) and x.value == value for x in n.comparators)
                        for n in comparisons)
            self.assertGreaterEqual(count, 2, value)

if __name__ == '__main__':
    unittest.main()
