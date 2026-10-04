"""Offline checks for independent models and concurrent training reports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from img_classification.models import CNN, ResNet


class BenchmarkTests(unittest.TestCase):
    def test_independent_models_keep_matched_initialization_and_skip_behavior(self):
        self.assertNotIn(CNN, ResNet.__mro__)
        torch.manual_seed(42)
        cnn = CNN(padding=1).eval()
        torch.manual_seed(42)
        resnet = ResNet().eval()
        self.assertEqual(cnn.state_dict().keys(), resnet.state_dict().keys())
        for key, value in cnn.state_dict().items():
            self.assertTrue(torch.equal(value, resnet.state_dict()[key]), key)
        inputs = torch.rand(2, 3, 32, 32)
        cnn_output, resnet_output = cnn(inputs), resnet(inputs)
        self.assertEqual(resnet_output.shape, (2, 10))
        self.assertFalse(torch.allclose(cnn_output, resnet_output))
        resnet_output.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in resnet.parameters()))

    def test_entrypoint_pairs_live_epochs_and_exports_test_plots(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            cache = work / 'data/processed'
            cache.mkdir(parents=True)
            for split in ('train', 'test'):
                torch.save({'images': torch.randint(0, 256, (8, 3, 32, 32), dtype=torch.uint8),
                            'labels': torch.arange(8), 'class_names': [str(i) for i in range(10)]},
                           cache / f'{split}.pt')
            env = {**os.environ, 'PYTHONPATH': str(root / 'src'), 'MPLBACKEND': 'Agg',
                   'MPLCONFIGDIR': str(work / 'mpl'), 'IMG_CLASSIFICATION_DATA_DIR': str(work / 'data'),
                   'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            output = work / 'result'
            run = subprocess.run([sys.executable, str(root / 'train.py'), '--epochs', '2',
                                  '--batch-size', '2', '--device', 'cpu', '--output-dir', str(output)],
                                 cwd=work, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            lines = (output / 'training.log').read_text().splitlines()
            paired = [line for line in lines if line.startswith('Epoch ')]
            self.assertEqual(len(paired), 2)
            for epoch, line in enumerate(paired, 1):
                self.assertIn(f'Epoch {epoch}/2 | CNN: loss=', line)
                self.assertIn('| ResNet: loss=', line)
            for filename in ('test_predictions.png', 'test_accuracy.png', 'benchmark.png', 'architectures.txt'):
                self.assertGreater((output / filename).stat().st_size, 0)
            splits = []
            for name in ('cnn', 'resnet'):
                metrics = json.loads((output / name / 'metrics.json').read_text())
                self.assertEqual(len(metrics['history']), 2)
                self.assertTrue(0 <= metrics['test_accuracy'] <= 1)
                splits.append(json.loads((output / name / 'split.json').read_text()))
            self.assertEqual(*splits)
