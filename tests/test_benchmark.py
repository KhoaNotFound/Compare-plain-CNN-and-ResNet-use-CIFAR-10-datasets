"""Offline checks for independent models and concurrent training reports."""
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from img_classification.models import CNN, ResNet
from img_classification.benchmark import model_devices, train_models
from img_classification.gradients import GradientMonitor


class BenchmarkTests(unittest.TestCase):
    def test_gradient_statistics_and_invalid_batches(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.Conv2d(1, 1, 1))
        monitor = GradientMonitor(model)
        for first, last in ((1., 2.), (3., 2.), (float('inf'), 0.)):
            model[0].weight.grad = torch.full_like(model[0].weight, first)
            model[1].weight.grad = torch.full_like(model[1].weight, last)
            monitor.record()
        result = monitor.summary()
        self.assertEqual(result['layers']['0']['rms_mean'], 2.)
        self.assertEqual(result['layers']['0']['rms_std'], 1.)
        self.assertEqual(result['layers']['0']['rms_cv'], .5)
        self.assertEqual(result['layers']['0']['nonfinite_batches'], 1)
        self.assertAlmostEqual(result['layers']['1']['near_zero_fraction'], 1 / 3)
        self.assertAlmostEqual(result['first_last_rms_ratio'], 1.5)
        json.dumps(result, allow_nan=False)
        zero = GradientMonitor(model)
        for layer in model:
            layer.weight.grad.zero_()
        zero.record()
        self.assertIsNone(zero.summary()['first_last_rms_ratio'])
        self.assertIsNone(zero.summary()['layers']['0']['rms_cv'])

    def test_monitor_does_not_change_updates_and_unscales_amp(self):
        from img_classification.train import train_one_epoch
        from torch.utils.data import DataLoader, TensorDataset
        import copy

        model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1),
                                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        loader = DataLoader(TensorDataset(torch.ones(2, 3, 2, 2, dtype=torch.uint8),
                                         torch.tensor([0, 1])), batch_size=2)
        for amp in (False, True):
            plain, measured = copy.deepcopy(model), copy.deepcopy(model)
            monitor = GradientMonitor(measured)
            for network, observer in ((plain, None), (measured, monitor)):
                train_one_epoch(network, loader, torch.nn.CrossEntropyLoss(),
                                torch.optim.SGD(network.parameters(), lr=.1), torch.device('cpu'),
                                torch.amp.GradScaler('cpu', init_scale=128) if amp else None,
                                observer)
            for p, q in zip(plain.parameters(), measured.parameters()):
                self.assertTrue(torch.equal(p, q))
            self.assertAlmostEqual(monitor.summary()['layers']['0']['rms_mean'],
                                   measured[0].weight.grad.float().square().mean().sqrt().item())

    def test_device_assignment(self):
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=2):
            self.assertEqual(model_devices('auto'), {'cnn': 'cuda:0', 'resnet': 'cuda:1'})
            self.assertEqual(model_devices('cuda'), {'cnn': 'cuda:0', 'resnet': 'cuda:1'})
            self.assertEqual(model_devices('cuda', True), {'cnn': 'cuda:0', 'resnet': 'cuda:1'})
            self.assertEqual(model_devices('cpu'), {'cnn': 'cpu', 'resnet': 'cpu'})
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=1):
            self.assertEqual(model_devices('auto'), {'cnn': 'cuda:0', 'resnet': 'cuda:0'})
            with self.assertRaisesRegex(ValueError, 'PyTorch sees 1'):
                model_devices('auto', True)
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(model_devices('auto'), {'cnn': 'cpu', 'resnet': 'cpu'})
            with self.assertRaisesRegex(ValueError, 'no CUDA GPU'):
                model_devices('cuda')
            with self.assertRaisesRegex(ValueError, 'PyTorch sees 0'):
                model_devices('auto', True)
            with self.assertRaisesRegex(ValueError, 'cannot be used'):
                model_devices('cpu', True)

    def test_children_receive_distinct_devices(self):
        children = [Mock(stdout=io.StringIO(''), wait=Mock(return_value=0),
                         poll=Mock(return_value=0)) for _ in range(2)]
        with tempfile.TemporaryDirectory() as temporary, \
                patch('img_classification.benchmark.subprocess.Popen', side_effect=children) as spawn:
            train_models(Path(temporary), ['--epochs', '1'], dict(os.environ),
                         {'cnn': 'cuda:0', 'resnet': 'cuda:1'})
            for call, name, device in zip(spawn.call_args_list, ('cnn', 'resnet'), ('cuda:0', 'cuda:1')):
                command = call.args[0]
                self.assertEqual(command[command.index('--model') + 1], name)
                self.assertEqual(command[command.index('--device') + 1], device)

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
            self.assertIn('Benchmark source:', run.stdout)
            self.assertIn('Model: cnn; PID:', run.stdout)
            self.assertIn('Model: resnet; PID:', run.stdout)
            self.assertEqual(json.loads((output / 'devices.json').read_text()),
                             {'cnn': 'cpu', 'resnet': 'cpu'})
            lines = (output / 'training.log').read_text().splitlines()
            paired = [line for line in lines if line.startswith('Epoch ')]
            self.assertEqual(len(paired), 2)
            for epoch, line in enumerate(paired, 1):
                self.assertIn(f'Epoch {epoch}/2 | CNN: loss=', line)
                self.assertIn('| ResNet: loss=', line)
            for filename in ('test_predictions.png', 'test_accuracy.png', 'benchmark.png',
                             'architectures.txt', 'gradients.png', 'gradients.csv'):
                self.assertGreater((output / filename).stat().st_size, 0)
            splits = []
            for name in ('cnn', 'resnet'):
                metrics = json.loads((output / name / 'metrics.json').read_text())
                self.assertEqual(len(metrics['history']), 2)
                for row in metrics['history']:
                    self.assertEqual(len(row['gradients']['layers']), 8)
                    self.assertEqual(row['gradients']['batches'], 4)
                    self.assertTrue(all(s['rms_mean'] >= 0 for s in row['gradients']['layers'].values()))
                self.assertTrue(0 <= metrics['test_accuracy'] <= 1)
                splits.append(json.loads((output / name / 'split.json').read_text()))
            self.assertEqual(*splits)
