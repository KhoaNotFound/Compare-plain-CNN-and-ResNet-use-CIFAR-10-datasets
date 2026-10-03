"""Boundary tests for experiment configuration and the data contract."""
import tempfile
import unittest
from pathlib import Path
import torch

from img_classification.config import TrainConfig, load_train_config
from img_classification.dataset import load_cached_dataset
from img_classification.model import CNN as LegacyCNN
from img_classification.models import CNN


class ConfigTests(unittest.TestCase):
    def test_overrides_and_rejects_typo(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.toml"
            path.write_text("[training]\nepochs = 7\namp = false\n")
            config = load_train_config(path, epochs=2, lr=None)
            self.assertEqual(config.epochs, 2)
            self.assertFalse(config.amp)
            path.write_text("[training]\nbatch_szie = 32\n")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_train_config(path)

    def test_invalid_values(self):
        for values in ({"lr": float("nan")}, {"lr": float("inf")}, {"epochs": True},
                       {"seed": -1}, {"val_fraction": 0}, {"val_fraction": 1}, {"amp": "false"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainConfig(**values)

    def test_old_model_import_is_same_class(self):
        self.assertIs(CNN, LegacyCNN)

    def test_cache_rejects_empty_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.pt"
            for labels in (torch.tensor([], dtype=torch.long), torch.tensor([10]), torch.tensor([-1])):
                torch.save({"images": torch.zeros((len(labels), 3, 32, 32), dtype=torch.uint8),
                            "labels": labels, "class_names": [str(i) for i in range(10)]}, path)
                with self.assertRaises(ValueError):
                    load_cached_dataset(path)
