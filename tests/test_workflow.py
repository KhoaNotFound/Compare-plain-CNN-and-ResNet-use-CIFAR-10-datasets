"""Offline checks for source packaging and the training artifact contract."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from tools import kaggle_workflow as workflow

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_bundle_is_self_contained_private_and_excludes_local_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "src/img_classification"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text('VALUE = "__MANIFEST__"\n')
            (root / ".env").write_text("SECRET=not-for-upload")
            with patch.object(workflow, "ROOT", root), patch.object(workflow, "BUILD_ROOT", root / "build/kaggle"):
                folder = workflow.build("example", 2, 32, 0.001)
            metadata = json.loads((folder / "kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertTrue(metadata["enable_gpu"])
            runner = (folder / "runner.py").read_text()
            compile(runner, "runner.py", "exec")
            tree = ast.parse(runner)
            assignments = {node.targets[0].id: node.value for node in tree.body
                           if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
            sources = ast.literal_eval(assignments["sources"])
            self.assertEqual(list(sources), ["src/img_classification/__init__.py"])
            self.assertNotIn("SECRET=", runner)
            manifest = ast.literal_eval(assignments["manifest"])
            self.assertEqual(manifest["kernel"], metadata["id"])
            # Execute the generated TOML writer to catch runner escaping/boolean bugs.
            with_block = next(node for node in tree.body if isinstance(node, ast.With))
            writer = [node for node in with_block.body if isinstance(node, ast.Expr)
                      and isinstance(node.value, ast.Call)
                      and isinstance(node.value.func, ast.Attribute)
                      and isinstance(node.value.func.value, ast.Name)
                      and node.value.func.value.id == "config_path"]
            config_path = root / "training.toml"
            exec(compile(ast.Module(body=writer, type_ignores=[]), "writer", "exec"),
                 {"config_path": config_path, "manifest": manifest})
            self.assertEqual(tomllib.loads(config_path.read_text())["training"], manifest["training"])

    def test_invalid_parameters(self):
        for username, epochs, batch, lr in [("a/b", 1, 2, .01), ("user", 0, 2, .01),
                                            ("user", 1, 0, .01), ("user", 1, 2, float("nan"))]:
            with self.assertRaises(ValueError):
                workflow.build(username, epochs, batch, lr)

    def test_training_from_another_working_directory(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            caches = root / "data/processed"
            caches.mkdir(parents=True)
            for split in ("train", "test"):
                torch.save({"images": torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8),
                            "labels": torch.tensor([0, 1, 2, 3]),
                            "class_names": [str(i) for i in range(10)]}, caches / f"{split}.pt")
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg",
                   "MPLCONFIGDIR": str(root / "mpl"), "IMG_CLASSIFICATION_DATA_DIR": str(root / "data"),
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            # Existing tensor caches must work without a raw dataset or network.
            subprocess.run([sys.executable, "-m", "img_classification.prepare_data"],
                           cwd=root, env=env, check=True, capture_output=True, text=True)
            subprocess.run([sys.executable, "-m", "img_classification.train", "--epochs", "1",
                            "--batch-size", "2", "--device", "cpu", "--output-dir", str(root / "result")],
                           cwd=root, env=env, check=True, capture_output=True, text=True)
            result = root / "result"
            metrics = json.loads((result / "metrics.json").read_text())
            self.assertEqual(len(metrics["history"]), 1)
            self.assertGreaterEqual(metrics["test_accuracy"], 0)
            self.assertTrue((result / "predictions.png").stat().st_size > 0)
            checkpoint = torch.load(result / "checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertIn("model_state_dict", checkpoint)
            self.assertTrue((result / "best.pt").is_file())
            self.assertEqual(metrics["best_epoch"], 1)
            self.assertIn("validation_accuracy", metrics["history"][0])
            self.assertFalse(metrics["amp_enabled"])
            split = json.loads((result / "split.json").read_text())
            self.assertFalse(set(split["train_indices"]) & set(split["validation_indices"]))
            self.assertEqual(sorted(split["train_indices"] + split["validation_indices"]), list(range(4)))
            repeated = subprocess.run([sys.executable, "-m", "img_classification.train", "--epochs", "1",
                                      "--device", "cpu", "--output-dir", str(result)],
                                     cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("Run already exists", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
