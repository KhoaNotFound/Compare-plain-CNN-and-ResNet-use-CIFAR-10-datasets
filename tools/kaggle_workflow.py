"""Build, submit, inspect, and fetch isolated Kaggle runs using only the stdlib."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / "build" / "kaggle"
# Bootstrap once for this standalone stdlib tool; application modules need no path hacks.
sys.path.insert(0, str(ROOT / "src"))
from img_classification.config import TrainConfig, load_train_config

# Embed only the source allowlist, never credentials, local data, or the venv.
# A script kernel uploads code_file, not every Python file beside it.
RUNNER = '''import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sources = __SOURCES__
manifest = __MANIFEST__
project = Path(tempfile.mkdtemp(prefix="img-classification-"))
for relative, content in sources.items():
    destination = project / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")

output = Path("/kaggle/working")
output.mkdir(parents=True, exist_ok=True)
(output / "run.json").write_text(json.dumps(manifest, indent=2))
# Retain the exact submitted source alongside results.
(output / "source.json").write_text(json.dumps(sources, indent=2))
os.environ["PYTHONPATH"] = str(project / "src")
os.environ["IMG_CLASSIFICATION_DATA_DIR"] = str(project / "data")
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(project / "mpl-cache")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HOME"] = str(project / "hf-cache")

with (output / "train.log").open("w", buffering=1) as log:
    def run(command):
        log.write("$ " + " ".join(command) + "\\n")
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        if process.wait():
            raise RuntimeError(f"Command failed ({process.returncode}); see train.log")

    # Preserve Kaggle's CUDA-enabled torch installation.
    run([sys.executable, "-c", "import sys, torch; assert sys.version_info >= (3, 12), 'Python 3.12+ required'; assert tuple(map(int, torch.__version__.split('.')[:2])) >= (2, 4), 'PyTorch 2.4+ required'; assert torch.cuda.is_available(), 'No CUDA GPU allocated'; print('GPU:', torch.cuda.get_device_name(0)); print('PyTorch:', torch.__version__)"])
    missing = [package for module, package in
               [("datasets", "datasets>=3,<6"), ("PIL", "pillow>=10"), ("numpy", "numpy>=1.26,<3"), ("matplotlib", "matplotlib>=3.8,<4")]
               if importlib.util.find_spec(module) is None]
    if missing:
        run([sys.executable, "-m", "pip", "install", *missing])
    with (output / "environment.txt").open("w") as environment:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=environment, check=True)
    run([sys.executable, "-m", "img_classification.prepare_data"])
    config_path = project / "training.toml"
    config_path.write_text("[training]\\n" + "\\n".join(
        f"{key} = {str(value).lower() if isinstance(value, bool) else value}"
        for key, value in manifest["training"].items()))
    run([sys.executable, "-m", "img_classification.train", "--device", "cuda",
         "--config", str(config_path), "--output-dir", str(output)])
'''


def build(username: str, epochs: int, batch_size: int, lr: float, *, config: TrainConfig | None = None) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", username):
        raise ValueError("Use your Kaggle username, not an email or URL.")
    if epochs < 1 or batch_size < 1 or not 0 < lr < float("inf"):
        raise ValueError("epochs, batch-size and lr must be positive finite values")
    config = config or TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    slug = "cifar10-" + run_id
    sources = {path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
               for path in sorted((ROOT / "src" / "img_classification").rglob("*.py"))}
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    manifest = {"run_id": run_id, "kernel": f"{username}/{slug}",
                "source_sha256": digest, "training": config.to_dict()}
    folder = BUILD_ROOT / run_id
    folder.mkdir(parents=True, exist_ok=False)
    # Insert source last so source text is never treated as template directives.
    runner = RUNNER.replace("__MANIFEST__", repr(manifest), 1).replace("__SOURCES__", repr(sources), 1)
    compile(runner, "runner.py", "exec")
    (folder / "runner.py").write_text(runner, encoding="utf-8")
    (folder / "run.json").write_text(json.dumps(manifest, indent=2))
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": manifest["kernel"], "title": slug, "code_file": "runner.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }, indent=2))
    (BUILD_ROOT / "latest.json").write_text(json.dumps({"run_dir": str(folder)}))
    return folder


def kaggle_command() -> list[str]:
    if shutil.which("kaggle"):
        return ["kaggle"]
    if importlib.util.find_spec("kaggle"):
        return [sys.executable, "-m", "kaggle"]
    raise RuntimeError("Kaggle CLI missing. Install it with: uv tool install kaggle")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build", help="Snapshot source and parameters locally; no upload")
    builder.add_argument("--username", required=True)
    builder.add_argument("--config", type=Path, default=ROOT / "configs/cifar10.toml")
    builder.add_argument("--epochs", type=int)
    builder.add_argument("--batch-size", type=int)
    builder.add_argument("--lr", type=float)
    for name in ("push", "status", "pull"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, help="Defaults to the latest locally built run")
        if name == "push":
            command.add_argument("--accelerator", default="NvidiaTeslaT4")
    args = parser.parse_args()
    if args.command == "build":
        config = load_train_config(args.config, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        folder = build(args.username, config.epochs, config.batch_size, config.lr, config=config)
        print(f"Built: {folder}\nReview runner.py and kernel-metadata.json, then run push.")
        return
    folder = args.run_dir
    if folder is None:
        pointer = BUILD_ROOT / "latest.json"
        if not pointer.exists():
            parser.error("No local run exists. Run build first.")
        folder = Path(json.loads(pointer.read_text())["run_dir"])
    manifest = json.loads((folder / "run.json").read_text())
    kernel = manifest["kernel"]
    cli = kaggle_command()
    if args.command == "push":
        subprocess.run([*cli, "kernels", "push", "-p", str(folder), "--accelerator", args.accelerator], check=True)
    elif args.command == "status":
        subprocess.run([*cli, "kernels", "status", kernel], check=True)
    else:
        destination = ROOT / "outputs" / "kaggle" / manifest["run_id"]
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run([*cli, "kernels", "output", kernel, "-p", str(destination)], check=True)
        print(f"Results: {destination}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
