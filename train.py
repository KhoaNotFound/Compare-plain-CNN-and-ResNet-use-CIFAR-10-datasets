"""One-command CIFAR-10 CNN/ResNet benchmark: uv run train.py."""
from pathlib import Path
import sys

from img_classification.benchmark import main

if __name__ == '__main__':
    if '--config' not in sys.argv and not any(arg.startswith('--config=') for arg in sys.argv):
        sys.argv.extend(['--config', str(Path(__file__).parent / 'configs/cifar10.toml')])
    main()
