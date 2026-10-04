"""Train independent CNN/ResNet concurrently with paired live epoch logs."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from itertools import zip_longest
from queue import Queue
import re
from threading import Thread

from .artifacts import save_json
from .config import load_train_config


def model_devices(requested: str, require_two_gpus: bool = False) -> dict[str, str]:
    """Assign one visible GPU per model when at least two are available."""
    import torch

    if requested == 'cpu':
        if require_two_gpus:
            raise ValueError('--require-two-gpus cannot be used with --device cpu')
        return dict(cnn='cpu', resnet='cpu')
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if require_two_gpus and count < 2:
        raise ValueError(f'Two CUDA GPUs required, but PyTorch sees {count}. '
                         'Check the Kaggle accelerator and CUDA_VISIBLE_DEVICES.')
    if not count:
        if requested == 'cuda':
            raise ValueError('CUDA requested, but no CUDA GPU is available.')
        return dict(cnn='cpu', resnet='cpu')
    return dict(cnn='cuda:0', resnet='cuda:1' if count >= 2 else 'cuda:0')


def train_models(output: Path, common: list[str], env: dict[str, str],
                 devices: dict[str, str]) -> None:
    """Stream both child processes and pair metrics by epoch, never by arrival order."""
    events = Queue()
    processes = {}
    readers = []
    epochs = {}

    def read_output(name, process):
        with (output / f'{name}.log').open('w') as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                events.put((name, line.rstrip()))
        events.put((name, None))

    print('\nEpoch | CNN: loss / train / validation | ResNet: loss / train / validation', flush=True)
    try:
        for name in ('cnn', 'resnet'):
            processes[name] = subprocess.Popen(
                [sys.executable, '-u', '-m', 'img_classification.train', '--model', name,
                 '--output-dir', str(output / name), '--device', devices[name], *common], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            reader = Thread(target=read_output, args=(name, processes[name]), daemon=True)
            reader.start()
            readers.append(reader)
        finished = set()
        with (output / 'training.log').open('w') as log:
            while len(finished) < 2:
                name, line = events.get()
                if line is None:
                    code = processes[name].wait()
                    if code:
                        raise RuntimeError(f'{name.upper()} training failed (exit {code}); see {output / (name + ".log")}')
                    finished.add(name)
                    continue
                match = re.fullmatch(r'Epoch (\d+)/(\d+): (.*)', line)
                if match:
                    epoch, total, metrics = match.groups()
                    row = epochs.setdefault(int(epoch), {})
                    row[name] = metrics
                    if len(row) < 2:
                        continue
                    line = f"Epoch {epoch}/{total} | CNN: {row['cnn']} | ResNet: {row['resnet']}"
                else:
                    line = f'[{name.upper()}] {line}'
                print(line, flush=True)
                log.write(line + '\n')
                log.flush()
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for reader in readers:
            reader.join(timeout=10)
        for process in processes.values():
            process.stdout.close()


def architecture_comparison() -> str:
    """Show the independent models without changing the training RNG."""
    import torch
    from .models import CNN, ResNet

    with torch.random.fork_rng(devices=[]):
        cnn = CNN(padding=1)
        resnet = ResNet()
    left = ['CNN — models/cnn.py', *str(cnn).splitlines(),
            'Forward: Conv → BN → ReLU (8 layers) → pool → head']
    right = ['ResNet — models/resnet.py', *str(resnet).splitlines(),
             'Forward: stem → 3 blocks ReLU(F(x) + x) → pool → head']
    width = max(map(len, left))
    return '\n'.join(f'{a:<{width}} | {b}' for a, b in zip_longest(left, right, fillvalue=''))


def plot_test_comparison(output: Path) -> None:
    """Compare the same test examples and distinguish sample/full-test accuracy."""
    import matplotlib.pyplot as plt
    import torch

    previews = {name: torch.load(output / name / 'test_preview.pt', weights_only=True,
                                 map_location='cpu') for name in ('cnn', 'resnet')}
    cnn, resnet = previews.values()
    if not torch.equal(cnn['images'], resnet['images']) or not torch.equal(cnn['labels'], resnet['labels']):
        raise ValueError('Test preview examples differ between models')
    count = len(cnn['images'])
    fig, axes = plt.subplots((count + 3) // 4, 4, figsize=(16, 4 * ((count + 3) // 4)), squeeze=False)
    for index, ax in enumerate(axes.flat):
        ax.axis('off')
        if index >= count:
            continue
        label = cnn['labels'][index].item()
        ax.imshow(cnn['images'][index].permute(1, 2, 0).clamp(0, 1))
        ax.set_title(f"True: {cnn['class_names'][label]}")
        for row, (name, preview) in enumerate(previews.items()):
            prediction = preview['predictions'][index].item()
            correct = prediction == label
            ax.text(0.5, -0.06 - row * 0.10,
                    f"{name.upper()}: {preview['class_names'][prediction]} ({'correct' if correct else 'wrong'})",
                    color='green' if correct else 'red', ha='center', va='top', transform=ax.transAxes)
    summary = ' | '.join(f"{name.upper()} full test: {preview['test_accuracy']:.2%}; "
                         f"shown subset: {(preview['labels'] == preview['predictions']).float().mean().item():.2%}"
                         for name, preview in previews.items())
    fig.suptitle(f'First {count} held-out test images — best validation checkpoints\n{summary}')
    fig.tight_layout(rect=(0, 0.03, 1, 0.94), h_pad=4)
    fig.savefig(output / 'test_predictions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar([name.upper() for name in previews],
                  [100 * preview['test_accuracy'] for preview in previews.values()])
    ax.bar_label(bars, fmt='%.2f%%', padding=4)
    ax.set(ylim=(0, 105), ylabel='Accuracy (%)', title='Full held-out test accuracy')
    fig.tight_layout()
    fig.savefig(output / 'test_accuracy.png', dpi=150)
    plt.close(fig)
    print(summary, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--val-fraction', type=float)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--require-two-gpus', action='store_true',
                        help='Fail before data preparation unless each model can use its own GPU')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        config = load_train_config(args.config, epochs=args.epochs, batch_size=args.batch_size,
                                  lr=args.lr, seed=args.seed, val_fraction=args.val_fraction, amp=args.amp)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    import torch
    print(f'Benchmark source: {Path(__file__).resolve()}', flush=True)
    print(f'Python: {sys.executable}; PyTorch: {torch.__version__}; '
          f'visible CUDA GPUs: {torch.cuda.device_count()}; '
          f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}', flush=True)
    try:
        devices = model_devices(args.device, args.require_two_gpus)
    except ValueError as error:
        parser.error(str(error))
    output = args.output_dir or Path('outputs/benchmark') / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    if output.exists() and any(output.iterdir()):
        parser.error(f'Run already exists: {output}. Choose a fresh --output-dir.')
    output.mkdir(parents=True, exist_ok=True)
    architectures = architecture_comparison()
    print(architectures, flush=True)
    (output / 'architectures.txt').write_text(architectures + '\n')
    # Each model gets a fresh process, RNG, loader, optimizer and CUDA allocator.
    env = {**os.environ, 'MPLBACKEND': 'Agg'}
    save_json(config.to_dict(), output / 'config.json')
    save_json(devices, output / 'devices.json')
    print(f"Devices: CNN → {devices['cnn']}; ResNet → {devices['resnet']}", flush=True)
    if devices['cnn'] == devices['resnet'] == 'cuda:0':
        print('Only one CUDA GPU is visible: both models will share it. '
              'Use --require-two-gpus to reject this fallback.', flush=True)
    subprocess.run([sys.executable, '-m', 'img_classification.prepare_data'], env=env, check=True)
    common = []
    for key, value in config.to_dict().items():
        if key == 'amp':
            common.append('--amp' if value else '--no-amp')
        else:
            common.extend(['--' + key.replace('_', '-'), str(value)])
    results = []
    histories = {}
    train_models(output, common, env, devices)
    for name in ('cnn', 'resnet'):
        metrics = json.loads((output / name / 'metrics.json').read_text())
        histories[name] = metrics['history']
        results.append({key: metrics[key] for key in (
            'model_name', 'parameters', 'best_epoch', 'best_validation_accuracy', 'test_accuracy',
            'train_seconds', 'training_wall_seconds', 'train_images_per_second',
            'peak_gpu_memory_mb', 'device', 'gpu_name', 'amp_enabled', 'torch_version', 'cuda_version')})
        save_json({'config': config.to_dict(), 'results': results}, output / 'benchmark.json')
    splits = [json.loads((output / name / 'split.json').read_text()) for name in ('cnn', 'resnet')]
    if splits[0] != splits[1]:
        raise RuntimeError('Benchmark splits differ')
    with (output / 'benchmark.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    report = ['# CNN vs ResNet', '',
              '| Model | Parameters | Best val | Test | Train (s) | Images/s | Peak CUDA MiB |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in results:
        memory = row['peak_gpu_memory_mb']
        report.append(f"| {row['model_name']} | {row['parameters']} | {row['best_validation_accuracy']:.2%} | "
                      f"{row['test_accuracy']:.2%} | {row['train_seconds']:.2f} | "
                      f"{row['train_images_per_second']:.1f} | {f'{memory:.1f}' if memory is not None else 'N/A'} |")
    delta = 100 * (results[1]['test_accuracy'] - results[0]['test_accuracy'])
    report += ['', f'ResNet − CNN test accuracy: {delta:+.2f} percentage points.', '',
               'Matched padded CNN vs small ResNet; only three identity additions differ.',
               'One seed; this is not statistical evidence of superiority.',
               f"Concurrent devices: CNN={devices['cnn']}, ResNet={devices['resnet']}. "
               'Timings include shared host resources and GPU contention when using the same GPU.',
               'Train seconds include batches, transfers and optimizer steps, including first-epoch startup.',
               'Wall seconds also include validation and checkpoints, excluding data preparation and final test.',
               'Peak CUDA memory is allocated tensor memory during training/validation, not reserved GPU memory.',
               'Evaluation uses float32; training AMP follows config on CUDA.']
    (output / 'benchmark.md').write_text('\n'.join(report) + '\n')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, history in histories.items():
        epochs = [row['epoch'] for row in history]
        for ax, metric, title in zip(axes, ('train_loss', 'validation_accuracy', 'train_seconds'),
                                    ('Training loss', 'Validation accuracy', 'Training seconds / epoch')):
            ax.plot(epochs, [row[metric] for row in history], label=name)
            ax.set(xlabel='Epoch', title=title)
            ax.legend()
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / 'benchmark.png', dpi=160)
    plt.close(fig)
    plot_test_comparison(output)
    print('\n'.join(report), flush=True)
    print(f'\nArtifacts: {output.resolve()}', flush=True)
    for filename in ('test_predictions.png', 'test_accuracy.png', 'benchmark.png'):
        print(f'Plot: {(output / filename).resolve()}', flush=True)


if __name__ == '__main__':
    main()
