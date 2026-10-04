"""Run matched CNN/ResNet training sequentially and export measured results."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import save_json
from .config import load_train_config


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
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        config = load_train_config(args.config, epochs=args.epochs, batch_size=args.batch_size,
                                  lr=args.lr, seed=args.seed, val_fraction=args.val_fraction, amp=args.amp)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    output = args.output_dir or Path('outputs/benchmark') / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    if output.exists() and any(output.iterdir()):
        parser.error(f'Run already exists: {output}. Choose a fresh --output-dir.')
    output.mkdir(parents=True, exist_ok=True)
    # Each model gets a fresh process, RNG, loader, optimizer and CUDA allocator.
    env = {**os.environ, 'MPLBACKEND': 'Agg'}
    save_json(config.to_dict(), output / 'config.json')
    subprocess.run([sys.executable, '-m', 'img_classification.prepare_data'], env=env, check=True)
    common = ['--device', args.device]
    for key, value in config.to_dict().items():
        if key == 'amp':
            common.append('--amp' if value else '--no-amp')
        else:
            common.extend(['--' + key.replace('_', '-'), str(value)])
    results = []
    histories = {}
    for name in ('cnn', 'resnet'):
        print(f'\n=== Benchmark: {name.upper()} ===', flush=True)
        subprocess.run([sys.executable, '-m', 'img_classification.train', '--model', name,
                        '--output-dir', str(output / name), *common], env=env, check=True)
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
               'One seed, fixed CNN-then-ResNet order; this is not statistical evidence of superiority.',
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
    print('\n'.join(report), flush=True)
    print(f'\nArtifacts: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
