"""Observe unscaled Conv weight gradients without modifying training."""
import csv
from pathlib import Path

import torch


class GradientMonitor:
    """Equal-weight batch statistics; RMS avoids dependence on tensor size."""

    def __init__(self, model: torch.nn.Module, threshold: float = 1e-8):
        self.weights = {name: layer.weight for name, layer in model.named_modules()
                        if isinstance(layer, torch.nn.Conv2d)}
        self.threshold = threshold
        self.samples = []

    @torch.no_grad()
    def record(self):
        rows = []
        for weight in self.weights.values():
            if weight.grad is None:
                raise RuntimeError('Missing Conv weight gradient')
            grad = weight.grad.detach().float()
            finite = torch.isfinite(grad).all()
            safe = torch.where(finite, grad, torch.zeros_like(grad))
            rows.append(torch.stack((safe.square().mean().sqrt(),
                                     (safe.abs() <= self.threshold).float().mean(),
                                     finite.float())))
        self.samples.append(torch.stack(rows))

    def summary(self) -> dict:
        values = torch.stack(self.samples).cpu()
        layers = {}
        for index, name in enumerate(self.weights):
            valid = values[:, index, 2].bool()
            rms = values[valid, index, 0].double()
            mean = rms.mean().item() if len(rms) else None
            std = rms.std(unbiased=False).item() if len(rms) else None
            layers[name] = {
                'rms_mean': mean, 'rms_std': std,
                'rms_cv': std / mean if mean else None,
                'near_zero_fraction': values[valid, index, 1].mean().item() if len(rms) else None,
                'finite_batches': len(rms), 'nonfinite_batches': int((~valid).sum()),
            }
        first, last = list(layers.values())[0], list(layers.values())[-1]
        ratio = (first['rms_mean'] / last['rms_mean']
                 if first['rms_mean'] is not None and last['rms_mean'] else None)
        return {'batches': len(self.samples), 'near_zero_threshold': self.threshold,
                'first_last_rms_ratio': ratio, 'layers': layers}


def export_gradient_comparison(output: Path, histories: dict) -> None:
    import matplotlib.pyplot as plt

    rows = [{'model': name, 'epoch': row['epoch'], 'layer': layer,
             'near_zero_threshold': row['gradients']['near_zero_threshold'],
             'first_last_rms_ratio': row['gradients']['first_last_rms_ratio'], **stats}
            for name, history in histories.items() for row in history
            for layer, stats in row['gradients']['layers'].items()]
    with (output / 'gradients.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for name, history in histories.items():
        epochs = [row['epoch'] for row in history]
        gradients = [row['gradients'] for row in history]
        layer_names = list(gradients[0]['layers'])
        for ax, layer in zip(axes[0, :2], (layer_names[0], layer_names[-1])):
            ax.plot(epochs, [g['layers'][layer]['rms_mean'] for g in gradients], label=name)
        axes[0, 2].plot(epochs, [g['first_last_rms_ratio'] for g in gradients], label=name)
        axes[1, 0].plot(epochs, [g['layers'][layer_names[0]]['rms_cv'] for g in gradients], label=name)
        axes[1, 1].plot(epochs, [g['layers'][layer_names[0]]['near_zero_fraction'] for g in gradients], label=name)
        ax = axes[1, 2]
        for index, layer in enumerate(layer_names):
            ax.plot(epochs, [g['layers'][layer]['rms_mean'] for g in gradients],
                    color=f'C{index}', linestyle='-' if name == 'cnn' else '--',
                    label=f'{name}: {layer}')
    titles = ('First Conv gradient RMS', 'Last Conv gradient RMS', 'First / last Conv RMS',
              'First Conv RMS CV across batches', 'First Conv fraction |grad| <= 1e-8',
              'All Conv gradients (CNN solid, ResNet dashed)')
    for ax, title in zip(axes.flat, titles):
        ax.set(xlabel='Epoch', title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, ncol=2 if ax is axes[1, 2] else 1)
    # Symlog preserves exact zero gradients instead of hiding them on a log axis.
    for ax in (axes[0, 0], axes[0, 1], axes[1, 2]):
        ax.set_yscale('symlog', linthresh=1e-8)
    axes[0, 2].set_yscale('symlog', linthresh=1e-4)
    axes[0, 2].axhline(1, color='gray', linestyle=':')
    fig.tight_layout()
    fig.savefig(output / 'gradients.png', dpi=160)
    plt.close(fig)
