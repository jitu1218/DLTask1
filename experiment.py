"""Reproducible, leakage-aware neural network regularization experiments."""
import argparse
import copy
import csv
import html
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

VARIANTS = {
    'baseline': dict(dropout=0., l2=0., early=False),
    'dropout': dict(dropout=.3, l2=0., early=False),
    'l2': dict(dropout=0., l2=.001, early=False),
    'early_stopping': dict(dropout=0., l2=0., early=True),
    'combined': dict(dropout=.3, l2=.001, early=True),
}


def load_data():
    """One fixed stratified split; fit preprocessing ONLY on training rows."""
    x, y = load_digits(return_X_y=True)
    train, test = train_test_split(np.arange(len(y)), test_size=.2,
                                   random_state=42, stratify=y)
    train, val = train_test_split(train, test_size=.25,
                                 random_state=42, stratify=y[train])
    scaler = StandardScaler().fit(x[train])
    parts = {name: (torch.tensor(scaler.transform(x[ids]), dtype=torch.float32),
                    torch.tensor(y[ids], dtype=torch.long))
             for name, ids in [('train', train), ('val', val), ('test', test)]}
    return parts, {'train': train, 'val': val, 'test': test}, scaler


def network(dropout):
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(64, 10))


def evaluate(model, data):
    """Comparable predictive CE: dropout off, L2 penalty excluded."""
    model.eval()
    with torch.no_grad():
        logits = model(data[0])
        loss = nn.functional.cross_entropy(logits, data[1]).item()
        probabilities = logits.softmax(1).numpy()
    pred = probabilities.argmax(1)
    return {'loss': loss, 'accuracy': accuracy_score(data[1].numpy(), pred),
            'macro_f1': f1_score(data[1].numpy(), pred, average='macro')}, probabilities


def fit(data, config, seed, epochs=120, patience=15):
    torch.manual_seed(seed)
    model = network(config['dropout'])
    # Explicit L2 below applies to weights only, not biases. Adam has no decay.
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    order_rng = torch.Generator().manual_seed(seed)
    history, best, best_epoch, waiting = [], float('inf'), 0, 0
    best_state = None
    x, y = data['train']
    for epoch in range(1, epochs + 1):
        model.train()
        for ids in torch.randperm(len(y), generator=order_rng).split(64):
            optimizer.zero_grad()
            ce = nn.functional.cross_entropy(model(x[ids]), y[ids])
            penalty = sum(p.square().sum() for name, p in model.named_parameters()
                          if name.endswith('weight'))
            objective = ce + .5 * config['l2'] * penalty
            objective.backward()
            optimizer.step()
        train_metrics, _ = evaluate(model, data['train'])
        val_metrics, _ = evaluate(model, data['val'])
        history.append(dict(epoch=epoch, train_loss=train_metrics['loss'],
                            val_loss=val_metrics['loss'],
                            train_accuracy=train_metrics['accuracy'],
                            val_accuracy=val_metrics['accuracy']))
        if val_metrics['loss'] < best - 1e-4:
            best, best_epoch, waiting = val_metrics['loss'], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waiting += 1
        if config['early'] and waiting >= patience:
            break
    if config['early']:
        model.load_state_dict(best_state)
    return model, history, best_epoch if config['early'] else epoch


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output, epochs=120, seeds=(11, 22, 33)):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'models').mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data, indices, scaler = load_data()
    np.savez(output / 'split_and_scaler.npz', **indices,
             mean=scaler.mean_, scale=scaler.scale_)
    manifest = dict(dataset='sklearn bundled digits', split_seed=42,
                    split_sizes={k: len(v) for k, v in indices.items()},
                    epochs=epochs, seeds=list(seeds), variants=VARIANTS,
                    learning_rate=.001, batch_size=64, patience=15, min_delta=.0001,
                    python=platform.python_version(), torch=torch.__version__,
                    numpy=np.__version__, sklearn=sklearn.__version__)
    (output / 'config.json').write_text(json.dumps(manifest, indent=2))
    runs, histories = [], []
    # Stage 1: training/validation only; no test metrics until selection is locked.
    for name, config in VARIANTS.items():
        for seed in seeds:
            model, hist, selected_epoch = fit(data, config, seed, epochs)
            val, _ = evaluate(model, data['val'])
            train, _ = evaluate(model, data['train'])
            row = dict(model=name, seed=seed, epochs_run=len(hist),
                       selected_epoch=selected_epoch, train_loss=train['loss'],
                       val_loss=val['loss'], val_accuracy=val['accuracy'])
            runs.append((model, row))
            histories.extend(dict(model=name, seed=seed, **h) for h in hist)
            torch.save(model.state_dict(), output / 'models' / f'{name}_{seed}.pt')
            print(f'{name:16s} seed={seed} epochs={len(hist):3d} val_loss={val["loss"]:.4f}', flush=True)
    selected = min(VARIANTS, key=lambda n: np.mean([r['val_loss'] for _, r in runs if r['model'] == n]))
    (output / 'selection.json').write_text(json.dumps({
        'selected_variant': selected, 'criterion': 'lowest mean validation cross-entropy across seeds',
        'test_used_for_selection': False}, indent=2))
    predictions = []
    for model, row in runs:
        metrics, prob = evaluate(model, data['test'])
        row.update({f'test_{k}': v for k, v in metrics.items()})
        row['generalization_gap'] = row['val_loss'] - row['train_loss']
        for index, truth, probabilities in zip(indices['test'], data['test'][1].numpy(), prob):
            predictions.append(dict(model=row['model'], seed=row['seed'], index=int(index),
                                    truth=int(truth), prediction=int(probabilities.argmax()),
                                    **{f'p_{i}': float(p) for i, p in enumerate(probabilities)}))
    rows = [r for _, r in runs]
    write_csv(output / 'metrics.csv', rows)
    write_csv(output / 'histories.csv', histories)
    write_csv(output / 'test_predictions.csv', predictions)
    make_plots(output, histories, rows, predictions, seeds)
    report(output, rows, selected, manifest)
    print(f'Report: {output / "REPORT.md"}')


def make_plots(output, histories, rows, predictions, seeds):
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, VARIANTS):
        for seed in seeds:
            hist = [r for r in histories if r['model'] == name and r['seed'] == seed]
            ax.plot([r['epoch'] for r in hist], [r['train_loss'] for r in hist],
                    color='#087f8c', alpha=.65, label='Train (evaluation mode)' if seed == seeds[0] else None)
            ax.plot([r['epoch'] for r in hist], [r['val_loss'] for r in hist],
                    color='#d66b35', alpha=.65, label='Validation' if seed == seeds[0] else None)
        ax.set(title=name.replace('_', ' ').title(), xlabel='Epoch', ylabel='Cross-entropy')
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.05, .8, 'Each line is one training seed.\nSame split and architecture for all runs.\nDropout disabled during curve evaluation.\nL2 excluded from plotted predictive loss.\nEarly-stopped curves end at stopping epoch;\nmetrics use restored best weights.', va='top', linespacing=1.8)
    fig.suptitle('Regularization experiments • training and validation loss', fontsize=17)
    fig.savefig(output / 'learning_curves.png', dpi=160)
    for ax in list(axes.flat)[:5]:
        ax.set_yscale('log')
    fig.suptitle('Learning curves • logarithmic loss axis reveals late-epoch gaps', fontsize=17)
    fig.savefig(output / 'learning_curves_detail.png', dpi=160)
    plt.close(fig)
    names = list(VARIANTS)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    means = [np.mean([r['test_accuracy'] for r in rows if r['model'] == n]) for n in names]
    stds = [np.std([r['test_accuracy'] for r in rows if r['model'] == n], ddof=1) if len(seeds)>1 else 0 for n in names]
    ax.bar(names, means, yerr=stds, capsize=5, color='#087f8c')
    ax.set(ylim=(0, 1), ylabel='Test accuracy', title='Held-out test accuracy • mean ± sample SD across training seeds')
    for i, value in enumerate(means):
        ax.text(i, value-.07, f'{value:.1%}', ha='center', color='white')
    fig.savefig(output / 'test_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    for ax, name in zip(axes, names):
        p = [r for r in predictions if r['model'] == name and r['seed'] == seeds[0]]
        matrix = confusion_matrix([r['truth'] for r in p], [r['prediction'] for r in p], labels=list(range(10)))
        ax.imshow(matrix, cmap='Blues')
        ax.set(title=name, xlabel='Predicted digit', ylabel='True digit', xticks=range(10), yticks=range(10))
    fig.suptitle(f'Test confusion matrices • representative seed {seeds[0]} (not cherry-picked)')
    fig.savefig(output / 'confusion_matrices.png', dpi=160)
    plt.close(fig)


def report(output, rows, selected, manifest):
    lines = ['# Model validation and regularization report', '',
             '## Protocol', '',
             f'Dataset: 1,797 handwritten digit images, 64 input pixels, 10 classes. Stratified split sizes: {manifest["split_sizes"]}. No dataset was supplied, so the bundled digits dataset is used.',
             'StandardScaler is fitted on training data only. A 64→128→64→10 ReLU network is trained with Adam (learning rate 0.001), batch size 64. Each variant uses the same split and paired initialization/shuffle seeds. No synthetic label noise is added.',
             f'Maximum epochs: {manifest["epochs"]}; seeds: {manifest["seeds"]}. Dropout p=0.3. L2 objective: mean cross-entropy + (0.001/2) × sum of squared weight parameters; biases excluded. This is explicit L2, not AdamW decay. Early stopping uses validation CE, patience 15 and min_delta 0.0001, restoring the best checkpoint.',
             'Hyperparameters were specified before inspecting test results. Only the regularization method is compared; this is not an exhaustive hyperparameter search. Baseline, dropout-only and L2-only use final-epoch weights; early stopping variants use restored weights.',
             '', '## Results', '', '| Variant | Validation CE | Test CE | Test accuracy | Test macro F1 | Val−train CE gap |',
             '|---|---:|---:|---:|---:|---:|']
    for name in VARIANTS:
        group = [r for r in rows if r['model'] == name]
        cells = []
        for metric in ['val_loss', 'test_loss', 'test_accuracy', 'test_macro_f1', 'generalization_gap']:
            values = [r[metric] for r in group]
            sd = np.std(values, ddof=1) if len(values)>1 else 0
            cells.append(f'{np.mean(values):.4f} ± {sd:.4f}')
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    base = [r for r in rows if r['model']=='baseline']
    means = {name: {m: float(np.mean([r[m] for r in rows if r['model']==name]))
                    for m in ['test_accuracy', 'test_loss', 'val_loss']}
             for name in VARIANTS}
    accuracy_leader = max(means, key=lambda n: means[n]['test_accuracy'])
    early_epochs = [r['epochs_run'] for r in rows if r['model']=='early_stopping']
    combined_epochs = [r['epochs_run'] for r in rows if r['model']=='combined']
    lines += ['', 'Values are mean ± sample SD across training seeds on one fixed split; these are not confidence intervals or independent test sets.', '',
              '## Interpretation', '',
              f'**Validation-selected variant: {selected}.** This choice was locked using mean validation cross-entropy before the test evaluation. Test results are reported for all variants as requested, but do not change the selection.',
              f'The baseline mean validation-minus-training CE gap is {np.mean([r["generalization_gap"] for r in base]):.4f}. A growing gap with falling training loss and flattening/rising validation loss is evidence of overfitting; a gap alone does not establish the best stopping point. Inspect the full curves rather than accuracy alone.',
              f'The highest mean test accuracy belongs to {accuracy_leader} ({means[accuracy_leader]["test_accuracy"]:.2%}). L2 test CE is {means["l2"]["test_loss"]:.4f}, compared with baseline {means["baseline"]["test_loss"]:.4f}. This illustrates why classification accuracy and probability-sensitive loss can favor different choices.',
              f'Early-stopping-only runs executed {early_epochs} epochs; combined runs executed {combined_epochs}. Reaching the epoch budget means the patience rule did not trigger, not that early stopping is broken. Baseline curves flatten at low validation loss, so these runs do not demonstrate a dramatic late-epoch validation-loss explosion.',
              'Dropout makes hidden activations unavailable during training; L2 penalizes large weights. Either can reduce variance but can also underfit if too strong. Early stopping limits training duration and returns the best validation checkpoint. Regularization is not guaranteed to improve every metric or every seed.',
              'For this comparison, prefer the validation-selected variant when predictive log-loss matters. Accuracy and macro F1 provide complementary classification views. The dataset is small and random splits may share handwriting characteristics; this does not establish performance for new writers or production images. Repeat with writer-separated data and an untouched external test set before deployment.', '',
              '## Visuals', '', '![Learning curves](learning_curves.png)', '', '![Log-scale learning curves](learning_curves_detail.png)', '',
              '![Test comparison](test_comparison.png)', '', '![Confusion matrices](confusion_matrices.png)', '',
              '## References', '',
              '- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html',
              '- https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html',
              '- https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    render_html(output, lines)


def render_html(output, lines):
    parts, in_table = [], False
    for line in lines:
        if in_table and not line.startswith('|'):
            parts.append('</table>')
            in_table = False
        if line.startswith('|---'):
            continue
        if line.startswith('|'):
            header = not in_table
            if header:
                parts.append('<table>')
                in_table = True
            tag = 'th' if header else 'td'
            parts.append('<tr>' + ''.join(f'<{tag}>{html.escape(x.strip())}</{tag}>'
                                          for x in line.strip('|').split('|')) + '</tr>')
        elif line.startswith('!['):
            alt, src = line[2:].split('](')
            parts.append(f'<img src="{html.escape(src[:-1])}" alt="{html.escape(alt)}">')
        elif line.startswith('# '):
            parts.append('<h1>' + html.escape(line[2:]) + '</h1>')
        elif line.startswith('## '):
            parts.append('<h2>' + html.escape(line[3:]) + '</h2>')
        elif line:
            parts.append('<p>' + html.escape(line.replace('**', '')) + '</p>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Model Validation Report</title><style>
    body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:24px;color:#183438;background:#f6f8f8}
    h1,h2{color:#087f8c}table{border-collapse:collapse;width:100%;font-size:14px;background:white}
    th,td{border:1px solid #d7e2e2;padding:10px;text-align:left}th{background:#e3f1f1}
    img{max-width:100%;background:white;margin:18px 0}p{max-width:1050px}
    </style><body>''' + '\n'.join(parts) + '</body></html>'
    (output / 'REPORT.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    args = parser.parse_args()
    if args.epochs < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('epochs must be positive and seeds must be distinct')
    run(args.output, args.epochs, args.seeds)
