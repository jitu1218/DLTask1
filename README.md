# Neural Network Validation & Regularization Suite

A complete college assignment with actual experiment outputs, learning curves,
metrics, saved model weights, a generated analysis report, and validation tests.
No GPU or separate dataset download is required after dependencies are installed.

## Start on Windows (PowerShell)

Use Python 3.11 or 3.12. Extract the ZIP and open a terminal inside the
`validation_suite` folder, where `experiment.py` is located.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python experiment.py
```

Open `results/REPORT.md` in VS Code's Markdown preview (Ctrl+Shift+V), or read
`results/REPORT.html` in your browser. PNG plots can be opened directly.
The included outputs are already generated; running the script regenerates
CSV/PNG/Markdown/HTML outputs. The HTML report is regenerated too.

On Linux/macOS use `.venv/bin/python` instead of `.venv\Scripts\python`.

For a smaller CPU-only PyTorch installation, install it first:

```powershell
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
```

Package installation needs internet access. A `Failed to resolve pypi.org` error
means DNS/connectivity failed; restore the connection and retry. It does not by
itself mean the specified package version is missing.

## Run options

Full run: five variants × three seeds × at most 120 epochs. Runtime depends on CPU.
All numerical computation runs on CPU with one PyTorch thread.

```powershell
.venv\Scripts\python experiment.py --epochs 120 --seeds 11 22 33 --output results
.venv\Scripts\python -m unittest discover -s tests -v
```

Quick smoke run (not a substitute for the full comparison):

```powershell
.venv\Scripts\python experiment.py --epochs 5 --seeds 11 --output smoke_results
```

Use a new output directory for changed settings to preserve the submitted run.
To experiment with dropout and L2 strengths, edit `VARIANTS` in `experiment.py`.
Do not tune them against the supplied test scores. For further tuning, treat the
current test as already inspected and obtain a fresh external test set for an
unbiased final estimate.

## Dataset and validation design

No dataset was attached to the assignment, so this project uses scikit-learn's
bundled real handwritten digits dataset: 1,797 images, 8×8 pixels, 10 classes.
Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

- Fixed, stratified 60/20/20 train/validation/test split (rounding produces
  1,077 training, 360 validation and 360 test samples).
- StandardScaler is fitted only on the training subset.
- Architecture: 64 inputs → 128 ReLU → 64 ReLU → 10 logits.
- Baseline: no dropout, no L2, fixed epoch budget.
- Dropout-only: p=0.3 after each hidden activation.
- L2-only: explicit weight penalty, lambda=0.001; biases not penalized.
- Early stopping-only: patience 15, minimum improvement 0.0001; restore best weights.
- Combined: dropout + L2 + early stopping.
- Paired training seeds 11, 22, 33, same architecture and optimizer settings.
- All training/validation curves use evaluation mode, with dropout off and
  no L2 added to predictive loss, allowing a fair loss comparison.
- Select variant by mean validation cross-entropy; then evaluate every variant
  on the held-out test set. Test metrics never affect training or selection.

L2 here is `CE + lambda/2 * sum(weight²)`, explicitly differentiated using autograd.
Adam's own weight decay is disabled. This avoids confusing L2 with decoupled AdamW.

## Outputs

| File | Purpose |
|---|---|
| `results/REPORT.md` | Generated findings and limitations |
| `results/REPORT.html` | Readable snapshot of supplied report with plots |
| `results/learning_curves.png` | Training/validation CE for every variant and seed |
| `results/test_comparison.png` | Mean test accuracy and sample SD |
| `results/confusion_matrices.png` | Confusion matrices for the first prespecified seed |
| `results/metrics.csv` | Per-run validation/test metrics and generalization gaps |
| `results/histories.csv` | Every epoch's loss and accuracy |
| `results/test_predictions.csv` | Original sample index, true label, prediction, probabilities |
| `results/config.json` | Parameters, split counts, software versions |
| `results/selection.json` | Validation-based selection recorded before test evaluation |
| `results/split_and_scaler.npz` | Split indices and training scaler parameters |
| `results/models/*.pt` | All 15 model state dictionaries |

The small dataset is useful for an instructional comparison. Repeated seeds on
one split measure training variability, not uncertainty across populations.
A random split does not establish generalization to unseen writers. No production
accuracy claim is made. Regularization may help one metric and hurt another.

## Submission

Upload this extracted project to your own GitHub repository, including `results`
and the README; submit that repository link if the portal asks for a link.
Do not submit a localhost address. No online repository has been created for you.
