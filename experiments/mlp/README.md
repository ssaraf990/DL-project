# MLP experiment

This experiment trains `models/mlp/model.py` on the official ESC-50 split:
folds 1-4 contain 1,600 training clips and fold 5 contains 400 held-out clips.
No fold-5 sample is used to fit either the feature scaler or model parameters.

## Configuration

- Seed: 42
- Epochs: 40
- Batch size: 32
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 0.0001
- Loss: cross-entropy on raw logits
- Device: Apple Silicon MPS when available, otherwise CPU

## Reproduction

Place the official ESC-50 `audio/` and `meta/esc50.csv` under `data/raw/`,
install `requirements.txt`, and run from the repository root:

```bash
python experiments/mlp/train_mlp.py
```

The script verifies the split, extracts MFCCs through the common pipeline,
fits preprocessing on training data only, trains the model, evaluates the best
checkpoint, and writes all outputs to `results/mlp/`.

Actual metrics and runtime are recorded in `results/mlp/metrics.json` and
`results/mlp/metrics.csv`; per-epoch measurements are in
`results/mlp/training_history.csv`.

## Actual run results

The completed 40-epoch CPU run selected epoch 38 and achieved **38.00%**
accuracy on the 400 held-out fold-5 clips. Macro/weighted precision was
**36.54%**, macro/weighted recall was **38.00%**, and macro/weighted F1 was
**35.61%**. Training took 19.95 seconds after 62.49 seconds of real MFCC
extraction. Exact unrounded values are retained in `results/mlp/metrics.json`.

The best checkpoint is `results/mlp/mlp_esc50_fold5.pt`. It contains the model
state plus the training-only scaler mean and scale. Model checkpoints are
excluded from Git by the repository's existing `*.pt` policy, so reproduction
requires rerunning the command above.

## Limitations

- Averaging MFCCs over time removes temporal ordering and short-lived events.
- ESC-50 supplies only 32 training clips per class in folds 1-4.
- No audio augmentation or hyperparameter sweep was performed.
- Fold 5 is evaluated each epoch for the required protocol and selects the
  checkpoint; it is therefore a validation fold rather than a pristine final
  test set.
