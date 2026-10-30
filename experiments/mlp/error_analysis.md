# MLP error analysis

This analysis comes from the best checkpoint (epoch 38) on the 400 held-out
fold-5 clips. See `results/mlp/top_confusions.csv` for the underlying counts
and `results/mlp/confusion_matrix.png` for all 50 classes.

## Findings

- Overall accuracy was 38.00%; macro F1 was 35.61%.
- The strongest recalls were crow and glass breaking (100%), followed by
  rooster, rain, and clock alarm (87.5%).
- Eight classes had zero recall: water drops, drinking/sipping, wood-door
  creaks, washing machine, clock tick, airplane, fireworks, and hand saw.
- The largest directed confusions were airplane → thunderstorm, fireworks →
  crow, crying baby → siren, and crickets → clock alarm (four clips each).
- Other recurring confusions included coughing → sneezing, pouring water →
  brushing teeth, sea waves → clapping, and frog → wood-door knock (three
  clips each).

## Interpretation

Time averaging makes the baseline insensitive to when a sound occurs. That
particularly hurts transient or rhythm-dependent events such as fireworks,
clock ticks, hand-saw strokes, and door creaks. Several mistakes also pair
classes with broadly similar stationary spectra (airplane/thunderstorm and
coughing/sneezing). The epoch-38 training accuracy was 56.75%, an 18.75-point
gap over fold 5, indicating moderate overfitting despite dropout, BatchNorm,
and weight decay.

Future work should preserve temporal statistics (for example mean plus
standard deviation or frame-level pooling), add training-only audio
augmentation, and report full five-fold cross-validation before drawing
strong conclusions about model ranking.
