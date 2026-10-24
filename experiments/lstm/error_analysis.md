# LSTM Model Error Analysis (Held-out Fold 5 Evaluation)

- **Assigned Team Member:** Ryan Gupta (Student ID: 230953288)
- **Model Architecture:** 2-Layer Bidirectional LSTM (`ESC50LSTM`)
- **Input Representation:** Sequential MFCC (`216 x 40`)
- **Evaluation Subset:** ESC-50 Official Fold 5 (400 test audio clips, 8 per class)
- **Overall Accuracy:** 48.75%

---

## 1. Strongest Performing Sound Classes

The LSTM does best on sounds with a distinctive, repeating temporal shape or a clear spectral signature that survives MFCC compression:

| Class Name | Category | Precision | Recall | F1-Score | Key Acoustic Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **glass_breaking** | Interior / Domestic | 0.8889 | 1.0000 | **0.9412** | Sharp broadband onset followed by a decaying high-frequency tail; a clear time pattern. |
| **clock_alarm** | Interior / Domestic | 0.8750 | 0.8750 | **0.8750** | Strictly periodic beeping; the LSTM can follow the on/off rhythm across frames. |
| **crying_baby** | Human non-speech | 0.7273 | 1.0000 | **0.8421** | Long harmonic vocal bursts separated by inhalations. |
| **rooster** | Animals | 0.7778 | 0.8750 | **0.8235** | Characteristic rising-falling crow contour over ~1–2 s. |
| **rain** | Natural soundscapes | 0.7000 | 0.8750 | **0.7778** | Stationary broadband texture; mean pooling gives a stable summary. |
| **church_bells** | Exterior / Urban | 1.0000 | 0.6250 | **0.7692** | Strong tonal partials with slow decay. |
| **crow** | Animals | 0.7500 | 0.7500 | **0.7500** | Repeated harsh caws with regular spacing. |

---

## 2. Weakest Performing Sound Classes

| Class Name | Category | Precision | Recall | F1-Score | Primary Confusions (Fold 5) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **airplane** | Exterior / Urban | 0.0000 | **0.0000** | 0.0000 | All 8 clips predicted as `thunderstorm` (4) or `train` (4): all three are long low-frequency rumbles. |
| **fireworks** | Exterior / Urban | 0.0000 | **0.0000** | 0.0000 | `engine` (4), `keyboard_typing` (2): sparse bursts over background noise. |
| **helicopter** | Exterior / Urban | 0.0000 | **0.0000** | 0.0000 | Spread over `clock_tick`, `crackling_fire`, `wind`, `train`: the rotor pulse is not separated from other repetitive textures. |
| **washing_machine** | Interior / Domestic | 0.0000 | **0.0000** | 0.0000 | Each clip predicted as a different class: no consistent learned pattern. |
| **clock_tick** | Interior / Domestic | 0.0909 | **0.1250** | 0.1053 | Confused with `drinking_sipping`, `water_drops`, `door_wood_knock`: all short isolated clicks. |
| **vacuum_cleaner** | Interior / Domestic | 0.1250 | **0.1250** | 0.1250 | `crickets` (3): both are continuous high-frequency drones. |
| **engine** | Exterior / Urban | 0.1538 | 0.2500 | 0.1905 | Also *receives* the `fireworks` false positives, so precision drops. |

---

## 3. Most Frequent Confusion Pairs

| True Class | Predicted As | Count (of 8) | Likely Cause |
| :--- | :--- | :---: | :--- |
| airplane | thunderstorm | 4 | Both are sustained low-frequency rumbles with slow amplitude changes. |
| airplane | train | 4 | Steady mechanical drone; MFCC envelopes are near-identical. |
| fireworks | engine | 4 | Background noise floor dominates the averaged MFCC frames between bursts. |
| sea_waves | clapping | 3 | Broadband noise bursts with similar spectral envelope. |
| hen | cat | 3 | Short harmonic animal vocalisations in a similar pitch range. |
| vacuum_cleaner | crickets | 3 | Continuous high-frequency energy with little temporal variation. |
| cow | coughing | 3 | Low-pitched voiced bursts. |

---

## 4. Discussion

1. **Machine and engine noises are the main failure mode.** Airplane, helicopter, washing machine, vacuum cleaner and engine all have an F1 below 0.2. These are *stationary* sounds: the MFCC frame at t = 1 s looks like the frame at t = 4 s, so the LSTM's temporal modelling adds nothing. The fine spectral detail that separates them (e.g. rotor harmonics vs. jet noise) is largely lost when 128 Mel bands are compressed into 40 cepstral coefficients.
2. **Temporally structured sounds are where the LSTM is strong.** Glass breaking, clock alarm, crying baby and rooster all have a clear "shape over time", which is exactly the inductive bias of a recurrent model.
3. **Overfitting limits everything.** Training accuracy reaches 95% while validation accuracy stalls at ~47%, and validation loss increases after epoch ~10. With 32 training clips per class, a 640k-parameter recurrent model can memorise individual sequences.
4. **Comparison with the spectrogram models.** The CNN (63.25%) and CNN-LSTM (57.75%) both beat the LSTM on fold 5. This supports the project hypothesis that local time-frequency pattern learning matters more than long-range temporal modelling for ESC-50.

### Possible improvements (future tuning)

- Training-set-only augmentation (time shift, time/frequency masking, mixup) to reduce overfitting.
- Adding delta and delta-delta MFCCs so each frame carries local rate-of-change information.
- Smaller hidden size or stronger dropout, plus early stopping on a validation split carved from the training folds.
