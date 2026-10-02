# CNN Model Error Analysis (ESC-50 Fold 5)

- **Assigned Team Member:** Satyam Saraf (Student ID: 230953178)
- **Model Architecture:** 4-Block 2D CNN with Global Average Pooling
- **Input Representation:** 2D Log-Mel Spectrogram (`128 x 216`)
- **Evaluation Subset:** ESC-50 Official Fold 5 (400 test audio clips)
- **Overall Accuracy:** 63.25%

---

## 1. Top Performing Classes (High Recall & F1-Score)

The CNN model demonstrated exceptional classification capability on distinct harmonic and continuous acoustic textures:

| Class Name | Category | Precision | Recall | F1-Score | Key Acoustic Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **crow** | Animals | 1.0000 | 1.0000 | **1.0000** | Distinct repetitive harsh cawing pattern across high-frequency bands. |
| **toilet_flush** | Interior / Domestic | 1.0000 | 1.0000 | **1.0000** | High-energy broad-spectrum noise envelope followed by distinctive water resonant decay. |
| **church_bells** | Urban / Exterior | 0.8889 | 1.0000 | **0.9412** | Strong fundamental pitch harmonics and long reverberant decay tails. |
| **hand_saw** | Urban / Exterior | 1.0000 | 0.8750 | **0.9333** | High-frequency periodic rhythmic friction noise. |
| **rooster** | Animals | 0.7273 | 1.0000 | **0.8421** | Distinct high-pitched crowing call with unique formant structure. |
| **rain** | Natural soundscapes | 0.7273 | 1.0000 | **0.8421** | Stationary broadband Gaussian-like noise texture across time. |

---

## 2. Low Performing Classes (Challenging Categories)

Certain sound categories exhibited lower classification recall due to temporal brevity, ambient background similarity, or overlapping spectral profiles:

| Class Name | Category | Precision | Recall | F1-Score | Primary Confusion & Vulnerability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **pig** | Animals | 0.1429 | **0.1250** | 0.1333 | Short low-frequency grunts confused with human non-speech / domestic background noise. |
| **fireworks** | Urban / Exterior | 0.2500 | **0.1250** | 0.1667 | Impulsive explosive pops confused with `crackling_fire`, `can_opening`, and `gunshots`. |
| **helicopter** | Urban / Exterior | 0.3333 | **0.2500** | 0.2857 | Low-frequency rhythmic thumping confused with `airplane` and `engine` noise. |
| **airplane** | Urban / Exterior | 0.5000 | **0.2500** | 0.3333 | Continuous low-frequency jet rumble confused with `wind` and `helicopter`. |
| **frog** | Animals | 0.5000 | **0.2500** | 0.3333 | Low-pitched croaking confused with `crickets` and `pig`. |
| **water_drops** | Natural soundscapes | 1.0000 | **0.2500** | 0.4000 | High precision (no false positives) but low recall because brief isolated drops were missed or confused with silence. |

---

## 3. Class Confusion Patterns & Acoustic Observations

1. **Stationary Broadband Noise Confusion:**
   - Background environmental noise textures like `wind`, `rain`, `pouring_water`, and `vacuum_cleaner` share similar energy distributions across Mel bands. While `rain` and `pouring_water` achieved strong scores, `wind` was occasionally misclassified as low-frequency mechanical noise (`engine`/`airplane`).

2. **Transient vs. Continuous Signals:**
   - The 2D CNN with $2 \times 2$ MaxPool operations effectively captures continuous spatial-frequency textures, but brief transient clicks (`mouse_click`, `water_drops`, `can_opening`) lose temporal resolution during deep pooling steps.

3. **Harmonic vs. Non-Harmonic Differentiation:**
   - Tonal animal vocalizations (`crow`, `cat`, `rooster`) and tonal urban sounds (`church_bells`, `siren`) produced sharp horizontal ridge patterns in log-Mel spectrograms, making them easily distinguishable for 2D convolutional filters.
