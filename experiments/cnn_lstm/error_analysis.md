# CNN-LSTM Hybrid Model Error Analysis (Held-out Fold 5 Evaluation)

- **Assigned Team Member:** Karan Shah (Student ID: 230953540)
- **Model Architecture:** 3-Block 2D CNN + 2-Layer LSTM Hybrid (`ESC50CNNLSTM`)
- **Input Representation:** 2D Log-Mel Spectrogram (`128 x 216`)
- **Evaluation Subset:** ESC-50 Official Fold 5 (400 test audio clips)
- **Overall Accuracy:** 57.75%

---

## 1. Strongest Performing Sound Classes

The CNN-LSTM model demonstrated strong performance on continuous natural soundscapes and distinct rhythmic/tonal acoustic events:

| Class Name | Category | Precision | Recall | F1-Score | Key Acoustic Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **rain** | Natural soundscapes | 0.8889 | 1.0000 | **0.9412** | Continuous stationary noise texture with stable temporal features across LSTM frames. |
| **door_wood_knock** | Interior / Domestic | 1.0000 | 0.8750 | **0.9333** | Distinct transient percussive sequence easily captured by CNN-LSTM sequence modeling. |
| **crow** | Animals | 0.8000 | 1.0000 | **0.8889** | High-pitch repeating vocal call with strong spectral harmonic structure. |
| **cow** | Animals | 0.8750 | 0.8750 | **0.8750** | Low-frequency sustained tonal lowing sound. |
| **pouring_water** | Natural soundscapes | 0.7273 | 1.0000 | **0.8421** | Continuous fluid sound texture with distinctive high-frequency spectral envelope. |
| **coughing** | Human non-speech | 0.7778 | 0.8750 | **0.8235** | Impulsive vocal bursts with distinct temporal repetition. |
| **rooster** | Animals | 0.6667 | 1.0000 | **0.8000** | High-energy crowing sequence. |

---

## 2. Weakest Performing Sound Classes

The hybrid model struggled on brief isolated transient sounds or classes with low signal-to-noise ratios:

| Class Name | Category | Precision | Recall | F1-Score | Primary Vulnerability & Confusions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **pig** | Animals | 0.0000 | **0.0000** | **0.0000** | Short low-frequency grunts completely misclassified into domestic ambient sounds. |
| **water_drops** | Natural soundscapes | 0.0000 | **0.0000** | **0.0000** | Extremely short isolated droplets lost after CNN temporal downsampling ($T=216 \to T'=27$). |
| **fireworks** | Urban / Exterior | 0.0000 | **0.0000** | **0.0000** | Explosive pops confused with `crackling_fire`, `can_opening`, and `gunshots`. |
| **door_wood_creaks**| Interior / Domestic | 0.2000 | **0.1250** | 0.1538 | Continuous squeaking friction noise confused with `chainsaw` and `mouse_click`. |
| **breathing** | Human non-speech | 1.0000 | **0.1250** | 0.2222 | Very faint low-amplitude airflow sound with high false negative rate. |
| **helicopter** | Urban / Exterior | 0.2857 | **0.2500** | 0.2667 | Low-frequency rhythmic thumping confused with `airplane` and `train`. |

---

## 3. Empirical Error Patterns & Plausible CNN-LSTM Limitations

1. **Impact of Temporal Downsampling in CNN Layers:**
   - In the hybrid pipeline, 3 pooling layers reduce the temporal dimension from $T=216$ down to $T'=27$ steps before feeding vectors into the LSTM.
   - While this compresses continuous soundscapes (`rain`, `pouring_water`) effectively, it causes severe information loss for ultra-short transient signals like `water_drops` and `pig` grunts (resulting in 0% recall).

2. **Sequential Dependency vs. Over-Parameterization:**
   - The combined parameter footprint of 2D CNN layers plus a 2-layer LSTM ($hidden\_size=128$) increases model capacity. On a dataset with 1,600 training samples, the hybrid model achieves 79.56% training accuracy but experiences slight overfitting compared to the purely convolutional 2D CNN baseline.

3. **Noise Texture vs. Tonal Signals:**
   - Tonal signals with distinct temporal sequences (`church_bells`, `clock_alarm`, `door_wood_knock`) are easily modeled by the recurrent LSTM layers, whereas non-stationary broadband noise (`wind`, `fireworks`) presents ambiguous sequence representations.
