# Noise-Removal
# AIM:
To study the effect of noise on an audio signal by adding slight Gaussian noise and to reduce the noise using the spectral subtraction method in the frequency domain. 

# APPARATUS REQUIRED:
* Python (Google Colab)

* Libraries:

  * pydub (audio file handling)
  * librosa (signal processing)
  * numpy (numerical operations)
  * matplotlib (plotting)
* Input audio file (WAV/MP3)

# THEORY:

* Audio signals may get corrupted by noise during recording or transmission.
* In this experiment, white Gaussian noise is added to the original signal.
* The noisy signal is processed using the spectral subtraction method.
* The signal is converted into frequency domain using Short-Time Fourier Transform (STFT).
* Noise is estimated from the initial frames of the signal.
* The estimated noise spectrum is subtracted from the noisy signal spectrum.
* Negative magnitude values are set to zero to maintain signal validity.
* The cleaned signal is converted back to time domain using Inverse STFT (ISTFT).
* Thus, noise is reduced without using conventional filters.

 # PROGRAM:
 ```
# ==============================
# STEP 1: Upload audio
# ==============================
from google.colab import files
uploaded = files.upload()

# ==============================
# STEP 2: Install & import
# ==============================
!pip install pydub librosa

from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio

# ==============================
# STEP 3: Load audio
# ==============================
filename = list(uploaded.keys())[0]

audio = AudioSegment.from_file(filename)
audio.export("input.wav", format="wav")

# Load with librosa
input_audio, fs = librosa.load("input.wav", sr=None)

# Normalize
input_audio = input_audio / np.max(np.abs(input_audio))

# ==============================
# STEP 4: Add VERY SLIGHT noise
# ==============================
noise_level = 0.08   # 🔥 small noise

noise = noise_level * np.random.randn(len(input_audio))
noisy_audio = input_audio + noise

# ==============================
# STEP 5: Spectral Subtraction 🔥
# ==============================
# STFT
stft = librosa.stft(noisy_audio)
magnitude, phase = np.abs(stft), np.angle(stft)

# Estimate noise (first few frames)
noise_est = np.mean(magnitude[:, :10], axis=1, keepdims=True)

# Subtract noise
clean_magnitude = magnitude - noise_est

# Avoid negative values
clean_magnitude = np.maximum(clean_magnitude, 0)

# Reconstruct
clean_stft = clean_magnitude * np.exp(1j * phase)
filtered_audio = librosa.istft(clean_stft)

# Normalize
filtered_audio = filtered_audio / np.max(np.abs(filtered_audio))

# ==============================
# STEP 6: Play audio
# ==============================
print("🔊 Original")
display(Audio(input_audio, rate=fs))

print("🔊 Noisy (slight noise)")
display(Audio(noisy_audio, rate=fs))

print("🔊 Filtered (very low noise)")
display(Audio(filtered_audio, rate=fs))

# ==============================
# STEP 7: Plot
# ==============================
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(input_audio)
plt.title("Original")

plt.subplot(3,1,2)
plt.plot(noisy_audio)
plt.title("Noisy")

plt.subplot(3,1,3)
plt.plot(filtered_audio)
plt.title("Filtered (Noise Strongly Reduced)")

plt.tight_layout()
plt.show() 
```
# OUTPUT

# Audio signal

<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/07e5d388-1d29-4c82-ba03-028fe92601a1" />

# Waveform

<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/6d8ca895-61dc-4e8c-848f-f6de3814fc82" />

# RESULT
Therefore the noise added to the audio signal is successfully reduced using the spectral subtraction method, and the filtered audio is obtained with improved clarity compared to the noisy signal.
