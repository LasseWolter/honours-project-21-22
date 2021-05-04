import librosa.display
import matplotlib.pyplot as plt

# Load the audio file 
AUDIO_FILE= './short_audio.wav'
samples, sample_rate = librosa.load(AUDIO_FILE, sr=None)

plt.figure(figsize=(14,5))
librosa.display.waveplot(samples, sr=sample_rate)
plt.show()
