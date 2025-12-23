from scipy.io import wavfile
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

notes = [
    "C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb",
    "G", "G#/Ab", "A", "A#/Bb", "B"
]

octaves = [
    [65.4063, 69.2956, 73.4162, 77.7817, 82.4068, 87.3070, 92.4986,
     97.9988, 103.8262, 110.0, 116.5409, 123.4708],   # octave 2
    [130.8128, 138.5913, 146.8324, 155.5635, 164.8138, 174.6141, 184.9972,
     195.9977, 207.6523, 220.0, 233.0819, 246.9417],  # octave 3
    [261.6256, 277.1826, 293.6648, 311.1270, 329.6276, 349.2282, 369.9944,
     391.9954, 415.3047, 440.0, 466.1638, 493.8833],   # octave 4
    [523.2511, 554.3653, 587.3295, 622.2540, 659.2551, 698.4565, 739.9888,
     783.9909, 830.6094, 880.0, 932.3275, 987.7666]    # octave 5
]
all_freqs = np.array(octaves).flatten()
all_notes = [note + str(octave+2) for octave in range(len(octaves)) for note in notes]


fs, chord = wavfile.read('short3.wav')



# Regular Chords
# N = len(chord)
# fft_magnitude = np.abs(fft(chord)[:N//2])
# delta_f = fs / N
# freqs = np.arange(N//2) * delta_f



#Short Chords
N = len(chord)
N_fft = 4000
chord_padded = np.zeros(N_fft)
chord_padded[:N] = chord

fft_values = np.fft.fft(chord_padded)
fft_magnitude = np.abs(fft_values[:N_fft//2])

delta_f = fs / N_fft
freqs = np.arange(N_fft//2) * delta_f


N_notes = 10
peaks, _ = find_peaks(fft_magnitude, height=0.1*np.max(fft_magnitude), distance=5)
peak_magnitudes = fft_magnitude[peaks]
top_freqs = freqs[peaks]

sorted_indices = np.argsort(peak_magnitudes)[::-1]
top_peaks = peaks[sorted_indices]
top_freqs = freqs[top_peaks]
top_magnitudes = fft_magnitude[top_peaks]


# Map top frequencies to closest notes
detected_notes = []
for f in top_freqs:
    idx = (np.abs(all_freqs - f)).argmin()
    detected_notes.append(all_notes[idx])
seen = set()

detected_notes_unique = []
for note in detected_notes:
    if note not in seen:
        detected_notes_unique.append(note)
        seen.add(note)
print(detected_notes_unique)




# Plot
plt.figure(figsize=(12,6))
plt.plot(freqs, fft_magnitude, label='FFT Magnitude')
plt.scatter(top_freqs, top_magnitudes, color='red', label='Detected Notes')
for i, f in enumerate(top_freqs):
    plt.text(f, top_magnitudes[i]+0.05*np.max(fft_magnitude),
             detected_notes[i], color='red', rotation=45, ha='center')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Spectrum with Detected Notes")
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()
plt.show()
