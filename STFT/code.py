import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from math import e, pi, sin, cos, log
import sys
from scipy.io import wavfile
import numpy as np
j = 1j


def stft(x, window_size, step_size, sample_rate, window_type='rect'):
    # return a Short-Time Fourier Transform of x, using the specified window
    # size and step size.
    # return your result as a list of lists, where each internal list
    # represents the DFT coefficients of one window.  I.e., output[n][k] should
    # represent the kth DFT coefficient from the nth window.
    output = []
    if window_type == 'hann':
        window = np.hanning(window_size)
    elif window_type == 'rect':
        window = np.ones(window_size)
    for i in range(0, len(x), step_size):
        segment = x[i:i+window_size]
        if len(segment) < window_size:
            break
        segment = segment * window  # apply window
        output.append(np.fft.fft(segment))
    return output


def k_to_hz(k, window_size, step_size, sample_rate):
    # return the frequency in Hz associated with bin number k in an STFT with
    # the parameters given above.
    return k*sample_rate/window_size


def hz_to_k(freq, window_size, step_size, sample_rate):
    # return the k value associated with the given frequency in Hz, in an STFT
    # with the parameters given above, rounded to the nearest integer.
    return round(freq*window_size/sample_rate)


def timestep_to_seconds(i, window_size, step_size, sample_rate):
    # return the real-world time in seconds associated with the center of the
    # ith window in an STFT using the parameters given above, rounded to the
    # nearest .01 seconds.
    time_seconds = (i * step_size + window_size / 2) / sample_rate
    return round(time_seconds, 2)

def transpose(x):
    # return the transpose of the input, which is given as a list of lists
    n_rows = len(x)
    n_cols = len(x[0])
    output = []
    for i in range(n_cols):
        new_row = []
        for j in range(n_rows):
            new_row.append(x[j][i])
        output.append(new_row)
    return output

def spectrogram(X, window_size, step_size, sample_rate):
    # X is the output of the stft function (a list of lists of DFT
    # coefficients) this function should return the spectrogram (magnitude
    # squared of the STFT).
    # it should be a list that is indexed first by k and then by i, so that
    # output[k][i] represents frequency bin k in analysis window i.
    X_T = transpose(X)
    output = []
    for freq_bin in X_T:
        new_row = []
        for coeff in freq_bin:
            new_row.append(abs(coeff)**2)
        output.append(new_row)
    
    return output



def plot_spectrogram(sgram, window_size, step_size, sample_rate,name,f_max=None):
    # the code below will uses matplotlib to display a spectrogram.  it uses
    # your k_to_hz and timestep_to_seconds functions to label the horizontal
    # and vertical axes of the plot.
    # amplitudes are plotted on a log scale, since human perception of loudness
    # is roughly logarithmic.

    width = len(sgram[0])
    height = len(sgram)//2 + 1  # only positive frequencies

    if f_max is not None:
        k_max = hz_to_k(f_max, window_size, step_size, sample_rate)
        height = min(k_max, height)          # update height
        sgram = sgram[:height]

    plt.imshow([[log(i + sys.float_info.min) for i in j] for j in sgram],
               aspect=width/height, origin='lower')

    plt.title(f"Spectrogram of {name} with window_size = {window_size}, step_size = {step_size}")
    
    plt.axis([0, width-1, 0, height-1])

    ticks = ticker.FuncFormatter(lambda x, pos: '{0:.1f}'.format(timestep_to_seconds(x, window_size, step_size, sample_rate)))
    plt.gca().xaxis.set_major_formatter(ticks)
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:.0f}'.format(k_to_hz(y, window_size, step_size, sample_rate)))
    plt.gca().yaxis.set_major_formatter(ticks)

    plt.xlabel('time [s]')
    plt.ylabel('frequency [Hz]')

    plt.colorbar()
    plt.show()


def draw_spectogram(samples,name, window_size, step_size, sample_rate):

    stft_x = stft(samples, window_size, step_size, sample_rate)

    spectrogram_x = spectrogram(stft_x, window_size, step_size, sample_rate)
    plot_spectrogram(spectrogram_x,window_size, step_size, sample_rate,name,4000)


fs, samples = wavfile.read("police.wav")

# Convert stereo to mono if needed
if samples.ndim == 2:
    samples = samples.mean(axis=1)

draw_spectogram(samples[:len(samples)//4],"Police Siren", 2048,512,fs)
