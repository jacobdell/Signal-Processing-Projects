from scipy.io import wavfile
import numpy as np


fs, y = wavfile.read("echo")

A = 0.9
n_0 = 1323


y = y.astype(float)
y = y / np.max(np.abs(y))



# Calculate A and n0 if not given
corr = np.correlate(y, y, mode='full')
mid = len(corr) // 2
corr_positive = corr[mid+1:]  # ignore lag 0
lag = np.argmax(corr_positive) + 1  # shift back


R0 = corr[mid]
Rn0 = corr[mid + lag]
A_est = Rn0 / R0



# Time Domain Approach

def causal_convolution(x, y):
    """
    Compute the causal discrete-time convolution of two 1D sequences.

    This function implements linear convolution directly in the time domain
    using nested loops, assuming both input sequences are causal
    (i.e., zero for negative indices).

    Parameters
    ----------
    x : array-like
        First input sequence (e.g., impulse response).
    y : array-like
        Second input sequence (e.g., input signal).

    Returns
    -------
    out : list
        The linear convolution of x and y, with length len(x) + len(y).

    """
    out = [0] * (len(x) + len(y))
    for i in range(len(x)):
        for j in range(len(y)):
            out[i + j] += y[j] * x[i]
    return out


def create_kernel_N_non_zeroes(N, A, n_0):
    """
    Construct a truncated echo-canceling impulse response.

    The impulse response consists of N nonzero samples spaced by n_0,
    with alternating signs and exponentially decaying amplitude:

        h[n] = sum_{k=0}^{N-1} (-A)^k delta[n - k n_0]

    Parameters
    ----------
    N : int
        Number of nonzero taps in the impulse response.
    A : float
        Echo attenuation factor (|A| < 1 for stability).
    n_0 : int
        Echo delay in samples.

    Returns
    -------
    out : list
        A discrete-time impulse response containing N nonzero samples.

    Notes
    -----
    This is a finite-length approximation of the ideal infinite impulse
    response used for echo cancellation.
    """
    out = [0] * ((N - 1) * n_0 + 1)
    for i in range(N):
        out[i * n_0] = ((-1) ** i) * (A ** i)
    return out

def remove_echo(N,sample):
    """
    Remove echo from an audio signal using time-domain convolution.

    This function constructs a truncated echo-canceling filter and
    convolves it with the input signal to suppress echoes.

    Parameters
    ----------
    N : int
        Number of nonzero taps used in the echo-canceling filter.
    sample : array-like
        Input audio signal containing echo.

    Returns
    -------
    None
        The echo-reduced signal is written to a WAV file.
    """
    h_N = create_kernel_N_non_zeroes(N,A,n_0)
    conv = causal_convolution(h_N,sample)
    name = f"first_{N}_non_zero.wav"
    wavfile.write(conv, fs, name)


#remove_echo(6,y)



# Frequency Domain Approach

Y = np.fft.fft(y)
N = len(Y)
Omega = 2 * np.pi * np.arange(N) / N
H = 1 / (1 + A * np.exp(-1j * Omega * n_0))
X = Y * H
x = np.fft.ifft(X)

#wavfile.write(x,fs,"Echo_free.wav")
