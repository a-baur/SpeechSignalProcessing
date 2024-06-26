import soundfile as sf
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import get_window
from code_exercise1 import (
    get_index_from_time,
    my_windowing,
    get_fundamental_frequency,
)


def convert_to_samples(milliseconds: int, sampling_freq: int):
    """
    Convert a millisecond duration into the number of samples given the sampling frequency.

    :param milliseconds: duration to be converted to number of samples
    :param sampling_freq: the sampling frequency
    :return: number of samples
    """
    return int(milliseconds * (10 ** (-3)) * sampling_freq)


def compute_stft(
        v_signal: np.ndarray,
        fs: int,
        frame_length: int,
        frame_shift: int,
        v_analysis_window: np.ndarray
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the short-time Fourier transform.

    :param v_signal: vector containing the time domain signal
    :param fs: the sampling frequency in Hz
    :param frame_length: frame length in milliseconds
    :param frame_shift: frame shift in milliseconds
    :param v_analysis_window: vector that contains the spectral analysis window
    :return: a tuple containing
        - a matrix which stores the complex short-time spectra in each row
        - a vector which contains the frequency axis in Hz corresponding to the rows of the matrix
        - time steps around which a frame is centered
    """
    v_windows, v_time = my_windowing(v_signal, fs, frame_length, frame_shift)

    v_windows = np.apply_along_axis(lambda x: x * v_analysis_window, 1, v_windows)
    m_stft = np.fft.fft(v_windows)
    v_freq = np.fft.fftfreq(frame_length, d=1/fs) # TODO: frequencies correct? convert_to_samples(.., , fs)
    print(v_freq)
    print(m_stft)
    m_stft = m_stft[:, :m_stft.shape[1] // 2 +1]
    v_freq = v_freq[:frame_length // 2 ]
    print(frame_length // 2 +1) # TOOD: sollte eigentlich 257 sein und nciht 17 (also gleich fenster breite)
    print(v_time.shape)
    # print(m_stft.shape, v_freq.shape, v_time.shape)
    # print(convert_to_samples(frame_length, fs))
    return m_stft, v_freq, v_time
# Questions regarding 4:
# Why are the computed spectra complex conjugate symmetric?
# -> because real valued signals only consists of real valued frequencies and the complex parts need to resolve each other
# What may be the advantage of only considering one half of the spectrum?
# -> we know, that it is symmetric, so one half suffices to recunstruct the signal. Less data to store
# How can you compute the frequency for each spectral bin? 
# -> What is a spectral bin?
# How many sampling points does the spectrum have after you removed the mirrored part while including the Nyquist frequency bin?
# -> 0.5*frame_length + 1 = 257
1


def compute_istft(stft: np.ndarray, sampling_rate: int, frame_shift: int, synthesis_window: np.ndarray) -> [np.ndarray]:
    """
    Compute the inverse short-time Fourier transform.

    :param stft: STFT transformed signal
    :param sampling_rate: the sampling rate in Hz
    :param frame_shift: the frame shift used to compute the STFT in milliseconds
    :param synthesis_window: a numpy array containing a synthesis window function (length must match with time domain
    signal segments that were used to compute the STFT)
    :return: a numpy array containing the time domain signal
    """

    # compute inverse rFFT and apply synthesis window
    time_frames = np.fft.irfft(stft)
    num_frames, samples_per_frame = time_frames.shape
    assert samples_per_frame == len(synthesis_window), "Synthesis window must match the number of samples per frame."
    time_frames *= synthesis_window

    # compute output size
    samples_per_shift = convert_to_samples(frame_shift, sampling_rate)
    output_len = samples_per_frame + (num_frames - 1) * samples_per_shift
    time_signal = np.zeros((output_len))

    # reconstruct signal by adding overlapping windowed segments
    for i in range(num_frames):
        time_signal[i * samples_per_shift:i * samples_per_shift + samples_per_frame] += time_frames[i]

    return time_signal


def plot_spectrogram(data, samplerate, frame_length, frame_shift, window="hann", fig=None, ax=None):
    v_analysis_window = get_window(window, get_index_from_time(frame_length / 1000, samplerate), fftbins=True)
    m_stft, v_freq, v_time = compute_stft(data, samplerate, frame_length, frame_shift, v_analysis_window)

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        draw = True
    else:
        draw = False
    im = ax.imshow(
        10 * np.log10(np.maximum(np.square(np.abs(m_stft.T)), 1e-15)),
        cmap='viridis',
        extent=[v_time[0], v_time[-1], v_freq[0], v_freq[-1]],  # noqa
        aspect='auto',
        origin='lower',
    )
    #fig.colorbar(im, orientation='vertical', pad=0.2)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(f"Spectrogram (frame length: {frame_length}ms, frame shift: {frame_shift}ms)")
    if draw:
        plt.show()
    return ax
# Questions regarding 2:
# Why is the magnitude plotted in dB? Why is it reasonable to introduce a lower limit? What is the lower limit in the command given above in dB?
# -> because the human ear perceives sound logarithmically, some frequencies cannot be perceived or because log cannot be <= 0 and become very large, -150 DB, weil man nochmal mal 10 macht

#b) Identify the voiced, unvoiced and silence segments in the spectrogram of the speech signal by eye.
#Describe their appearance and what distinguishes them
# -> voiced: lines, unvoiced: equally spread noise
# Is it possible to identify the different voicing types more easily in comparison to the time domain representation?
#-> yes, because the frequencies are more clearly visible. Especially the frequencies, because before guessing and pi mal daumen calculating

# c) Produce the same plot as in a) but this time using a frame length corresponding to 8 ms and a frame shift of 2 ms. Further, create a plot for a frame length of 128 ms and a frame shift of 32 ms.
# How well can you distinguish single sinusoidal components? Short impulses? Explain the influence of the different parameter settings.
# -> 1. more "crips" time, but smeared frequencies, 2. better frequency resolution, smeared time 

# d) Only for the speech signal estimate the fundamental ...
# Do the estimated fundamental frequencies follow the harmonic structures in the spectrogram? You may also want to plot higher harmonics by multiplying your estimated fundamental frequencies with a positive integer value. This way, you can see the precision of the estimated frequencies more precisely
# -> looks similar.

def plot_with_different_parameters(data, samplerate):

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    plot_spectrogram(
        data,
        samplerate,
        frame_length=32,
        frame_shift=8,
        window="hann",
        fig=fig,
        ax=axes[0],
    )

    plot_spectrogram(
        data,
        samplerate,
        frame_length=8,
        frame_shift=2,
        window="hann",
        fig=fig,
        ax=axes[1],
    )

    plot_spectrogram(
        data,
        samplerate,
        frame_length=128,
        frame_shift=32,
        window="hann",
        fig=fig,
        ax=axes[2],
    )

    plt.tight_layout()
    plt.show()


def plot_with_fundamental_frequency(data, samplerate, frame_length, frame_shift, window="hann", harmonies=8):
    """
    Plot the spectrogram of the input signal and overlay the fundamental frequency and its harmonics.
    The fundamental frequency is estimated using autocorrelation.

    :param data: Input signal
    :param samplerate: Sampling rate of the input signal
    :param frame_length: Frame length in milliseconds
    :param frame_shift: Frame shift in milliseconds
    :param window: Window function to be used for spectral analysis
    :param harmonies: Number of harmonics to be plotted
    :return: None
    """
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    plot_spectrogram(data, samplerate, frame_length, frame_shift, window, fig, ax)
    fundamental_frequencies, v_time_frame = get_fundamental_frequency(data, samplerate, frame_length, frame_shift)
    for i in range(harmonies):
        ax.plot(v_time_frame, np.multiply(fundamental_frequencies, i + 1), color='red', linestyle='--')
    plt.show()


def plot_reconstructed_signal(data, samplerate, frame_length, frame_shift, window="hann", playback=False):
    analysis_window = np.sqrt(get_window(window, convert_to_samples(frame_length, samplerate), fftbins=False)) # fftbins = periodic

    stft, freq, time = compute_stft(data, samplerate, frame_length, frame_shift, analysis_window)
    reconstructed_signal = compute_istft(stft, samplerate, frame_shift, analysis_window)

    if playback:
        sd.play(reconstructed_signal, samplerate, blocking=True)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    ax.plot(reconstructed_signal, color='red')
    ax.plot(data, alpha=0.5, color='blue')

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Reconstructed signal")
    ax.legend(["Reconstructed signal", "Original signal"])

    plt.show()
# Is it possible to perfectly reconstruct the input signal? Are there parts where a perfect reconstruction is not possible when a √Hann-window is used as analysis and synthesis window
# -> yes, but window needs to be chosen such that the overlapping windows can be summed up to a constant. at the beginning and at the end it is not possiblek, because the overlapping windows do not sum to a constant.
# What happens, when you unset the parameter periodic in the window generation? Which error can you observe in the reconstructed signal? Explain the difference in the window function which causes this behavior.
# -> no error


if __name__ == "__main__":
    #data, samplerate = sf.read('Audio/phone.wav')
    data, samplerate = sf.read('Audio/speech1.wav')

    # plot_spectrogram(data, samplerate, frame_length=32, frame_shift=8)  # 2a
    # plot_with_different_parameters(data, samplerate)  # 2c
    # plot_with_fundamental_frequency(data, samplerate, frame_length=32, frame_shift=8, harmonies=16)  # 2d
    # plot_reconstructed_signal(data, samplerate, frame_length=32, frame_shift=16, playback=False)  # 3
    
    v_test_singal = np.ones(2048)
    plot_reconstructed_signal(v_test_singal, 16000, 32, 16, playback=False) 
