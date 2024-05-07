import soundfile as sf
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from code_exercise1 import my_windowing, get_index_from_time, get_input_split_size
from scipy.signal import get_window
import scipy as sp

def compute_LP_coefficients(signal, M=12):
    R = np.correlate(signal, signal, mode='full')
    R = R[len(R)//2:]
    R = -R
    c = R[:M]
    b = R[1:M+1]
    a = sp.linalg.solve_toeplitz((c, c), b)
    return a
    
    
    
    

def plot_frequency_response(a, fs, numPoints, whole=True):
    """
    Plot the frequency response of the LP filter.

    :param a: the LP coefficients
    :param fs: the sampling frequency
    """
    w, h = sp.signal.freqz(1, a, numPoints, whole=whole, fs=fs)
    plt.plot(w, abs(h)) # for the size use numPoints, the segment length in sampels
    plt.ylabel('Amplitude ')
    plt.xlabel('Frequency ')
    plt.show()
    
def plot_dft_and_filter(dft, a, fs, numPoints, gain=1):
    w, h = sp.signal.freqz(gain, a, numPoints, whole=True, fs=fs)
    h = h[:len(h)//2]
    plt.plot(20 * np.log10(np.abs(dft)), label='DFT')
    plt.plot(20 * np.log10(abs(h)), label='Filter')
    plt.ylabel('Amplitude ')
    plt.xlabel('Frequency ')
    plt.legend()
    
    plt.show()

def plot_residual_signal(a, signal):
    print(a)
    a = np.concatenate(([1], a))
    e = sp.signal.lfilter(a, 1, signal)
    plt.plot(signal, label='Original Signal')
    plt.plot(e, label='Residual Signal')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def plot_pre_emphasized_signal_and_normal(signal, pre_emphasized):
    plt.plot(signal, label='Original Signal')
    plt.plot(pre_emphasized, label='Pre-emphasized Signal')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    
if __name__ == '__main__':
    # No. 1
    data, samplerate = sf.read('Audio/speech1.wav')
    
    # No. 2
    # we need to look at the signal and select one voiced and one unvoiced segment
    # voiced is at v-time of 176, meaning it starts at 176-16=160 and ends at 176+16=192, this is index 10
    # unvoiced is at v-time 544, meaning it starts at 544-16=528 and ends at 544+16=560, this is index 33
    windows, v_time = my_windowing(data, samplerate, 32, 16)
    voiced_segment = windows[10]
    unvoiced_segment = windows[33]
    
    voiced_window = get_window('hann', get_index_from_time(32 / 1000, samplerate))
    unvoiced_window = get_window('hann', get_index_from_time(32 / 1000, samplerate))
    
    # No. 3
    a_voiced = compute_LP_coefficients(voiced_segment)
    a_unvoiced = compute_LP_coefficients(unvoiced_segment)
    
    # No. 4
    # plot_frequency_response(a_voiced, samplerate, len(voiced_segment))
    # plot_frequency_response(a_unvoiced, samplerate, len(unvoiced_segment))
    
    # No. 5
    dft_voiced = np.fft.rfft(voiced_segment)
    dft_unvoiced = np.fft.rfft(unvoiced_segment)
    
    # plot_dft_and_filter(dft_voiced, a_voiced, samplerate, len(voiced_segment))
    # plot_dft_and_filter(dft_unvoiced, a_unvoiced, samplerate, len(unvoiced_segment))
    
    # No. 6
    
    # plot_residual_signal(a_voiced, voiced_segment)
    # plot_residual_signal(a_unvoiced, unvoiced_segment) 
    
    # No. 7
    
    # maybe modify the gain?
    # plot_dft_and_filter(dft_voiced, a_voiced, samplerate, len(voiced_segment), gain=0.1)
    # plot_dft_and_filter(dft_unvoiced, a_unvoiced, samplerate, len(unvoiced_segment), gain=0.1)
    
    # plot_dft_and_filter(dft_voiced, a_voiced, samplerate, len(voiced_segment), gain=0.02)
    # plot_dft_and_filter(dft_unvoiced, a_unvoiced, samplerate, len(unvoiced_segment), gain=0.02)
    
    # No. 8
    
    # a_voiced_2 = compute_LP_coefficients(voiced_segment, M=2)
    # a_voiced_16 = compute_LP_coefficients(voiced_segment, M=16)
    # a_voiced_32 = compute_LP_coefficients(voiced_segment, M=32)
    
    # plot_dft_and_filter(dft_voiced, a_voiced_2, samplerate, len(voiced_segment))
    # plot_dft_and_filter(dft_voiced, a_voiced_16, samplerate, len(voiced_segment))
    # plot_dft_and_filter(dft_voiced, a_voiced_32, samplerate, len(voiced_segment))
    
    # No. 9
    # b are the numerators
    # a are the denominators
    # alpha is 0.95
    
     
    pre_emphasized_voiced = sp.signal.lfilter([1, -0.95], 1, voiced_segment)
    pre_emphasized_unvoiced = sp.signal.lfilter([1, -0.95], 1, unvoiced_segment)

    plot_pre_emphasized_signal_and_normal(voiced_segment, pre_emphasized_voiced)
    plot_pre_emphasized_signal_and_normal(unvoiced_segment, pre_emphasized_unvoiced)
    
    
    
    
    
    
    
        
