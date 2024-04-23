import soundfile as sf
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Exercise 1.1
# data, samplerate = sf.read('speech1.wav')
# print(f"Sampling frequency is:{1/samplerate}")

# plt.plot(data)
# plt.show()
# 2000-2400 haben wir uns angeguckt
# wir haben mit dem Auge bestimmt: 70 samples zwischen den peaks
# da die sampling rate 16000 ist, haben wir 70/16000 = 0.004375s
# IN Hz sind das 228.57Hz
# -> female speaker (around 200hz)

data, samplerate = sf.read('speech2.wav')
# print(f"Sampling frequency is:{1/samplerate}")

# plt.plot(data)
# plt.show()
# beim zweiten haben wir das erste "vokale" angeguckt
# (150 sampels pro periode)
# und sind da auf 106.66HZ gekommen
# -> male speaker (around 100 Hz)
# sd.play(data, samplerate, blocking=True)

#exercise 1.2
def my_windowing ( v_signal : np . ndarray , sampling_rate : int , frame_length : int , frame_shift : int ):
    frame_length_to_seconds = frame_length/1000
    frame_length_to_ticks = int(frame_length_to_seconds * sampling_rate)
    frame_shift_to_seconds = frame_shift/1000
    frame_shift_to_ticks = int(frame_shift_to_seconds * sampling_rate)
    
    all_windows = sliding_window_view(v_signal, window_shape=frame_length_to_ticks)
    
    v_time_frame = [i*frame_shift + 0.5*frame_length for i in range(0, all_windows.shape[0])]
    
    return all_windows[::frame_shift_to_ticks], np.array(v_time_frame)

windows = my_windowing(data, samplerate, 1000, 500)

for audio in windows:
    sd.play(audio, samplerate, blocking=True)
    
def get_input_split_size(signal_length, sampling_rate, frame_length, frame_shift):
    frame_length_to_seconds = frame_length/1000
    frame_length_to_ticks = int(frame_length_to_seconds * sampling_rate)
    frame_shift_to_seconds = frame_shift/1000
    frame_shift_to_ticks = int(frame_shift_to_seconds * sampling_rate)
    # wir gucken, wie viele frame_shifts in das singal passsen, aber müssen aufpassen, dass es gross genug ist für die frame_length_to_ticks
    return (signal_length - frame_length_to_ticks)//frame_shift_to_ticks
    
    
