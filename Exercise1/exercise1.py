import soundfile as sf
import matplotlib.pyplot as plt
import sounddevice as sd

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
sd.play(data, samplerate, blocking=True)
