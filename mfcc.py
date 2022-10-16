import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import os

script_dir = os.path.dirname(__file__)
rel_path = "A_logs\\1.wav"
abs_file_path = os.path.join(script_dir, rel_path)
frequency_sampling, audio_signal = wavfile.read(abs_file_path)

audio_signal = audio_signal[:15000]

features_mfcc = mfcc(audio_signal, frequency_sampling)

print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])

features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')

plt.show()