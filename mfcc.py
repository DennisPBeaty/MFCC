import numpy as np
import matplotlib.pyplot as plot
from scipy.io import wavfile
import python_speech_features
import os
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

keys = ["A", "S", "D", "Space"]

for key in keys:
    for i in range(1,21):
        script_dir = os.path.dirname(__file__)
        rel_path = key + "_logs\\" + str(i) + ".wav"
        abs_file_path = os.path.join(script_dir, rel_path)
        freq_data, audio_data = wavfile.read(abs_file_path)

        audio_data = audio_data[:20000]

        mfcc_data = python_speech_features.mfcc(audio_data, freq_data)

        mfcc_data = mfcc_data.T
        plot.matshow(mfcc_data)
        plot.title('MFCC')

        plot.savefig("outputs\\" + key + "_data\\" + str(i) + ".png")
        plot.close()