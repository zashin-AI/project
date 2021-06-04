# data load

import librosa
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import gzip
import os
import sys
import numpy as np
sys.path.append('E:/nmb/nada/python_import/')
from feature_handling import load_data_mel

# female
# pathAudio_F = 'E:\\nmb\\nmb_data\\mindslab\\minslab_f\\f_total_chunk\\mindslab_2m5s\\'
# load_data_mel(pathAudio_F, 'wav', 0)

# male
# pathAudio_M = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\mindslab_2m5s\\'
# load_data_mel(pathAudio_M, 'wav', 1)

# file check
x1 = np.load('E:\\nmb\\nmb_data\\npy\\mindslab_2m5s_f_data.npy')
print(x1.shape) # (288, 128, 862)

y1 = np.load('E:\\nmb\\nmb_data\\npy\\mindslab_2m5s_f_label.npy')
print(y1.shape) # (288,)
print(y1[:10])  # [0 0 0 0 0 0 0 0 0 0]

x2 = np.load('E:\\nmb\\nmb_data\\npy\\mindslab_2m5s_m_data.npy')
print(x2.shape) # (48, 128, 862)

y2 = np.load('E:\\nmb\\nmb_data\\npy\\mindslab_2m5s_m_label.npy')
print(y2.shape) # (48,)
print(y2[:10])  # [1 1 1 1 1 1 1 1 1 1]
