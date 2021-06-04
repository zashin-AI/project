import librosa
from pydub import AudioSegment
import soundfile as sf
import os
import sys
sys.path.append('E:/nmb/nada/python_import/')
from voice_handling import import_test, voice_split, voice_split_1m, voice_split_term

import_test()
# ==== it will be great ====

# ---------------------------------------------------------------
# voice_split: 하나로 합쳐진 wav 파일을 5초씩 잘라서 dataset으로 만들기
# def voice_split(origin_dir, threshold, out_dir):
# **** example ****
# origin_dir(하나의 wav파일이 있는 경로+파일명) = 'D:/nmb_test/test_sum/test_01_wav_sum.wav'
# threshold(몇초씩 자를지 5초는 5000) = 5000
# out_dir(5초씩 잘려진 wav 파일을 저장할 경로) = 'D:/nmb_test/test_split/'

origin_dir = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\nslab_m\\m_total_chunk\\mindslab_m_total\\_noise\\'
# origin_dir = 'E:\\nmb\\nmb_data\\mindslab\\minslab_f\\f_total_chunk\\nslab_f\\f_total_chunk\\mindslab_f_total\\_noise\\'
threshold = 5000 # 몇초씩 자를 것인지 설정
out_dir = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\m_total_chunk\\mindslab_2m5s\\'
# out_dir = 'E:\\nmb\\nmb_data\\mindslab\\minslab_f\\f_total_chunk\\mindslab_2m5s\\'
end_threshold = 120000 # 끝나는 지점(2분)

infiles = librosa.util.find_files(origin_dir)

for files in infiles :
    voice_split_1m(origin_dir=files, threshold=threshold, end_threshold =end_threshold,out_dir=out_dir)
