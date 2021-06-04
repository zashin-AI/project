import librosa
from pydub import AudioSegment
import soundfile as sf
import os
from voice_handling import import_test, voice_sum

# ---------------------------------------------------------------
# voice_sum: 오디오 한 wav 파일로 합쳐서 저장하기
# def voice_sum(form, pathaudio, save_dir, out_dir):
# **** example ****
# form(파일 형식): 'wav' or 'flac'
# audio_dir(여러 오디오가 있는 파일경로) = 'C:/nmb/nmb_data/F1F2F3/F3/'
# save_dir(flac일 경우 wav파일로 저장할 경로) = 'C:/nmb/nmb_data/F1F2F3/F3_to_wave/'
# out_dir(wav파일을 합쳐서 저장할 경로+파일명까지) = "C:/nmb/nmb_data/combine_test/F3_sum.wav"

# 1) wav일 때
# filelist = ['f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14']
filelist = ['m1','m2']

for name in filelist : 
    filename = name

    path_wav = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\' + filename
    path_out = 'E:\\nmb\\nmb_data\\mindslab\\minslab_m\\' + filename + '.wav'
    voice_sum(form='wav', audio_dir=path_wav, save_dir=None, out_dir=path_out)

