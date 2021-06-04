import sys
sys.path.append('E:/nmb/nada/python_import/')
from split_silence import split_silence_hm
    
audio_dir = 'E:\\nmb\\nmb_data\\mindslab\\minslab_f\\f_2m\\'
split_silence_dir = "E:\\nmb\\nmb_data\\mindslab\\minslab_f\\f_total_chunk\\"
sum_dir = "E:\\nmb\\nmb_data\\mindslab\\minslab_f\\f_total_chunk\\total\\"

split_silence_hm(audio_dir, split_silence_dir, sum_dir)