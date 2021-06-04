# 묵음 제거하기

import sys
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence   
# split_on_silence 참고 사이트,  https://github.com/jiaaro/pydub/issues/169  
# 기존 split_on_silence 는 copy 해두고, https://github.com/jiaaro/pydub/blob/master/pydub/silence.py 이 사람이 만든  split_on_silence 복사/수정
import librosa
from voice_handling import import_test, voice_sum

# split_silence_hm : 오디오 파일 침묵구간 마다 오디오 자름 > 자른 오디오 저장 > 합친 오디오 저장
def split_silence_hm(audio_dir, split_silence_dir, sum_dir) :
    '''
    Args : 
        audio_dir : 여러 오디오('wav')가 있는 파일경로
        split_silence_dir : 묵음 부분 마다 자른 오디오 파일을 저장할 파일 경로
        sum_dir : 묵음 부분 마다 자른 오디오 파일을 합쳐서 저장할 파일경로
    '''
    
    # audio_dir에 있는 모든 파일을 가져온다.
    audio_dir = librosa.util.find_files(audio_dir, ext=['wav'])

    # 폴더 생성하기
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)

    # audio_dir에 있는 파일을 하나 씩 불러온다.
    for path in audio_dir :
        print("묵음을 없앨 파일 ", path)

        # 오디오 불러오기
        sound_file = AudioSegment.from_wav(path)

        # 파일 이름만 가져오기
        _, w_id = os.path.split(path)
        w_id = w_id[:-4]

        # 가장 최소의 dbfs가 무엇인지
        # dbfs : 아날로그 db과는 다른 디지털에서의 db 단위, 0일 때가 최고 높은 레벨
        dbfs = sound_file.dBFS

        # silence 부분 마다 자른다. 
        audio_chunks = split_on_silence(sound_file,  
            min_silence_len= 200,
            silence_thresh= dbfs - 16 ,
            # keep_silence= 100
            keep_silence= 0
        )

        # 파일 명으로 새로운 폴더를 생성한다.
        createFolder(split_silence_dir + w_id)

        # silence 부분 마다 자른 거 wav로 저장
        for i, chunk in enumerate(audio_chunks):        
            out_file = split_silence_dir + w_id + "\\" + w_id+ f"_{i}.wav"
            # print ("exporting", out_file)
            chunk.export(out_file, format="wav")

        # 묵음을 기준으로 자른 오디오 파일을 하나의 파일로 합친다.
        path_wav = split_silence_dir + w_id + "\\" 
        print("묵음으로 잘린 파일이 저장된 곳", path_wav) 
        path_out = sum_dir + w_id + '_silence_total.wav'
        print("오디오 합친 파일 경로 ", path_out) 
        voice_sum(form='wav', audio_dir=path_wav, save_dir=None, out_dir=path_out)
        # voice_handling.py 55번째 줄, 아래처럼 수정해야 돌아감
        # combined = combined.append(wav, crossfade=0) 
