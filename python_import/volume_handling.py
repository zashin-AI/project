# 볼륨 조절

from pydub import AudioSegment, effects
import os, librosa

def volume_normal(origin_dir, out_dir):
    '''
    Args :
        origin_dir : 불러올 wav파일이 있는 경로
        out_dir : 오디오 볼륨을 정규화 시킨 파일 저장 경로
    '''
    origin_dir = librosa.util.find_files(origin_dir, ext=['wav'])

    for path in origin_dir:
        audio = AudioSegment.from_wav(path)
        _, w_id = os.path.split(path)
        w_id = w_id[:-4]
        # 오디오 볼륨 정규화(db : -1과 1사이)
        normalizedsound = effects.normalize(audio) 
        filename = out_dir + w_id + '_volume_normal.wav'
        normalizedsound.export(filename, format='wav')
        
    print('--voulum normal done--')

def volume_updown(path, volume, outpath):
    '''
    Args :
        path : wav 파일 불러올 경로
        volume(decibel) : 데시벨 up & down 숫자로 설정 ex) 10 or -10
        filename : 파일 이름 설정
        outpath : 저장할 폴더 경로
    '''
    audio = AudioSegment.from_wav(path)
    _, w_id = os.path.split(path)
    w_id = w_id[:-4]

    audio = audio + volume

    if volume > 0 :
        filename = outpath + w_id + f'+{volume}db_up.wav'
    elif volume < 0 :
        filename = outpath + w_id + f'{volume}db_down.wav'
    else : 
        pass
    audio.export(filename, format='wav')
        
    print('--voulum up & down done--')