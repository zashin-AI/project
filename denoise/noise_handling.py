import librosa
import soundfile as sf
import os
import numpy as np
import noisereduce as nr

# 기존 저장 시 data 에서 reduce_noise 로 바꿔서 재업로드함. 2021-05-11

def denoise_tim(
    load_dir,
    out_dir,
    noise_min,
    noise_max,
    n_fft,
    hop_length,
    win_length
):

    '''
    Args :
        load_dir : c:/nmb/nmb_data/audio_data/ 로 해야함
        out_dir : 저장 할 파일 경로
        noise_min : 노이즈 최소값
        noise_max : 노이즈 최대값
        n_fft : n_fft
        hop_length : hop_length
        win_length : win_length

    e.g. :
        denoise_tim(
            'c:/nmb/nmb_data/audio_data/',
            'c:/nmb/nmb_data/audio_data_noise/',
            5000, 15000,
            512, 128, 512
        )
    '''

    for (path, dir, files) in os.walk(load_dir): # 하위 디렉토리와 파일 체크
        for filename in files:
            ext = os.path.splitext(filename)[-1] # 확장자명만을 취함
            ext_dir = os.path.splitext(path)[0][27:] + '_denoise/' # str 화 된 디렉토리 경로내에서 특정 폴더의 이름만 반환
            if ext == '.wav':
                try:
                    if not(os.path.isdir(out_dir + ext_dir)): # wav 파일인 경우 새 폴더 생성
                        os.makedirs(os.path.join(out_dir + ext_dir))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        print("Failed to create directory!!!!!")
                        raise
                data, sr = librosa.load("%s/%s" % (path, filename)) # 파일 로드

                noise_part = data[noise_min:noise_max] # 원본 데이터 시간만큼의 노이즈 생성

                reduce_noise = nr.reduce_noise( # 노이즈 제거
                    audio_clip=data, 
                    noise_clip=noise_part,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length)
                sf.write(out_dir + ext_dir + filename[:-4] + '_denoise.wav', reduce_noise, sr) # 노이즈 제거 한 파일 생성
                print("%s/%s" % (path, filename) + ' done') # 완료 된 경우에 출력

if __name__ == '__main__':
    denoise_tim(
        load_dir = 'c:/nmb/nmb_data/audio_data/',
        out_dir = 'c:/nmb/nmb_data/audio_data_denoise/',
        noise_min = 5000,
        noise_max = 15000,
        n_fft = 512,
        hop_length = 128,
        win_length = 512
    )