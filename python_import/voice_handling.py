# 오디오 합치기와 나누기를 함수로 정의

import librosa
from pydub import AudioSegment
import soundfile as sf
import os

# 테스트 한번 불러서 출력하고 가져가세요~
def import_test():
    print('==== it will be great ====')

# ---------------------------------------------------------------
# voice_sum: 오디오를 한 wav 파일로 합쳐서 저장하기

def voice_sum(form, audio_dir, save_dir, out_dir):

    '''
    Args : 
        voice_sum 함수 : 오디오를 한 wav 파일로 합쳐서 저장하기
        form(파일 형식): 'wav' or 'flac'
        audio_dir(여러 오디오가 있는 파일경로) = 'C:/nmb/nmb_data/F1F2F3/F3/'
        save_dir(flac일 경우 wav파일로 저장할 경로) = 'C:/nmb/nmb_data/F1F2F3/F3_to_wave/'
        out_dir(wav파일을 합쳐서 저장할 경로+파일명) = "C:/nmb/nmb_data/combine_test/F3_sum.wav"
    '''
    if form =='flac':
        infiles = librosa.util.find_files(audio_dir)
        for infile in infiles:
            # flac 파일의 이름 불러오기
            _, w_id = os.path.split(infile)
            # 이름에서 뒤에 .flac 떼기
            w_id = w_id[:-5]
            # flac 파일의 data, sr 불러오기
            w_data, w_sr = sf.read(infile)
            # 같은 이름으로 wav 형식으로 저장하기
            sf.write(save_dir + w_id + '.wav', w_data, w_sr, format='WAV', endian='LITTLE', subtype='PCM_16')
        print('==== flac to wav done ====')
        # 경로안의 모든 파일 불러오기
        infiles = librosa.util.find_files(save_dir)
        # 모든 파일을 wav로 불러오기
        wavs = [AudioSegment.from_wav(wav) for wav in infiles]
        # 맨 처음 시작 지정
        combined = wavs[0]
        # 1번부터 이어 붙이기
        for wav in wavs[1:]:
            combined = combined.append(wav) 
        # out_dir 경로에 wav 형식으로 내보내기
        combined.export(out_dir, format='wav')
        print('==== wav sum done ====')

    if form == 'wav':
        infiles = librosa.util.find_files(audio_dir)
        wavs = [AudioSegment.from_wav(wav) for wav in infiles]
        combined = wavs[0]
        for wav in wavs[1:]:
            combined = combined.append(wav) 
        combined.export(out_dir, format='wav')
        print('==== wav sum done ====')


# ---------------------------------------------------------------
# voice_split: 하나로 합쳐진 wav 파일을 5초씩 잘라서 dataset으로 만들기

def voice_split(origin_dir, threshold, out_dir):
    
    '''
    Args : 
        voice_split 함수: 하나로 합쳐진 wav 파일을 5초씩 잘라서 dataset으로 만들기
        origin_dir(하나의 wav파일이 있는 경로+파일명) = 'D:/nmb_test/test_sum/test_01_wav_sum.wav'
        threshold(몇초씩 자를지 5초는 5000) = 5000
        out_dir(5초씩 잘려진 wav 파일을 저장할 경로) = 'D:/nmb_test/test_split/'
    '''

    audio = AudioSegment.from_file(origin_dir)
    _, w_id = os.path.split(origin_dir)
    w_id = w_id[:-4]
    # 임계점 설정(1s = 1000ms)
    start = 0
    threshold = threshold
    end = 0
    counter = 0
    # 본격적인 잘라서 저장하기
    while start < len(audio):
        end += threshold
        print(start, end)
        chunk = audio[start:end]
        filename = out_dir + w_id + f'{counter}.wav'
        chunk.export(filename, format='wav')
        counter += 1
        start += threshold
    print('==== wav split done ====')



# 5초씩 12개 총 1분으로 자르기 위해 end_thresholdf를 만든 함수
def voice_split_1m(origin_dir, threshold, end_threshold, out_dir):
    audio = AudioSegment.from_file(origin_dir)
    _, w_id = os.path.split(origin_dir)
    w_id = w_id[:-4]
    # 임계점 설정(1s = 1000ms)
    start = 0
    threshold = threshold
    end = 0
    counter = 0
    end_threshold = end_threshold
    # 본격적인 잘라서 저장하기
    while start < end_threshold:
        end += threshold
        print(start, end)
        chunk = audio[start:end]
        filename = out_dir + w_id + f'_{counter}.wav'
        chunk.export(filename, format='wav')
        counter += 1
        start += threshold
    print('==== wav split done ====')



# 원하는 초만 자르고 싶어서 만든 함수
def voice_split_term(origin_dir, out_dir, start, end):
    '''
    Args :
        voice_split_term : 음성 파일에서 원하는 부분을 추출해주는 함수
        origin_dir : 파일 불러올 경로
        out_dir : 저장할 경로
        start : 시작하는 부분(msec)
        end : 끝나는 부분(msec)
    '''
    audio = AudioSegment.from_file(origin_dir)
    _, w_id = os.path.split(origin_dir)
    w_id = w_id[:-4]
    start = start
    end = end
    print(start, end)
    chunk = audio[start:end]
    filename = out_dir + w_id + '.wav'
    chunk.export(filename, format='wav')
    print('==== wav split done ====')