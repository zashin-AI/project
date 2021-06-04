import librosa
from pydub import AudioSegment
import soundfile as sf
import os

audio_dir = 'C:\\nmb\\nmb_data\\STT\\P\\'
save_dir =  'C:\\nmb\\nmb_data\\STT\\P_WAV\\'

def flac_to_wav(form, audio_dir, save_dir):
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

flac_to_wav(form='flac', audio_dir=audio_dir, save_dir=save_dir)
