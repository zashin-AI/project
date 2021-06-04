# 원하는 특성으로 저장하는 함수

import librosa
import numpy as np
import sklearn
import soundfile as sf

def load_data_mfcc(filepath, filename, labels):

    '''
    Args : 
        filepath : 파일 불러 올 경로
        filename : 불러올 파일 확장자명 e.g. wav, flac....
        labels : label 번호 (여자 0, 남자 : 1)
    '''
    count = 1
    dataset = list()
    label = list()

    def normalize(x, axis = 0):
        return sklearn.preprocessing.minmax_scale(x, axis = axis)

    files = librosa.util.find_files(filepath, ext=[filename])
    files = np.asarray(files)
    for file in files:
        y, sr = librosa.load(file, sr=22050, duration=5.0)
        length = (len(y) / sr)
        if length < 5.0 : pass
        else:
            mels = librosa.feature.mfcc(y, sr=sr)
            mels = librosa.amplitude_to_db(mels, ref=np.max)
            mels = normalize(mels, axis = 1)

            dataset.append(mels)
            label.append(labels)
            print(str(count))
            
            count+=1
        
    if labels == 0:
        out_name = 'female'
        out_dir = 'c:/nmb/nmb_data/npy/'
        np.save(
            out_dir + out_name + '_mfcc_data.npy',
            arr = dataset
        )
        np.save(
            out_dir + out_name + '_mfcc_label.npy',
            arr = label
        )
    elif labels == 1:
        out_name = 'male'
        out_dir = 'c:/nmb/nmb_data/npy/'
        np.save(
            out_dir + out_name + '_mfcc_data.npy',
            arr = dataset
        )
        np.save(
            out_dir + out_name + '_mfcc_label.npy',
            arr = label
        )

    data = np.load(
        out_dir + out_name + '_mfcc_data.npy'
    )
    lab = np.load(
        out_dir + out_name + '_mfcc_label.npy'
    )

    return data, lab



def load_data_mel(filepath, filename, labels):

    '''
    Args : 
        filepath : 파일 불러 올 경로
        filename : 불러올 파일 확장자명 e.g. wav, flac....
        labels : label 번호 (여자 0, 남자 : 1)
    '''

    count = 1

    dataset = list()
    label = list()

    def normalize(x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis=axis)

    files = librosa.util.find_files(filepath, ext=[filename])
    files = np.asarray(files)
    for file in files:
        y, sr = librosa.load(file, sr=22050, duration=5.0)
        length = (len(y) / sr)
        if length < 5.0 : pass
        else:
            mels = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, hop_length=128)
            mels = librosa.amplitude_to_db(mels, ref=np.max)

            dataset.append(mels)
            label.append(labels)
            print(str(count))
            
            count+=1

    if labels == 0:
        out_name = 'female'
        out_dir = 'c:/nmb/nmb_data/npy/'
        np.save(
            out_dir + out_name + '_mel_data.npy',
            arr = dataset
        )
        np.save(
            out_dir + out_name + '_mel_label.npy',
            arr = label
        )
    elif labels == 1:
        out_name = 'male'
        out_dir = 'c:/nmb/nmb_data/npy/'
        np.save(
            out_dir + out_name + '_mel_data.npy',
            arr = dataset
        )
        np.save(
            out_dir + out_name + '_mel_label.npy',
            arr = label
        )

    data = np.load(
        out_dir + out_name + '_mel_data.npy'
    )
    lab = np.load(
        out_dir + out_name + '_mel_label.npy'
    )

    return data, lab