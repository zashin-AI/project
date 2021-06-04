import librosa
import numpy as np
import json
import os
from datetime import datetime

#get the number of classes from the number of folders in the audio dir
# 오디오 디렉토리 내의 폴더 수에서 클래스의 수를 가져옵니다
def get_n_classes(audio_path):
    root, dirs, files = next(os.walk(audio_path))
    n_classes = len(dirs)
    print(f'Found {n_classes} different classes in {audio_path}')
    return n_classes

#load the audio. Pad the audio if the file is shorter than the maximum architecture capacity
# 오디오를 로드합니다. 파일이 아키텍처의 최대 용량보다 짧은 경우 오디오를 패딩르로 채우는 부분
def load_audio(audio_path, sr, audio_size_samples):
    X_audio, _ = librosa.load(audio_path, sr = sr, duration=5.0)
    if X_audio.size < audio_size_samples:
        padding = audio_size_samples - X_audio.size
        X_audio = np.pad(X_audio, (0, padding), mode = 'constant')
    elif (X_audio.size >= audio_size_samples):
        X_audio = X_audio[0:audio_size_samples]
    return X_audio

#save the label names for inference
# 추론을 위해 라벨 이름을 저장
def save_label_names(audio_path, save_folder):
    label_names = {}
    for i, folder in enumerate(next(os.walk(audio_path))[1]):
        label_names[i] = folder
    #save the dictionary to use it later with the standalone generator
    #나중에 generator과 함께 사용하기 위해 폴더 안에 있는 라벨링 이름 저장하는 부분
    with open(os.path.join(save_folder, 'label_names.json'), 'w') as outfile:
        json.dump(label_names, outfile)
        
#create the dataset from the audio path folder
#오디오 경로 폴더에서 dataset를 만드는 부분
def create_dataset(audio_path, sample_rate, architecture_size, labels_saving_path):
    
    if architecture_size == 'audio_size':   # architecture_size : 오디오 길이
        audio_size_samples = 114688
    
    #save the label names in a dict
    # 라벨 이름을 딕셔너리 안에 저장
    save_label_names(audio_path, labels_saving_path)
    audio = []
    labels_names = []
    for folder in next(os.walk(audio_path))[1]:
        for wavfile in os.listdir(audio_path+folder):
            audio.append(load_audio(audio_path = f'{audio_path}{folder}/{wavfile}', sr = sample_rate, audio_size_samples = audio_size_samples))
            labels_names.append(folder)
    audio_np = np.asarray(audio)
    audio_np = np.expand_dims(audio_np, axis = -1)
    labels = np.unique(labels_names, return_inverse=True)[1]
    labels_np = np.expand_dims(labels, axis = -1)
    
    return audio_np, labels_np

#create folder with current date (to avoid overriding the synthesised audio/model when resuming the training)
#현재 날짜에 폴더를 만듭니다 (훈련을 재개 할 때 합성 된 오디오 / 모델을 덮어 쓰지 않도록하기 위해)
def create_date_folder(checkpoints_path):
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    date = datetime.now()
    day = date.strftime('%d-%m-%Y_')
    path = f'{checkpoints_path}{day}{str(date.hour)}h'
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(f'{path}/synth_audio'):
        os.mkdir(f'{path}/synth_audio')     
    return path

#save the training arguments used to the checkpoints folder (it make it easier retrieve the hyperparameters afterwards)
#사용한 훈련 인수를 체크 포인트 폴더에 저장합니다 (나중에 하이퍼 매개 변수를 쉽게 검색 할 수 있도록합니다)
def write_parameters(sampling_rate, n_batches, batch_size, audio_path, checkpoints_path, 
                architecture_size, path_to_weights, resume_training, override_saved_model, synth_frequency, 
                save_frequency, latent_dim, discriminator_learning_rate, generator_learning_rate,
                discriminator_extra_steps):
    print(f'Saving the training parameters to disk in {checkpoints_path}/training_parameters.txt')
    arguments = open(f'{checkpoints_path}/training_parameters.txt', "w")
    arguments.write(f'sampling_rate = {sampling_rate}\n')
    arguments.write(f'n_batches = {n_batches}\n')
    arguments.write(f'batch_size = {batch_size}\n')
    arguments.write(f'audio_path = {audio_path}\n')
    arguments.write(f'checkpoints_path = {checkpoints_path}\n')
    arguments.write(f'architecture_size = {architecture_size}\n')
    arguments.write(f'path_to_weights = {path_to_weights}\n')
    arguments.write(f'resume_training = {resume_training}\n')
    arguments.write(f'override_saved_model = {override_saved_model}\n')
    arguments.write(f'synth_frequency = {synth_frequency}\n')
    arguments.write(f'save_frequency = {save_frequency}\n')
    arguments.write(f'latent_dim = {latent_dim}\n')
    arguments.write(f'discriminator_learning_rate = {discriminator_learning_rate}\n')
    arguments.write(f'generator_learning_rate = {generator_learning_rate}\n')
    arguments.write(f'discriminator_extra_steps = {discriminator_extra_steps}\n')
    arguments.close()
