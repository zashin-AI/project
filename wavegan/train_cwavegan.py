import cwawegan_architecture_copy
import wgangp
import utils_copy
from tensorflow.keras.optimizers import Adam

def train_model(sampling_rate = 22050,
                n_batches = 10,
                batch_size = 4,
                audio_path = 'audio/',
                checkpoints_path = 'checkpoints2/',
                architecture_size = 'audio_size',
                resume_training = False,
                path_to_weights = 'checkpoints2/model_weights.h5',
                override_saved_model = False,
                synth_frequency = 500,
                save_frequency = 500,
                latent_dim = 100,
                discriminator_learning_rate = 0.00004,
                generator_learning_rate = 0.00004,
                discriminator_extra_steps = 5
                ):

    '''
    Train the conditional WaveGAN architecture.
    Args:
        sampling_rate (int): Sampling rate of the loaded/synthesised audio.
        n_batches (int): Number of batches to train for.
        batch_size (int): batch size (for the training process).
        audio_path (str): Path where your training data (wav files) are store. 
            Each class should be in a folder with the class name
        checkpoints_path (str): Path to save the model / synth the audio during training
        architecture_size (str) = size of the wavegan architecture. Eeach size processes the following number 
            of audio samples: 'small' = 16384, 'medium' = 32768, 'large' = 65536"
        resume_training (bool) = Restore the model weights from a previous session?
        path_to_weights (str) = Where the model weights are (when resuming training)
        override_saved_model (bool) = save the model overwriting 
            the previous saved model (in a past epoch)?. Be aware the saved files could be large!
        synth_frequency (int): How often do you want to synthesise a sample during training (in batches).
        save_frequency (int): How often do you want to save the model during training (in batches).
        latent_dim (int): Dimension of the latent space.
        discriminator_learning_rate (float): Discriminator learning rate.
        generator_learning_rate (float): Generator learning rate.
        discriminator_extra_steps (int): How many steps the discriminator is trained per step of the generator.
    '''

    '''
    조건부 WaveGAN 아키텍처를 훈련시킵니다.
    인수 :
        sampling_rate (int) :로드 / 합성 된 오디오의 샘플링 속도.
        n_batches (int) : 학습 할 배치 수입니다.
        batch_size (int) : 배치 크기 (학습 프로세스 용).
        audio_path (str) : 훈련 데이터 (wav 파일)가 저장되는 경로입니다.
            각 클래스는 클래스 이름이있는 폴더에 있어야합니다.
        checkpoints_path (str) : 훈련 기간 동안 모델을 저장할 경로 / 오디오를 합성하는 경로
        architecture_size (str) = wavegan 아키텍처의 크기. Eeach 크기는 다음 수를 처리합니다.
            오디오 샘플 수 : 'small'= 16384, 'medium'= 32768, 'large'= 65536 "
        resume_training (bool) = 이전 세션에서 모델 가중치를 복원 하시겠습니까?
        path_to_weights (str) = 모델 가중치가있는 위치 (학습 재개시)
        override_saved_model (bool) = 모델 덮어 쓰기 저장
            이전에 저장된 모델 (과거 epoch안) ?. 저장된 파일이 클 수 있습니다!
        synth_frequency (int) : 학습 중에 샘플을 합성하려는 빈도 (배치 단위).
        save_frequency (int) : 훈련 중 모델을 얼마나 자주 저장 하시겠습니까 (일괄).
        latent_dim (int) : 잠재 공간의 차원.
        Discriminator 학습률 (float) : 판별 학습률.
        generator_learning_rate (float) : 발전기 학습률.
        판별 자 _extra_steps (int) : 판별자가 생성기의 단계 당 훈련 된 단계 수.
    '''

    #get the number of classes from the audio folder
    #오디오 폴더로부터 얻은 클래스의 수
    n_classes = utils_copy.get_n_classes(audio_path)
    
    #build the discriminator
    #discriminator(판별자) 구축
    discriminator = cwawegan_architecture_copy.discriminator(architecture_size=architecture_size,
                                                    n_classes = n_classes)
    #build the generator
    #generator(생성자) 구축
    generator = cwawegan_architecture_copy.generator(architecture_size=architecture_size,
                                                z_dim = latent_dim,
                                                n_classes = n_classes)
                                                
    #set the optimizers
    #discriminator(판별자), generator(생성자) optimizers 설정
    discriminator_optimizer = Adam(learning_rate = discriminator_learning_rate)
    generator_optimizer = Adam(learning_rate = generator_learning_rate)
    
    #build the gan
    gan = wgangp.WGANGP(latent_dim=latent_dim, discriminator=discriminator, generator=generator,
                    n_classes = n_classes, discriminator_extra_steps = discriminator_extra_steps,
                    d_optimizer = discriminator_optimizer, g_optimizer = generator_optimizer)

    # Compile the wgan model
    gan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer)

    #make a folder with the current date to store the current session to
    # 현재 세션을 저장하는 현재 날짜의 폴더(checkpoint)를 만드는 부분
    # avoid overriding past synth audio files and checkpoints
    checkpoints_path = utils_copy.create_date_folder(checkpoints_path)
    
    #save the training parameters used to the checkpoints folder,
    #it makes it easier to retrieve the parameters/hyperparameters afterwards
    utils_copy.write_parameters(sampling_rate, n_batches, batch_size, audio_path, checkpoints_path, architecture_size,
                path_to_weights, resume_training, override_saved_model, synth_frequency, save_frequency,
                latent_dim, discriminator_learning_rate, generator_learning_rate,
                discriminator_extra_steps)
    
    #create the dataset from the class folders in '/audio'
    audio, labels = utils_copy.create_dataset(audio_path, sampling_rate, architecture_size, checkpoints_path)

    #load the desired weights in path (if resuming training)
    if resume_training == True:
        print(f'Resuming training. Loading weights in {path_to_weights}')
        gan.load_weights(path_to_weights)
    
    #train the gan for the desired number of batches
    gan.train(x = audio, y = labels, batch_size = batch_size, batches = n_batches, 
                 synth_frequency = synth_frequency, save_frequency = save_frequency,
                 checkpoints_path = checkpoints_path, override_saved_model = override_saved_model,
                 sampling_rate = sampling_rate, n_classes = n_classes)

if __name__ == '__main__':
    train_model(sampling_rate = 22050,
                n_batches = 100,
                batch_size = 8,
                audio_path = 'audio/',
                checkpoints_path = 'checkpoints2/',
                architecture_size = 'audio_size',
                path_to_weights = 'model_weights.h5',
                resume_training = False,
                override_saved_model = True,
                synth_frequency = 10,
                save_frequency = 10,
                latent_dim = 100,
                discriminator_learning_rate = 0.0002,
                generator_learning_rate = 0.0002),
