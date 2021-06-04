import librosa
import librosa.display
import numpy as np
import soundfile as sf
import os

def pitch_change(loaddir, n_steps, outdir):
    
    """
    Args :
        loaddir : 파일 로드 경로
        n_steps : 음정 조절 (양수 - 음정 높임, 음수 - 음정 내림)
        outdir  : 파일 저장 경로
    """
    
    dataset = list()
    rateset = list()
    count = 1

    files = librosa.util.find_files(loaddir, ext=['wav'])
    files = np.asarray(files)
    
    if n_steps > 0:
        outpath = '/octave_up/'
        try:
            if not(os.path.isdir(outdir + outpath)):
                os.makedirs(os.path.join(outdir + outpath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise
        for file in files:
            data, rate = librosa.load(file, sr=22050)
            data = librosa.effects.pitch_shift(
            data, rate, n_steps = n_steps
        )
            sf.write(
                outdir + outpath + 'octave_up_' + str(count) + '.wav',
                data, rate
            )
            print(count)
            count += 1

        dataset.append(data)
        rateset.append(rate)

    elif n_steps < 0 :
        outpath = '/octave_down/'
        try:
            if not(os.path.isdir(outdir + outpath)):
                os.makedirs(os.path.join(outdir + outpath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise
        for file in files:
            data, rate = librosa.load(file, sr=22050)
            data = librosa.effects.pitch_shift(
            data, rate, n_steps = n_steps
        )
            sf.write(
                outdir + outpath + 'octave_down_' + str(count) + '.wav',
                data, rate
            )
            print(count)
            count += 1
            
        dataset.append(data)
        rateset.append(rate)
    else :
        pass
    
    dataset = np.array(dataset)
    rateset = np.array(rateset)

    return dataset, rateset