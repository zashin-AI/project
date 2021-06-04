from pydub import AudioSegment
import soundfile as sf
import librosa.display
import librosa
import os


def speed_change(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(
        sound.raw_data, 
        overrides={
            "frame_rate": int(sound.frame_rate * speed)
        }
    )
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

sound = AudioSegment.from_file('E:\\nmb\\nmb_data\\predict\\M3.wav')
out_file = "E:\\nmb\\nmb_data\\predict\\M3_slow.wav"
slow_sound = speed_change(sound, 1.0) 
slow_sound.export(out_file, format="wav")