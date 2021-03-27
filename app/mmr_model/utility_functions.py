import matplotlib
matplotlib.use('Agg')
import glob
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pylab
from pydub import AudioSegment
import predict_mir


# Generate spectrogram
def prepare_song(file):
    if ".wav" in file:
        file = music_wav_format(file)
    song_name = replace_format(file)
    sig, fs = librosa.load(file)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    spec = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max))
    pylab.savefig(song_name, bbox_inches=None, pad_inches=0)
    pylab.close()
    return song_name


def music_wav_format(song):
    song = song.split(".")
    AudioSegment.from_wav(song[0] + ".wav").export(song[0] + ".mp3", format="mp3")
    return song[0] + ".mp3"


def mood_prediction(file):
    song = prepare_song(file)
    emotion, value = predict_mir.predict_music_mood(song)
    os.remove(song)
    print("Mood: " + str(emotion) + " - Value: " + str(value))
    return emotion, value

def replace_format(song_name):
    if ".mp3" in song_name:
        return song_name.replace("mp3", "png", 1)
    else:
        return song_name.replace("au", "png", 1)

# Generate spectrograms from songs
def prepare_music_dataset():
    classes = ["happy", "sad", "angry"]
    for cl in classes:
        print("class --- " + cl)
        spec_path = '/mmr_model/music/spectrograms/' + cl + '/'
        os.makedirs(spec_path)
        music_files = glob.glob('/mmr_model/music/music/' + cl + '/*')
        for music in music_files:
            spec_path = '/mmr_model/music/spectrograms/' + cl + '/'
            song_name = os.path.basename(music)
            song_name = replace_format(song_name)
            sig, fs = librosa.load(music)
            pylab.axis('off')
            pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
            spec = librosa.feature.melspectrogram(y=sig, sr=fs)
            librosa.display.specshow(librosa.power_to_db(spec, ref=np.max))
            spec_path = spec_path + song_name
            print(spec_path)
            pylab.savefig(spec_path, bbox_inches=None, pad_inches=0)
            pylab.close()

prepare_music_dataset()

def display_spectrograms():
    sig, fs = librosa.load('/mmr_model/music/tests/angry1.mp3')
    spec = librosa.feature.melspectrogram(y=sig, sr=fs)
    plt.figure(figsize=(10, 4))
    d = librosa.power_to_db(spec, ref=np.max)
    librosa.display.specshow(d, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()




