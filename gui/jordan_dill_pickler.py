'''
Saves the CNN preprocessor as a dill pickle file for GUI
loading.
'''

from typing import Tuple
from os import path
import keras
import librosa
import numpy as np
from matplotlib import pyplot as plt
from dill import dump


model = keras.models.load_model('../trained_cnn.keras')


def preproc_jordan(x, sample_rate) -> Tuple[float, float, float]:
    '''
    Preprocessing for Jordan's CNN model. Maps raw audio data to
    a 3-tuple for predictions (f, m, nb)
    '''

    counter: int = 0
    while path.exists(f'melspec_{counter}.png'):
        counter += 1

    out: str = f'melspec_{counter}.png'
    s = librosa.feature.melspectrogram(y=x, sr=sample_rate)
    plt.clf()

    librosa.display.specshow(
        librosa.power_to_db(s), x_axis='time',
        y_axis='mel', fmin=50, fmax=280, cmap='gray')

    plt.gcf().set_dpi(64)
    plt.gca().set_position((0, 0, 1, 1))

    plt.savefig(out)

    x = [keras.preprocessing.image.img_to_array(
            keras.preprocessing.image.load_img(
                out,
                target_size=(64, 64)))]

    y_pred = model.predict(np.array(x))

    return y_pred[0]


if __name__ == '__main__':
    with open('preproc_jordan.dill', 'wb') as f:
        dump(preproc_jordan, f, recurse=True)
