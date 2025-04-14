'''
A basic GUI for model execution
Jordan Dehmel, 2025, MIT license
'''

from typing import Optional
from os import path
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np
import librosa
import keras
import TKinterModernThemes as TKMT
import sounddevice as sd
from scipy.io.wavfile import write


def audio_to_png(filename: str, overwrite: bool = False) -> str:
    '''
    If the given mp3 file does not already have a saved
    spectrograph image, creates it. Either way, returns the path
    of the image.
    :param filename: The mp3 file to convert
    :param overwrite: If true, never fetches from cache
    :returns: The filepath of the spectograph version
    '''

    out: str = filename + '.png'
    if path.exists(out) and not overwrite:
        return out

    x, sample_rate = librosa.load(
        filename, res_type='kaiser_fast')
    s = librosa.feature.melspectrogram(y=x, sr=sample_rate)

    plt.clf()

    librosa.display.specshow(
        librosa.power_to_db(s, ref=np.max), x_axis='time',
        y_axis='mel', fmin=50, fmax=280, cmap='gray')

    plt.gcf().set_dpi(64)
    plt.gca().set_position((0, 0, 1, 1))

    plt.savefig(out)

    return out


class GenderClassifierGUI(TKMT.ThemedTKinterFrame):
    '''
    A basic graphical user interface for handling our
    pre-trained gender classification models
    '''

    def __init__(self):
        '''
        Initialize, call the main screen
        '''

        super().__init__('Voice GUI', 'park', 'light')

        self.__model_filepath: Optional[str] = None
        self.__model: Optional[keras.models.Model] = None

        self.__about_text: str = (
            'This project was made as a learning\n'
            + 'exercise during Spring 2025 at\n'
            + 'Colorado Mesa University for\n'
            + '"Python Machine Learning" w/ Dr.\n'
            + 'Ram Basnet. It is licensed under\n'
            + 'the MIT license.\n\n'
            + 'Jordan Dehmel, Aidan Meens,\n'
            + 'Evelyn Drollinger-Smith')

        self.root.report_callback_exception = \
            self.report_callback_exception

        self.__main_screen()
        self.run()

    def report_callback_exception(self, _, val, __) -> None:
        '''
        Overload the default tkinter error handling, which is
        just printing and not halting
        '''

        self.__clear()
        self.Label(text='An error occurred!')

        print(val)
        self.Label(text=val)

        self.__model = None
        self.__model_filepath = None

        self.Button(
            text='Back', command=self.__main_screen)
        self.Button(
            text='Quit', command=self.root.quit)

    def __clear(self) -> None:
        '''
        Clear the screen
        '''

        for child in self.root.winfo_children():
            child.destroy()

    def __main_screen(self) -> None:
        '''
        Display the main screen from which you can access the
        others
        '''

        self.__clear()

        self.Label(text='Gender Classification ML Project')

        def on_model_load_button_pushed():
            '''
            Callback lambda for loading new models
            '''

            self.__model = None
            self.__model_filepath = filedialog.askopenfilename(
                filetypes=[('Keras Models', '*.keras')])

        self.Button(
            text='Select model',
            command=on_model_load_button_pushed)

        self.Button(
            text='Use existing audio file',
            command=self.__audio_file_page)

        self.Button(
            text='Record and classify audio',
            command=self.__live_audio_page)

        self.Button(
            text='About',
            command=self.__about_page)

    def __load_model(self) -> None:
        '''
        Shows a loading screen, loads the model, then returns.
        This halts all other execution until it finishes!
        '''

        assert self.__model_filepath is not None, \
            'Select a model first!'

        if self.__model is None:
            self.__clear()
            self.Label(text='Loading model...')

            self.__model = \
                keras.models.load_model(self.__model_filepath)

    def __results_page(self, image_filepath) -> None:
        x = [keras.preprocessing.image.img_to_array(
                keras.preprocessing.image.load_img(
                    image_filepath,
                    target_size=(64, 64)))]

        y_pred = self.__model.predict(np.array(x))

        # From OneHotEncoder:
        # 'female' 'male' 'non-binary'
        f, m, nb = y_pred[0]
        f = round(100.0 * f)
        m = round(100.0 * m)
        nb = round(100.0 * nb)

        self.__clear()

        self.Label(
            text='Predictions w/ confidence:')
        self.Label(text=f'Female: {f}%')
        self.Label(text=f'Male: {m}%')
        self.Label(text=f'Nonbinary: {f}%')

        self.Button(
            text='Back',
            command=self.__main_screen)

    def __audio_file_page(self) -> None:
        '''
        The page where you load a pre-existing audio file
        '''

        self.__load_model()
        self.__clear()

        def on_file_load_button_pressed():
            input_filepath = filedialog.askopenfilename(
                filetypes=[('Voice Clips', '*.mp3')])

            self.__clear()
            self.Label(text='Processing...')
            self.__results_page(audio_to_png(input_filepath))

        self.Button(
            text='Select file',
            command=on_file_load_button_pressed)

        self.Button(
            text='Back',
            command=self.__main_screen)

    def __live_audio_page(self) -> None:
        '''
        The page from which you can record a voice clip and have
        it classified
        '''

        self.__load_model()
        self.__clear()

        def on_record_button_press():
            self.__clear()
            self.Label(text='Recording...')

            audio_data = np.array([])

            def callback(indata, _, __, ___, ____):
                nonlocal audio_data
                audio_data += indata

            strm = sd.Stream(channels=1, blocksize=2048,
                             callback=callback)

            def on_stop_button_press():
                strm.close()
                recording_path = 'recorded.wav'

                print(len(audio_data))

                write(recording_path, 44100, audio_data)
                self.__results_page(audio_to_png(recording_path))

            self.Button(text='Stop recording',
                        command=on_stop_button_press)

        self.Button(text='Start recording',
                    command=on_record_button_press)

        self.Button(
            text='Back',
            command=self.__main_screen)

    def __about_page(self) -> None:
        '''
        Displays help text
        '''

        self.__clear()

        self.Label(text=self.__about_text, size=12, sticky='e')

        self.Button(text='Back',
                    command=self.__main_screen)


# If main, run this as a script
if __name__ == "__main__":
    GenderClassifierGUI()
