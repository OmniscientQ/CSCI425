'''
A basic GUI for model execution
Jordan Dehmel, 2025, MIT license

Inexplicably, you have to run with pavucontrol as follows:
`pavucontrol & python3 gui/main.py`

If you don't you'll get all zeros from the mic. I have no idea
why that is true: pavucontrol is just a volume manager.
'''

from typing import Optional
from os import path, remove
from tkinter import filedialog
import numpy as np
import librosa
import keras
import dill
# Teenage Kutant Minja Turtles
import TKinterModernThemes as TKMT
import sounddevice as sd
from scipy.io.wavfile import write


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

        # If provided, is a function mapping raw audio input to
        # a tensor to be processed by the model. This is loaded
        # from a pickle.
        self.__preprocessor_fn = lambda x: x

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

        def on_preprocessor_load_button_pushed():
            '''
            Callback lambda for loading preprocessors
            '''

            fp = filedialog.askopenfilename(filetypes=[
                ('Dill Pickles', '*.dill')])

            with open(fp, 'rb') as f:
                self.__preprocessor_fn = \
                    dill.load(f)

        self.Button(
            text='Select model (required)',
            command=on_model_load_button_pushed)

        self.Button(
            text='Select preprocessor (depends on model)',
            command=on_preprocessor_load_button_pushed)

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

    def __results_page(self, audio_filepath) -> None:
        x, sample_rate = \
            librosa.load(audio_filepath, res_type='kaiser_fast')

        x = self.__preprocessor_fn(x, sample_rate)
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
                filetypes=[
                    ('mp3 files', '*.mp3'),
                    ('wav files', '*.wav')])

            self.__clear()
            self.Label(text='Processing...')
            self.__results_page(input_filepath)

        self.Label(text='Load existing audio clip')

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

            audio_data = []

            def callback(indata, _, __, status):
                nonlocal audio_data
                audio_data.append(indata.copy())

                if status:
                    print(status)

            strm = sd.InputStream(callback=callback)

            def on_stop_button_press():
                nonlocal audio_data, strm

                strm.stop()
                strm.close()
                del strm

                recording_path = 'recorded.wav'

                audio_data = np.concat(audio_data)
                write(recording_path, 44100, audio_data)
                self.__results_page(recording_path)

                # Clean up local data
                if path.exists(recording_path):
                    remove(recording_path)

            self.Button(text='Stop recording',
                        command=on_stop_button_press)

            strm.start()

        self.Label(text='Analyze new recording')

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
