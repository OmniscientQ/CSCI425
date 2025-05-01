# CSCI425

Python Machine Learning Project

> "Absolutely despicable that gingerbread men are forced to live
> in houses made of their own flesh."

> "Before guns were invented, armies had to throw bullets at
> each other and if a bullet touched you, you had to sit out
> until the next war."

## Project Members

Jordan Dehmel

Aidan Meens

Evelyn Drollinger-Smith

# Containerization

To enter the included container, run `make docker` or
`make podman`.

# Abstract

## Neural Network Classification of Gender in Audio Streams

This project aims to train a neural network model to classify
audio stream samples along a spectrum from masculine to
feminine. The primary data will be a randomly-selected subset of
the Mozilla Common Voice Data Set. The model produced will be
useful primarily in helping transgender users to learn to use
their voices in ways that match their proper gender, as the
model will be able to provide realtime feedback.

# Process Description / Log

1. Download "Common Voice Corpus 20.0" from
    [mozilla](https://commonvoice.mozilla.org/en/datasets)
    - Unzip data to a large hard drive: This is ~100GB
2. Use [python](./exploration/resampling.py) to reduce the
    sample size
    - Load the index file `validated.tsv` (~500MB)
        ```py
        index: pd.DataFrame = \
            pd.read_csv(
                path.join(PATH, 'validated.tsv'), sep='\t')
        ```
    - Drop unnecessary columns
        ```py
        index.drop(
            ['client_id', 'sentence_id', 'sentence_domain',
            'age', 'accents', 'variant', 'locale', 'segment'],
            axis=1, inplace=True)
        ```
    - Examine data proportions and determine resampled size
        ```py
        m: int = \
            len(index[index['gender'] == 'male_masculine'])
        f: int = \
            len(index[index['gender'] == 'female_feminine'])
        nb: int = \
            len(index[index['gender'] == 'non-binary'])
        ```
        - There are proportionally few non-binary samples: About
            0.35% of the US is non-binary, but far fewer are
            represented here. If we resample the 22 samples we
            have to be 0.35%, we find that we should have a
            sample size of NUMBER GOES HERE. Adjusting for the
            current world gender proportions, NUMBER of those
            should be male and NUMBER should be female.
    - Calculate score (upvotes - downvotes)
        ascending
        ```py
        index['score'] = index['up_votes'] - index['down_votes']
        ```
    - Select new subsamples
        ```py
        male_sample = \
            index[
                index['gender'] == 'male_masculine'
            ].sort_values('score', ascending=False)[:new_m_size]

        female_sample = \
            index[
                index['gender'] == 'female_feminine'
            ].sort_values('score', ascending=False)[:new_f_size]

        nb_sample = index[index['gender'] == 'non-binary']
        ```
    - Combine and save to new index `csv`
        ```py
        resampled = \
            pd.concat([male_sample, female_sample, nb_sample])

        resampled.to_csv(
            path.join(PATH, 'resampled_validated.csv'))
        ```
    - Copy only clip files needed by the new dataset
        ```py
        if not path.exists(path.join(PATH, 'resampled_clips')):
            os.makedirs(path.join(PATH, 'resampled_clips'))

        total: int = len(resampled)
        cur: int = 0
        for index, data in resampled.iterrows():
            print(f'Copying file {data["path"]}: '
                f'{100.0 * cur / total}%')
            cur += 1
            if not path.exists(path.join(PATH, 'resampled_clips',
                                        data['path'])):
                shutil.copyfile(path.join(PATH, 'clips', data['path']),
                                path.join(PATH, 'resampled_clips',
                                        data['path']))
        ```
    - This brings us from ~100GB to ~1GB: Much better
        - We will perform all further operations on this
            smaller, more representative, dataset

# Creating Models and Running the GUI

In order to use the GUI, you must first create the models (e.g.
`./trained_cnn.keras`) locally. This can be done by simply
running all the cells in the model Jupyter notebooks.

After running the Jupyter notebooks, the models should be saved
locally: However, this does not necessarily mean that the GUI
will work yet! First, you must run
`python3 gui/aidan_dill_pickler.py` and
`python3 gui/jordan_dill_pickler.py` to create local `dill`
files of the models. While `.keras` files are the actual models
themselves, `.dill` archives represent arbitrary Python
functions and thus allow any needed preprocessing (e.g.
converting audio data to Mel-spectrogram).

The GUI expects models to be of the `*.dill` form as created by
the dill pickler scripts. Therefore, if no `dill` archives
exist, the GUI will have nothing to load.
