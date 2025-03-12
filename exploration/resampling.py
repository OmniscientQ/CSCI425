'''
The script that created the resampled dataset
'''

import sys
from os import path
import os
import shutil
import pandas as pd


PATH: str = '/run/media/jorb/BigBackups/voice-data/cv-corpus-20.0-2024-12-06-en/cv-corpus-20.0-2024-12-06/en/'


def main() -> int:
    '''
    Main fn
    :returns: Exit code
    '''

    # Load index: This will be ~500MB
    index: pd.DataFrame = \
        pd.read_csv(path.join(PATH, 'validated.tsv'), sep='\t')

    # ['client_id', 'path', 'sentence_id', 'sentence',
    #  'sentence_domain', 'up_votes', 'down_votes', 'age',
    #  'gender', 'accents', 'variant', 'locale', 'segment']
    print(f'Initial columns: {index.columns}')

    index.drop(
        ['client_id', 'sentence_id', 'sentence_domain',
         'age', 'accents', 'variant', 'locale', 'segment'],
        axis=1, inplace=True)

    index.dropna(subset=['gender'], inplace=True)
    index = index[index['gender'] != 'do_not_wish_to_say']

    # male_masculine     857281
    # female_feminine    302137
    # non-binary             22
    # transgender             9 (not a helpful category)
    # World *sex* data via data.worldbank.org:
    # Female: 49.7%
    # Male:   50.0%
    # Other research says 1.2 million nonbinary in USA
    # Total of 346714662 people in USA (about 0.35% are NB)
    print(index['gender'].value_counts())

    n: int = len(index)

    m: int = len(index[index['gender'] == 'male_masculine'])
    f: int = len(index[index['gender'] == 'female_feminine'])
    nb: int = len(index[index['gender'] == 'non-binary'])

    print('Ratios:')

    print(f'M is {round(100.0 * m / n, 3)}% of the sample, should be 49.95%')
    print(f'F is {round(100.0 * f / n, 3)}% of the sample, should be 49.7%')
    print(f'NB is {round(100.0 * nb / n, 3)}% of the sample, should be 0.35%')

    nb_corrected_size: int = round(100.0 * nb / 0.35)
    print('In order for NB population to be correct proportion,'
          f' we should have {nb_corrected_size} samples')

    new_m_size: int = round(nb_corrected_size * 0.4995)
    new_f_size: int = round(nb_corrected_size * 0.4970)

    print(f'Of {nb_corrected_size}, {new_m_size} should be M')
    print(f'Of {nb_corrected_size}, {new_f_size} should be F')

    print(
        f'This would represent {100.0 * nb_corrected_size / n}%'
        ' of the total sample')

    index['score'] = index['up_votes'] - index['down_votes']
    print(index)

    male_sample = \
        index[index['gender'] == 'male_masculine'].sort_values(
            'score', ascending=False)[:new_m_size]
    print(male_sample)

    female_sample = \
        index[index['gender'] == 'female_feminine'].sort_values(
            'score', ascending=False)[:new_f_size]
    print(female_sample)

    nb_sample = index[index['gender'] == 'non-binary']
    print(nb_sample)

    resampled = \
        pd.concat([male_sample, female_sample, nb_sample])
    print(resampled)

    resampled.to_csv(path.join(PATH, 'resampled_validated.csv'))

    # Resample clips
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

    return 0


if __name__ == '__main__':
    sys.exit(main())
