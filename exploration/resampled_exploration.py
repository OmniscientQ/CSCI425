'''
Visualization stuff for the resampled dataset, which should be
smaller and more representative of reality
'''

import sys
from os import path
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def main() -> int:
    '''
    Main fn. Returns 0 on success.
    '''

    dataset_root_path: str = input('Path: ')

    assert path.exists(dataset_root_path), 'Nonexistant path!'

    if path.exists(path.join(dataset_root_path, 'en')):
        dataset_root_path = path.join(dataset_root_path, 'en')

    assert path.exists(path.join(dataset_root_path, 'resampled_validated.csv'))
    assert path.exists(path.join(dataset_root_path, 'clip_durations.tsv'))

    validated: pd.DataFrame = \
        pd.read_csv(path.join(dataset_root_path, 'resampled_validated.csv'))
    durations: pd.DataFrame = \
        pd.read_csv(path.join(dataset_root_path, 'clip_durations.tsv'),
                    sep='\t', header=0)
    durations.rename(columns={'clip': 'path'}, inplace=True)

    validated = validated.merge(durations, on='path', how='left')
    validated.to_csv(
        path.join(dataset_root_path,
                  'resampled_validated_with_durations.csv'))

    sns.histplot(data=validated['gender'])
    plt.title(path.join(dataset_root_path, 'resampled_validated.csv'))
    plt.savefig('resampled_gender_dist.png')

    plt.clf()
    sns.histplot(validated['duration[ms]'] / 1_000.0)
    plt.xlabel('Index')
    plt.ylabel('Duration (seconds)')
    plt.savefig('durations.png')

    for _, row in validated.iterrows():
        print('Asserting existence of', path.join(dataset_root_path, row['path']))
        assert path.exists(path.join(dataset_root_path, 'clips', row['path']))

    return 0


if __name__ == '__main__':
    sys.exit(main())
