'''
Data exploration for our voice dataset
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

    assert path.exists(path.join(dataset_root_path, 'validated.tsv'))

    validated: pd.DataFrame = \
        pd.read_csv(
            path.join(dataset_root_path, 'validated.tsv'),
            sep='\t')

    validated = validated[['path', 'gender']].dropna()

    sns.histplot(data=validated['gender'])
    plt.title(path.join(dataset_root_path, 'validated.tsv'))
    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
