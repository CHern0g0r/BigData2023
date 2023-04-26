import os
import pandas as pd

from sklearn.model_selection import train_test_split
from argparse import ArgumentParser


def main(data_path):
    csv_path = os.path.join(data_path, 'penguins_size.csv')
    df = pd.read_csv(csv_path).dropna()
    traindf, testdf = train_test_split(
        df, random_state=42, test_size=0.2
    )
    traindf.to_csv(os.path.join(data_path, 'train.csv'), index=False)
    testdf.to_csv(os.path.join(data_path, 'test.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', default='../data')
    args = parser.parse_args()
    main(args.data_path)
