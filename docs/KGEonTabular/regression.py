import sys
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import numpy as np
import argparse

random_state = 1
num_splits = 10


# USE https://pycaret.gitbook.io/docs/
def perform_kfold_regression(X, y):
    print(f'Tabular data:\tX:{X.shape} | y:{y.shape}')
    mean_squared_errors = []
    for train_index, test_index in KFold(n_splits=num_splits, shuffle=True, random_state=random_state).split(X):
        clf = LinearRegression().fit(X[train_index], y[train_index])
        err = clf.predict(X[test_index]) - y[test_index]
        mean_squared_errors.append(np.mean(sum(err ** 2)))

    fold_errors = pd.Series(mean_squared_errors)
    print(f'Linear Regression: results of {num_splits} splits')
    print(fold_errors.describe())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_kg", type=str, default="Example/train.txt", nargs="?",
                        help="The name of a Knowledge graph in the ntriple format.")
    parser.add_argument("--path_tabular_csv", type=str, default='california.csv', nargs="?",
                        help="The name of a Knowledge graph in the ntriple format.")
    parser.add_argument("--path_entity_embeddings", type=str,
                        default="Experiments/2022-12-08 08:19:51.814609/QMult_entity_embeddings.csv")

    args = parser.parse_args()
    df_csv = pd.read_csv(args.path_tabular_csv, index_col=0)
    # (1) Standard tabular data.
    # (2) KFOLD Linear Regression on (1)
    perform_kfold_regression(X=df_csv.loc[:, df_csv.columns.drop('labels')].values, y=df_csv['labels'].values)

    # (3) LOAD generate knowledge graph ( does not contain label information
    # (3.1) Select all unique subject entities. A subject entity is an index denoting the row of the input tabular data.
    kg = pd.read_csv(args.path_kg,
                     delim_whitespace=True,
                     header=None,
                     usecols=[0, 1, 2],
                     names=['subject', 'relation', 'object'],
                     dtype=str)
    # (4) Load KGE
    kge = pd.read_csv(args.path_entity_embeddings, index_col=0)
    n, dim = kge.shape
    # (5) Construct tabular data from (4)
    unique_subject_entities = kg['subject'].unique()
    emb_dataset = []
    print(
        f'Constructing tabular data from embeddings by sequentially iterating over {len(unique_subject_entities)} entities...')
    # () Iterate over subject entities
    for ind, i in (pbar := tqdm(enumerate(unique_subject_entities), file=sys.stdout)):
        # Select all object ocurred with an input subject,
        objects = kg[kg['subject'] == i]['object'].tolist()
        emb = []
        pbar.set_description(f'Entity {ind}.')
        for ob in objects:
            try:
                emb.extend(kge.loc[ob].tolist())
            except:
                print(f'NOT FOUND{ob}')
                exit(1)
        emb_dataset.append(emb)
        # Store embeddings of all objects
    # (7) KFOLD Linear Regression on KGE only
    print('KGE')
    perform_kfold_regression(X=np.array(emb_dataset), y=df_csv['labels'].values)
    # (8)
    print('TABULAR & KGE')
    perform_kfold_regression(X=np.hstack(
        (df_csv.loc[:, df_csv.columns.drop('labels')].values, # Tabular Repr.
         np.array(emb_dataset))),                             # KGE Repr
                             y=df_csv['labels'].values)
