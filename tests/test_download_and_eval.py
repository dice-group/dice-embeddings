from dicee import KGE
import pandas as pd
from dicee.static_funcs import get_er_vocab
from dicee.eval_static_funcs import evaluate_link_prediction_performance_with_reciprocals

import pytest


class Download_Eval:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_sample(self):
        # (1) Download a pre-trained model and store it a newly created directory (KINSHIP-Keci-dim128-epoch256-KvsAll)
        model = KGE(url="https://files.dice-research.org/projects/DiceEmbeddings/KINSHIP-Keci-dim128-epoch256-KvsAll")
        # (2) Make a prediction
        print(model.predict(h="person49", r="term12", t="person39", logits=False))

        # Load the train, validation, test datasets
        train_triples = pd.read_csv("KGs/KINSHIP/train.txt",
                                    sep="\s+",
                                    header=None, usecols=[0, 1, 2],
                                    names=['subject', 'relation', 'object'],
                                    dtype=str).values.tolist()
        valid_triples = pd.read_csv("KGs/KINSHIP/valid.txt",
                                    sep="\s+",
                                    header=None, usecols=[0, 1, 2],
                                    names=['subject', 'relation', 'object'],
                                    dtype=str).values.tolist()
        test_triples = pd.read_csv("KGs/KINSHIP/test.txt",
                                   sep="\s+",
                                   header=None, usecols=[0, 1, 2],
                                   names=['subject', 'relation', 'object'],
                                   dtype=str).values.tolist()
        # Compute the mapping from each unique entity and relation pair to all entities, i.e.,
        # e.g. V_{e_i,r_j} = {x | x \in Entities s.t. e_i, r_j, x) \in Train \cup Val \cup Test}
        # This mapping is used to compute the filtered MRR and Hit@n
        er_vocab = get_er_vocab(train_triples + valid_triples + test_triples)

        result = model.get_eval_report()

        evaluate_link_prediction_performance_with_reciprocals(model, triples=train_triples, er_vocab=er_vocab)
        evaluate_link_prediction_performance_with_reciprocals(model, triples=valid_triples, er_vocab=er_vocab)
        evaluate_link_prediction_performance_with_reciprocals(model, triples=test_triples, er_vocab=er_vocab)

