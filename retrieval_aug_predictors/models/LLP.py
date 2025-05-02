"""Link prediction for numerical literals

python -m LLP.py --kg_path path/to/KGs/LitWD1K --base_url http://harebell.cs.upb.de:8501/v1 --temperature 0.1 --seed 42 --llm_model tentris

Prerequisites: LitWD1K should contain the following files generated after some pre-processing (ask Alkid for them):
               - numerical_literals_test.txt and numerical_literals_train.txt.
               - train_in_labels.txt

"""
import argparse
import os
from random import shuffle

import dspy
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
api_key = os.environ.get("TENTRIS_TOKEN")

def evaluate_interval_predictions(y_true, y_pred, y_min, y_max, eps=1e-9):
    """
    Measures Interval Coverage Rate (ICR), Relative Interval Width (RIW), Mean Absolute Percentage Error (MAPE)
    and Normalized Root Mean Squared Error (NRMSE).
    Args:
        y_true: ground truth values (shape: [N])
        y_pred: most confident predictions (point predictions) (shape: [N])
        y_min: lower bounds of prediction interval (shape: [N])
        y_max: upper bounds of prediction interval (shape: [N])
        eps: epsilon
    """
    print(y_true)
    print(y_pred)
    print(y_min)
    print(y_max)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_min = np.array(y_min)
    y_max = np.array(y_max)

    # 1. Interval Coverage Rate (ICR)
    within_interval = (y_true >= y_min) & (y_true <= y_max)
    icr = np.mean(within_interval)

    # 2. Relative Interval Width (RIW)
    riw = np.mean((y_max - y_min) / (y_true + eps))

    # 3.
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps)))

    # 4. Mean Squared Error (MSE) of point prediction
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(y_true) - np.min(y_true) + eps)

    return {
        "ICR": icr,
        "RIW": riw,
        "MAPE": mape,
        "NRMSE": nrmse,
    }

class AVP(dspy.Signature):
    context = dspy.InputField(desc="Triples in the knowledge graph used as indirect context to perform reasoning and predict the literal value given subject and predicate.")
    subject: str = dspy.InputField(desc="The subject in the semantic triple")
    predicate: str = dspy.InputField(desc="The predicate in the semantic triple")
    min_value: float = dspy.OutputField(desc="The minimum numerical literal value of the semantic triple to predict")
    literal: float = dspy.OutputField(desc="The most likely numerical literal value of the semantic triple to predict")
    max_value: float = dspy.OutputField(desc="The maximum numerical literal value of the semantic triple to predict")

class LLP:
    def __init__(self, path, base_url, api_key, temperature, seed, llm_model):
        self.kg_path = path
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.llm_model = llm_model
        self.train_triples = [] # train triples on the numeric train set
        self.test_triples = []  # test triples on the numeric train set
        self.e_r_e_train_triples = [] # train triples on the entity-relation-entity train set
        with open(self.kg_path + "/numerical_literals_train.txt", "r") as f:
            for line in f:
                triple = line.strip().split("\t")
                self.train_triples.append((triple[0], triple[1], triple[2]))

        with open(self.kg_path + "/numerical_literals_test.txt", "r") as f1:
            for line in f1:
                triple = line.strip().split("\t")
                self.test_triples.append((triple[0], triple[1], triple[2]))

        with open(self.kg_path + "/train_in_labels.txt", "r") as f2:
            for line in f2:
                triple = line.strip().split("\t")
                self.e_r_e_train_triples.append((triple[0], triple[1], triple[2]))

        self.test_hr = [(s, p) for s, p, _ in self.test_triples] # only for numeric_literals test dataset
        self.triples = self.train_triples + self.test_triples  # only for numeric_literals datasets (test + train)
        self.entity_relation_to_literals = dict() #  only for numeric_literals dataset
        for s, p, o in self.triples:
            self.entity_relation_to_literals.setdefault((s, p), []).append(o)

    def get_context(self, h, r, triples_with_h_threshold = 100,  triples_with_r_threshold = 10):
        """Select triples that goes to the context argument."""
        ctx = ""
        triples_with_r = []
        triples_with_h = []
        # () store triples for h and r -- this is a custom strategy
        for triple in self.train_triples:
            if triple[0] == h:
                triples_with_h.append(triple[0] + "\t" + triple[1] + "\t" + triple[2] + "\n")
            if triple[1] == r:
                triples_with_r.append(triple[0] + "\t" + triple[1] + "\t" + triple[2] + "\n")

        triples_with_h_from_e_r_e_train_set = [triple[0] + "\t" + triple[1] + "\t" + triple[2] + "\n" for triple in self.e_r_e_train_triples if triple[0] == h]
        triples_with_h.extend(triples_with_h_from_e_r_e_train_set)

        # () shuffle so the literal-containing triples are mixed with the other ones and to avoid bias in re-evaluation
        shuffle(triples_with_h)
        shuffle(triples_with_r)

        # () put the triples in the context window given the thresholds
        for i in range(len(triples_with_h)):
            if i >= triples_with_h_threshold:
                break
            if triples_with_h[i] not in ctx:
                ctx += triples_with_h[i]

        for i in range(len(triples_with_r)):
            if i >= triples_with_r_threshold:
                break
            if triples_with_r[i] not in ctx:
                ctx += triples_with_r[i]

        return ctx

    def evaluate(self):
        lm = dspy.LM(model=f"openai/{self.llm_model}", api_key=self.api_key, api_base=self.base_url,
                     temperature=self.temperature, seed=self.seed, cache=True, cache_in_memory=True)
        dspy.configure(lm=lm)

        cot_model = dspy.ChainOfThought(AVP)

        y_pred = []
        y_min = []
        y_max = []
        y_true = []
        seen_set = set()
        for h, r in tqdm(self.test_hr, desc="Generating predictions and evaluating"):
            try:
                float(self.entity_relation_to_literals[(h, r)][0])
            except ValueError:
                continue # not considering dates in this evaluation, only numeric literals of type float/int
            if (h, r) in seen_set:
                continue
            else:
                seen_set.add((h, r))

            # () generate the prediction for the given hr pair
            results = cot_model(context=self.get_context(h, r), subject=h, predicate=r)
            y_pred.append(results.literal)
            y_min.append(results.min_value)
            y_max.append(results.max_value)

            # () set y_true for this pair. Mean value if more than one literal is found per hr pair.
            if len(self.entity_relation_to_literals[(h, r)]) == 1:
                y_true.append(float(self.entity_relation_to_literals[(h, r)][0]))
            elif len(self.entity_relation_to_literals[(h, r)]) > 1:
                literals = [float(value) for value in self.entity_relation_to_literals[(h, r)]]
                y_true.append(sum(literals) / len(literals))

        return evaluate_interval_predictions(y_true, y_pred, y_min, y_max)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg_path", type=str, default="../../KGs/LitWD1K")
    parser.add_argument("--base_url", type=str, default="http://harebell.cs.upb.de:8501/v1")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm_model", type=str, default="tentris")
    args = parser.parse_args()
    if args.api_key is None:
        args.api_key = os.environ.get("TENTRIS_TOKEN")

    model = LLP(args.kg_path, args.base_url, args.api_key, args.temperature, args.seed, args.llm_model)

    eval_results = model.evaluate()

    print(eval_results)
