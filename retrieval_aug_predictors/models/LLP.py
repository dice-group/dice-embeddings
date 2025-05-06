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
import pandas as pd
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
    results = dict()
    for r in y_pred.keys():
        y_true_r = np.array(y_true[r])
        y_pred_r = np.array(y_pred[r])
        y_min_r = np.array(y_min[r])
        y_max_r = np.array(y_max[r])

        # 1. Mean value
        mean = np.mean(y_true_r)

        # 2. Standard deviation
        sd = np.std(y_true_r)
        # 3. Mean Prediction value
        mean_pred = np.mean(y_pred_r)

        # 4. Interval Coverage Rate (ICR)
        within_interval = (y_true_r >= y_min_r) & (y_true_r <= y_max_r )
        icr = np.mean(within_interval)

        # 5. Interval Width (RIW)
        iw = np.mean((y_max_r  - y_min_r))


        results[r] = { "Mean": mean, "Standard Deviation": sd, "Predictions Mean":mean_pred, "Interval Width": iw, "Interval Coverage Rate": icr}

    return results

class AVP(dspy.Signature):
    context = dspy.InputField(desc="Triples in the knowledge graph used as indirect context to perform reasoning and predict the literal value given subject and predicate.")
    subject: str = dspy.InputField(desc="The subject in the semantic triple")
    predicate: str = dspy.InputField(desc="The predicate in the semantic triple")
    min_value: float = dspy.OutputField(desc="The minimum numerical literal value for the given subject and predicate. This should always be less than the max_value.")
    point_pred_value: float = dspy.OutputField(desc="The most likely numerical literal value for the given subject and predicate.")
    max_value: float = dspy.OutputField(desc="The maximum numerical literal value for the given subject and predicate. This should always be greater than the min_value.")

class LLP:
    def __init__(self, path, base_url, api_key, temperature, seed, llm_model):
        self.kg_path = path
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.llm_model = llm_model
        self.numeric_triples = [] # triples on numeric_triples.txt
        self.train_triples = [] # triples on train.txt
        with open(self.kg_path + "/numeric_literals_in_labels.txt", "r") as f:
            for line in f:
                triple = line.strip().split("\t")
                self.numeric_triples.append((triple[0], triple[1], triple[2]))

        with open(self.kg_path + "/train_in_labels.txt", "r") as f2:
            for line in f2:
                triple = line.strip().split("\t")
                self.train_triples.append((triple[0], triple[1], triple[2]))

        self.pred_sample = ["population",
                       "age_of_majority_years_old",
                       "total_fertility_rate",
                       "inflation_rate",
                       "human_development_index",
                       "life_expectancy_second",
                       "real_gdp_growth_rate",
                       "mass_kilogram",
                       "height_metre",
                       "coordinate_location_longitude"]
        # self.pred_sample = ["population","human_development_index"]

        self.eval_sample = [(s, p) for s, p, _ in self.numeric_triples if p in self.pred_sample]
        self.entity_relation_to_literals = dict()
        for s, p, o in self.numeric_triples:
            self.entity_relation_to_literals.setdefault((s, p), []).append(o)

    def get_context(self, h, r, triples_with_h_threshold = 100,  triples_with_r_threshold = 10):
        """Select triples that goes to the context argument."""
        ctx = ""
        triples_with_r = []
        triples_with_h = []
        # () store triples for h and r (ofc excluding the triple containing hr) -- this is a custom strategy
        for triple in self.numeric_triples:
            if triple[0] == h and triple[1] != r:
                triples_with_h.append(triple[0] + "\t" + triple[1] + "\t" + triple[2] + "\n")
            if triple[1] == r and triple[0] != h:
                triples_with_r.append(triple[0] + "\t" + triple[1] + "\t" + triple[2] + "\n")

        triples_with_h_from_train_set = [triple[0] + "\t" + triple[1] + "\t" + triple[2] + "\n" for triple in self.train_triples if triple[0] == h]
        triples_with_h.extend(triples_with_h_from_train_set)

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

    def evaluate(self, save = True):
        lm = dspy.LM(model=f"openai/{self.llm_model}", api_key=self.api_key, api_base=self.base_url,
                     temperature=self.temperature, seed=self.seed, cache=True, cache_in_memory=True)
        dspy.configure(lm=lm)

        cot_model = dspy.ChainOfThought(AVP)

        y_pred = dict()
        y_min = dict()
        y_max = dict()
        y_true = dict()
        heads = dict()

        seen_set = set()
        for h, r in tqdm(self.eval_sample, desc="Generating predictions and evaluating"):
            try:
                float(self.entity_relation_to_literals[(h, r)][0])
            except ValueError:
                continue # dates not considered although there is none in the selected eval_sample.
            if (h, r) in seen_set:
                continue # if already seen (for triples that share multiple hr pair) skip. Although there is none in the selected eval_sample.
            else:
                seen_set.add((h, r))

            if r not in y_pred:
                y_pred[r] = []
                y_min[r] = []
                y_max[r] = []
                y_true[r] = []
                heads[r] = []

            heads[r].append(h)
            # () generate the prediction for the given hr pair
            results = cot_model(context=self.get_context(h, r), subject=h, predicate=r)
            y_pred[r].append(results.point_pred_value)
            y_min[r].append(results.min_value)
            y_max[r].append(results.max_value)

            # () set y_true for this pair. Mean value if more than one literal is found per hr pair.
            if len(self.entity_relation_to_literals[(h, r)]) == 1:
                y_true[r].append(float(self.entity_relation_to_literals[(h, r)][0]))
            elif len(self.entity_relation_to_literals[(h, r)]) > 1:
                literals = [float(value) for value in self.entity_relation_to_literals[(h, r)]]
                y_true[r].append(sum(literals) / len(literals))

        e_r = evaluate_interval_predictions(y_true, y_pred, y_min, y_max)

        if save:
            data_for_csv = []
            for r in y_pred:
                data_for_csv.append([r, "y_pred"] + y_pred[r])
                data_for_csv.append([r, "y_true"] + y_true[r])
                data_for_csv.append([r, "y_min"] + y_min[r])
                data_for_csv.append([r, "y_max"] + y_max[r])
                data_for_csv.append([r, "heads"] + heads[r])
            rng = max([len(y_pred[r]) for r in self.pred_sample])
            df = pd.DataFrame(data_for_csv, columns=["Predicate", "Set"] + ["Value" + str(i) for i in range(rng)])
            df.to_csv("llp_results.csv", index=False)

        return e_r


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

    [print(_) for _ in eval_results.items()]
