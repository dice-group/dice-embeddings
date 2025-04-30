"""Link prediction for numerical literals"""
import os
import dspy
import numpy as np
from dotenv import load_dotenv
from dicee.knowledge_graph import KG

# # () some random examples
# contexts= ["John has_age 43 Mary \n has_parent John \n Amanda has_spouse John",
#            "John has_age 43 Mary \n has_parent John \n Amanda has_spouse John \n Amanda has_age 38",
#            "John has_age 43 Mary \n has_parent John \n Amanda has_spouse John"]
# subjects = ["Mary","Mary","Amanda"]
# predicates = ["has_age","has_age","has_age"]
# y_true = [12, 12, 38]

# {'ICR': 0.6666666666666666, 'IW': 30.666666666666668, 'MSE': 206.0, 'Winkler': 70.66666666666667}
# {'ICR': 0.6666666666666666, 'IW': 22.333333333333332, 'MSE': 79.0, 'Winkler': 62.333333333333336}

load_dotenv()
api_key = os.environ.get("TENTRIS_TOKEN")

def evaluate_interval_predictions(y_true, y_pred, y_min, y_max, alpha=0.1):
    """
    Measures Interval Coverage Rate (ICR), Interval Width (IW), Mean Squared Error (MSE) and Winkler Score (WS).
    Args:
        y_true: ground truth values (shape: [N])
        y_pred: most confident predictions (point predictions) (shape: [N])
        y_min: lower bounds of prediction interval (shape: [N])
        y_max: upper bounds of prediction interval (shape: [N])
        alpha: confidence level
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_min = np.array(y_min)
    y_max = np.array(y_max)

    # 1. Interval Coverage Rate (ICR)
    within_interval = (y_true >= y_min) & (y_true <= y_max)
    icr = np.mean(within_interval)

    # 2. Interval Width (IW)
    iw = np.mean(y_max - y_min)

    # 3. Mean Squared Error (MSE) of point prediction
    mse = np.mean((y_pred - y_true) ** 2)

    # 4. Winkler Score
    winkler_scores = np.where(
        within_interval,
        y_max - y_min,
        np.where(
            y_true < y_min,
            (y_max - y_min) + (2 / alpha) * (y_min - y_true),
            (y_max - y_min) + (2 / alpha) * (y_true - y_max)
        )
    )
    winkler = np.mean(winkler_scores)

    return {
        "ICR": icr,
        "IW": iw,
        "MSE": mse,
        "Winkler": winkler
    }

class AVP(dspy.Signature):
    context = dspy.InputField(desc="Triples in the knowledge graph used as indirect context to perform reasoning and predict the literal value given subject and predicate.")
    subject: str = dspy.InputField(desc="The subject in the semantic triple")
    predicate: str = dspy.InputField(desc="The predicate in the semantic triple")
    min_value: float = dspy.OutputField(desc="The minimum numerical literal value of the semantic triple to predict")
    literal: float = dspy.OutputField(desc="The most likely numerical literal value of the semantic triple to predict")
    max_value: float = dspy.OutputField(desc="The maximum numerical literal value of the semantic triple to predict")

class LLP:
    def __init__(self, path, base_url, api_key, temperature, seed, llm_model, use_val: bool = False):
        self.kg_path = path
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.llm_model = llm_model
        self.train_triples = []
        self.test_triples = []
        with open(self.kg_path + "/numerical_literals_train.txt", "r") as f:
            for line in f:
                triple = line.strip().split("\t")
                self.train_triples.append((triple[0], triple[1], triple[2]))

        with open(self.kg_path + "/numerical_literals_test.txt", "r") as f1:
            for line in f1:
                triple = line.strip().split("\t")
                self.test_triples.append((triple[0], triple[1], triple[2]))

        self.test_hr = [(s, p) for s, p, _ in self.test_triples]
        self.triples = self.train_triples + self.test_triples
        self.entity_relation_to_literals = dict()
        for s, p, o in self.triples:
            self.entity_relation_to_literals.setdefault((s, p), []).append(o)

    def get_context(self, h, r, triples_with_h_threshold = 100,  triples_with_r_threshold = 10):
        """Select triples that goes to the context argument."""
        ctx = ""
        triples_with_r = []
        triples_with_h = []
        for triple in self.train_triples:
            if triple[0] == h:
                triples_with_h.append(triple[0] + "\t" + triple[1] + "\t" + triple[2] + "\n")
            if triple[1] == r:
                triples_with_r.append(triple[0] + "\t" + triple[1] + "\t" + triple[2] + "\n")

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
        count = 0
        for h, r in self.test_hr:
            try:
                float(self.entity_relation_to_literals[(h, r)][0])
            except ValueError:
                continue # we will not consider dates in this evaluation
            if (h, r) in seen_set:
                continue
            else:
                seen_set.add((h, r))
            results = cot_model(context=self.get_context(h, r), subject=h, predicate=r)
            y_pred.append(results.literal)
            y_min.append(results.min_value)
            y_max.append(results.max_value)
            if len(self.entity_relation_to_literals[(h, r)]) > 1:
                y_true.append(float(self.entity_relation_to_literals[(h, r)][0]))
            else:
                literals = [float(value) for value in self.entity_relation_to_literals[(h, r)]]
                y_true.append(sum(literals) / len(literals))
            count += 1
            print(count)

        return evaluate_interval_predictions(y_true, y_pred, y_min, y_max)


if __name__ == "__main__":

    api_k = os.environ.get("TENTRIS_TOKEN")
    model = LLP("../../KGs/LitWD1K", "https://api.tentris.ai/v1", api_k, 0.1, 42, "gpt-3.5-turbo")
    eval_results = model.evaluate()

    print(eval_results)
