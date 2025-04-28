"""Link prediction for numerical literals"""
import os
import dspy
import numpy as np
from dotenv import load_dotenv
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
    context = dspy.InputField(desc="Triples in the knowledge graph used as context to perform reasoning and predict the literal value given subject and predicate.")
    subject: str = dspy.InputField(desc="The subject in the semantic triple")
    predicate: str = dspy.InputField(desc="The predicate in the semantic triple")
    min_value: float = dspy.OutputField(desc="The minimum numerical literal value of the semantic triple to predict")
    literal: float = dspy.OutputField(desc="The most likely numerical literal value of the semantic triple to predict")
    max_value: float = dspy.OutputField(desc="The maximum numerical literal value of the semantic triple to predict")


lm = dspy.LM(model=f"openai/tentris", api_key=api_key, api_base="http://harebell.cs.upb.de:8501/v1",
             seed=42, temperature=1.0,
             cache=True, cache_in_memory=True)
dspy.configure(lm=lm)

model = dspy.ChainOfThought(AVP)

# some random examples
contexts= ["John has_age 43 Mary \n has_parent John \n Amanda has_spouse John",
           "John has_age 43 Mary \n has_parent John \n Amanda has_spouse John \n Amanda has_age 38",
           "John has_age 43 Mary \n has_parent John \n Amanda has_spouse John"]
subjects = ["Mary","Mary","Amanda"]
predicates = ["has_age","has_age","has_age"]
y_true = [12, 12, 38]

y_pred = []
y_min = []
y_max = []


for i in range(3):
    results= model(context=contexts[i], subject=subjects[i], predicate=predicates[i])
    y_pred.append(results.literal)
    y_min.append(results.min_value)
    y_max.append(results.max_value)
    print(results.min_value)
    print(results.literal)
    print(results.max_value)
    print("---------------")

print(evaluate_interval_predictions(y_true, y_pred, y_min, y_max))
