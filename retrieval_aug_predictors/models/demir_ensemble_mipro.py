"""

python -m retrieval_aug_predictors.models.demir_ensemble_mipro --dataset_dir KGs/Countries-S1 --out "countries_s1_results.json" && cat countries_s1_results.json
{
    "H@1": 0.9583333333333334,
    "H@3": 1.0,
    "H@10": 1.0,
    "MRR": 0.9791666666666666
}

python -m retrieval_aug_predictors.models.demir_ensemble_mipro --dataset_dir KGs/Countries-S2 --out "countries_s2_results.json" && cat countries_s2_results.json
{
    "H@1": 0.6666666666666666,
    "H@3": 1.0,
    "H@10": 1.0,
    "MRR": 0.8333333333333334
}
python -m retrieval_aug_predictors.models.demir_ensemble_mipro --dataset_dir KGs/Countries-S3 --out "countries_s3_results.json" && cat countries_s3_results.json
{
    "H@1": 0.6666666666666666,
    "H@3": 1.0,
    "H@10": 1.0,
    "MRR": 0.8333333333333334
}
"""
import dspy
import torch
import json
import random
from tqdm import tqdm
from typing import List, Tuple, Dict, Set
from dspy.teleprompt import MIPROv2
from retrieval_aug_predictors.models import KG, AbstractBaseLinkPredictorClass
from retrieval_aug_predictors.arguments import parser
from retrieval_aug_predictors.utils import sanity_checking
from dicee.evaluator import evaluate_lp_k_vs_all
from dotenv import load_dotenv
import os
from dspy.evaluate import Evaluate
import math
import pandas as pd
from typing import List,Iterable
import dspy
import pydantic

pd.set_option('display.max_columns', None)

# --- Constants ---
load_dotenv()
pd.set_option('display.max_columns', None)

SAVE_DIR_BASE = "DemirEnsembleMIPRO_Optimized"
TEST_SIZE = 0.2

# --- Pydantic Model for Structured Output ---
class PredictionScore(pydantic.BaseModel):
    entity: str = pydantic.Field(..., description="The predicted entity.")
    score: pydantic.confloat(ge=0.0, le=1.0) = pydantic.Field(...,
                                                              description="Confidence score between 0.0 and 1.0.")
class MultiLabelLinkPrediction(dspy.Signature):
    """Predicts multiple tail entities for a given subject and predicate, along with confidence scores."""
    subject: str = dspy.InputField(desc="The subject entity.")
    predicate: str = dspy.InputField(desc="The relationship type.")
    entities: List[str] = dspy.InputField(desc="List of all possible entities in the knowledge graph.")
    target: List[str] = dspy.OutputField(desc="List of suitable entities (str)")

class Scorer(dspy.Signature):
    """Predicts multiple tail entities for a given subject and predicate, along with confidence scores."""
    #reasoning: str = dspy.InputField(desc="Explanation behind prediction.")
    predicted_entities: List[str] = dspy.InputField(desc="A list of predicted entities")
    target: List[float] = dspy.OutputField(desc="List of scores for each entity")


class MultiLabelLinkPredictor(dspy.Module):
    """DSPy Module wrapping a ChainOfThought program for link prediction."""
    def __init__(self, entities: List[str]):
        super().__init__()
        self.entities = sorted(list(set(entities)))
        self.finder = dspy.Predict(MultiLabelLinkPrediction)
        self.scorer = dspy.Predict(Scorer)

    def forward(self, subject: str, predicate: str) -> dspy.Prediction:
        retrieved_entities = self.finder(subject=subject, predicate=predicate, entities=self.entities)
        scores=self.scorer(#reasoning=retrieved_entities.reasoning,
                           predicted_entities=retrieved_entities.target)

        final=dspy.Prediction(#reasoning_entity=retrieved_entities.reasoning,
                              #reasoning_scores=scores.reasoning,
                        target=[ (retrieved_entities.target[idx],score) for idx,score in enumerate(scores.target)])
        return final
# --- Data Preparation ---
def generate_train_test_split(examples: List[dspy.Example], test_size: float, seed: int) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """Splits dspy.Examples into training and testing sets."""
    shuffled_examples = examples.copy()
    random.Random(seed).shuffle(shuffled_examples) # Use seeded random for reproducibility
    split_idx = int(len(shuffled_examples) * (1 - test_size))
    train_examples = shuffled_examples[:split_idx]
    test_examples = shuffled_examples[split_idx:]
    return train_examples, test_examples

def generate_examples_for_mipro(data: Dict[Tuple[str, str], List[str]], all_entities: Set[str], seed: int) -> List[dspy.Example]:
    """
    Generates dspy.Example objects for MIPRO optimization.
    Includes true entities (score 1.0) and subsampled false entities (score 0.0).
    """
    examples = []
    for (h, r), true_entity_list in tqdm(data.items(), desc="Generating Examples for MIPRO"):
        true_entities_set = set(true_entity_list)
        y_true = list(sorted([(entity, 1.0) for entity in true_entities_set]))
        examples.append(dspy.Example(subject=h, predicate=r, target=y_true).with_inputs("subject", "predicate"))

        continue

        if not false_entities_set or subsample_size == 0:
             # If no false entities or cannot sample, only use true ones
             examples.append(dspy.Example(subject=h, predicate=r, target=y_true).with_inputs("subject", "predicate"))
        else:
            # Create multiple examples by subsampling different false entities
            # (Original code created multiple examples per (h,r) pair)
            # Let's create one balanced example per (h,r) for simplicity in MIPRO training
            y_false_sample = [(e, 0.0) for e in local_random.sample(list(false_entities_set), subsample_size)]
            target_combined = list(sorted(y_true + y_false_sample)) # Combine and sort
            examples.append(dspy.Example(subject=h, predicate=r, target=target_combined).with_inputs("subject", "predicate"))
    return examples

def quality_score_closeness(y: List[Tuple[str, float]],
                            prediction: List[Tuple[str, float]],
                            penalty_factor: float = 1.0) -> float:
    """
    Calculates a quality score based on how closely the predicted scores match
    the ground truth scores for entities specified in 'y'.

    The score reflects the average "closeness" for items in 'y'. Closeness
    is measured using an exponential decay function based on the absolute
    difference between the ground truth score and the predicted score.

    Args:
        y: Ground truth list of tuples (entity_name, ground_truth_score).
           Represents the known true entities against which to evaluate.
        prediction: Predicted list of tuples (entity_name, predicted_score).
        penalty_factor: Controls how quickly the closeness score decays as the
                        difference between ground truth and predicted score increases.
                        Higher values penalize differences more severely. Defaults to 2.0.

    Returns:
        A quality score between 0.0 and 1.0.
        - 1.0 indicates a perfect score match for all entities in 'y' found in 'prediction'.
        - Scores decrease as the difference between y_score and prediction_score increases.
        - Entities in 'y' but missing from 'prediction' contribute 0.0 to the average closeness.
        - Returns 0.0 if 'y' is empty, as no evaluation is possible.
    """
    if not y:
        # Cannot evaluate quality if there are no ground truth items
        return 0.0

    # Create a dictionary for efficient lookup of prediction scores
    prediction_map = {entity: score for entity, score in prediction}

    total_closeness = 0.0
    for entity, y_score in y:
        if entity in prediction_map:
            prediction_score = prediction_map[entity]
            # Calculate the absolute difference in scores
            difference = abs(y_score - prediction_score)
            # Calculate closeness using exponential decay
            # closeness = 1 means scores match, decreases towards 0 as difference grows
            closeness = math.exp(-penalty_factor * difference)
            total_closeness += closeness
        else:
            # The entity from y was not found in the prediction. Assign minimum closeness (0.0).
            total_closeness += 0.0
    # Calculate the average closeness across all evaluated items from y
    average_closeness = total_closeness / len(y)
    return average_closeness

def dspy_quality_score_closeness(y: dspy.Example, yhat: dspy.Prediction,trace=None)-> float:
    return quality_score_closeness(y.target, yhat.target)

class DemirEnsembleMPRO(AbstractBaseLinkPredictorClass):
    """
    Ensemble predictor using MIPROv2 optimized base prompts.
    Combines predictions from models trained/optimized with different settings (e.g., temperature).
    """

    def __init__(self, knowledge_graph: KG, base_url: str, api_key: str, llm_model: str,
                 temperature: float, seed: int, use_val: bool = True,
                 ensemble_temperatures=None,
                 save_dir: str = SAVE_DIR_BASE):
        super().__init__(knowledge_graph, name="DemirEnsembleMIPRO")

        # Configuration
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.seed = seed
        self.use_val = use_val
        self.save_dir = save_dir
        if ensemble_temperatures is None:
            self.ensemble_temperatures = [i * 0.1 for i in range(1)]
        else:
            self.ensemble_temperatures = ensemble_temperatures

        self.mipro_optimizer_temperature = temperature
        # Seed random for reproducibility
        random.seed(self.seed)
        # Data Preparation
        self._prepare_data()
        # Create and Optimize/Load Predictors
        os.makedirs(self.save_dir, exist_ok=True)
        self.predictors: List[MultiLabelLinkPredictor] = self._create_and_optimize_predictors()

        print(f"\n--- {self.__class__.__name__} initialized with {len(self.predictors)} predictors ---")
    def _prepare_data(self):
        """Loads, processes, and prepares training/validation data."""
        print("Preparing data...")
        train_set = [(self.idx_to_entity[h], self.idx_to_relation[r], self.idx_to_entity[t])
                     for h, r, t in self.kg.train_set.tolist()]
        val_set = [(self.idx_to_entity[h], self.idx_to_relation[r], self.idx_to_entity[t])
                   for h, r, t in self.kg.valid_set.tolist()]

        self.triples = train_set + (val_set if self.use_val else [])

        # Group triples by (subject, predicate)
        self.entity_relation_to_entities: Dict[Tuple[str, str], List[str]] = {}
        self.all_entities: Set[str] = set()
        for s, p, o in self.triples:
            self.all_entities.add(s)
            self.all_entities.add(o)
            self.entity_relation_to_entities.setdefault((s, p), []).append(o)

        print(f"Prepared data: {len(self.triples)} triples, {len(self.all_entities)} unique entities.")

    def _create_and_optimize_predictors(self) -> List[MultiLabelLinkPredictor]:
        """
        Creates predictor instances for each temperature, optimizes them using MIPRO,
        and saves/loads the optimized state.
        """
        predictors = []
        for idx, temp in enumerate(self.ensemble_temperatures):
            print(f"\n--- Optimizing/Loading Predictor for Temperature: {temp:.1f} ---")
            save_filename = os.path.join(self.save_dir, f"predictor_temp_{temp:.1f}.json")
            results_filename = os.path.join(self.save_dir, f"eval_results_temp_{temp:.1f}.csv")
            # Initialize the base predictor for this temperature
            base_predictor = MultiLabelLinkPredictor(entities=list(self.all_entities))

            if os.path.exists(save_filename):
                print(f"Loading optimized predictor from {save_filename}...")
                # Need to configure LM *before* loading if LM state isn't saved
                lm = dspy.LM(model=f"openai/{self.llm_model}", api_key=self.api_key,
                             api_base=self.base_url, temperature=temp, # Use the specific temp
                             cache=True, cache_in_memory=True) # Seed might not be needed for loading
                dspy.configure(lm=lm)
                # Load requires the class, not an instance
                # optimized_predictor = MultiLabelLinkPredictor.load(save_filename)
                # Load modifies the instance in place
                base_predictor.load(save_filename)
                optimized_predictor = base_predictor
                print("Loaded successfully.")
            else:
                print("Optimizing predictor using MIPROv2...")
                print("Generating examples for MIPRO...")
                all_examples = generate_examples_for_mipro(data=self.entity_relation_to_entities,
                                                           all_entities=self.all_entities, seed=idx)
                train_examples, test_examples = generate_train_test_split(all_examples, test_size=TEST_SIZE,
                                                                               seed=idx)
                print(f"Generated {len(all_examples)} examples ({len(train_examples)} train, {len(test_examples)} test).")

                optimized_predictor = self._compile_predictor_for_temperature(
                    predictor_to_optimize=base_predictor,
                    train_examples=train_examples,
                    test_examples=test_examples,
                    temperature=temp, save_filename=save_filename)
                # --- Optional: Evaluate after optimization ---
                # print(f"Evaluating optimized predictor for temp {temp:.1f}...")
                # evaluator = Evaluate(devset=self.test_examples[:50], # Limit evaluation size
                #                      metric=dspy_binary_cross_entropy,
                #                      num_threads=1, # Use more threads if possible
                #                      display_progress=True,
                #                      return_outputs=True)
                # score, results_data = evaluator(program=optimized_predictor)
                # print(f"Post-Optimization BCE Loss (Avg): {score:.4f}")
                #
                # # Save detailed results
                # data_to_save = []
                # for example, prediction, loss in results_data:
                #     row = {
                #         "subject": example.subject,
                #         "predicate": example.predicate,
                #         "target": example.target, # Ground truth
                #         "prediction": prediction.target, # Model prediction (tuples)
                #         "reasoning": prediction.get('rationale', prediction.get('reasoning', '')), # Get reasoning if available
                #         "bce_loss": loss
                #     }
                #     data_to_save.append(row)
                # df = pd.DataFrame(data_to_save)
                # df.to_csv(results_filename, index=False)
                # print(f"Detailed evaluation results saved to {results_filename}")
                # --- End Optional Evaluation ---
            predictors.append(optimized_predictor)
            dspy.configure(lm=None) # Reset global LM config after use

        return predictors
    def _compile_predictor_for_temperature(self, predictor_to_optimize: MultiLabelLinkPredictor,
                                           train_examples:List[dspy.Example],test_examples,temperature: float,save_filename:str) -> MultiLabelLinkPredictor:
        """Configures LM and runs MIPROv2 compilation."""
        assert isinstance(predictor_to_optimize, MultiLabelLinkPredictor)
        assert isinstance(train_examples, list) and isinstance(train_examples[0],dspy.Example)
        assert isinstance(test_examples, list) and isinstance(test_examples[0],dspy.Example)
        # Configure DSPy LM specifically for this optimization run
        lm = dspy.LM(model=f"openai/{self.llm_model}", api_key=self.api_key,
                     api_base=self.base_url, seed=self.seed, temperature=temperature,
                     cache=True, cache_in_memory=True)
        dspy.configure(lm=lm)
        #print(x)
        #tp = dspy.MIPROv2(metric=dspy.SemanticF1(), auto="medium", num_threads=24)
        #optimized_rag = tp.compile(RAG(), trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2)

        #evaluate = Evaluate(devset=self.test_examples,metric=dspy_quality_score_closeness,
        #                    num_threads=6, display_table=False, display_progress=True, provide_traceback=True)
        #print(evaluate(predictor_to_optimize))

        #for data_example in self.train_examples:
        #    yhat = predictor_to_optimize(**data_example.inputs())
        #    print(dspy_quality_score_closeness(y=data_example.labels(),yhat=yhat))
        #    break
        # Generate examples needed for optimization outside the loop
        tp = dspy.MIPROv2(metric=dspy_quality_score_closeness, auto="light")
        optimized_predictor = tp.compile(predictor_to_optimize.deepcopy(), trainset=train_examples[:],
                                         valset=test_examples[:],
                                         requires_permission_to_run=False)
        optimized_predictor.finder.lm=lm
        optimized_predictor.scorer.lm=lm
        optimized_predictor.save(save_filename)
        """
        print(f"Starting MIPROv2 compilation for temperature {temperature:.1f}...")
        optimized_predictor = mipro_optimizer.compile(
            student=predictor_to_optimize.deepcopy(), # Optimize a copy
            trainset=self.train_examples,
            valset=self.test_examples, # Use test set as validation for MIPRO
            num_trials=MIPRO_NUM_TRIALS,
            max_bootstrapped_demos=2, # Limit number of demos MIPRO adds
            max_labeled_demos=2, # Limit number of demos MIPRO adds
            seed=self.seed,
            requires_permission_to_run=False # Auto-run if needed
        )
        print("MIPROv2 compilation finished.")
        """
        return optimized_predictor
    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Generate predictions for (head, relation) pairs against all entities.
        Averages scores from all predictors in the ensemble.
        """
        batch_predictions = []
        num_entities = len(self.idx_to_entity)

        # Use tqdm for progress visualization
        for hr in tqdm(x.tolist(), desc="Predicting Batches (K vs All)"):
            idx_h, idx_r = hr
            h = self.idx_to_entity.get(idx_h, None)
            r = self.idx_to_relation.get(idx_r, None)
            if h is None or r is None:
                 print(f"Warning: Unknown index in query: h={idx_h}, r={idx_r}. Skipping.")
                 batch_predictions.append([0.0] * num_entities) # Predict all zeros
                 continue

            # Use a dictionary to accumulate scores by entity name for easier handling
            accumulated_scores: Dict[str, float] = {}
            num_predictors_used = 0

            # Get predictions from each predictor in the ensemble
            for i, predictor in enumerate(self.predictors):
                try:
                    # Ensure correct LM is configured if predictor doesn't hold it internally
                    # This might be needed if .save()/.load() doesn't handle LM state well.
                    # Re-configure based on predictor's intended temperature (if stored) or loop temp?
                    # Assuming predictor holds its state or configuration is handled globally/contextually.
                    # lm = dspy.LM(...) # Potentially reconfigure LM here if needed per predictor
                    # dspy.configure(lm=lm)

                    prediction = predictor.forward(subject=h, predicate=r)
                    num_predictors_used += 1

                    # prediction.target should be List[Tuple[str, float]] thanks to predictor.forward()
                    for entity, score in prediction.target:
                        accumulated_scores[entity] = accumulated_scores.get(entity, 0.0) + score

                except Exception as e:
                    print(f"Error during prediction with predictor {i} for ({h}, {r}): {e}")
                    # Optionally skip this predictor's contribution or handle error

            # --- Convert accumulated scores to the final output format ---
            final_scores = [0.0] * num_entities
            if num_predictors_used > 0:
                for entity_name, total_score in accumulated_scores.items():
                    if entity_name in self.entity_to_idx:
                        idx_entity = self.entity_to_idx[entity_name]
                        final_scores[idx_entity] = total_score / num_predictors_used # Average score
                    # else:
                    #    print(f"Warning: Predicted entity '{entity_name}' not in entity index map.")

            batch_predictions.append(final_scores)

        return torch.FloatTensor(batch_predictions)
    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        # This method is required by the abstract class but not the focus here.
        # If needed, adapt it similar to forward_k_vs_all but for specific triples.
        raise NotImplementedError("forward_triples needs implementation if used.")


# --- Main Execution Block ---
if __name__ == "__main__":
    args = parser.parse_args()

    # Important: add_reciprocal=False in KvsAll implies that inverse relation has been introduced.
    # Therefore, The link prediction results are based on the missing tail rankings only!
    print(f"Loading KG from: {args.dataset_dir}")
    kg = KG(dataset_dir=args.dataset_dir, separator="\s+", eval_model=args.eval_model, add_reciprocal=False)
    sanity_checking(args, kg) # Assuming this function exists and checks args
    print("Initializing DemirEnsemble with MIPROv2...")
    model = DemirEnsembleMPRO(knowledge_graph=kg, base_url=args.base_url, api_key=args.api_key,
                              llm_model=args.llm_model_name, temperature=args.temperature,
                              seed=args.seed, use_val=True)
    print("Starting evaluation...")
    # Limit evaluation size if needed (e.g., during testing)
    eval_triples = kg.test_set[:args.eval_size] if args.eval_size > 0 else kg.test_set
    print(f"Evaluating on {len(eval_triples)} test triples...")
    results: dict = evaluate_lp_k_vs_all(model=model, triple_idx=eval_triples,
                                         er_vocab=kg.er_vocab,
                                         info='Eval KvsAll (DemirEnsembleMIPRO) Starts')
    print(results)
    if args.out and results:
        print(f"\nEvaluation Results:\n{json.dumps(results, indent=4)}")
        # Writing the dictionary to a JSON file
        with open(args.out, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        print(f"Results have been saved to {args.out}")
    elif results:
        print(f"\nEvaluation Results:\n{json.dumps(results, indent=4)}")
    else:
        print("Evaluation did not produce results.")