"""
python -m retrieval_aug_predictors.models.RALP_mipro --dataset_dir KGs/Countries-S1 --out "countries_s1_results.json" && cat countries_s1_results.json

"""
import dspy
import torch
import json
from tqdm import tqdm
from typing import List, Tuple, Dict, Set, Any, Protocol
from retrieval_aug_predictors.models import KG, AbstractBaseLinkPredictorClass
from retrieval_aug_predictors.arguments import parser
from retrieval_aug_predictors.utils import sanity_checking
from dicee.evaluator import evaluate_lp_k_vs_all
from dotenv import load_dotenv
import os
from dspy.evaluate import Evaluate
import math
import pandas as pd
import random
from typing import TypeAlias, Union, Literal, Optional
# --- Constants ---
load_dotenv()
pd.set_option('display.max_columns', None)

SAVE_DIR_BASE = "RALP_MPRO_Optimized"
TEST_SIZE = 0.1
# --- Type Definitions ---
# Represents the graph: {(subject, predicate): {object1, object2}}
# Using str for entities/predicates as in the input code
GraphType: TypeAlias = Dict[Tuple[str, str], Set[str]]
# Represents a single triple
TripleType: TypeAlias = Tuple[str, str, str]
# Define type for the new return structure
ResultsByHopType: TypeAlias = Dict[int, Union[Set[TripleType], Set[str]]]

# --- Data Preparation ---
def generate_train_test_split(examples: List[dspy.Example], test_size: float, seed: int) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """Splits dspy.Examples into training and testing sets."""
    shuffled_examples = examples.copy()
    random.Random(seed).shuffle(shuffled_examples) # Use seeded random for reproducibility
    split_idx = int(len(shuffled_examples) * (1 - test_size))
    train_examples = shuffled_examples[:split_idx]
    test_examples = shuffled_examples[split_idx:]
    return train_examples, test_examples

def generate_examples_for_mipro(data: Dict[Tuple[str, str], Set[str]], all_entities: Set[str], seed: int) -> List[dspy.Example]:
    """
    Generates dspy.Example objects for MIPRO optimization.
    Includes true entities (score 1.0) and subsampled false entities (score 0.0).
    """
    examples = []
    for (h, r), true_entity_list in data.items():
        true_entities_set = set(true_entity_list)
        y_true = list(sorted([(entity, 1.0) for entity in true_entities_set]))
        examples.append(dspy.Example(subject=h, predicate=r,target=y_true).with_inputs("subject", "predicate"))

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


def cross_entropy_loss(y: List[Tuple[str, float]],
                       prediction: List[Tuple[str, float]],
                       epsilon: float = 1e-15) -> float:
    """
    Calculates the cross-entropy loss between ground truth probabilities and predicted
    probabilities for entities specified in 'y'.

    Cross-entropy loss measures how different two probability distributions are.
    Lower values indicate better alignment between the distributions.

    Args:
        y: Ground truth list of tuples (entity_name, ground_truth_probability).
           These should be probabilities in the range [0, 1] and can represent
           either binary labels (0 or 1) or probability distributions.
        prediction: Predicted list of tuples (entity_name, predicted_probability).
           These should be probabilities in the range [0, 1].
        epsilon: Small constant to avoid numerical instability with log(0).
                 Defaults to 1e-15.

    Returns:
        The cross-entropy loss value. Lower values indicate better predictions.
        - Returns 0.0 if 'y' is empty, as no evaluation is possible.
        - For missing predictions, assumes a default probability of 0.5.
    """
    if not y:
        # Cannot evaluate loss if there are no ground truth items
        return 0.0

    # Create a dictionary for efficient lookup of prediction scores
    prediction_map = {entity: score for entity, score in prediction}

    total_loss = 0.0
    for entity, y_prob in y:
        # Get prediction probability, default to 0.5 if not found
        pred_prob = prediction_map.get(entity, 0.5)

        # Clip probabilities to avoid log(0) or log(1)
        y_prob = max(min(y_prob, 1.0 - epsilon), epsilon)
        pred_prob = max(min(pred_prob, 1.0 - epsilon), epsilon)

        # Calculate cross-entropy for this entity
        # CE = -y * log(p) - (1-y) * log(1-p)
        entity_loss = -1 * (y_prob * math.log(pred_prob) + (1 - y_prob) * math.log(1 - pred_prob))
        total_loss += entity_loss

    # Calculate average loss across all entities
    average_loss = total_loss / len(y)
    return average_loss
def dspy_quality_score_closeness(y: dspy.Example, yhat: dspy.Prediction,trace=None)-> float:
    return cross_entropy_loss(y.target, yhat.target)

def traverse_beam_by_hop(
    graph: GraphType,
    start_entities: Union[Set[str], List[str], str],
    hops: int,
    beam_width: int,
    return_triples_only: bool = True
) -> ResultsByHopType:
    """
    Performs multi-hop graph traversal using beam search, returning results by hop.

    Explores the graph outwards for a specified number of hops. At each hop,
    it finds triples originating from the subjects currently in the beam.
    The objects of these triples become candidates for the next hop's beam,
    limited by 'beam_width'. Results are stored per hop.

    Args:
        graph: The graph data structure.
        start_entities: A set, list, or single string entity (node) to start the
                        traversal from.
        hops: The number of hops (steps) to explore outwards. A hop finds
              triples (S, P, O) where S is in the current beam.
        beam_width: The maximum number of entities (objects found) to
                    consider as subjects for the *next* hop.
        return_triples_only: If True (default), the dictionary values will be
                             sets of triples found *during* each hop. If False,
                             the values will be the set of entities constituting
                             the beam selected *after* each hop (i.e., the beam
                             for the *next* hop).

    Returns:
        A dictionary where keys are hop numbers (int, starting from 1) and
        values are the results for that hop.
        - If return_triples_only is True: Dict[int, Set[TripleType]]
        - If return_triples_only is False: Dict[int, Set[str]] (entities in beam for next hop)
        The dictionary will contain entries only for hops that were actually executed.
    """
    # Ensure start_entities is a set for efficient handling
    if isinstance(start_entities, str):
        current_beam: Set[str] = {start_entities}
    elif isinstance(start_entities, list):
        current_beam: Set[str] = set(start_entities)
    elif isinstance(start_entities, set):
         current_beam: Set[str] = start_entities
    else:
         # Raise an error for unsupported type
         raise TypeError("start_entities must be a string, list of strings, or set of strings")


    # Dictionary to store results per hop
    results_by_hop: ResultsByHopType = {}
    # Keep track of all triples found across hops to avoid adding duplicates
    # to the results sets if return_triples_only=True, although the beam
    # logic itself doesn't strictly need this separation anymore. Let's keep
    # it simple and store only the unique triples *found in this specific hop*.
    all_found_triples_ever: Set[TripleType] = set()


    for i in range(hops):
        hop_number = i + 1
        # If the beam is empty, we can't explore further
        if not current_beam:
            break

        # Store candidate objects for the next beam and triples found in this hop
        next_beam_candidates: Set[str] = set()
        triples_this_hop: Set[TripleType] = set()

        # Explore from all entities currently in the beam
        for subject in current_beam:
            # Find triples in the graph where 'subject' is the subject
            for (s, p), objects in graph.items():
                if s == subject:
                    for o in objects:
                        triple = (s, p, o)
                        # Add triple if it's genuinely new in this traversal
                        # This ensures triples_this_hop only contains triples first discovered in this hop.
                        if triple not in all_found_triples_ever:
                             triples_this_hop.add(triple)
                             all_found_triples_ever.add(triple) # Mark as seen globally

                        # Always consider the object as a candidate for the next beam,
                        # even if the triple itself was seen before via another path.
                        next_beam_candidates.add(o)

        # --- Beam Selection ---
        # Select entities for the *next* hop's beam
        if len(next_beam_candidates) > beam_width:
            sorted_candidates = sorted(list(next_beam_candidates), key=str)
            next_beam = set(sorted_candidates[:beam_width])
        else:
            next_beam = next_beam_candidates

        # --- Store Results for Current Hop ---
        if return_triples_only:
            # Store the unique triples discovered *during* this hop
             results_by_hop[hop_number] = triples_this_hop
        else:
            # Store the beam selected to proceed *from* this hop
            results_by_hop[hop_number] = next_beam

        # Update the beam for the next iteration
        current_beam = next_beam

    return results_by_hop

class Composer(dspy.Signature):
    """Finds multiple tail entities for a given subject and predicate"""
    subject: str = dspy.InputField(desc="An entity as a subject in a triple.")
    predicate: str = dspy.InputField(desc="A predicate in a triple.")
    context: str = dspy.InputField(desc="Background knowledge.")
    entities: List[str] = dspy.InputField(desc="A list of all possible entities in the knowledge graph.")
    target: List[str] = dspy.OutputField(desc="Candidates found in entities.")

class Scorer(dspy.Signature):
    """Predict likelihoods of multiple tail entities for a given subject and predicate."""
    predicted_entities: List[str] = dspy.InputField(desc="A list of predicted entities")
    target: List[float] = dspy.OutputField(desc="A list of scores for each entity")

class MultiLabelLinkPredictor(dspy.Module):
    def __init__(self, entities: List[str],knowledge_graph:GraphType):
        super().__init__()
        self.entities = sorted(list(set(entities)))
        self.knowledge_graph = knowledge_graph
        # Using dspy.ChainOfThought increases the context length drastically.
        self.finder = dspy.ChainOfThought(Composer)
        self.scorer = dspy.ChainOfThought(Scorer)

    def _graph_based_content_builder(self,subject:str,hops:int=5):
        assert hops>=0
        hop_to_triples = dict()
        graph_report = traverse_beam_by_hop(graph=self.knowledge_graph,
                                            start_entities=subject,
                                            hops=hops,
                                            beam_width=len(self.entities), return_triples_only=True)
        # Accumulate triples over hops: assert hop_to_triples[i].issubset(hop_to_triples[i+1])
        for k,v in graph_report.items():
            hop_to_triples[k] = v | set().union(*hop_to_triples.values())
        return hop_to_triples

    def forward(self, subject: str, predicate: str) -> dspy.Prediction:
        intermediate_predictions=dict()
        # graph_report = self._graph_based_content_builder(subject=subject)
        contextual_triples={ (s,p,o) for (s,p), os in self.knowledge_graph.items() for o in os}
        contextual_triples = [f"{s} {p} {o}" for s, p, o in sorted(contextual_triples)][:100]
        context = "\n".join(contextual_triples)
        retrieved_entities = self.finder(subject=subject,
                                         predicate=predicate,
                                         context=context,
                                         entities=self.entities)
        scores = self.scorer(predicted_entities=retrieved_entities.target)
        for idx, score in enumerate(scores.target):
            entity = retrieved_entities.target[idx]
            intermediate_predictions.setdefault(entity, []).append(score)
        predictions=dict()
        for k,v in intermediate_predictions.items():
            predictions[k]=sum(v)/len(v)
        return dspy.Prediction(target=[(k,v) for k,v in predictions.items()])



class RALP_MPRO(AbstractBaseLinkPredictorClass):
    """
    (ex DemirEnsembleMPRO)
    Ensemble predictor using MIPROv2 optimized base prompts.
    Combines predictions from models trained/optimized with different settings (e.g., temperature).
    """
    def __init__(self, knowledge_graph: KG, base_url: str, api_key: str, llm_model: str,
                 temperature: float, seed: int, use_val: bool = True, ensemble_temperatures=None,
                 save_dir: str = SAVE_DIR_BASE,auto: Optional[Literal["light", "medium", "heavy"]] = "light",num_ensemble:int=10):
        super().__init__(knowledge_graph, name="RALP_MPRO")

        # Configuration
        self.base_url = base_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.auto=auto
        self.seed = seed
        self.use_val = use_val
        self.save_dir = save_dir
        self.num_ensemble = num_ensemble
        self.ensemble_temperatures = [i * 0.1 for i in range(self.num_ensemble)] if (ensemble_temperatures
                                                                     is None) else ensemble_temperatures
        assert isinstance(self.ensemble_temperatures, list) and isinstance(self.ensemble_temperatures[0], float) and 1.0 > self.ensemble_temperatures[0] >= 0.0
        self.mipro_optimizer_temperature = temperature
        # Seed random for reproducibility.
        random.seed(self.seed)
        # Data Preparation.
        self._prepare_data()
        # Create and Optimize/Load Predictors.
        os.makedirs(self.save_dir, exist_ok=True)
        # Train predictors.
        self.predictors: List[MultiLabelLinkPredictor] = self._create_and_optimize_predictors()
        print(f"\n--- {self.__class__.__name__} initialized with {len(self.predictors)} predictors ---")

    def _create_and_optimize_predictors(self) -> List[MultiLabelLinkPredictor]:
        """
        Creates predictor instances for each temperature, optimizes them using MIPRO,
        and saves/loads the optimized state.
        """
        predictors = []
        for idx, temp in enumerate(self.ensemble_temperatures):
            print(f"\n--- Optimizing/Loading Predictor for Temperature: {temp:.1f} ---")
            dataset_name=self.kg.dataset_dir.split("/")[-1]
            save_filename = os.path.join(self.save_dir, f"{dataset_name}_predictor_temp_{temp:.1f}.json")
            # @TODO: CD: Later, save the details eval results as csv controlled by a flag attribute.
            results_filename = os.path.join(self.save_dir, f"eval_results_temp_{temp:.1f}.csv")
            # Initialize the base predictor for this temperature
            base_predictor = MultiLabelLinkPredictor(entities=list(self.all_entities),
                                                     knowledge_graph=self.entity_relation_to_entities)
            if os.path.exists(save_filename):
                print(f"Loading optimized predictor from {save_filename}...")
                # Need to configure LM *before* loading if LM state isn't saved
                lm = dspy.LM(model=f"openai/{self.llm_model}", api_key=self.api_key,
                             api_base=self.base_url, temperature=temp, # Use the specific temp
                             cache=True, cache_in_memory=True) # Seed might not be needed for loading
                dspy.configure(lm=lm)
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
                    temperature=temp, save_filename=save_filename,auto=self.auto)
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
                                           train_examples:List[dspy.Example],test_examples,temperature: float,
                                           save_filename:str,
                                           auto: Optional[Literal["light", "medium", "heavy"]] = "heavy") -> MultiLabelLinkPredictor:
        """Configures LM and runs MIPROv2 compilation."""

        assert isinstance(predictor_to_optimize, MultiLabelLinkPredictor)
        assert isinstance(train_examples, list) and isinstance(train_examples[0],dspy.Example)
        assert isinstance(test_examples, list) and isinstance(test_examples[0],dspy.Example)
        assert auto in ("light", "medium", "heavy")
        # Configure DSPy LM specifically for this optimization run
        lm = dspy.LM(model=f"openai/{self.llm_model}", api_key=self.api_key,api_base=self.base_url,
                     seed=self.seed, temperature=temperature,
                     cache=True, cache_in_memory=True)
        dspy.configure(lm=lm)
        # Test Run
        for s,p,o in self.triples:
            print(s,p,o)
            yhat = predictor_to_optimize.forward(subject=s,predicate=p)
            print(yhat)
            break
        # Test Run
        Evaluate(devset=test_examples[:6],metric=dspy_quality_score_closeness,
                 num_threads=1, display_table=False,
                 display_progress=True, provide_traceback=True)(predictor_to_optimize)
        # Generate examples needed for optimization outside the loop
        optim = dspy.MIPROv2(metric=dspy_quality_score_closeness,
                             auto=auto)
        optimized_predictor = optim.compile(predictor_to_optimize.deepcopy(), trainset=train_examples[:],
                                         valset=test_examples[:],
                                         requires_permission_to_run=False)
        optimized_predictor.finder.lm=lm
        optimized_predictor.scorer.lm=lm
        optimized_predictor.save(save_filename)
        return optimized_predictor

    class ProcessingType(Protocol):
        """Specifies the type of the callable argument for `_process_scores` method"""
        def __call__(self, x: List[float], y: List, **kwargs: Any) -> None: ...

    def _process_scores(self, x: torch.LongTensor, process: ProcessingType,
                       storage: List = None) -> List:
        """Run the predictors to predict tail entities, score the prediction and
        further process the scores depending on the `process` argument.

        Args:
            x (torch.LongTensor): Batch of (h, r) indices to predict triples for.
            process (ProcessingType): Callable function to further process the scores.
            storage (List): List to store the processed scores.

        Returns:
            List: a list of processed scores.
        """
        num_entities = len(self.idx_to_entity)

        # TODO: AB: Why do we config this LM if we already have configured LM for each predictor in self.predictors?
        lm = dspy.LM(model=f"openai/{self.llm_model}", api_key=self.api_key, api_base=self.base_url,
                     seed=self.seed, temperature=self.mipro_optimizer_temperature,
                     cache=True, cache_in_memory=True)
        dspy.configure(lm=lm)
        if storage is None:
            storage = []
        # Use tqdm for progress visualization
        for hr in tqdm(x.tolist(), desc="Predicting Batches (K vs All)"):
            idx_h, idx_r = hr
            h = self.idx_to_entity.get(idx_h, None)
            r = self.idx_to_relation.get(idx_r, None)
            if h is None or r is None:
                print(f"Warning: Unknown index in query: h={idx_h}, r={idx_r}. Skipping.")
                storage.append([0.0] * num_entities)  # Predict all zeros
                continue

            # Use a dictionary to accumulate scores by entity name for easier handling
            accumulated_scores: Dict[str, float] = {}
            num_predictors_used = 0

            # Get predictions from each predictor in the ensemble
            for i, predictor in enumerate(self.predictors):
                prediction = predictor.forward(subject=h, predicate=r)
                num_predictors_used += 1

                # prediction.target should be List[Tuple[str, float]] thanks to predictor.forward()
                for entity, score in prediction.target:
                    accumulated_scores[entity] = accumulated_scores.get(entity, 0.0) + score

            # --- Convert accumulated scores to the final output format ---
            final_scores = [0.0] * num_entities
            if num_predictors_used > 0:
                for entity_name, total_score in accumulated_scores.items():
                    if entity_name in self.entity_to_idx:
                        idx_entity = self.entity_to_idx[entity_name]
                        final_scores[idx_entity] = total_score / num_predictors_used  # Average score
                    # else:
                    #    print(f"Warning: Predicted entity '{entity_name}' not in entity index map.")

            process(final_scores, storage, h=h, r=r)

        return storage

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Generate predictions for (head, relation) pairs against all entities.
        Averages scores from all predictors in the ensemble.
        """

        def process(scores, storage, **kwargs):
            storage.append(scores)

        batch_predictions = self._process_scores(x, process)

        return torch.FloatTensor(batch_predictions)
    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        # This method is required by the abstract class but not the focus here.
        # If needed, adapt it similar to forward_k_vs_all but for specific triples.
        raise NotImplementedError("forward_triples needs implementation if used.")

    def get_predicted_triples(self, x: torch.LongTensor, top_k: int = 1):
        """
        Retrieve the top predicted triples for the given batch of (h, r) indices.
        Args:
            x (torch.LongTensor): Batch of (h, r) indices to predict triples for.
            top_k (int): Number of top entities to return per (h, r).
        Returns:
            List[Tuple[str, str, List[Tuple[str, float]]]]:
                A list of (subject, relation, [(entity, score)]), where the scores are
                sorted to get the top_k predicted triples.
        """

        def process(scores, storage, **kwargs):
            # Extract top-k entities (along with scores)
                rng = range(len(scores))
                top_entity_indices = sorted(rng, key=lambda i: scores[i], reverse=True)[:top_k]
                top_entities = [(self.idx_to_entity[idx], scores[idx]) for idx in top_entity_indices]

                # Append formatted triple: (subject, predicate, [(entity, score)])
                storage.append((kwargs.get('h'), kwargs.get('r'), top_entities))

        predicted_triples = self._process_scores(x, process)

        return predicted_triples

    def _prepare_data(self):
        """Loads, processes, and prepares training/validation data."""
        print("Preparing data...")
        train_set = [(self.idx_to_entity[h], self.idx_to_relation[r], self.idx_to_entity[t])
                     for h, r, t in self.kg.train_set.tolist()]
        val_set = [(self.idx_to_entity[h], self.idx_to_relation[r], self.idx_to_entity[t])
                   for h, r, t in self.kg.valid_set.tolist()]

        self.triples = train_set + (val_set if self.use_val else [])

        # Group triples by (subject, predicate)
        self.entity_relation_to_entities: Dict[Tuple[str, str], Set[str]] = {}
        self.all_entities: Set[str] = set()
        for s, p, o in self.triples:
            self.all_entities.add(s)
            self.all_entities.add(o)
            self.entity_relation_to_entities.setdefault((s, p), set()).add(o)

        print(f"Prepared data: {len(self.triples)} triples, {len(self.all_entities)} unique entities.")


# --- Main Execution Block ---
if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Loading KG from: {args.dataset_dir}")
    kg = KG(dataset_dir=args.dataset_dir, separator="\s+", eval_model=args.eval_model, add_reciprocal=False)
    sanity_checking(args, kg) # Assuming this function exists and checks args
    print("Initializing RALP with MIPROv2...")
    model = RALP_MPRO(knowledge_graph=kg, base_url=args.base_url, api_key=args.api_key,
                              llm_model=args.llm_model_name, temperature=args.temperature,
                              seed=args.seed, use_val=True)

    if not args.print_top_predictions: # Evaluate
        print("Starting evaluation...")
        # Limit evaluation size if needed (e.g., during testing)
        eval_triples = kg.test_set[:args.eval_size] if args.eval_size > 0 else kg.test_set
        print(f"Evaluating on {len(eval_triples)} test triples...")
        results: dict = evaluate_lp_k_vs_all(model=model, triple_idx=eval_triples,
                                             er_vocab=kg.er_vocab,
                                             info='Eval KvsAll (RALP_MPRO) Starts')
    else: # Print prediction generated from the train set
        x = kg.train_set[:, [0, 1]]
        results = model.get_predicted_triples(x, args.k)
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