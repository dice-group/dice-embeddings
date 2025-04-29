import os
import dspy
import json
from typing import List, Tuple, Callable, Protocol, Any
def sanity_checking(args,kg):
    if args.eval_size is not None:
        assert len(kg.test_set) >= args.eval_size, (f"Evaluation size cant be greater than the "
                                                    f"total amount of triples in the test set: {len(kg.test_set)}")
    else:
        args.eval_size = len(kg.test_set)
    if args.api_key is None:
        args.api_key = os.environ.get("TENTRIS_TOKEN")



class MultiLabelLinkPredictionWithScores(dspy.Signature):
    examples = dspy.InputField(
        desc="Few-shot examples of (subject, predicate) -> [{'entity': entity1, 'score': score1}, ...].")
    subject:str = dspy.InputField(desc="The subject entity.")
    predicate:str = dspy.InputField(desc="The relationship type.")

    # Updated OutputField requesting JSON
    objects_with_scores = dspy.OutputField(
        desc="A JSON string representing a list of objects. "
             "Each object in the list should be a dictionary with 'entity' (string) and 'score' (float, 0.0-1.0) keys.")


class BasicMultiLabelLinkPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(MultiLabelLinkPredictionWithScores)

    def forward(self, subject, predicate, few_shot_examples) -> List[Tuple[str, float]]:
        # Format examples more structured with clearer JSON expectations
        example_str = "Given a subject entity and relationship, predict all possible object entities with confidence scores.\n\n"
        example_str += "Examples:\n"

        # Add more context to each example
        for idx, ((s, p), o_list) in enumerate(few_shot_examples.items()):
            formatted_objects = [{"entity": o, "score": 1.0} for o in
                                 o_list]  # Assuming high confidence in training examples
            example_str += f"Example {idx + 1}:\n"
            example_str += f"Input: subject='{s}', predicate='{p}'\n"
            example_str += f"Output: {json.dumps(formatted_objects)}\n\n"

        # Add more explicit instructions for the model
        example_str += "Rules:\n"
        example_str += "1. Return all relevant entities with confidence scores between 0.0 and 1.0\n"
        example_str += "2. Higher scores indicate higher confidence\n"
        example_str += "3. Sort results by confidence score in descending order\n"
        example_str += "4. Use knowledge of real-world relationships when appropriate\n\n"

        # Log the prompt for analysis (optional, can be removed in production)
        if hasattr(self, 'debug') and self.debug:
            print(f"PROMPT:\n{example_str}\n")
            print(f"QUERY: subject='{subject}', predicate='{predicate}'")

        # Make the prediction
        dspy_pred = self.predictor(examples=example_str, subject=subject, predicate=predicate)

        try:
            results = json.loads(dspy_pred.objects_with_scores)
            # Sort by score descending
            return [(i["entity"], i["score"]) for i in sorted(results, key=lambda x: x["score"], reverse=True)]
        except json.JSONDecodeError:
            # Handle malformed JSON gracefully
            print(f"Warning: Received malformed JSON: {dspy_pred.objects_with_scores}")
            return []
