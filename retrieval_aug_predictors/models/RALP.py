import re
import torch
from openai import OpenAI
from retrieval_aug_predictors.models import KG, AbstractBaseLinkPredictorClass


class RALP(AbstractBaseLinkPredictorClass):
    def __init__(self, knowledge_graph: KG = None,
                 name="ralp-1.0",
                 base_url="http://tentris-ml.cs.upb.de:8501/v1",
                 api_key=None,
                 llm_model="tentris",
                 temperature: float = 1, seed: int = 42) -> None:
        super().__init__(knowledge_graph, name)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.llm_model = llm_model
        self.temperature = temperature
        self.seed = seed

    def extract_float(self, text):
        """Extract the float number from a string. Used mainly to filter the LLM-output for the scoring task."""
        pattern = r"-?\d*\.\d+|-?\d+\.\d*"
        match = re.search(pattern, text)
        return float(match.group()) if match else 0.0

    def ru(self, entity):
        """Remove underscore from the entity (as str)."""
        return entity.replace("_", " ")

    def get_score(self, triple: tuple, triples_h: str) -> float:
        system_prompt = """You are an expert in knowledge graphs and link prediction. Your task is to assign a plausibility score (from 0 to 1) to a given triple (subject, predicate, object) based on a set of known training triples for the same subject. 

        - A score of 1.0 means the triple is highly likely to be true.  
        - A score of 0.0 means the triple is highly unlikely to be true.  
        - Intermediate values (e.g., 0.4, 0.7) reflect varying levels of plausibility.


        **Guidelines for scoring:**
        1. **Exact Match:** If the triple already exists in the training set or if the facts clearly state that the triple must be true assign a score close to 1.0.
        2. **Pattern Matching:** If the predicate-object pair frequently occurs for the given subject, assign a high score.
        3. **Semantic Similarity:** If the object is semantically close to known objects for the subject-predicate pair, assign a moderate to high score.
        4. **Rare or Unseen Combinations:** If the triple does not follow the learned patterns, assign a low score.
        5. **Contradictions:** If the triple contradicts existing facts (perform your own reasoning), assign a very low score.

        You must analyze the given triple and the training triples, apply the reasoning above, and output only a single **floating-point score** between **0.0 and 1.0**, without any explanation or additional text.
        Do not depend only on triples provided to you, also use your own knowledge as an AI assistant to reason about the truthness of the given triple as a fact.
        You are strictly required to provide only the score as an answer and do not explain it."""

        user_prompt = f"""Here is the triple we want to evaluate:
        (subject: {triple[0]}, predicate: {triple[1]}, object: {triple[2]})

        Here are the known training triples for the subject "{triple[0]}":
        {triples_h}

        Assign a score to the given triple based on the provided training triples.
        """
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed=42,
            temperature=self.temperature
        )

        # Extract the response content
        content = response.choices[0].message.content
        return self.extract_float(content)

    def forward_k_vs_all(self, x):
        raise NotImplementedError("RALP needs to implement it")

    def forward_triples(self, indexed_triples: torch.LongTensor):
        n, d = indexed_triples.shape
        # For the time being
        assert d == 3
        assert n == 1
        scores = []
        for triple in indexed_triples.tolist():
            idx_h, idx_r, idx_t = triple
            h, r, t = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r], self.idx_to_entity[idx_t]

            # Retrieve triples where 'h' is a subject or an object
            triples_h = [trp for trp in self.kg.train_set if (trp[0] == idx_h or trp[2] == idx_h)]

            # Format the triples into structured string output that will be used in the prompt.
            triples_h_str = ""
            for trp in triples_h:
                triples_h_str += f'- ("{self.ru(self.idx_to_entity[trp[0]])}", "{self.ru(self.idx_to_relation[trp[1]])}", "{self.ru(self.idx_to_entity[trp[2]])}") \n'

            # Get the score from the LLM
            score = self.get_score((h, r, t), triples_h_str)
            scores.append([score])
        return torch.FloatTensor(scores)
