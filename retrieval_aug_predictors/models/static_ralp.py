"""
python -m retrieval_aug_predictors.models.static_ralp --enrich_training --dataset_dir KGs/Countries-S1 --out "countries_s1_results.json" && cat countries_s1_results.json
{
    "H@1": 1.0,
    "H@3": 1.0,
    "H@10": 1.0,
    "MRR": 1.0
}

python -m retrieval_aug_predictors.models.static_ralp --dataset_dir KGs/Countries-S2 --out "countries_s2_results.json" && cat countries_s2_results.json
{
    "H@1": 0.625,
    "H@3": 0.9583333333333334,
    "H@10": 0.9583333333333334,
    "MRR": 0.7921296296296297
}
python -m retrieval_aug_predictors.models.static_ralp --dataset_dir KGs/Countries-S3 --out "countries_s3_results.json" && cat countries_s3_results.json
{
    "H@1": 0.7083333333333334,
    "H@3": 0.9583333333333334,
    "H@10": 0.9583333333333334,
    "MRR": 0.8337962962962964
}
"""

import dspy
import numpy as np
import torch
import json
from typing import List, Tuple, Set
from tqdm import tqdm
from retrieval_aug_predictors.models import KG, AbstractBaseLinkPredictorClass
from retrieval_aug_predictors.arguments import parser
from retrieval_aug_predictors.utils import sanity_checking, MultiLabelLinkPredictionWithScores, BasicMultiLabelLinkPredictor
from dicee.evaluator import evaluate_lp_k_vs_all
from dotenv import load_dotenv
load_dotenv()

class StaticRALP(AbstractBaseLinkPredictorClass):
    def __init__(self,knowledge_graph, base_url,api_key,temperature, seed,llm_model,use_val:bool=False):
        super().__init__(knowledge_graph,name="Demir")
        self.temperature = temperature
        self.seed = seed
        self.lm = dspy.LM(model=f"openai/{llm_model}", api_key=api_key,
                          api_base=base_url,
                          seed=seed,
                          temperature=temperature,
                          cache=True,cache_in_memory=True)
        dspy.configure(lm=self.lm)
        self.train_set: List[Tuple[str]] = [(self.idx_to_entity[idx_h],
                                             self.idx_to_relation[idx_r],
                                             self.idx_to_entity[idx_t]) for idx_h, idx_r, idx_t in
                                            self.kg.train_set.tolist()]
        # Validation dataset
        self.val_set: List[Tuple[str]] = [(self.idx_to_entity[idx_h],
                                           self.idx_to_relation[idx_r],
                                           self.idx_to_entity[idx_t]) for idx_h, idx_r, idx_t in
                                          self.kg.valid_set.tolist()]
        self.triples = self.train_set + self.val_set if use_val else self.train_set

        self.entity_relation_to_entities=dict()
        for s,p,o in self.triples:
            self.entity_relation_to_entities.setdefault((s,p),[]).append(o)
        # @TODO: rename entity_relation_to_entities to something else
        # @TODO: given (s,p), we need to return relevant triples, not the whole graph.
        self.scoring_func = BasicMultiLabelLinkPredictor()

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        raise NotImplementedError("RCL needs to implement it")
    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        batch_predictions=[]
        for hr in x.tolist():
            idx_h, idx_r = hr
            h, r = self.idx_to_entity[idx_h], self.idx_to_relation[idx_r]
            predictions = self.scoring_func.forward(
                subject=h,
                predicate=r,
                few_shot_examples=self.entity_relation_to_entities)
            scores=[-100]*len(self.idx_to_entity)
            for entity,score in predictions:
                try:
                    idx_entity=self.entity_to_idx[entity]
                except KeyError:
                    print(f"Entity:{entity} not found")
                    continue
                scores[idx_entity]=score
            batch_predictions.append(scores)
        return torch.FloatTensor(batch_predictions)


def find_missing_triples(model:AbstractBaseLinkPredictorClass,train_set:np.ndarray,threshold:float=0.5)->List[Tuple[str,str,str]]:
    triples={tuple(triple) for triple in train_set.tolist()}
    founded_triples=set()
    t:Tuple[int,int,int]
    for t in tqdm(triples):
        idx_s,idx_p,idx_o=t
        #
        s,p,o=model.idx_to_entity[idx_s],model.idx_to_relation[idx_p],model.idx_to_entity[idx_o]
        # print(s,p,o)
        # Likelihood for each entities.
        predictions=model.forward_k_vs_all(x=torch.LongTensor([[idx_s,idx_p]]))
        # Work with the 1D row only
        scores = predictions[0]
        mask = scores > threshold
        matching_indices = torch.nonzero(mask, as_tuple=True)[0]  # Get just the indices, shape: [num_matches]
        # print("Scores > 0.5:")
        # print(scores[matching_indices])
        # print("Indices with score > 0.5:")
        entities=[ model.idx_to_entity[i]for i in matching_indices.tolist()]
        for new_o in entities:
            founded_triples.add((s,p,new_o))
    # Only return missing triples
    return founded_triples - triples

# test the dspy model -> remove later
if __name__ == "__main__":
    args=parser.parse_args()
    # Important: add_reciprocal=False in KvsAll implies that inverse relation has been introduced.
    # Therefore, The link prediction results are based on the missing tail rankings only!
    kg = KG(dataset_dir=args.dataset_dir, separator="\s+", eval_model=args.eval_model, add_reciprocal=False)
    sanity_checking(args,kg)
    model = StaticRALP(knowledge_graph=kg, base_url=args.base_url, api_key=args.api_key, llm_model=args.llm_model_name, temperature=args.temperature, seed=args.seed)

    if args.enrich_train:
        missing_triples:Set[Tuple[str,str,str]] = find_missing_triples(model=model,train_set=kg.train_set)
        # Write the triples to a file
        paths = args.dataset_dir+"/missing_triples.txt"
        with open(paths, "w", encoding="utf-8") as f:
            for s, p, o in missing_triples:
                f.write(f"{s}\t{p}\t{o}\n")
        print(f"Missing triples written to {paths}")

    results:dict = evaluate_lp_k_vs_all(model=model, triple_idx=kg.test_set[:args.eval_size],
                         er_vocab=kg.er_vocab, info='Eval KvsAll Starts', batch_size=args.batch_size)
    if args.out and results:
        # Writing the dictionary to a JSON file
        print(results)
        with open(args.out, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        print(f"Results has been saved to {args.out}")