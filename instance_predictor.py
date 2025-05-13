import torch
import argparse
import pandas as pd
from retrieval_aug_predictors.models import KG
from retrieval_aug_predictors.arguments import parser
from retrieval_aug_predictors.models.RALP import RALP  

def predict_instances(args):
    # 1. Load Knowledge Graph with reciprocal relations
    print(f"Loading KG from: {args.dataset_dir}")
    kg = KG(
        dataset_dir=args.dataset_dir,
        separator="\s+",
        eval_model=args.eval_model,
        add_reciprocal=True  # To have inverse relations
    )
    
    # 2. Initialize RALP model
    print("Initializing RALP model...")
    model = RALP(
        knowledge_graph=kg,
        base_url=args.base_url,
        api_key=args.api_key,
        llm_model=args.llm_model_name,
        temperature=args.temperature,
        seed=args.seed,
        use_val=True
    )
    
    try:
        # 3. Get indices for target class and inverse predicate
        class_row = kg.entity_to_idx.loc[kg.entity_to_idx['entity'] == args.class_name]
        if class_row.empty:
            raise KeyError(f"'{args.class_name}'")
        class_idx = class_row.index[0]  
        
        inv_predicate_row = kg.relation_to_idx.loc[kg.relation_to_idx['relation'] == args.inverse_predicate]
        if inv_predicate_row.empty:
            raise KeyError(f"'{args.inverse_predicate}'")
        inv_predicate_idx = inv_predicate_row.index[0]  
        
    except KeyError as e:
        print(f"Error: {str(e)} not found in knowledge graph")
        print("Available classes:", kg.entity_to_idx['entity'].head(5).tolist(), "...")
        print("Available predicates:", kg.relation_to_idx['relation'].head(20).tolist(), "...")
        return

    # 4. Create query tensor (class_index, inverse_predicate_index)
    query = torch.LongTensor([[class_idx, inv_predicate_idx]])

    # 5. Get predictions
    print(f"\nPredicting instances of {args.class_name}...")
    scores = model.forward_k_vs_all(query)
    
    # 6. Process all results (not just top_k)
    instances = []
    for idx in range(len(scores[0])):
        score = scores[0][idx].item()
        if idx < len(kg.entity_to_idx):  # Ensure index is valid
            entity = kg.entity_to_idx.iloc[idx]['entity']
            instances.append((entity, score))
    
    # Sort instances by score in descending order
    instances.sort(key=lambda x: x[1], reverse=True)
    
    # 7. Display results
    print(f"\nAll instances of {args.class_name} (sorted by score):")
    for i, (entity, score) in enumerate(instances, 1):
        print(f"{i}. {entity} (score: {score:.4f})")

if __name__ == "__main__":
    parser.description = "Instance Prediction with RALP"
    
    parser.add_argument("--class_name", type=str, default="<http://example.com/foo#car>",
                        help="Target class name (e.g., 'http://example.com/foo#car')")
    parser.add_argument("--inverse_predicate", type=str, default="<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>_inverse",
                        help="Inverse predicate name (e.g., 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type_inverse')")
    
    parser.set_defaults(dataset_dir="KGs/Trains")
    
    args = parser.parse_args()
    
    predict_instances(args)