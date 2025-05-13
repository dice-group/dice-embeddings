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

"""
python instance_predictor.py

Predicting instances of <http://example.com/foo#car>...

All instances of <http://example.com/foo#car> (sorted by score):
1. <http://example.com/foo#car_71> (score: 1.0000)
2. <http://example.com/foo#car_14> (score: 0.9992)
3. <http://example.com/foo#car_21> (score: 0.9992)
4. <http://example.com/foo#car_72> (score: 0.9992)
5. <http://example.com/foo#car_101> (score: 0.9984)
6. <http://example.com/foo#car_22> (score: 0.5016)
7. <http://example.com/foo#car_82> (score: 0.5016)
8. <http://example.com/foo#car_12> (score: 0.5008)
9. <http://example.com/foo#car_31> (score: 0.5008)
10. <http://example.com/foo#car_32> (score: 0.5008)
11. <http://example.com/foo#car_42> (score: 0.5008)
12. <http://example.com/foo#car_43> (score: 0.5008)
13. <http://example.com/foo#car_51> (score: 0.5008)
14. <http://example.com/foo#car_53> (score: 0.5008)
15. <http://example.com/foo#car_61> (score: 0.5008)
16. <http://example.com/foo#car_62> (score: 0.5008)
17. <http://example.com/foo#car_91> (score: 0.5008)
18. <http://example.com/foo#car_102> (score: 0.5000)
19. <http://example.com/foo#car_11> (score: 0.5000)
20. <http://example.com/foo#car_13> (score: 0.5000)
21. <http://example.com/foo#car_23> (score: 0.5000)
22. <http://example.com/foo#car_33> (score: 0.5000)
23. <http://example.com/foo#car_41> (score: 0.5000)
24. <http://example.com/foo#car_44> (score: 0.5000)
25. <http://example.com/foo#car_52> (score: 0.5000)
26. <http://example.com/foo#car_73> (score: 0.5000)
27. <http://example.com/foo#car_81> (score: 0.5000)
28. <http://example.com/foo#car_92> (score: 0.5000)
29. <http://example.com/foo#car_93> (score: 0.5000)
30. <http://example.com/foo#car_94> (score: 0.5000)
31. <http://example.com/foo#train> (score: 0.0016)
32. <http://example.com/foo#circle> (score: 0.0016)
33. <http://example.com/foo#east2> (score: 0.0016)
34. <http://example.com/foo#east4> (score: 0.0016)
35. <http://example.com/foo#u_shaped> (score: 0.0016)
36. <http://example.com/foo#west8> (score: 0.0016)
37. <http://example.com/foo> (score: 0.0008)
38. <http://example.com/foo#car> (score: 0.0008)
39. <http://example.com/foo#closed> (score: 0.0008)
40. <http://example.com/foo#double> (score: 0.0008)
41. <http://example.com/foo#jagged> (score: 0.0008)
42. <http://example.com/foo#long> (score: 0.0008)
43. <http://example.com/foo#open_car> (score: 0.0008)
44. <http://example.com/foo#short> (score: 0.0008)
45. <http://example.com/foo#has_car> (score: 0.0008)
46. <http://example.com/foo#load> (score: 0.0008)
47. <http://example.com/foo#load_count> (score: 0.0008)
48. <http://example.com/foo#wheels> (score: 0.0008)
49. <http://example.com/foo#hasShape> (score: 0.0008)
50. <http://example.com/foo#east1> (score: 0.0008)
51. <http://example.com/foo#east3> (score: 0.0008)
52. <http://example.com/foo#east5> (score: 0.0008)
53. <http://example.com/foo#elipse> (score: 0.0008)
54. <http://example.com/foo#hexagon> (score: 0.0008)
55. <http://example.com/foo#one> (score: 0.0008)
56. <http://example.com/foo#rectangle> (score: 0.0008)
57. <http://example.com/foo#three> (score: 0.0008)
58. <http://example.com/foo#triangle> (score: 0.0008)
59. <http://example.com/foo#two> (score: 0.0008)
60. <http://example.com/foo#west10> (score: 0.0008)
61. <http://example.com/foo#west6> (score: 0.0008)
62. <http://example.com/foo#west7> (score: 0.0008)
63. <http://example.com/foo#west9> (score: 0.0008)
64. <http://example.com/foo#zero> (score: 0.0008)
65. <http://www.w3.org/2002/07/owl#Ontology> (score: 0.0000)
66. <http://www.w3.org/2002/07/owl#Class> (score: 0.0000)
67. <http://www.w3.org/2002/07/owl#ObjectProperty> (score: 0.0000)
68. <http://example.com/foo#shape> (score: 0.0000)
69. <http://www.w3.org/2002/07/owl#Thing> (score: 0.0000)
"""