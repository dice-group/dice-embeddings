import torch
from core import KGE
from core.knowledge_graph import KG

# (1) Load a pre-trained model; YAGO3-10. construct_ensemble combines all models in this folder.
pre_trained_kge = KGE(path_of_pretrained_model_dir='Experiments/2022-03-18 09:54:53.214382', construct_ensemble=False)
# (2) Compute Triple Score.
heads, relations, tails = ['Telmo_Zarra'], ['diedIn'], ['Bilbao']
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=heads, logits=True))
pre_trained_kge.train_triples_lbfgs_negative(head_entity=heads, relation=relations, tail_entity=heads, repeat=2)
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=heads, logits=True))
print('\n')
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=tails, logits=True))
pre_trained_kge.train_triples_lbfgs_positive(head_entity=heads, relation=relations, tail_entity=tails, repeat=2)
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=tails, logits=True))
exit(1)
# (3) Compute  ? r t, h r ?, and h ? t scores.
print(pre_trained_kge.predict_topk(head_entity=heads, relation=relations))
print(pre_trained_kge.predict_topk(head_entity=heads, tail_entity=tails))
print(pre_trained_kge.predict_topk(tail_entity=tails, relation=relations))
# (4) Save.
pre_trained_kge.save()
# (5) Train Model on KGs/SubYAGO3-10/train.txt.
kg = KG("KGs/SubYAGO3-10", entity_to_idx=pre_trained_kge.entity_to_idx, relation_to_idx=pre_trained_kge.relation_to_idx)
pre_trained_kge.train(kg, lr=.1, epoch=2, batch_size=32, neg_sample_ratio=2, num_workers=64)
pre_trained_kge.save()
