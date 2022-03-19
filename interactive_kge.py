import torch
from core import KGE
from core.knowledge_graph import KG

# (1) Load a pre-trained model; YAGO3-10. construct_ensemble combines all models in this folder.
pre_trained_kge = KGE(path_of_pretrained_model_dir='Experiments/2022-03-19 11:47:08.946813', construct_ensemble=False)
# (2) Compute Triple Score.
heads, relations = ['Chatou'], ['isLocatedIn']
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=['France'], logits=True))
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=['Yvelines'], logits=True))
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=['Île-de-France'], logits=True))
print(pre_trained_kge.predict_topk(head_entity=heads, relation=relations))
pre_trained_kge.train_k_vs_all(head_entity=heads, relation=relations, iteration=10, lr=.1)
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=['France'], logits=True))
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=['Yvelines'], logits=True))
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=['Île-de-France'], logits=True))
print(pre_trained_kge.predict_topk(head_entity=heads, relation=relations))
exit(1)
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=heads, logits=True))
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=heads, logits=True))
# (3) Train on (2) in kvsall setting
pre_trained_kge.train_k_vs_all(head_entity=heads, relation=relations, iteration=10, lr=.01)
# (4) Compute the new score
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=heads, logits=True))
# (5) Compute score of false triple
heads, relations, tails = ['France'], ['isLocatedIn'], ['Chatou']
print(pre_trained_kge.predict_topk(head_entity=heads, relation=relations))


exit(1)
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=tails, logits=True))
pre_trained_kge.train_triples_lbfgs_negative(head_entity=heads, relation=relations, tail_entity=heads, repeat=2)
print(pre_trained_kge.triple_score(head_entity=heads, relation=relations, tail_entity=tails, logits=True))
print('\n')
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
