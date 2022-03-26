"""
Large-Scale KGE Challenges:

+ Time requirement: Training a KGE model on 4 x 10^7 triples with large batch (50 000) for a single epoch takes up to 4 hours with negative sampling with 1 negative ratio.
We can't run a model 1000 epochs we can increase the negative ratio arbitrarily.

+ Memory requirement: Theoretically, we can use KvsAll or 1vsAll to accelerate the convergence process during training.
Yet, having more than 10^7 entities does not allow mini batch size more than 3-5.

###############################################################
Motivated by the workshop (Large-scale Machine Learning and Stochastic Algorithms by Leon Bottou)

Assume that training set contains 10 copies of the 100 same examples

BATCH: Blindly computes redundant gradients, 1 epoch on large set \equiv 1 epoch on small set without redundant

Online: Take advantage of redundancy
###############################################################

Applying KvsAll on two consecutive example :  [ (h, r)_i + |E| ] + [ (h, r)_j + |E| ]
"""
from core import KGE
from core.knowledge_graph import KG
import random
import time

# (1) Load a pretrained model (say we ran it only for a single epoch).
pre_trained_kge = KGE(path_of_pretrained_model_dir='Experiments/2022-03-26 15:54:06.034352', construct_ensemble=False)
# (2) Sample random entities
start_time = time.time()
for ith, entity in enumerate(pre_trained_kge.entity_to_idx.sample(10, random_state=random.randint(1, 1_000_000)).index):
    # Retrieve relations occurring with (2))
    relations = pre_trained_kge.get_relations(entity)
    print(f'Example:{ith}\t Learning over {entity} and {len(relations)} relations')
    # z:= {(h,r), y)_i }_i ^n s.t. n:=| {r | (h,r,x) OR (x,r,h) \in G }|
    for relation in relations:
        pre_trained_kge.train_k_vs_all(head_entity=[entity], relation=[relation], iteration=10, lr=.0001)
        pre_trained_kge.train_k_vs_all_lbfgs(head_entity=[entity], relation=[relation], iteration=1)
        print(pre_trained_kge.predict_topk(head_entity=[entity], relation=[relation], k=10))
        print()

print(f'Online KGE training took {time.time() - start_time}')
# Save model
pre_trained_kge.save()
exit(1)
for relation in pre_trained_kge.relation_to_idx.sample(10).index:
    relation = [relation]
    pre_trained_kge.train_k_vs_all(head_entity=["http://dbpedia.org/resource/Albert_Einstein"], relation=relation,
                                   iteration=1, lr=.0001)

for entity in pre_trained_kge.entity_to_idx.sample(1).index:
    entity = [entity]
    # Random relations
    for relation in pre_trained_kge.relation_to_idx.sample(1).index:
        relation = [relation]
        pre_trained_kge.train_k_vs_all(head_entity=entity, relation=relation, iteration=1, lr=.001)

print(pre_trained_kge.predict_topk(head_entity=["Brad_Pitt"], relation=['actedIn'], k=25))

pre_trained_kge.train_triples(head_entity=["http://dbpedia.org/resource/Albert_Einstein"],
                              relation=['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'],
                              tail_entity=["http://dbpedia.org/class/yago/WikicatTanzanianPeople"],
                              iteration=1,
                              lr=.001,
                              labels=[0.0])

exit(1)

# pre_trained_kge.train_cbd(["Brad_Pitt"], iteration=100, num_copies_in_batch=1)
print(pre_trained_kge.triple_score(head_entity=["Brad_Pitt"], relation=['isLocatedIn'], tail_entity=["Brad_Pitt"],
                                   logits=False))
# print(pre_trained_kge.predict_missing_tail_entity(head_entity=["Brad_Pitt"], relation=['actedIn'], k=100))
pre_trained_kge.train_triples(head_entity=["Brad_Pitt"], relation=['isLocatedIn'], tail_entity=["Brad_Pitt"],
                              iteration=1,
                              lr=.0001,
                              labels=[0.0])
print(pre_trained_kge.triple_score(head_entity=["Brad_Pitt"], relation=['isLocatedIn'], tail_entity=["Brad_Pitt"],
                                   logits=True))

exit(1)
print(pre_trained_kge.triple_score(head_entity=["Chatou"], relation=['isLocatedIn'], tail_entity=["France"],
                                   logits=False))
print(pre_trained_kge.triple_score(head_entity=["Chatou"], relation=['isLocatedIn'], tail_entity=["France"],
                                   without_norm=True, logits=False))

exit(1)
# (2) Sample few entities
num_samples = 10
num_iter = 10
for i in pre_trained_kge.entity_to_idx.sample(10).index:
    # 20 times Albert_Einstein r 110MillionEntities s.t. r \in {r | Albert_Einstein r x \in G }
    pre_trained_kge.train_cbd([i], iteration=num_iter)
pre_trained_kge.save()
exit(1)

# (3) train CBD
# Online CBD Learning
num_iter = 50
num_copies_in_batch = 1
p = ["Chatou"]
for r in pre_trained_kge.relation_to_idx.index.to_list():
    if r == 'isLocatedIn':
        print(p, r, end='\t')
        res = pre_trained_kge.predict_topk(head_entity=p, relation=[r], k=25)
        print(res)
# 20 times Albert_Einstein r 110MillionEntities s.t. r \in {r | Albert_Einstein r x \in G }
pre_trained_kge.train_cbd(p, iteration=num_iter, num_copies_in_batch=num_copies_in_batch, lr=.01)
for r in pre_trained_kge.relation_to_idx.index.to_list():
    if r == 'isLocatedIn':
        print(p, r, end='\t')
        res = pre_trained_kge.predict_topk(head_entity=p, relation=[r], k=25)
        print(res)

# pre_trained_kge = KGE(path_of_pretrained_model_dir='seedQMult')
pre_trained_kge = KGE(path_of_pretrained_model_dir='Experiments/2022-03-19 16:40:54.529443')
# Randomly sample an entity and construc input data based on relations occured with it
for ith, entity in enumerate(pre_trained_kge.entity_to_idx.sample(100).index):
    relations = pre_trained_kge.get_relations(entity)
    print(f'Example:{ith}\t Learning over {entity} and {len(relations)} relations')
    for relation in relations:
        pre_trained_kge.train_k_vs_all(head_entity=[entity], relation=[relation], iteration=10, lr=.0001)
########### Selective
num_iter = 50
num_copies_in_batch = 10
# Online KvsAll Learning
for p in [["http://dbpedia.org/resource/Albert_Einstein"]]:
    pre_trained_kge.train_k_vs_all(head_entity=["http://dbpedia.org/resource/Albert_Einstein"],
                                   relation=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
                                   iteration=num_iter, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/citizenship"],
                                   iteration=num_iter, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/nationality"],
                                   iteration=num_iter, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/birthPlace"],
                                   iteration=num_iter, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/deathPlace"],
                                   iteration=num_iter, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/knownFor"], iteration=num_iter,
                                   lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/wikiPageWikiLink"],
                                   iteration=num_iter, lr=.1)
pre_trained_kge.save()
