from core import KGE
from core.knowledge_graph import KG
import torch

# pre_trained_kge = KGE(path_of_pretrained_model_dir='DBpediaQMult')
pre_trained_kge = KGE(path_of_pretrained_model_dir='Experiments/2022-03-19 16:40:54.529443')
persons = [["http://dbpedia.org/resource/Albert_Einstein"]]

num_iter = 50
num_copies_in_batch = 10
# Online KvsAll Learning
for p in persons:
    # 10 times p r 110MillionEntities => 110 10^7 data points.

    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
                                   iteration=num_iter,
                                   num_copies_in_batch=num_copies_in_batch, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/citizenship"],
                                   iteration=num_iter,
                                   num_copies_in_batch=num_copies_in_batch, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/nationality"],
                                   iteration=num_iter,
                                   num_copies_in_batch=num_copies_in_batch, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/birthPlace"],
                                   iteration=num_iter,
                                   num_copies_in_batch=num_copies_in_batch, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/deathPlace"],
                                   iteration=num_iter,
                                   num_copies_in_batch=num_copies_in_batch, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/knownFor"], iteration=num_iter,
                                   num_copies_in_batch=num_copies_in_batch, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/wikiPageWikiLink"],
                                   iteration=num_iter, num_copies_in_batch=num_copies_in_batch, lr=.1)
pre_trained_kge.save()
# Online CBD Learning
num_iter = 20
num_copies_in_batch = 10
for p in ["http://dbpedia.org/resource/Albert_Einstein"]:
    # 20 times Albert_Einstein r 110MillionEntities s.t. r \in {r | Albert_Einstein r x \in G }
    pre_trained_kge.train_cbd([p], iteration=num_iter, num_copies_in_batch=num_copies_in_batch, lr=.1)
for r in pre_trained_kge.relation_to_idx.index.to_list():
    print(p, r)
    res = pre_trained_kge.predict_topk(head_entity=[p], relation=[r], k=3)
    print(res)
pre_trained_kge.save()
