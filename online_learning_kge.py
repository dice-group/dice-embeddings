from core import KGE
from core.knowledge_graph import KG
import torch
pre_trained_kge = KGE(path_of_pretrained_model_dir='DBpediaQMult')
persons = [["http://dbpedia.org/resource/Albert_Einstein"], ["http://dbpedia.org/resource/John_von_Neumann"], ["http://dbpedia.org/resource/Stephen_Hawking"]]
for p in persons:
    # 10 times p r ? against all entities.
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/citizenship"], iteration=100, repeat=10, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/nationality"], iteration=100, repeat=10, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/birthPlace"], iteration=100, repeat=10, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/deathPlace"], iteration=100, repeat=10, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/knownFor"], iteration=100, repeat=10, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/wikiPageWikiLink"], iteration=100, repeat=10, lr=.1)
pre_trained_kge.save()