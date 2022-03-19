from core import KGE
from core.knowledge_graph import KG
import torch

# pre_trained_kge = KGE(path_of_pretrained_model_dir='DBpediaQMult')
pre_trained_kge = KGE(path_of_pretrained_model_dir='Experiments/2022-03-19 16:40:54.529443')
persons = [["http://dbpedia.org/resource/Albert_Einstein"], ["http://dbpedia.org/resource/John_von_Neumann"],
           ["http://dbpedia.org/resource/Stephen_Hawking"], ["http://dbpedia.org/resource/Germany"],
           ["http://dbpedia.org/resource/Turkey"]]

num_iter = 50
repeat = 10
for p in persons:
    # 10 times p r 110MillionEntities => 110 10^7 data points.

    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://www.w3.org/1999/02/22-rdf-syntax-ns#type"],
                                   iteration=num_iter,
                                   repeat=repeat, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/citizenship"],
                                   iteration=num_iter,
                                   repeat=repeat, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/nationality"],
                                   iteration=num_iter,
                                   repeat=repeat, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/birthPlace"],
                                   iteration=num_iter,
                                   repeat=repeat, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/deathPlace"],
                                   iteration=num_iter,
                                   repeat=repeat, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/knownFor"], iteration=num_iter,
                                   repeat=repeat, lr=.1)
    pre_trained_kge.train_k_vs_all(head_entity=p, relation=["http://dbpedia.org/ontology/wikiPageWikiLink"],
                                   iteration=num_iter, repeat=repeat, lr=.1)
pre_trained_kge.save()
