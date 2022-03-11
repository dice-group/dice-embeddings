from core import KGE
from core.knowledge_graph import KG

# (1) Train a knowledge graph embedding model on a dataset (example from UMLS)
# (2) Give the path of serialized (1).


pre_trained_kge = KGE(path_of_pretrained_model_dir='Experiments/2022-03-11 16:25:17.923404')


def predictions():
    # (3) Triple score: <entity, relation,entity>
    triple_score = pre_trained_kge.predict(head_entity=['eicosanoid'],
                                           relation=['interacts_with'],
                                           tail_entity=['eicosanoid'])
    print(triple_score)

    # (4) Randomly sampled triple score: <entity, relation,entity>
    triple_score = pre_trained_kge.predict(head_entity=pre_trained_kge.sample_entity(1),
                                           relation=pre_trained_kge.sample_relation(1),
                                           tail_entity=pre_trained_kge.sample_entity(1))
    print(triple_score)

    # (5) Head entity prediction : <?,relation,entity>
    scores_and_entities = pre_trained_kge.predict(relation=['interacts_with'], tail_entity=['eicosanoid'], k=10)
    print([i for i in scores_and_entities])
    # (6) Tail entity prediction : <entity,relation,?>
    scores_and_entities = pre_trained_kge.predict(head_entity=['eicosanoid'],
                                                  relation=['interacts_with'], k=10)
    print([i for i in scores_and_entities])
    # (7) Relation prediction : <entity,?relation>
    scores_and_relations = pre_trained_kge.predict(head_entity=['eicosanoid'],
                                                   tail_entity=['eicosanoid'], k=10)
    print([i for i in scores_and_relations])


# Selective Continual Training
# Load DBpedia page of something
import requests as req
import os

triples = req.get("https://dbpedia.org/data/Albert_Einstein.ntriples").text
os.system('mkdir Dummy')
with open('Dummy/train', 'w') as w:
    w.writelines(triples)

kg = KG("Dummy", entity_to_idx=pre_trained_kge.entity_to_idx, relation_to_idx=pre_trained_kge.relation_to_idx)
pre_trained_kge.train(kg)
