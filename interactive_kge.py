from core import KGE

# (1) Train a knowledge graph embedding model on a dataset (example from UMLS)
# (2) Give the path of serialized (1).
pre_trained_kge = KGE(path_of_pretrained_model_dir='Experiments/2022-03-11 08:30:52.896174')
# (3) Head entity prediction : <?,relation,entity>
scores_and_entities = pre_trained_kge.predict(relation=['interacts_with'], tail_entity=['eicosanoid'], k=10)
print([i for i in scores_and_entities])
# (4) Tail entity prediction : <entity,relation,?>
scores_and_entities = pre_trained_kge.predict(head_entity=['eicosanoid'],
                                              relation=['interacts_with'], k=10)
print([i for i in scores_and_entities])
# (5) Relation prediction : <entity,?relation>
scores_and_relations = pre_trained_kge.predict(head_entity=['eicosanoid'],
                                               tail_entity=['eicosanoid'], k=10)
print([i for i in scores_and_relations])
# (6) Triple score: <entity, relation,entity>
triple_score = pre_trained_kge.predict(head_entity=['eicosanoid'],
                                       relation=['interacts_with'],
                                       tail_entity=['eicosanoid'])
print(triple_score)

# (7) Randomly sampled triple score: <entity, relation,entity>
triple_score = pre_trained_kge.predict(head_entity=pre_trained_kge.sample_entity(1),
                                       relation=pre_trained_kge.sample_relation(1),
                                       tail_entity=pre_trained_kge.sample_entity(1))
print(triple_score)
