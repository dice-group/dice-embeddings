"""
Large-Scale KGE Challenges:

+ Time requirement: Training a KGE model on 4 x 10^7 triples even with a large batch (50 000)
for a single epoch can take hours (maybe days) with negative sampling with 1-5 negative ratio.
We can't run a model 1000 epochs we can increase the negative ratio arbitrarily.

+ Memory requirement: Theoretically, we can use KvsAll or 1vsAll to accelerate the convergence process during training.
Yet, having more than 10^7 entities does not allow mini batch size more than 3-5.
Even with negative sampling, we can have a large batch

###############################################################
Motivated by the workshop (Large-scale Machine Learning and Stochastic Algorithms by Leon Bottou)

Assume that training set contains 10 copies of the 100 same examples
BATCH: Blindly computes redundant gradients, 1 epoch on large set \equiv 1 epoch on small set without redundant
Online: Take advantage of redundancy
###############################################################

Our idea:

Combine online learning and model averagining

1- Train a model for 1 epoch say
2- Retrain this model on selected sets of data points.
3. Carry out 2 several times
4. Merge retrained model into one and go to (1)
Applying KvsAll on two consecutive example :  [ (h, r)_i + |E| ] + [ (h, r)_j + |E| ]

"""
from core import KGE
from core.knowledge_graph import KG
import random
import time

# (1) Load a pretrained model (say we ran it only for a single epoch). YAGO
pre_trained_kge = KGE(path_of_pretrained_model_dir='Experiments/2022-03-26 15:54:06.034352', construct_ensemble=False)


def select_cbd_entities_given_rel_and_obj():
    relation = "wasBornIn"
    object = "Detroit"
    # Select x:= {x | (x,wasBornIn,y} or (y,wasBornIn,x}
    id_rel = pre_trained_kge.relation_to_idx.loc[relation].values[0]
    id_tail = pre_trained_kge.entity_to_idx.loc[object].values[0]
    id_heads = pre_trained_kge.train_set[
        (pre_trained_kge.train_set['relation'] == id_rel) & (pre_trained_kge.train_set['object'] == id_tail)]['subject']
    entities = {pre_trained_kge.entity_to_idx.iloc[i].name for i in id_heads.to_list()}
    start_time = time.time()
    for ith, entity in enumerate(entities):
        pre_trained_kge.train_cbd(head_entity=[entity], iteration=1, lr=.01)
        print(entity)
        print(pre_trained_kge.predict_topk(head_entity=[entity], relation=[relation], k=10))
        if ith == 1:
            break
    print(f'Online KGE training took {time.time() - start_time}')


def select_entities_given_rel():
    relation = 'wasBornIn'
    # Select x:= {x | (x,wasBornIn,y} or (y,wasBornIn,x}
    id_rel = pre_trained_kge.relation_to_idx.loc[relation].values[0]
    x = pre_trained_kge.train_set[pre_trained_kge.train_set['relation'] == id_rel]
    entities = {pre_trained_kge.entity_to_idx.iloc[i].name for i in x['subject'].to_list() + x['object'].to_list()}
    start_time = time.time()
    for ith, entity in enumerate(entities):
        pre_trained_kge.train_cbd(head_entity=[entity], iteration=1, lr=.01)
        print(entity)
        print(pre_trained_kge.predict_topk(head_entity=[entity], relation=[relation], k=10))
        if ith == 1:
            break
    print(f'Online KGE training took {time.time() - start_time}')


# (2) Sample random entities
def train_on_randomly_sampled_entities():
    start_time = time.time()
    for ith, entity in enumerate(
            pre_trained_kge.entity_to_idx.sample(1, random_state=random.randint(1, 1_000_000)).index):
        # Retrieve relations occurring with (2))
        relations = pre_trained_kge.get_relations(entity)
        print(f'Example:{ith}\t Learning over {entity} and {len(relations)} relations')
        # z:= {(h,r), y)_i }_i ^n s.t. n:=| {r | (h,r,x) OR (x,r,h) \in G }|
        for relation in relations:
            pre_trained_kge.train_k_vs_all(head_entity=[entity], relation=[relation], iteration=1, lr=.0001)
            # print(pre_trained_kge.predict_topk(head_entity=[entity], relation=[relation], k=10))

    print(f'Online KGE training took {time.time() - start_time}')
    # Save model
    # pre_trained_kge.save()


def triple_score_update():
    # Wrong triple
    s = pre_trained_kge.triple_score(head_entity=["female"], relation=['hasGender'], tail_entity=["Marie_of_Romania"])

    pre_trained_kge.train_triples(head_entity=["female"],
                                  relation=['hasGender'],
                                  tail_entity=["Marie_of_Romania"],
                                  iteration=1,
                                  lr=.001,
                                  labels=[0.0])
    y = pre_trained_kge.triple_score(head_entity=["female"], relation=['hasGender'], tail_entity=["Marie_of_Romania"])

    assert s > y


def selected_cbd_learning():
    pre_trained_kge.train_cbd(head_entity=["Marie_of_Romania"], iteration=1, lr=.001)


select_entities_given_rel()
train_on_randomly_sampled_entities()
selected_cbd_learning()
triple_score_update()
triple_score_update()
selected_cbd_learning()
triple_score_update()
select_cbd_entities_given_rel_and_obj()
