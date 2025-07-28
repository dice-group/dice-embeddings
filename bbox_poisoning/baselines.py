import random
from collections import defaultdict

def poison_random(triples, k, corruption_type):
    entity_list = list(set([h for h, _, _ in triples] + [t for _, _, t in triples]))
    relation_list = list(set([r for _, r, _ in triples]))

    corrupted_triples = []
    triples_before_corruption = []

    for i in range(k):
        triple =  random.choice(triples)
        h, r, t = triple

        if corruption_type == 'all':
            corrupt_h = random.choice([i for i in entity_list if i != h])
            corrupt_r = random.choice([i for i in relation_list if i != r])
            corrupt_t = random.choice([i for i in entity_list if i != t])
            corrupted = (corrupt_h, corrupt_r, corrupt_t)
        if corruption_type == 'head':
            corrupt_h = random.choice([i for i in entity_list if i != h])
            corrupted = (corrupt_h, r, t)
        if corruption_type == 'rel':
            corrupt_r = random.choice([i for i in relation_list if i != r])
            corrupted = (h, corrupt_r, t)
        if corruption_type == 'tail':
            corrupt_t = random.choice([i for i in entity_list if i != t])
            corrupted = (h, r, corrupt_t)
        if corruption_type == 'head-tail':
            corrupt_h = random.choice([i for i in entity_list if i != h])
            corrupt_t = random.choice([i for i in entity_list if i != t])
            corrupted = (corrupt_h, r, corrupt_t)

        corrupted_triples.append(corrupted)
        triples_before_corruption.append(triple)

    remaining_triples = [item for item in triples if item not in triples_before_corruption]

    output  = remaining_triples + corrupted_triples

    return output, remaining_triples, corrupted_triples

"""

def poison_random(triples, k):
    entities = list(set([h for h, _, _ in triples] + [t for _, _, t in triples]))
    relations = list(set([r for _, r, _ in triples]))

    perturbs = []

    while len(perturbs) < k:
        corrupt_h = random.choice(entities)
        corrupt_r = random.choice(relations)
        corrupt_t = random.choice(entities)
        corrupted = (corrupt_h, corrupt_r, corrupt_t)

        if corrupted not in triples and corrupt_h != corrupt_t:
            perturbs.append(corrupted)

    return perturbs
"""
def poison_centrality(triples, k, corruption_type, entity_emb, relation_emb):
    entity_list = list(entity_emb.keys())
    relation_list = list(relation_emb.keys())

    entity_degree = defaultdict(int)
    for h, _, t in triples:
        entity_degree[h] += 1
        entity_degree[t] += 1

    triple_scores = []
    for triple in triples:
        h, _, t = triple
        score = entity_degree[h] + entity_degree[t]
        triple_scores.append((score, triple))

    #triple_scores.sort(key=lambda x: x[0], reverse=False) for low scores, which is ineffective
    triple_scores.sort(key=lambda x: x[0], reverse=True)
    selected_top_k = [t for _, t in triple_scores[:k]]

    after_edits = []
    before_edits = []

    for h, r, t in selected_top_k:
        for _ in range(10):  # Try up to 10 times to get a valid corruption
            if corruption_type == 'all':
                corrupt_h = random.choice(entity_list)
                corrupt_r = random.choice(relation_list)
                corrupt_t = random.choice(entity_list)
                corrupted = (corrupt_h, corrupt_r, corrupt_t)
            elif corruption_type == 'head':
                corrupt_h = random.choice(entity_list)
                corrupted = (corrupt_h, r, t)
            elif corruption_type == 'rel':
                corrupt_r = random.choice(relation_list)
                corrupted = (h, corrupt_r, t)
            elif corruption_type == 'tail':
                corrupt_t = random.choice(entity_list)
                corrupted = (h, r, corrupt_t)
            elif corruption_type == 'head-tail':
                corrupt_h = random.choice(entity_list)
                corrupt_t = random.choice(entity_list)
                corrupted = (corrupt_h, r, corrupt_t)
            else:
                raise ValueError(f"Unknown corruption type: {corruption_type}")

            if corrupted != (h, r, t) and corrupted not in triples:
                after_edits.append(corrupted)
                before_edits.append((h, r, t))
                break

    return after_edits, before_edits
