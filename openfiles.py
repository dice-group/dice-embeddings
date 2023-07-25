from pprint import pprint
import pickle

with open('/Users/sourabh/dice-embeddings/data/UMLS/ent2id.pkl', "rb") as f:
    ent2id = pickle.load(f)
with open('/Users/sourabh/dice-embeddings/data/UMLS/rel2id.pkl', "rb") as f:
    rel2id = pickle.load(f)

# with open('/Users/sourabh/Desktop/negCQD/creating queries/data/UMLS/test-queries.pkl', "rb") as f:
#     train_q = pickle.load(f)
# with open('/Users/sourabh/Desktop/negCQD/creating queries/data/UMLS/test-easy-answers.pkl', "rb") as f:
#     train_qa = pickle.load(f)
# with open('/Users/sourabh/Desktop/negCQD/creating queries/data/UMLS/test-hard-answers.pkl', "rb") as f:
#     train_ha = pickle.load(f)


with open('/Users/sourabh/dice-embeddings/data/UMLS/test-inp-queries.pkl', "rb") as f:
    train_p = pickle.load(f)
with open('/Users/sourabh/dice-embeddings/data/UMLS/test-inp-tp-answers.pkl', "rb") as f:
    train_pa = pickle.load(f)
with open('/Users/sourabh/dice-embeddings/data/UMLS/test-inp-fn-answers.pkl', "rb") as f:
    train_ppa = pickle.load(f)
with open('/Users/sourabh/dice-embeddings/data/UMLS/test-inp-unmapped-queries.pkl', "rb") as f:
    train_q = pickle.load(f)
with open('/Users/sourabh/dice-embeddings/data/UMLS/test-inp-tp-unmapped-answers.pkl', "rb") as f:
    train_qa = pickle.load(f)
with open('/Users/sourabh/dice-embeddings/data/UMLS/test-inp-fn-unmapped-answers.pkl', "rb") as f:
    train_qqa = pickle.load(f)
pprint(train_p)
pprint(train_pa)
pprint(train_ppa)

print(ent2id)
print(rel2id)

pprint(train_q)
pprint(train_qa)
pprint(train_qqa)
# 4,
#                                17,
#                                18,
#                                29,
#                                32,
#                                33,
#                                39,
#                                40,
#                                56,
#                                59,
#                                72,
#                                76,
#                                82,
#                                86,
#                                89,
#                                94,
#                                111,
#                                113,
#                                117,
#                                128},
# 'alga',
#                                                                    'amphibian',
#                                                                    'animal',
#                                                                    'archaeon',
#                                                                    'bacterium',
#                                                                    'bird',
#                                                                    'body_part_organ_or_organ_component',
#                                                                    'cell',
#                                                                    'cell_component',
#                                                                    'fish',
#                                                                    'fungus',
#                                                                    'human',
#                                                                    'invertebrate',
#                                                                    'mammal',
#                                                                    'organism',
#                                                                    'reptile',
#                                                                    'rickettsia_or_chlamydia',
#                                                                    'tissue',
#                                                                    'vertebrate',
#                                                                    'virus'
#('molecular_biology_research_technique', ('method_of', 'assesses_effect_of')), ('laboratory_or_test_result', ('measurement_of',))), (('disease_or_syndrome', ('conceptually_related_to', 'process_of')), ('clinical_attribute', ('property_of',))), (('finding', ('manifestation_of', 'isa')), ('mental_process', ('isa',))), (('organism_function', ('result_of', 'isa')), ('human_caused_phenomenon_or_process', ('result_of',))), (('receptor', ('complicates', 'process_of')), ('organism_attribute', ('property_of',))), (('laboratory_procedure', ('measures', 'interacts_with')), ('diagnostic_procedure', ('measures',))), (('body_location_or_region', ('adjacent_to', 'location_of')), ('body_part_organ_or_organ_component', ('location_of',))), (('organism_function', ('co-occurs_with', 'process_of')), ('congenital_abnormality', ('part_of',))), (('injury_or_poisoning', ('result_of', 'measures')), ('laboratory_procedure', ('measures',))), (('biologic_function', ('process_of', 'result_of')), ('mental_or_behavioral_dysfunction', ('result_of',)))}})