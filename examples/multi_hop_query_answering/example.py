# pip install dicee
# wget https://hobbitdata.informatik.uni-leipzig.de/KG/KGs.zip --no-check-certificate & unzip KGs.zip
from dicee.executer import Execute
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE
from dicee import QueryGenerator
from dicee.static_funcs import evaluate
from dicee.static_funcs import load_pickle
from dicee.static_funcs import load_json


# Set up parameters for training a KGE model
args = Namespace()

# Specify KGE model, optimizer, and scoring technique
args.model = 'Keci'
args.optim = 'Adam'
args.scoring_technique = "AllvsAll"

# Provide path to the knowledge graph file in RDF format
args.path_single_kg = "KGs/Family/family-benchmark_rich_background.owl"

# Choose RDF library backend(eg: rdflib or pandas)
args.backend = "rdflib"

# Set training epochs, batch size, learning rate, and embedding dimension
args.num_epochs = 200
args.batch_size = 1024
args.lr = 0.1
args.embedding_dim = 512

# Train the KGE model
result = Execute(args).start()

# Load the Model
pre_trained_kge = KGE(path=result['path_experiment_folder'])


#IMPLEMENTING MULTI-HOP QUERIES

# Question 1: Who are the siblings of F9M167?
# Query: ?E : \exist E.hasSibling(E, F9M167)
# Answer: [F9M157, F9F141], as (F9M167, hasSibling, F9M157) and (F9M167, hasSibling, F9F141)
predictions = pre_trained_kge.answer_multi_hop_query(query_type="1p",
                                                     query=('http://www.benchmark.org/family#F9M167',
                                                            ('http://www.benchmark.org/family#hasSibling',)),
                                                     tnorm="min", k=3)
top_entities = [topk_entity for topk_entity, query_score in predictions]


# (1) Who are the siblings of F9M167 ? => F9M167 hasSibling [F9M157, F9F141]
assert "http://www.benchmark.org/family#F9F141" in top_entities
assert "http://www.benchmark.org/family#F9M157" in top_entities



# Question 2: To whom a sibling of F9M167 is married to?
# Query: ?D : \exist E.Married(D, E) \land hasSibling(E, F9M167)
# Answer: [F9F158, F9M142] as (F9M157 #married F9F158) and (F9F141 #married F9M142)
predictions = pre_trained_kge.answer_multi_hop_query(query_type="2p",
                                                     query=("http://www.benchmark.org/family#F9M167",
                                                            ("http://www.benchmark.org/family#hasSibling",
                                                             "http://www.benchmark.org/family#married")),
                                                     tnorm="min", k=3)
top_entities = [topk_entity for topk_entity, query_score in predictions]
assert "http://www.benchmark.org/family#F9M142" in top_entities
assert "http://www.benchmark.org/family#F9F158" in top_entities


# Question 3: What are the type of people who are married to a sibling of F9M167?
# Query: ?T : \exist D.type(D,T) \land Married(D,E) \land hasSibling(E, F9M167)
# Answer : [('http://www.benchmark.org/family#Person', tensor(0.9999)),
#           ('http://www.benchmark.org/family#Male', tensor(0.9999)),
#           ('http://www.benchmark.org/family#Father', tensor(0.9999))]


#(3) Third hop info.
#    - F9M157 is [Brother Father Grandfather Male]
#    - F9M142 is [Male Grandfather Father]

predictions = pre_trained_kge.answer_multi_hop_query(query_type="3p", query=("http://www.benchmark.org/family#F9M167",
                                                                             (
                                                                             "http://www.benchmark.org/family#hasSibling",
                                                                             "http://www.benchmark.org/family#married",
                                                                             "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")),
                                                     tnorm="min", k=8)
top_entities = [topk_entity for topk_entity, query_score in predictions]
print(top_entities)
assert "http://www.benchmark.org/family#Person" in top_entities
assert "http://www.benchmark.org/family#Father" in top_entities
assert "http://www.benchmark.org/family#Male" in top_entities
