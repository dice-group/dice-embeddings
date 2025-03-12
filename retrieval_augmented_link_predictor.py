"""
# pip install openai
# Data Preparation
# Read a knowledge graph (Countries)

Let g:(train,val,test) be a tuple of three knowledge graphs
For g[i] \in E x R x E, where
E denotes a set of entities
R denotes a set of relations

# Get the test dataset
train : List[Tuple[str,str,str]] = g[0]
test  : List[Tuple[str,str,str]] = g[2]

1. Move to train into directed graph of networkx (https://networkx.org/documentation/stable/index.html) or igraph (https://python.igraph.org/en/stable/)
Although this is not necessary, they implement few functions that we would like to use in the next steps.


# Link Prediction

Let (h,r,t) be a test triple

## Predicting missing tail
Given (h,r) rank elements of E in te descending order of their relevance.

#### Getting information about an entity (h)

1. Getting k order neighbors of an entity
Let n_h := {(s,p,o)} denote a set of triples from the train set, where h==s or h==o.
n_h denotes the first order neighborhood (see https://python.igraph.org/en/stable/analysis.html#neighborhood)
we can extend this into k>1 to get a subgraph that is "about h".


2. Getting k order neighbors of a relation
Let m_r := {(s,p,o)} denote a set of triples from the train set, where p==r.
Similarly,
- m_h denotes the first order neighborhood of r
- we can extend this into k>1 to get a subgraph that is "about r".

For the time being, assume that k=3.

3. Assigning scores to entities based on information derived from (1) and (2)
Let G_hr denote a set of triples derived from (1) and (2)
Let E_hr denote a set of filtered entities (a concept from the link prediction evaluation)

3.1.Write a prompt based on G_hr, and E_hr so that LLM generates scores for each item of E_hr

3.2. We are done :)


"""

