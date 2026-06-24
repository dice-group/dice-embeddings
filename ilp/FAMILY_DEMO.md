# The Family Demo — why inductive link prediction matters



```bash
pytest ilp/tests/test_inductive_family.py -v -s
```


## The story in one sentence

> Teach the model the rule *"a parent is older than their child"* on one set of
> families, then watch it apply that rule correctly to **brand-new families it
> has never seen** — people whose names never appeared in training.

## The rule

Every family in the synthetic knowledge graph obeys a simple logical rule:

```
parentOf(X, Y)        ⟹   olderThan(X, Y)
grandparentOf(X, Y)   ⟹   olderThan(X, Y)
```

The training graph states the *full* family: the structure (`parentOf`,
`grandparentOf`, `marriedTo`, `siblingOf`) **and** the `olderThan` facts. So the
association `parentOf → olderThan` is learnable from the data.

## Step by step

1. **Generate a training knowledge graph.** ~100 three-generation families,
   each with complete structure and `olderThan` facts. People get opaque,
   globally-unique ids (`fam0007_p03`), and families never share a person.

2. **Train the inductive model.** Crucially, every person is anonymized to a
   random `[Z_*]` slot *at every sample draw* (see `dataset.py` /
   `vocab.py`). The embedding table contains only schema tokens and the
   anonymous `[Z_*]` pool — **never a row for any specific person**. The model
   therefore *cannot* memorize "`fam0007_p03` is old"; it can only learn the
   structural pattern: *whoever sits on the `parentOf` source side is the one
   who is `olderThan`.*

3. **Evaluate on a disjoint dataset shaped as `(context, query)` tuples.** Each
   tuple is a new family:
   - `context_triples`: the family's structure only (`parentOf`, `marriedTo`,
     …). **No `olderThan`.**
   - `query_triple`: one held-out `olderThan(parent, child)` fact.

   The eval families use offset ids so **no person overlaps training** — unseen
   entities, but the *same* relations.

4. **Score both directions.** For each query the test asks the model two
   questions over the family's context graph:
   - `P(parent olderThan child)` — should be **high**
   - `P(child olderThan parent)` — should be **low**

   The model reads the context, finds `parentOf(parent, child)`, and applies the
   transferred rule. The test asserts directional accuracy (parent scored older
   than child) on these never-seen people.

Because the people are anonymized and disjoint from training, getting the
direction right is only possible if the model learned a **transferable
relational rule**, not memorized facts. That is the definition of *inductive*
link prediction.

## Why a transductive embedding model cannot do this

Classical knowledge-graph embedding models, TransE, DistMult, ComplEx, RotatE,
and the like, are **transductive**: they learn one fixed embedding vector per
entity *string* and per relation seen during training. Scoring a triple
`(h, r, t)` is a function of the looked-up vectors `e_h`, `e_r`, `e_t`.

That design fails this task:

1. **No vectors for unseen entities.** Every person in the eval families is new.
   A transductive model has **no embedding** for `fam10007_p03`, there is
   simply no row in its entity table. It cannot even form a score for the query,
   let alone a correct one. 



