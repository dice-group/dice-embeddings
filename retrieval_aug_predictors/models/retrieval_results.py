from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.utils import jaccard_similarity, f1_set_similarity, concept_reducer, concept_reducer_properties
from owlapy.class_expression import (
    OWLObjectUnionOf,
    OWLObjectIntersectionOf,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
    OWLObjectOneOf,
)
import time
from typing import Tuple, Set
import pandas as pd
from owlapy import owl_expression_to_dl
from itertools import chain
from argparse import ArgumentParser
import os
from tqdm import tqdm
import random
import itertools
import ast

def execute(args):
    # (1) Initialize knowledge base.
    assert os.path.isfile(args.path_kg)
    symbolic_kb = KnowledgeBase(path=args.path_kg)
    random.seed(args.seed)

    ###################################################################
    # GENERATE DL CONCEPTS TO EVALUATE RETRIEVAL PERFORMANCES
    # (3) R: Extract object properties.
    object_properties = sorted({i for i in symbolic_kb.get_object_properties()})

    # (3.1) Subsample if required.
    if args.ratio_sample_object_prop and len(object_properties) > 0:
        object_properties = {i for i in random.sample(population=list(object_properties),
                                                      k=max(1, int(len(
                                                          object_properties) * args.ratio_sample_object_prop)))}

    object_properties = set(object_properties)

    # (4) R⁻: Inverse of object properties.
    object_properties_inverse = {i.get_inverse_property() for i in object_properties}

    # (5) R*: R UNION R⁻.
    object_properties_and_inverse = object_properties.union(object_properties_inverse)
    # (6) NC: Named owl concepts.
    nc = sorted({i for i in symbolic_kb.get_concepts()})

    if args.ratio_sample_nc and len(nc) > 0:
        # (6.1) Subsample if required.
        nc = {i for i in random.sample(population=list(nc), k=max(1, int(len(nc) * args.ratio_sample_nc)))}

    nc = set(nc)  # return to a set
    # (7) NC⁻: Complement of NC.
    nnc = {i.get_object_complement_of() for i in nc}

    # (8) NC*: NC UNION NC⁻.
    nc_star = nc.union(nnc)
    # (9) Retrieve 10 random Nominals.
    if len(set(symbolic_kb.individuals())) > args.num_nominals:
        nominals = set(random.sample(set(symbolic_kb.individuals()), args.num_nominals))
    else:
        nominals = symbolic_kb.individuals()
    # (10) All combinations of 3 for Nominals, e.g. {martin, heinz, markus}
    nominal_combinations = set(OWLObjectOneOf(combination) for combination in itertools.combinations(nominals, 3))

    # (11) NC UNION NC.
    unions = concept_reducer(nc, opt=OWLObjectUnionOf)
    # (12) NC INTERSECTION NC.
    intersections = concept_reducer(nc, opt=OWLObjectIntersectionOf)
    # (13) NC* UNION NC*.
    unions_nc_star = concept_reducer(nc_star, opt=OWLObjectUnionOf)
    # (14) NC* INTERACTION NC*.
    intersections_nc_star = concept_reducer(nc_star, opt=OWLObjectIntersectionOf)
    # (15) \exist r. C s.t. C \in NC* and r \in R* .
    exist_nc_star = concept_reducer_properties(
        concepts=nc_star,
        properties=object_properties_and_inverse,
        cls=OWLObjectSomeValuesFrom,
    )
    # (16) \forall r. C s.t. C \in NC* and r \in R* .
    for_all_nc_star = concept_reducer_properties(
        concepts=nc_star,
        properties=object_properties_and_inverse,
        cls=OWLObjectAllValuesFrom,
    )
    # (17) >= n r. C  and =< n r. C, s.t. C \in NC* and r \in R* .
    min_cardinality_nc_star_1, min_cardinality_nc_star_2, min_cardinality_nc_star_3 = (
        concept_reducer_properties(
            concepts=nc_star,
            properties=object_properties_and_inverse,
            cls=OWLObjectMinCardinality,
            cardinality=i,
        )
        for i in [1, 2, 3]
    )
    max_cardinality_nc_star_1, max_cardinality_nc_star_2, max_cardinality_nc_star_3 = (
        concept_reducer_properties(
            concepts=nc_star,
            properties=object_properties_and_inverse,
            cls=OWLObjectMaxCardinality,
            cardinality=i,
        )
        for i in [1, 2, 3]
    )
    # (18) \exist r. Nominal s.t. Nominal \in Nominals and r \in R* .
    exist_nominals = concept_reducer_properties(
        concepts=nominal_combinations,
        properties=object_properties_and_inverse,
        cls=OWLObjectSomeValuesFrom,
    )

    ###################################################################

    # Retrieval Results
    def concept_retrieval(retriever_func, c) -> Tuple[Set[str], float]:
        start_time = time.time()
        return {i.str for i in retriever_func.individuals(c)}, time.time() - start_time

    # () Collect the data.
    data = []
    # () Converted to list so that the progress bar works.
    concepts = list(
        chain(
            nc,  # named concepts          (C)
            nnc,  # negated named concepts  (\neg C)
            unions_nc_star,  # A set of Union of named concepts and negat
            intersections_nc_star,  #
            exist_nc_star,
            for_all_nc_star,
            min_cardinality_nc_star_1, min_cardinality_nc_star_1, min_cardinality_nc_star_3,
            max_cardinality_nc_star_1, max_cardinality_nc_star_2, max_cardinality_nc_star_3,
            exist_nominals))
    print("\n")
    print("#" * 50)
    print("Description of generated Concepts")
    print(f"NC denotes the named concepts\t|NC|={len(nc)}")
    print(f"NNC denotes the negated named concepts\t|NNC|={len(nnc)}")
    print(f"|NC UNION NC|={len(unions)}")
    print(f"|NC Intersection NC|={len(intersections)}")

    print(f"NC* denotes the union of named concepts and negated named concepts\t|NC*|={len(nc_star)}")
    print(f"|NC* UNION NC*|={len(unions_nc_star)}")
    print(f"|NC* Intersection NC*|={len(intersections_nc_star)}")
    print(f"|exist R* NC*|={len(exist_nc_star)}")
    print(f"|forall R* NC*|={len(for_all_nc_star)}")

    print(
        f"|Max Cardinalities|={len(max_cardinality_nc_star_1) + len(max_cardinality_nc_star_2) + len(max_cardinality_nc_star_3)}")
    print(
        f"|Min Cardinalities|={len(min_cardinality_nc_star_1) + len(min_cardinality_nc_star_1) + len(min_cardinality_nc_star_3)}")
    print(f"|exist R* Nominals|={len(exist_nominals)}")
    print("#" * 50, end="\n\n")

    # () Shuffled the data so that the progress bar is not influenced by the order of concepts.

    random.shuffle(concepts)
    # check if csv arleady exists and delete it cause we want to override it
    if os.path.exists(args.path_report):
        os.remove(args.path_report)
    file_exists = False
    # () Iterate over single OWL Class Expressions in ALCQIHO
    for expression in (tqdm_bar := tqdm(concepts, position=0, leave=True)):
        retrieval_y: Set[str]
        runtime_y: Set[str]
        # () Retrieve the true set of individuals and elapsed runtime.
        retrieval_y, runtime_y = concept_retrieval(symbolic_kb, expression)
        # () Retrieve a set of inferred individuals and elapsed runtime.
        # retrieval_neural_y, runtime_neural_y = concept_retrieval(neural_owl_reasoner, expression)
        # () Compute the Jaccard similarity.
        # jaccard_sim = jaccard_similarity(retrieval_y, retrieval_neural_y)
        # () Compute the F1-score.
        # f1_sim = f1_set_similarity(retrieval_y, retrieval_neural_y)
        # () Store the data.
        df_row = pd.DataFrame(
            [{
                "Expression": owl_expression_to_dl(expression),
                "Type": type(expression).__name__,
                "Symbolic_Retrieval": retrieval_y
            }])
        # Append the row to the CSV file
        df_row.to_csv(args.path_report, mode='a', header=not file_exists, index=False)
        file_exists = True
        # () Update the progress bar.
    # () Read the data into pandas dataframe
    df = pd.read_csv(args.path_report, index_col=0, converters={'Symbolic_Retrieval': lambda x: ast.literal_eval(x),
                                                                'Symbolic_Retrieval_Neural': lambda x: ast.literal_eval(
                                                                    x)})

    # () Group by the type of OWL concepts
    df_g = df.groupby(by="Type")
    print(df_g["Type"].count())


def get_default_arguments():
    parser = ArgumentParser()
    parser.add_argument("--path_kg", type=str, default="/home/alkid/PycharmProjects/dice-embeddings/KGs/Family/father.owl")
    parser.add_argument("--path_kge_model", type=str, default=None)
    parser.add_argument("--endpoint_triple_store", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ratio_sample_nc", type=float, default=0.2, help="To sample OWL Classes.")
    parser.add_argument("--ratio_sample_object_prop", type=float, default=0.1, help="To sample OWL Object Properties.")
    parser.add_argument("--min_jaccard_similarity", type=float, default=0.0, help="Minimum Jaccard similarity to be achieve by the reasoner")
    parser.add_argument("--num_nominals", type=int, default=10, help="Number of OWL named individuals to be sampled.")

    # H is obtained if the forward chain is applied on KG.
    parser.add_argument("--path_report", type=str, default="ALCQHI_Retrieval_Results.csv")
    return parser.parse_args()

if __name__ == "__main__":
    execute(get_default_arguments())