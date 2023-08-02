import sys, os

sys.path.append(os.getcwd())

from dicee.knowledge_graph_embeddings import KGE
from dicee.knowledge_graph import KG
import random
import typing


def read_triples_from_file(
    file_path: str, num_of_triples: int = 0
) -> typing.List[tuple]:
    triples_list = []
    counter = 0
    if num_of_triples == 0:
        not_use_random = True
    else:
        not_use_random = False

    with open(file_path, "r") as file:
        for line in file:
            if num_of_triples != 0 and counter == num_of_triples:
                break

            rand_num = random.randint(1, 10)
            if rand_num > 5 or not_use_random:
                triple = line.strip().split()
                if len(triple) == 3:
                    triples_list.append(tuple(triple))
                    counter += 1
                else:
                    print(f"Skipping line: {line.strip()} (not a valid triple)")

    return triples_list


if __name__ == "__main__":
    """
    To use lp_evaluate(), you need to create the instances of class KGE
    and class KG(to get the er_vocab,re_vocab and ee_vocab).
    Triples are randomly chosen from the test dataset file (by default all triples of given file).
    """
    # path = "E:\\DICEE\\dice-embeddings\\Experiments\\2023-08-02 12-38-15.345656"
    path = "E:\\DICEE\\dice-embeddings\\Experiments\\2023-08-02 14-52-57.261661"
    path_dataset_folder = "./KGs/UMLS"

    pre_trained_kge = KGE(path=path)
    kg = KG(
        data_dir=path_dataset_folder,
        path_for_deserialization=path,
        path_for_serialization=path,
        eval_model="train_val_test",
    )

    file_path = "./KGs/UMLS/test.txt"  # Replace this with the actual path to your file
    triples = read_triples_from_file(
        file_path
    )  # by default all the triples of the given file are used

    pre_trained_kge.lp_evaluate(triples, kg, filtered=True)
