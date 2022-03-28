import argparse
from core import KGE
from core.knowledge_graph import KG


def select_cbd_entities_given_rel_and_obj(args):
    pre_trained_kge = KGE(path_of_pretrained_model_dir=args.path_of_pretrained_model_dir, construct_ensemble=True)
    id_rel = pre_trained_kge.relation_to_idx.loc[args.relation].values[0]
    id_tail = pre_trained_kge.entity_to_idx.loc[args.object].values[0]
    id_heads = pre_trained_kge.train_set[
        (pre_trained_kge.train_set['relation'] == id_rel) & (pre_trained_kge.train_set['object'] == id_tail)]['subject']
    entities = {pre_trained_kge.entity_to_idx.iloc[i].name for i in id_heads.to_list()}
    del id_heads, id_rel, id_tail
    print('Num of selected entities: ', len(entities))
    for ith, entity in enumerate(entities):
        print('#' * 10)
        print(f"{ith}.th example {entity}", end="\t")
        pre_trained_kge.train_cbd(head_entity=[entity], iteration=args.iteration, lr=args.lr)
        if ith == args.num_example:
            break

    pre_trained_kge.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--path_of_pretrained_model_dir', type=str, default="OnlineDBQMult")
    parser.add_argument("--relation", default="http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    parser.add_argument("--object", type=str, default="http://dbpedia.org/ontology/Scientist")
    parser.add_argument("--num_example", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--lr", type=float, default=.01)
    select_cbd_entities_given_rel_and_obj(parser.parse_args())
