import pickle
import click
from collections import defaultdict
import os
import glob

def create_mappings(datapath):
    ent_set = set()
    rel_set = set()
    with open(os.path.join(datapath,"train.txt"), "r") as f:
        for line in f.readlines():
            # Split the line and extract entities and relationships
            ent1, rel, ent2 = line.strip().split('\t')

            # Add entities and relationships to their respective sets
            ent_set.add(ent1)
            ent_set.add(ent2)
            rel_set.add(rel)

    # Create ent2id and rel2id dictionaries
    ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
    rel2id = {rel: idx for idx, rel in enumerate(rel_set)}

    # Save ent2id and rel2id dictionaries to pickle files
    with open(os.path.join(datapath,"ent2id.pkl"), "wb") as f:
        pickle.dump(ent2id, f)

    with open(os.path.join(datapath,"rel2id.pkl"), "wb") as f:
        pickle.dump(rel2id, f)


def joinqueries(datapath,file_type):
    if file_type == "mapped":
        file_pattern = 'test-*-queries.pkl'

    elif file_type == "unmapped":
        file_pattern = 'test-*-unmapped-queries.pkl'

    else:
        raise ValueError("Invalid file_type. Must be 'mapped' or 'unmapped'")

    query_files = glob.glob(os.path.join(datapath, file_pattern))
    combined_queries ={}
    for file_path in query_files:
        with open(file_path, 'rb') as f:
            file_data = pickle.load(f)
        combined_queries.update(file_data)
        #os.remove(file_path)  # Remove the intermediate file
    with open(os.path.join(datapath, f'test-queries.pkl'), 'wb') as f:
        pickle.dump(combined_queries, f)

    # Combine easy and hard answers
    for answer_type in ['tp', 'fn']:
            combined_answer_type = "easy" if answer_type == "tp" else "hard"
            answer_files = glob.glob(os.path.join(datapath, f'test-*-{answer_type}-{file_type}-answers.pkl'))
            combined_answers = {}
            for file_path in answer_files:
                with open(file_path, 'rb') as f:
                    file_data = pickle.load(f)
                combined_answers.update(file_data)
                #os.remove(file_path)  # Remove the intermediate file
            with open(os.path.join(datapath, f'test-{combined_answer_type}-answers.pkl'), 'wb') as f:
                pickle.dump(combined_answers, f)
    print('queries joined')


def mapid(datapath):
    with open(os.path.join(datapath, 'ent2id.pkl'), "rb") as f:
        ent2id = pickle.load(f)
    with open(os.path.join(datapath, 'rel2id.pkl'), "rb") as f:
        rel2id = pickle.load(f)

    for file in ["train", "valid", "test"]:
        filepath = os.path.join(datapath, f"{file}.txt")
        filepath2 = os.path.join(datapath, f"{file}_id.txt")

        with open(filepath, "r") as in_file, open(filepath2, "w") as out_file:
            for line in in_file:
                ent1, rel, ent2 = line.strip().split('\t')
                ent1_id = ent2id[ent1]
                rel_id = rel2id[rel]
                ent2_id = ent2id[ent2]
                out_file.write(f"{ent1_id}\t{rel_id}\t{ent2_id}\n")


def unmap(datapath,query_type,query_structures):
    # Load ent2id dictionary from pickle file
    with open(os.path.join(datapath,"ent2id.pkl"), "rb") as f:
        ent2id = pickle.load(f)

    with open(os.path.join(datapath,"rel2id.pkl"), "rb") as f:
        rel2id = pickle.load(f)

    # Create id2ent dictionary
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}

    query_name_dict = {

        ("e", ("r",)): "1p",
        ("e", ("r", "r")): "2p",
        ("e", ("r", "r", "r",),): "3p",
        (("e", ("r",)), ("e", ("r",))): "2i",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3i",
        ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
        (("e", ("r", "r")), ("e", ("r",))): "pi",
        # negation
        (("e", ("r",)), ("e", ("r", "n"))): "2in",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))): "3in",
        ((("e", ("r",)), ("e", ("r", "n"))), ("r",)): "inp",
        (("e", ("r", "r")), ("e", ("r", "n"))): "pin",
        (("e", ("r", "r", "n")), ("e", ("r",))): "pni",

        # union
        (("e", ("r",)), ("e", ("r",)), ("u",)): "2u",
        ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)): "up",

    }
    name_query_dict = {value: key for key, value in query_name_dict.items()}
    # query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']
    # query_structure_tuple=name_query_dict[query_structure]
    for query_structure in query_structures:
        # Load queries
        with open(f"{datapath}/{query_type}-{query_structure}-queries.pkl", "rb") as f:
            queries = pickle.load(f)

        # Unmap queries and create a mapping from ID-based queries to text-based queries
        unmapped_queries_dict=defaultdict(set)
        query_id_to_text = {}
        for query_structure_tuple, query_set in queries.items():

            for query in query_set:
                unmapped_query = unmap_query(query_structure_tuple, query, id2ent, id2rel)
                unmapped_queries_dict[query_structure_tuple].add(unmapped_query)
                query_id_to_text[query] = unmapped_query

        # Save unmapped queries
        with open(f"{datapath}/{query_type}-{query_structure}-unmapped-queries.pkl", "wb") as f:
            pickle.dump(unmapped_queries_dict, f)

        # Loop through both answer types
        answer_types = ['tp', 'fn']
        for answer_type in answer_types:
            # Load answers
            with open(f"{datapath}/{query_type}-{query_structure}-{answer_type}-answers.pkl", "rb") as f:
                answers = pickle.load(f)

            # Unmap answers and update keys using the created mapping
            unmapped_answers_dict = defaultdict(set)
            for query, answer_set in answers.items():
                unmapped_answer_set = set()
                for answer in answer_set:
                    unmapped_answer = id2ent[answer]
                    unmapped_answer_set.add(unmapped_answer)
                unmapped_answers_dict[query_id_to_text[query]]=unmapped_answer_set

            # Save unmapped answers
            with open(f"{datapath}/{query_type}-{query_structure}-{answer_type}-unmapped-answers.pkl", "wb") as f:
                pickle.dump(unmapped_answers_dict, f)
        print('queries unmapped with structure %s' % (query_structure))
def unmap_query(query_structure, query, id2ent, id2rel):
    #2i
    if query_structure == (("e", ("r",)), ("e", ("r",))) :
        ent1, (rel1_id,) = query[0]
        ent2, (rel2_id,) = query[1]
        ent1 = id2ent[ent1]
        ent2 = id2ent[ent2]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        return ((ent1, (rel1,)), (ent2, (rel2,)))
    # 3i
    elif query_structure == (("e", ("r",)), ("e", ("r",)), ("e", ("r",))):
        ent1, (rel1_id,) = query[0]
        ent2, (rel2_id,) = query[1]
        ent3, (rel3_id,) = query[2]
        ent1 = id2ent[ent1]
        ent2 = id2ent[ent2]
        ent3 = id2ent[ent3]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        rel3 = id2rel[rel3_id]
        return ((ent1, (rel1,)), (ent2, (rel2,)), (ent3, (rel3,)))
    #2p
    elif query_structure == ("e", ("r", "r")):
        ent1, (rel1_id,rel2_id) = query
        ent1 = id2ent[ent1]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        return (ent1, (rel1,rel2))
    #3p
    elif query_structure == ("e", ("r", "r", "r")):
        ent1, (rel1_id,rel2_id,rel3_id) = query
        ent1 = id2ent[ent1]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        rel3 = id2rel[rel3_id]
        return (ent1, (rel1,rel2,rel3))
    #pi
    elif query_structure == (("e", ("r", "r")), ("e", ("r",))):
        ent1, (rel1_id,rel2_id) = query[0]
        ent2, (rel3_id,)=query[1]
        ent1 = id2ent[ent1]
        ent2 = id2ent[ent2]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        rel3 = id2rel[rel3_id]
        return ((ent1, (rel1,rel2)),(ent2,(rel3,)))
    #ip
    elif query_structure == ((("e", ("r",)), ("e", ("r",))), ("r",)):
        ent1, (rel1_id,) = query[0][0]
        ent2, (rel2_id,) = query[0][1]
        (rel3_id,)=query[1]
        ent1 = id2ent[ent1]
        ent2 = id2ent[ent2]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        rel3 = id2rel[rel3_id]
        return (((ent1, (rel1,)), (ent2, (rel2,))), (rel3,))
    #negation
    #2in
    elif query_structure == (("e", ("r",)), ("e", ("r", "n"))):
        ent1, (rel1_id,) = query[0]
        ent2, (rel2_id, negation) = query[1]
        ent1 = id2ent[ent1]
        ent2 = id2ent[ent2]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        return ((ent1, (rel1,)), (ent2, (rel2, "not")))
    #3in
    elif query_structure == (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))):
        ent1, (rel1_id,) = query[0]
        ent2, (rel2_id,) = query[1]
        ent3, (rel3_id,negation) = query[2]
        ent1 = id2ent[ent1]
        ent2 = id2ent[ent2]
        ent3 = id2ent[ent3]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        rel3 = id2rel[rel3_id]
        return ((ent1, (rel1,)),(ent2, (rel2,)), (ent3, (rel3, "not")))
    #pin
    elif query_structure == (("e", ("r", "r")), ("e", ("r", "n"))):
        ent1, (rel1_id, rel2_id) = query[0]
        ent2, (rel3_id,negation) = query[1]
        ent1 = id2ent[ent1]
        ent2 = id2ent[ent2]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        rel3 = id2rel[rel3_id]
        return ((ent1, (rel1, rel2)), (ent2, (rel3,"not")))
    #inp
    elif query_structure == ((("e", ("r",)), ("e", ("r", "n"))), ("r",)):
        ent1, (rel1_id,) = query[0][0]
        ent2, (rel2_id,negation) = query[0][1]
        (rel3_id,) = query[1]
        ent1 = id2ent[ent1]
        ent2 = id2ent[ent2]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        rel3 = id2rel[rel3_id]
        return (((ent1, (rel1,)), (ent2, (rel2,"not"))), (rel3,))
    #pni
    elif query_structure == (("e", ("r", "r", "n")), ("e", ("r",))):
        ent1, (rel1_id, rel2_id, negation) = query[0]
        ent2, (rel3_id, ) = query[1]
        ent1 = id2ent[ent1]
        ent2 = id2ent[ent2]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        rel3 = id2rel[rel3_id]
        return ((ent1, (rel1, rel2, "not")), (ent2, (rel3,)))
    #union
    #2u
    elif query_structure == (("e", ("r",)), ("e", ("r",)), ("u",)) :
        ent1, (rel1_id,) = query[0]
        ent2, (rel2_id,) = query[1]

        ent1 = id2ent[ent1]
        ent2 = id2ent[ent2]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        return ((ent1, (rel1,)), (ent2, (rel2,)),("union",))
    #up
    elif query_structure == ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)):
        ent1, (rel1_id,) = query[0][0]
        ent2, (rel2_id,) = query[0][1]
        (rel3_id,)=query[1]
        ent1 = id2ent[ent1]
        ent2 = id2ent[ent2]
        rel1 = id2rel[rel1_id]
        rel2 = id2rel[rel2_id]
        rel3 = id2rel[rel3_id]
        return (((ent1, (rel1,)), (ent2, (rel2,)),("union",)), (rel3,))

@click.command()
@click.option('--datapath', default='./KGs/UMLS')
@click.option('--map_to_ids', is_flag=True, default=False)
@click.option('--indexify_files', is_flag=True, default=False)
@click.option('--unmap_to_text', is_flag=True, default=False)
@click.option('--query_type', type=click.Choice(['train', 'valid','test']), default='test',
              help='query_type to be mapped or unmapped')
@click.option('--query_structures', type=str, default='2p,3p,2i,3i,pi,ip,2in,3in,pin,pni,inp,2u,up',
              help='comma-separated list of query structure names')
@click.option('--join_queries', is_flag=True, default=False)
@click.option('--file_type', type=str, help='mapped for ids or unmapped for strings',default='unmapped')
def main(datapath,map_to_ids,indexify_files,unmap_to_text,query_type,query_structures,join_queries,file_type):

     if map_to_ids is True:
        create_mappings(datapath)

     if indexify_files is True:
        mapid(datapath)

     query_structures_list = query_structures.split(',')
     if unmap_to_text is True:
        unmap(datapath,query_type,query_structures_list)

     if join_queries is True:
         joinqueries(datapath,file_type)

if __name__ == '__main__':
    main()