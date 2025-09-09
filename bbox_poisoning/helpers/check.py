import pandas as pd

def graph_stats_from_list(triples_list):

    if len(triples_list) == 0:
        df = pd.DataFrame(columns=["head", "relation", "tail"])
    elif isinstance(triples_list[0], str):
        rows = [ln.strip().split("\t") for ln in triples_list if ln.strip()]
        df = pd.DataFrame(rows, columns=["head", "relation", "tail"])
    else:
        df = pd.DataFrame(triples_list, columns=["head", "relation", "tail"])

    # Basic counts
    num_triples = len(df)

    s = pd.concat([df["head"], df["tail"]], ignore_index=True)
    s = s.dropna().astype(str).str.strip()  # optional: .str.lower() to normalize case
    entities = pd.unique(s)
    num_entities = entities.size

    num_relations = df["relation"].nunique()

    # Degrees
    out_deg = df["head"].value_counts().rename("out_degree")
    in_deg  = df["tail"].value_counts().rename("in_degree")
    degrees = pd.concat([out_deg, in_deg], axis=1).fillna(0).astype(int)
    degrees["total_degree"] = degrees["out_degree"] + degrees["in_degree"]
    degrees = degrees.sort_values(["total_degree","out_degree","in_degree"], ascending=[False,False,False])

    rel_freq = (df["relation"].value_counts()
                  .rename_axis("relation")
                  .reset_index(name="count")
                  .sort_values("count", ascending=False))

    summary = {
        "num_triples": num_triples,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "avg_out_degree": out_deg.sum() / max(1, num_entities),
        "avg_in_degree":  in_deg.sum()  / max(1, num_entities),
    }
    return {"summary": summary, "degrees": degrees, "relation_frequencies": rel_freq}


def triples_in_a_not_in_b(file_a, file_b):
    with open(file_a, "r", encoding="utf-8") as fa:
        a_lines = {line.strip() for line in fa}
    with open(file_b, "r", encoding="utf-8") as fb:
        b_lines = {line.strip() for line in fb}

    a_triples = [ln for ln in a_lines]
    b_triples = [ln for ln in b_lines]

    in_a_not_in_b = [t for t in a_triples if t not in b_triples]

    return in_a_not_in_b, len(in_a_not_in_b), a_triples, b_triples

#smaller_file = "./saved_models/UMLS/active_poisoning_whitebox_server/DistMult/high_close_fgsm/1304/random-one/0/train.txt"
#chosen_file = "./saved_models/UMLS/active_poisoning_whitebox_server/DistMult/high_close_fgsm/1564/random-one/0/train.txt"
#bigger_file = "./saved_models/UMLS/active_poisoning_whitebox_server/DistMult/high_close_fgsm/1825/random-one/0/train.txt"

smaller_file = "./saved_datasets/wo/UMLS/random/ComplEx/1251/random-one/0/train.txt"
chosen_file = "./saved_datasets/wo/UMLS/active_poisoning_whitebox/ComplEx/high_betw_global_fgsm/1251/random-one/0/train.txt"
bigger_file = "./saved_datasets/wo/UMLS/active_poisoning_whitebox/ComplEx/corrupted_bw/1251/random-one/0/train.txt"

clean_db = "./KGs/UMLS/train.txt"
with open(clean_db, "r", encoding="utf-8") as fa:
    clean_lines = {line.strip() for line in fa}
clean_triples = [ln for ln in clean_lines]

"""
in_a_not_in_b, len_in_a_not_in_b, a_triples, b_triples =  triples_in_a_not_in_b(file_2, file_1)
print("len_in_a_not_in_b: ", len_in_a_not_in_b)
print("len b_triples: ", len(b_triples))
print("len a_triples", len(a_triples))

print("**********************************")
print("Diff")
res = graph_stats_from_list(in_a_not_in_b)
print(res["summary"])
print(res["degrees"].head())
#print(res["relation_frequencies"])

print("**********************************")
print("Bigger")
res_a = graph_stats_from_list(a_triples)
print(res_a["summary"])
print(res_a["degrees"].head())
#print(res_a["relation_frequencies"])

print("**********************************")
print("Smaller")
res_b = graph_stats_from_list(b_triples)
print(res_b["summary"])
print(res_b["degrees"].head())
#print(res_b["relation_frequencies"])

print("**********************************")
print("Clean DB")
res_clean = graph_stats_from_list(clean_triples)
print(res_clean["summary"])
print(res_clean["degrees"].head())
#print(res_clean["relation_frequencies"])
"""
print("betweenness:")

in_bigger_not_in_db, len_in_bigger_not_in_db, bigger_triples, db_triples =  triples_in_a_not_in_b(bigger_file, clean_db)
#print("len_in_db_not_in_bigger: ", len_in_bigger_not_in_db)
#print("len db_triples: ", len(db_triples))
#print("len bigger_triples", len(bigger_triples))

res_in_bigger_not_in_db = graph_stats_from_list(in_bigger_not_in_db)
print(res_in_bigger_not_in_db["summary"])
print(res_in_bigger_not_in_db["degrees"].head())
#print(res_in_bigger_not_in_db["relation_frequencies"])

print("**********************************")
print("Random:")

in_smaller_not_in_db, len_in_smaller_not_in_db, smaller_triples, db_triples =  triples_in_a_not_in_b(smaller_file, clean_db)
#print("len_in_smaller_not_in_db: ", len_in_smaller_not_in_db)
#print("len db_triples: ", len(db_triples))
#print("len smaller_triples", len(smaller_triples))

res_in_smaller_not_in_db = graph_stats_from_list(in_smaller_not_in_db)
print(res_in_smaller_not_in_db["summary"])
print(res_in_smaller_not_in_db["degrees"].head())

print("**********************************")
print("FGSM_betweenness(global)")

in_chosen_not_in_db, len_in_chosen_not_in_db, chosen_triples, db_triples =  triples_in_a_not_in_b(chosen_file, clean_db)
#print("len_in_chosen_not_in_db: ", len_in_chosen_not_in_db)
#print("len db_triples: ", len(db_triples))
#print("len chosen_triples", len(chosen_triples))

res_in_chosen_not_in_db = graph_stats_from_list(in_chosen_not_in_db)
print(res_in_chosen_not_in_db["summary"])
print(res_in_chosen_not_in_db["degrees"].head())

print("**********************************")
print("Original DB:")
res_db_triples = graph_stats_from_list(db_triples)
print(res_db_triples["summary"])
print("top 5:")
print(res_db_triples["degrees"].head())