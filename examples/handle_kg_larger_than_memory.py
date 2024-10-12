"""Handle datasets much larger than your available RAM.

        HARDWARE SETUP
62.2 RAM
Intel Core U7

        DATA SETUP
dbpedia-generic-snapshot-2022-12 without literals

$ ls -lh dbpedia.nt
-rw-rw-r-- 1 cdemir cdemir 85G Sep 30 16:16 dbpedia.nt
$ wc -l dbpedia.nt
590550727 dbpedia.nt

        Indexing
$ python examples/handle_kg_larger_than_memory.py --path_dataset dbpedia.nt
Collecting subject entities...
Unique number of subjects:40890146
Collecting object entities...
Unique number of objects:117367061
Indexing relations
Batch Processing...
Step:0  Seen Triples: 59228337  Current Runtime: 265.289 secs
Step:1  Seen Triples: 118617771 Current Runtime: 296.688 secs
Step:2  Seen Triples: 177962380 Current Runtime: 326.925 secs
Step:3  Seen Triples: 237353864 Current Runtime: 355.622 secs
Step:4  Seen Triples: 296632299 Current Runtime: 383.619 secs
Step:5  Seen Triples: 355872505 Current Runtime: 411.644 secs
Step:6  Seen Triples: 415059906 Current Runtime: 440.054 secs
Step:7  Seen Triples: 474349288 Current Runtime: 468.443 secs
Step:8  Seen Triples: 533641626 Current Runtime: 495.682 secs
Step:9  Seen Triples: 578714291 Current Runtime: 517.766 secs
Step:10 Seen Triples: 590550727 Current Runtime: 529.929 secs
Total Runtime:529.9295048713684

$ python examples/handle_kg_larger_than_memory.py --path_dataset dbpedia.nt --path_entity dbpedia_entities.csv --path_relation dbpedia_relations.csv
Reading entities...     Unique number of entities:      127741846
Reading relations...    Unique number of relations:     12569
Batch Processing...
Step:0  Seen Triples: 59228337  Current Runtime: 35.948 secs
Step:1  Seen Triples: 118617771 Current Runtime: 62.114 secs
Step:2  Seen Triples: 177962380 Current Runtime: 88.316 secs
Step:3  Seen Triples: 237353864 Current Runtime: 114.177 secs
Step:4  Seen Triples: 296632299 Current Runtime: 140.772 secs
Step:5  Seen Triples: 355872505 Current Runtime: 166.082 secs
Step:6  Seen Triples: 415059906 Current Runtime: 193.374 secs
Step:7  Seen Triples: 474349288 Current Runtime: 218.604 secs
Step:8  Seen Triples: 533641626 Current Runtime: 245.580 secs
Step:9  Seen Triples: 578714291 Current Runtime: 265.677 secs
Step:10 Seen Triples: 590550727 Current Runtime: 275.785 secs
Total Runtime:275.7852234840393

"""
import argparse
import polars as pl
import time
import os


def get_default_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--path_dataset", type=str, default="KGs/UMLS")
    parser.add_argument("--path_dir", type=str, default="indexed_dataset")
    parser.add_argument("--path_entity", type=str, default="dbpedia_entities.csv")
    parser.add_argument("--path_relation", type=str, default="dbpedia_relations.csv")
    parser.add_argument("--batch_size", type=int, default=50_000_000)
    return parser.parse_args()

def run(args):
    start_time = time.time()
    if os.path.exists(args.path_entity) is False:
        # () Collect unique entities.
        # Note: Setting maintain_order=True in unique() increases the memory usage.
        lazy_df = pl.scan_csv(args.path_dataset,
                              has_header=False,
                              separator=" ",
                              new_columns=['subject', 'relation', 'object', "end"]).drop('end')
        # () Select unique subject entities.
        print("Collecting subject entities...")
        subjects = lazy_df.select(pl.col("subject").unique().alias("entity")).collect()
        print(f"Unique number of subjects:{len(subjects)}")
        # () Select unique object entities.
        print("Collecting object entities...")
        objects = lazy_df.select(pl.col("object").unique().alias("entity")).collect()
        print(f"Unique number of objects:{len(objects)}")
        # () Select unique entities.
        entity_to_idx = pl.concat([subjects, objects], how="vertical").unique()
        entity_to_idx = entity_to_idx.with_row_index("index").select(["index", "entity"])
        # () Write unique entities with indices.
        entity_to_idx.write_csv(file="dbpedia_entities.csv", include_header=True)
        del subjects, objects
    else:
        # () Read entities with indices.
        print("Reading entities...", end="\t")
        entity_to_idx = pl.read_csv(args.path_entity)
        print(f"Unique number of entities:\t{len(entity_to_idx)}")

    if os.path.exists(args.path_relation) is False:
        print("Indexing relations")
        # () Collect relations.
        lazy_df = pl.scan_csv(args.path_dataset,
                              has_header=False,
                              separator=" ",
                              new_columns=['subject', 'relation', 'object', "end"]).drop('end')
        relation_to_idx = lazy_df.select(pl.col("relation").unique()).collect(streaming=True).with_row_index(
            "index").select(["index", "relation"])
        relation_to_idx.write_csv(file="dbpedia_relations.csv", include_header=True)
    else:
        print("Reading relations...", end="\t")
        relation_to_idx = pl.read_csv(args.path_relation)
        print(f"Unique number of relations:\t{len(relation_to_idx)}")

    # Create directory if it does not exist, without raising an error
    os.makedirs(args.path_dir, exist_ok=True)

    start_row = 0
    end_row = 0
    reader = pl.read_csv_batched(source=args.path_dataset,
                                 batch_size=args.batch_size,
                                 has_header=False,
                                 separator=" ",
                                 schema_overrides=pl.Schema({"subject": pl.String,
                                                             "relation": pl.String(),
                                                             "object": pl.String(),
                                                             "end": pl.String()}))
    print("Batch Processing...")
    for i, batches in enumerate(iter(lambda: reader.next_batches(1), None)):
        # if next_batches(n>2), we need to apply the concat opt.
        pl_batch = pl.concat(batches).drop("end")
        end_row += len(pl_batch)

        # Step : Join on 'relation' to replace relation with its index
        df_merged = pl_batch.join(relation_to_idx, on="relation", how="left")
        df_merged = df_merged.select([pl.col("subject"), pl.col("index").alias("relation"), pl.col("object")])
        # Step :  Consider Left Table on subject and Right Table on entity with the left join
        # Returns all rows from the left table, and the matched rows from the right table
        df_merged = df_merged.join(entity_to_idx, left_on="subject", right_on="entity", how="left")
        df_merged = df_merged.drop("subject").rename({"index": "subject"})

        # Step 3: Join on 'object' to replace object with its index
        df_final = df_merged.join(entity_to_idx, left_on="object", right_on="entity", how="left")
        df_final = df_final.drop("object").rename({"index": "object"})

        # Step 4: Select the desired columns
        df_final = df_final.select([pl.col("subject"), pl.col("relation"), pl.col("object")])

        df_final.write_csv(f"{args.path_dir}/Start_row_{start_row}_End_row_{end_row}.csv",
                           include_header=False)

        start_row = end_row
        print(f"Step:{i}\tSeen Triples: {end_row}\tCurrent Runtime: {time.time() - start_time:.3f} secs")

    print(f"Total Runtime:{time.time() - start_time}")


if __name__ == '__main__':
    run(get_default_arguments())