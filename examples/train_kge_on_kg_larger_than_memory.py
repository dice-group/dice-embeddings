"""
Training a large knowledge graph embedding model on a knowledge graph that does not fit in memory.
Computation
(1) Read/Index/Store a knowledge graph
(1.1) Extract unique entities and assign unique integers them
(1.2) Extract unique relations and assign unique integers them
(1.4) Index an input knowledge graph batch wise
(1.4.1) Read a batch in memory, replace each subject, predicate, and object with their indices
(1.4.2) Store a batch into disk as CSV.

(2) Train a KGE model
(2.1) Initialize a KGE model
(2.2) Construct a memory map for the indexed knowledge graph that is a concatenation of index batch csv files.
(2.3) Initialize a dataset based on (2.2) that generates negative examples on the fly.

        SETUP
+ A machine with 62.2 RAM and Intel Core U7
+ dbpedia-generic-snapshot-2022-12 without literals having a size of 590,550,727 (wc -l dbpedia.nt)

        RUN
(dice) python examples/train_kge_on_kg_larger_than_memory.py --path_dataset dbpedia.nt --path_csv_index_entities dbpedia_index_entities.csv --path_csv_index_relations dbpedia_index_relations.csv --path_indexed_dataset indexed_dbpedia
Collecting subject entities...
Unique number of subjects:40890146
Collecting object entities...
Unique number of objects:117367061
Batch Processing...
Step:0  Seen Triples: 59228337  Total Runtime: 28.149 secs
Step:1  Seen Triples: 118617771 Total Runtime: 58.033 secs
Step:2  Seen Triples: 177962380 Total Runtime: 86.136 secs
Step:3  Seen Triples: 237353864 Total Runtime: 116.173 secs
Step:4  Seen Triples: 296632299 Total Runtime: 143.573 secs
Step:5  Seen Triples: 355872505 Total Runtime: 172.268 secs
Step:6  Seen Triples: 415059906 Total Runtime: 200.339 secs
Step:7  Seen Triples: 474349288 Total Runtime: 231.305 secs
Step:8  Seen Triples: 533641626 Total Runtime: 264.210 secs
Step:9  Seen Triples: 578714291 Total Runtime: 290.475 secs
Step:10 Seen Triples: 590550727 Total Runtime: 304.888 secs
Total Runtime:304.8876941204071
Memory usage of entity_to_idx dataframe is 7788.37 MB
Memory usage of relation_to_idx dataframe is 0.54 MB
Parsing CSV files...
Creating a memory-map to a graph stored in a binary file on disk.
Adding indexed_dbpedia/Start_row_0_End_row_59228337.np
Adding indexed_dbpedia/Start_row_59228337_End_row_118617771.np
Adding indexed_dbpedia/Start_row_118617771_End_row_177962380.np
Adding indexed_dbpedia/Start_row_177962380_End_row_237353864.np
Adding indexed_dbpedia/Start_row_237353864_End_row_296632299.np
Adding indexed_dbpedia/Start_row_296632299_End_row_355872505.np
Adding indexed_dbpedia/Start_row_355872505_End_row_415059906.np
Adding indexed_dbpedia/Start_row_415059906_End_row_474349288.np
Adding indexed_dbpedia/Start_row_474349288_End_row_533641626.np
Adding indexed_dbpedia/Start_row_533641626_End_row_578714291.np
Adding indexed_dbpedia/Start_row_578714291_End_row_590550727.np
Concatenated data saved to indexed_dbpedia/indexed_knowledge_graph.npy
Total number of parameters: 4088141280
Total memory size: 16352565120 bytes
Number of batch updates per epoch 11812
Batch [0 | 11812]       Loss [2.3336198329925537]       Forward/Backward/Update [8.225] BatchFetch [1.447]      RT [9.672]
Batch [1 | 11812]       Loss [2.3810811042785645]       Forward/Backward/Update [4.082] BatchFetch [1.500]      RT [5.582]
Batch [2 | 11812]       Loss [2.3562967777252197]       Forward/Backward/Update [3.949] BatchFetch [1.339]      RT [5.287]
Batch [3 | 11812]       Loss [2.368497371673584]        Forward/Backward/Update [4.032] BatchFetch [1.278]      RT [5.309]
Batch [4 | 11812]       Loss [2.3351900577545166]       Forward/Backward/Update [4.021] BatchFetch [1.404]      RT [5.424]
Batch [5 | 11812]       Loss [2.4003183841705322]       Forward/Backward/Update [3.938] BatchFetch [1.254]      RT [5.192]
...


"""
import argparse
import polars
import polars as pl
from torch.nn import functional as F
from dicee import Keci
import os
import torch
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
from typing import Tuple

class OnevsSampleDataset(Dataset):
    def __init__(self, mmap_kg:np.memmap, num_entities: int = None, neg_sample_ratio: int = 2):
        assert num_entities >= 1
        self.memmap_g = mmap_kg
        self.num_points = len(self.memmap_g)
        self.num_entities = 30_000 if num_entities > 2 ** 20 else num_entities
        self.neg_sample_ratio = neg_sample_ratio

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        # pytorch triple from the memory map of numpy.
        triple = torch.from_numpy(self.memmap_g[idx].copy())
        x = triple[:2]
        y = triple[-1].unsqueeze(-1)
        # Initialize weights for negative sampling. This corresponds to sampling with replacement
        negative_idx = torch.randint(low=0,
                                     high=self.num_entities,
                                     size=(self.neg_sample_ratio,),
                                     device="cpu")
        # Concatenate the true tail entity with the negative samples
        y_idx = torch.cat((y, negative_idx), 0).long()
        y_vec = torch.zeros(self.neg_sample_ratio + 1, device='cpu')
        y_vec[0] = 1
        return x, y_idx, y_vec

def get_default_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--path_dataset", type=str, default=None,
                        help="Path to a knowledge graph in the n-triple format without containing literals")
    parser.add_argument("--path_csv_index_entities", type=str, default=None, required=True)
    parser.add_argument("--path_csv_index_relations", type=str, default=None, required=True)
    parser.add_argument("--path_indexed_dataset", type=str, default=None, required=True)
    parser.add_argument("--preprocessing_batch_size", type=int, default=50_000_000)
    parser.add_argument("--batch_size", type=int, default=50_000, help="Batch size for the SGD training.")
    parser.add_argument("--neg_sample_ratio", type=int, default=100, help="Number of negative examples per positive example.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Number of training epochs.")
    return parser.parse_args()

def create_indexing(args)->Tuple[polars.DataFrame,polars.DataFrame]:
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
    entity_to_idx.write_csv(file=args.path_csv_index_entities, include_header=True)
    del subjects, objects
    # () Collect relations.
    relation_to_idx = lazy_df.select(pl.col("relation").unique()).collect(streaming=True).with_row_index(
        "index").select(["index", "relation"])
    relation_to_idx.write_csv(file=args.path_csv_index_relations, include_header=True)
    return entity_to_idx, relation_to_idx

def index_knowledge_graph_index(path_dataset:str=None,
                                path_indexed_dataset:str=None,
                                preprocessing_batch_size:int=None,
                                entity_to_idx:polars.DataFrame=None,
                                relation_to_idx:polars.DataFrame=None)->None:
    assert path_dataset is not None, f"path_dataset ({path_dataset}) must be specified)"
    assert path_indexed_dataset is not None, f"path_indexed_dataset ({path_indexed_dataset}) must be specified)"
    # Create directory if it does not exist, without raising an error
    os.makedirs(path_indexed_dataset, exist_ok=True)
    start_time=time.time()
    start_row = 0
    end_row = 0
    reader = pl.read_csv_batched(source=path_dataset,
                                 batch_size=preprocessing_batch_size,
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
        # why numpy?
        df_final.write_csv(f"{path_indexed_dataset}/Start_row_{start_row}_End_row_{end_row}.csv", include_header=False)

        start_row = end_row
        print(f"Step:{i}\tSeen Triples: {end_row}\tTotal Runtime: {time.time() - start_time:.3f} secs")

    print(f"Total Runtime:{time.time() - start_time}")

def prepare_dataset(args)->Tuple[int,int,np.memmap]:
    if (os.path.exists(args.path_indexed_dataset) and os.path.exists(args.path_csv_index_entities)
            and os.path.exists(args.path_csv_index_relations)):
            # () Read entities with indices.
            print("Reading entities...", end="\t")
            entity_to_idx = pl.read_csv(args.path_csv_index_entities)
            print(f"Unique number of entities:\t{len(entity_to_idx)}")
            print("Reading relations...", end="\t")
            relation_to_idx = pl.read_csv(args.path_csv_index_relations)
            print(f"Unique number of relations:\t{len(relation_to_idx)}")

    else:
        # Ensure that the folder exists.
        assert os.path.exists(args.path_dataset), f"Path ({args.path_dataset}) does not exist!"
        entity_to_idx, relation_to_idx = create_indexing(args)
        assert isinstance(args.preprocessing_batch_size,int),f"--preprocessing_batch_size must be int!"
        assert args.preprocessing_batch_size > 1_000_000, f"--preprocessing_batch_size must be greater than 10⁶ !"
        index_knowledge_graph_index(path_dataset=args.path_dataset,
                                    path_indexed_dataset=args.path_indexed_dataset,
                                    preprocessing_batch_size=args.preprocessing_batch_size,
                                    entity_to_idx=entity_to_idx,
                                    relation_to_idx=relation_to_idx)

    print(f"Memory usage of entity_to_idx dataframe is {round(entity_to_idx.estimated_size('mb'), 2)} MB")
    print(f"Memory usage of relation_to_idx dataframe is {round(relation_to_idx.estimated_size('mb'), 2)} MB")

    num_relations = len(relation_to_idx)
    num_entities = len(entity_to_idx)

    # Create a memory-mapped file for the concatenated array
    output_filename = f"{args.path_indexed_dataset}/indexed_knowledge_graph.npy"

    if os.path.exists(output_filename) is False:
        file_ranges = []
        print("Parsing CSV files...")
        for filename in os.listdir(args.path_indexed_dataset):
            if filename.endswith('.csv'):
                # Extract start and end row numbers from the filename
                parts = filename.replace('.csv', '').split('_')
                start_row = int(parts[2])
                end_row = int(parts[-1])
                file_path = os.path.join(args.path_indexed_dataset, filename)
                file_ranges.append((file_path, start_row, end_row))
        # Sort by start_row
        file_ranges = sorted(file_ranges, key=lambda x: x[1])
        file_shapes = []  # To keep track of the shapes of each numpy array
        print("Creating a memory-map to a graph stored in a binary file on disk.")
        created_files = []
        num_triples = 0
        for f in file_ranges:
            csv_path, start_row, end_row = f
            # csv to pt file name change
            new_name = csv_path.replace(".csv", ".np")
            np_triple = pl.read_csv(csv_path, has_header=False).to_numpy()
            # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html#numpy-ndarray-tofile
            np_triple.tofile(new_name)
            num_triples += np_triple.size // 3
            created_files.append(new_name)
            file_shapes.append(np_triple.shape)
        mmap_kg = np.memmap(output_filename, dtype=np.int64, mode='w+', shape=(num_triples, 3))
        # Concatenate all the data into the new memory map
        current_position = 0
        for file_path, shape in zip(created_files, file_shapes):
            g = np.memmap(file_path, mode="r", dtype=np.int64, shape=shape)
            print("Adding", file_path)
            mmap_kg[current_position:current_position + len(g):, :] = g
            current_position += len(g)
        # Flush the memory-mapped file to disk
        mmap_kg.flush()
        print(f"Concatenated data saved to {output_filename}")
    else:
        n = len(np.memmap(output_filename, dtype=np.int64, mode='r'))#, shape=(num_triples, 3))
        mmap_kg = np.memmap(output_filename, dtype=np.int64, mode='r', shape=(n//3, 3))

    return num_entities, num_relations, mmap_kg

def run(args):
    num_entities, num_relations, mmap_kg = prepare_dataset(args)
    model = Keci(args={"embedding_dim": args.embedding_dim, "p": 0, "q": 1,
                       "num_entities": num_entities, "num_relations": num_relations})

    # Number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Memory size in bytes
    total_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Total memory size: {total_memory} bytes")

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    rows_to_copy = 100
    sub_mmap_kg = np.memmap('destination_file.dat', dtype=mmap_kg.dtype, mode='w+', shape=(rows_to_copy, 3))
    # Copy the first few rows from the source to the destination
    sub_mmap_kg[:] = mmap_kg[:rows_to_copy]

    # Multi-class classification per triple.
    dataset = OnevsSampleDataset(mmap_kg=sub_mmap_kg,
                                 num_entities=num_entities,
                                 neg_sample_ratio=args.neg_sample_ratio)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    num_of_batches = len(dataloader)
    print(f"Number of batch updates per epoch {num_of_batches}")
    for epoch_id in range(args.num_epochs):
        iter_dataloader = iter(dataloader)
        for batch_id in range(num_of_batches):
            batch_time = time.time()
            batch_fetch_time = time.time()
            # () Get the next batch.
            try:
                x, y_idx, y_label = next(iter_dataloader)
            except StopIteration:
                continue
            batch_fetch_time = time.time() - batch_fetch_time
            rt_forward_backward_update = time.time()
            # () Clean gradients
            optimizer.zero_grad()
            # () - log(1/N) => -math.log(1/127741846) = 18.83
            logits = model.forward_k_vs_sample(x=x, target_entity_idx=y_idx)
            # ()
            batch_loss = F.binary_cross_entropy_with_logits(logits, y_label)
            # ()
            batch_loss_float = batch_loss.item()
            # ()
            batch_loss.backward()
            # ()
            optimizer.step()
            # ()
            rt_forward_backward_update = time.time() - rt_forward_backward_update
            # ()
            print(f"Epoch [{epoch_id} | {args.num_epochs}]\t"
                  f"Batch [{batch_id} | {num_of_batches}]\t"
                  f"Loss [{batch_loss_float}]\t"
                  f"Forward/Backward/Update [{rt_forward_backward_update:.3f}]\t"
                  f"BatchFetch [{batch_fetch_time:.3f}]\t"
                  f"RT [{time.time() - batch_time:.3f}]")
        # TODO: Epochs End Save the model
if __name__ == '__main__':
    run(get_default_arguments())




