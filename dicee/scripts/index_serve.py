"""
$ docker pull qdrant/qdrant && docker run -p 6333:6333 -p 6334:6334      -v $(pwd)/qdrant_storage:/qdrant/storage:z      qdrant/qdrant
$ dicee_vector_db --index --serve --path CountryEmbeddings --collection "countries_vdb"

"""
import argparse
import os
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct

from fastapi import FastAPI
import uvicorn


def get_default_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--path", type=str, required=True,
                        help="The path of a directory containing embedding csv file(s)")
    parser.add_argument("--index", action="store_true", help="A flag for indexing")
    parser.add_argument("--serve", action="store_true", help="A flag for serving")

    parser.add_argument("--collection", type=str, required=True,help="Named of the vector database collection")

    parser.add_argument("--vdb_host", type=str,default="localhost",help="Host of qdrant vector database")
    parser.add_argument("--vdb_port", type=int,default=6333,help="port number")
    parser.add_argument("--host",type=str, default="0.0.0.0",help="Host")
    parser.add_argument("--port", type=int, default=8000,help="port number")
    return parser.parse_args()


def index(args):
    client = QdrantClient(host=args.vdb_host, port=args.vdb_port)
    entity_to_idx = pd.read_csv(args.path + "/entity_to_idx.csv", index_col=0)
    assert entity_to_idx.index.is_monotonic_increasing, "Entity Index must be monotonically increasing!{}"
    entity_to_idx = {name: idx for idx, name in enumerate(entity_to_idx["entity"].tolist())}

    csv_files_holding_embeddings = [args.path + "/" + f for f in os.listdir(args.path) if "entity_embeddings.csv" in f]
    assert len(
        csv_files_holding_embeddings) == 1, f"There must be only single csv file containing entity_embeddings.csv prefix. Currently, :{len(csv_files_holding_embeddings)}"
    path_entity_embeddings_csv = csv_files_holding_embeddings[0]

    points = []
    embedding_dim = None
    for ith_row, (index_name, embedding_vector) in enumerate(
            pd.read_csv(path_entity_embeddings_csv, index_col=0, header=0).iterrows()):
        index_name: str
        embedding_vector: np.ndarray
        embedding_vector = embedding_vector.values

        points.append(PointStruct(id=entity_to_idx[index_name],
                                  vector=embedding_vector,
                                  payload={"name": index_name}))

        embedding_dim = len(embedding_vector)

    assert embedding_dim > 0
    # If the collection is not created, create it
    if args.collection in [i.name for i in client.get_collections().collections]:
        print("Deleting existing collection ", args.collection)
        client.delete_collection(collection_name=args.collection)

    print(f"Creating a collection {args.collection} with distance metric:Cosine")
    client.create_collection(collection_name=args.collection,
                             vectors_config=VectorParams(size=embedding_dim,
                                                         distance=Distance.COSINE))
    client.upsert(collection_name=args.collection, points=points)
    print("Completed!")


app = FastAPI()
# Create a neural searcher instance
neural_searcher = None

class NeuralSearcher:
    def __init__(self, args):
        self.collection_name = args.collection
        assert os.path.exists(args.path + "/entity_to_idx.csv"), f"{args.path + '/entity_to_idx.csv'} does not exist!"
        self.entity_to_idx = pd.read_csv(args.path + "/entity_to_idx.csv", index_col=0)
        assert self.entity_to_idx.index.is_monotonic_increasing, "Entity Index must be monotonically increasing!{}"
        self.entity_to_idx = {name: idx for idx, name in enumerate(self.entity_to_idx["entity"].tolist())}
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(host=args.vdb_host,port=args.vdb_port)
        # semantic search
        self.topk=5

    def get(self,entity:str=None):
        if entity is None:
            return {"Input {entity} cannot be None"}
        elif self.entity_to_idx.get(entity,None) is None:
            return {f"Input {entity} not found"}
        else:
            ids=[self.entity_to_idx[entity]]
            return self.qdrant_client.retrieve(collection_name=self.collection_name,ids=ids, with_vectors=True)

    def search(self, entity: str):
        return self.qdrant_client.query_points(collection_name=self.collection_name, query=self.entity_to_idx[entity],limit=self.topk)

@app.get("/")
async def root():
    return {"message": "Hello Dice Embedding User"}

@app.get("/api/search")
async def search_embeddings(q: str):
    return {"result": neural_searcher.search(entity=q)}

@app.get("/api/get")
async def retrieve_embeddings(q: str):
    return {"result": neural_searcher.get(entity=q)}


def serve(args):
    global neural_searcher
    neural_searcher = NeuralSearcher(args)
    uvicorn.run(app, host=args.host, port=args.port)

def main():
    args = get_default_arguments()
    if args.index:
        index(args)
    if args.serve:
        serve(args)

if __name__ == '__main__':
    main()