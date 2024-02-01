import argparse
from ..knowledge_graph_embeddings import KGE
from fastapi import FastAPI
import uvicorn
from qdrant_client import QdrantClient

app = FastAPI()
# Create a neural searcher instance
neural_searcher = None
def get_default_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--path_model", type=str, required=True,
                        help="The path of a directory containing pre-trained model")
    parser.add_argument("--collection_name", type=str, required=True, help="Named of the vector database collection")
    parser.add_argument("--collection_location", type=str, required=True, help="location")
    parser.add_argument("--host",type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()

@app.get("/")
async def root():
    return {"message": "Hello Dice Embedding User"}

@app.get("/api/search")
async def search_embeddings(q: str):
    return {"result": neural_searcher.search(entity=q)}

@app.get("/api/get")
async def retrieve_embeddings(q: str):
    return {"result": neural_searcher.get(entity=q)}

class NeuralSearcher:
    def __init__(self, args):
        self.collection_name = args.collection_name
        # Initialize encoder model
        self.model = KGE(path=args.path_model)
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(location=args.collection_location)

    def get(self,entity:str):
        return self.model.get_transductive_entity_embeddings(indices=[entity], as_list=True)[0]

    def search(self, entity: str):
        # Convert text query into vector
        vector=self.get(entity)

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,  # If you don't want any filters for now
            limit=5,  # 5 the most closest results is enough
        )
        return [{"hit": i.payload["name"], "score": i.score} for i in search_result]


def main():
    args = get_default_arguments()
    global neural_searcher
    neural_searcher = NeuralSearcher(args)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()
