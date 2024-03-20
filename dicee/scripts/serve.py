import argparse
from typing import Dict, List, Union
from ..knowledge_graph_embeddings import KGE
from fastapi import FastAPI
import uvicorn
from qdrant_client import QdrantClient

app = FastAPI()
# Create a neural searcher instance
neural_searcher = None
def get_default_arguments() -> argparse.Namespace:
    """
    Get default command-line arguments for a specific task.

    This function returns a set of default command-line arguments that are used for a specific task. The arguments
    include options for specifying the path to a pre-trained model, the name of a vector database collection,
    the location of the collection, host information, and port number.

    Returns
    -------
    argparse.Namespace
        A namespace containing the default command-line arguments.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--path_model",
        type=str,
        required=True,
        help="The path of a directory containing pre-trained model",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        required=True,
        help="Named of the vector database collection",
    )
    parser.add_argument(
        "--collection_location", type=str, required=True, help="location"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
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
    """
    A class for performing neural-based vector search using a pre-trained model and a vector database.

    This class is designed for searching for entities in a vector database using a neural network-based model.
    It initializes the model and the Qdrant client for performing vector searches.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace containing the configuration and settings for the searcher.

    Attributes
    ----------
    collection_name : str
        The name of the vector database collection to perform searches in.
    model : KGE
        An instance of the knowledge graph embedding model for encoding entities into vectors.
    qdrant_client : QdrantClient
        An instance of the Qdrant client for interacting with the vector database.

    Methods
    -------
    search(entity: str) -> List[Dict[str, Union[str, float]]]
        Search for the closest vectors to the input entity in the vector database.
    """

    def __init__(self, args):
        self.collection_name = args.collection_name
        # Initialize encoder model
        self.model = KGE(path=args.path_model)
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(location=args.collection_location)

    def search(self, entity: str) -> List[Dict[str, Union[str, float]]]:
        """
        Search for the closest vectors to the input entity in the vector database.

        Parameters
        ----------
        entity : str
            The entity for which to find the closest matches in the database.

        Returns
        -------
        List[Dict[str, Union[str, float]]]
            A list of dictionaries containing search results, where each dictionary has "hit" (str) and "score" (float) keys.
        """
        # Convert text query into vector
        vector = self.model.get_transductive_entity_embeddings(indices=[entity], as_list=True)[0]

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


if __name__ == "__main__":
    main()
