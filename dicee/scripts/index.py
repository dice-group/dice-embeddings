import argparse


def get_default_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--path_model", type=str, required=True,
                        help="The path of a directory containing pre-trained model")
    parser.add_argument("--collection_name", type=str, required=True,
                        help="Named of the vector database collection")
    parser.add_argument("--location", type=str, required=True,
                        help="location")
    return parser.parse_args()


def main():
    args = get_default_arguments()
    # docker pull qdrant/qdrant
    # docker run -p 6333:6333 -p 6334:6334      -v $(pwd)/qdrant_storage:/qdrant/storage:z      qdrant/qdrant
    # pip install qdrant-client

    from dicee.knowledge_graph_embeddings import KGE

    # Train a model on Countries dataset
    KGE(path=args.path_model).create_vector_database(collection_name=args.collection_name,
                                                     location=args.location,
                                                     distance="cosine")
    return "Completed!"


if __name__ == '__main__':
    main()
