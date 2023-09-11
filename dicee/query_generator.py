from typing import Dict, List, Tuple


class QueryGenerator:
    def __init__(self):
        """
        A mapping from type of queries (e.g. "2in") to a list, where
        an item in this list should be a tuple containing the query structure
        (e.g. ("e", ("r",)), ("e", ("r", "n"))) )  and its answer

        """
        self.queries: Dict[str, List[Tuple]] = None

    def save_queries(self, path=str) -> None:
        """
        Save the attribute of self.queries as a json file into a given path as a json file ?
        """
        pass

    def generator_queries(self) -> List:
        """
        Move create_queries.py into this function
        """
        pass
