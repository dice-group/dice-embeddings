from abc import ABC, abstractmethod
from dicee.knowledge_graph import KG
import torch
from typing import Tuple
class AbstractBaseLinkPredictorClass(ABC):
    def __init__(self, knowledge_graph: KG = None, name="dummy"):
        assert knowledge_graph is not None
        assert name is not None
        self.kg = knowledge_graph
        self.name = name

        # Create dictionaries
        self.idx_to_entity = self.kg.entity_to_idx.set_index(self.kg.entity_to_idx.index)['entity'].to_dict()
        self.entity_to_idx = {idx: entity for entity, idx in self.idx_to_entity.items()}
        #
        self.idx_to_relation = self.kg.relation_to_idx.set_index(self.kg.relation_to_idx.index)['relation'].to_dict()
        self.relation_idx = {idx: rel for rel, idx in self.idx_to_relation.items()}

    def eval(self):
        pass

    @abstractmethod
    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        pass

    @abstractmethod
    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        pass

    def __call__(self, x: torch.LongTensor | Tuple[torch.LongTensor, torch.LongTensor], y_idx: torch.LongTensor = None):
        """Predicting missing triples """

        if isinstance(x, tuple):
            # x, y_idx = x
            raise NotImplementedError(
                "Currently, We do not support KvsSample. KvsSample allows a model to assign scores only on the selected entities.")
            # return self.forward_k_vs_sample(x=x, target_entity_idx=y_idx)
        else:
            shape_info = x.shape
            if len(shape_info) == 2:
                batch_size, dim = x.shape
                if dim == 3:
                    return self.forward_triples(x)
                elif dim == 2:
                    # h, y = x[0], x[1]
                    # Note that y can be relation or tail entity.
                    return self.forward_k_vs_all(x=x)
            else:
                raise RuntimeError("Unsupported shape: {}".format(shape_info))
