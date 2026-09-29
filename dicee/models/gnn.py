"""GNN-encoder / DistMult-decoder knowledge graph embedding models.

``RGCN`` and ``GATv2`` each use a relational graph encoder to
refine the entity embedding table via message passing over the training
graph, then score triples with a shared DistMult decoder.

The training graph is attached with ``set_graph`` (``DICE_Trainer.prepare_graph_model``
does this after construction), stored beside the checkpoint as ``gnn_graph.pt``,
and restored via ``load_graph``. ``inverse_relation[r]`` gives the relation ID
stating the same fact in the other direction, or ``-1`` if there is none.

Requires ``torch_geometric``.
"""
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, RGCNConv

from .base_model import BaseKGE


class RelationalGNNEncoder(BaseKGE):
    """``BaseKGE`` subclass that refines entity embeddings via message passing.

    ``args["train_set_idx"]``, an ``(N, 3)`` array of indexed training triples,
    builds the message-passing graph. If omitted, attach it later with
    :meth:`set_graph` or :meth:`load_graph` before scoring.
    """

    name = "RelationalGNNEncoder"
    config_prefix = "gnn"
    graph_filename = "gnn_graph.pt"

    def __init__(self, args: dict):
        args = dict(args)
        # nn.Embedding's default N(0,1) scale compounds through message-passing layers.
        if args.get("init_param") is None:
            args["init_param"] = "xavier_normal"
        super().__init__(args)
        # Graph is a dataset property, not an architectural one, so it stays out of the checkpoint.
        for name in ("graph_triples", "inverse_relation", "edge_index", "edge_type"):
            self.register_buffer(name, None, persistent=False)
        if args.get("train_set_idx") is not None:
            self.set_graph(args["train_set_idx"], inverse_relations=args.get("inverse_relations"))

        self.gnn_num_layers = int(args.get("gnn_num_layers", 2))
        if self.gnn_num_layers < 0:
            raise ValueError("gnn_num_layers must be nonnegative; zero disables message passing")
        self.gnn_dropout = torch.nn.Dropout(self.hidden_dropout_rate)
        self.gnn_layers = self._build_layers()

    def set_graph(self, triples, num_entities: Optional[int] = None, num_relations: Optional[int] = None,
                  inverse_relations=None) -> "RelationalGNNEncoder":
        """Attach the indexed training triples this encoder passes messages over.

        Args:
            triples: ``(N, 3)`` array/tensor of indexed ``[h, r, t]`` training facts.
            num_entities: Entity vocabulary size to validate against (defaults to
                the model's own).
            num_relations: Relation vocabulary size to validate against (defaults
                to the model's own).
            inverse_relations: Mapping from a direct relation id to the id stating the
                same fact in the other direction, as created by reciprocal preprocessing.
        """
        device = self.device
        triples = torch.as_tensor(triples, dtype=torch.long, device=device).clone()
        ne = self.num_entities if num_entities is None else num_entities
        if triples.ndim != 2 or triples.shape[1] != 3 or not len(triples):
            raise ValueError(f"{self.name} requires a nonempty [N, 3] training graph")
        if num_relations is not None and int(num_relations) != self.num_relations:
            raise ValueError("Relation vocabulary does not match the one this encoder was built for")
        if triples.min() < 0 or triples[:, [0, 2]].max() >= ne or triples[:, 1].max() >= self.num_relations:
            raise ValueError("Graph IDs are outside the supplied vocabulary")
        pairs = {int(k): int(v) for k, v in (inverse_relations or {}).items()}
        if (len(set(pairs.values())) != len(pairs) or set(pairs) & set(pairs.values())
                or any(min(k, v) < 0 or max(k, v) >= self.num_relations for k, v in pairs.items())):
            raise ValueError("Inverse relation mapping must contain disjoint, valid direct/inverse pairs")
        mapping = torch.full((self.num_relations,), -1, dtype=torch.long, device=device)
        for direct, inverse in pairs.items():
            mapping[direct] = inverse
            mapping[inverse] = direct
        self.graph_triples = triples
        self.inverse_relation = mapping
        self.edge_index, self.edge_type = triples[:, [0, 2]].T, triples[:, 1]
        return self

    def save_graph(self, path) -> None:
        """Write the attached graph beside the checkpoint."""
        self._require_graph()
        torch.save(dict(version=1, num_entities=self.num_entities, num_relations=self.num_relations,
                        graph_triples=self.graph_triples.cpu(),
                        inverse_relation=self.inverse_relation.cpu()), path)

    def load_graph(self, path) -> "RelationalGNNEncoder":
        """Restore the graph written by :meth:`save_graph`."""
        if not Path(path).is_file():
            raise FileNotFoundError(f"{self.name} requires its graph artifact: {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("version") != 1:
            raise ValueError(f"Unsupported {self.name} graph artifact version")
        if (data["num_entities"], data["num_relations"]) != (self.num_entities, self.num_relations):
            raise ValueError(f"{self.name} graph vocabulary does not match the experiment")
        self.graph_triples = data["graph_triples"].to(self.device)
        self.inverse_relation = data["inverse_relation"].to(self.device)
        self.edge_index, self.edge_type = self.graph_triples[:, [0, 2]].T, self.graph_triples[:, 1]
        return self

    def _require_graph(self) -> Tuple[int, int]:
        if self.graph_triples is None:
            raise RuntimeError(
                f"Attach a training graph with set_graph() (or args['train_set_idx']) "
                f"before scoring {self.name}")
        return self.num_entities, self.num_relations

    def _build_layers(self) -> torch.nn.ModuleList:
        """Return the per-layer GNN modules. Must be implemented by subclasses."""
        raise NotImplementedError

    def _apply_layer(self, layer_idx: int, layer: torch.nn.Module, x: torch.FloatTensor,
                     edge_index: torch.LongTensor, edge_type: torch.LongTensor) -> torch.FloatTensor:
        """Apply one GNN layer (incl. any layer-specific activation). Subclasses implement this."""
        raise NotImplementedError

    def _training_edges(self, queries: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Return the propagation graph with this batch's own query edges removed.

        Withholds edges being predicted (R-GCN, Schlichtkrull et al. 2018, Section 4.2),
        including the mirrored fact under a reciprocal relation, so message passing can't
        answer a query over the edge it's scoring. A no-op outside training.

        Args:
            queries: ``(B, 2)`` tensor of indexed ``[head, relation]`` queries.
        """
        _, num_relations = self._require_graph()
        if not self.training:
            return self.edge_index, self.edge_type
        # (head, relation) flattened to one int so isin can test set membership.
        qkeys = queries[:, 0] * num_relations + queries[:, 1]
        head, tail, relation = self.edge_index[0], self.edge_index[1], self.edge_type
        remove = torch.isin(head * num_relations + relation, qkeys)
        reciprocal = self.inverse_relation[relation]
        has_reciprocal = reciprocal >= 0
        if bool(torch.any(has_reciprocal)):
            mirrored = torch.isin(tail * num_relations + reciprocal.clamp(min=0), qkeys)
            remove = remove | (has_reciprocal & mirrored)
        if not torch.any(remove):
            return self.edge_index, self.edge_type
        keep = ~remove
        return self.edge_index[:, keep], self.edge_type[keep]

    def encode(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor) -> torch.FloatTensor:
        """Run message passing over ``(edge_index, edge_type)`` once.

        Returns:
            ``(num_entities, embedding_dim)`` refined entity representations.
        """
        x = self.entity_embeddings.weight
        num_layers = len(self.gnn_layers)
        for layer_idx, layer in enumerate(self.gnn_layers):
            x = self._apply_layer(layer_idx, layer, x, edge_index, edge_type)
            if layer_idx < num_layers - 1:
                x = self.gnn_dropout(x)
        return x

    def score(self, head_ent_emb: torch.FloatTensor, rel_ent_emb: torch.FloatTensor,
              tail_ent_emb: torch.FloatTensor) -> torch.FloatTensor:
        """Score a triple with the DistMult decoder."""
        return (self.hidden_dropout(self.hidden_normalizer(head_ent_emb * rel_ent_emb)) * tail_ent_emb).sum(dim=1)

    def k_vs_all_score(self, emb_h: torch.FloatTensor, emb_r: torch.FloatTensor,
                       emb_E: torch.FloatTensor) -> torch.FloatTensor:
        """Score a head/relation batch against all entities with the DistMult decoder."""
        return torch.mm(self.hidden_dropout(self.hidden_normalizer(emb_h * emb_r)), emb_E.transpose(1, 0))

    def forward_triples(self, x: torch.LongTensor) -> torch.FloatTensor:
        idx_head_entity, idx_relation, idx_tail_entity = x[:, 0], x[:, 1], x[:, 2]
        encoded_ent_emb = self.encode(*self._training_edges(x[:, :2]))
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(encoded_ent_emb.index_select(0, idx_head_entity)))
        tail_ent_emb = self.normalize_tail_entity_embeddings(encoded_ent_emb.index_select(0, idx_tail_entity))
        rel_ent_emb = self.normalize_relation_embeddings(
            self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        return self.score(head_ent_emb, rel_ent_emb, tail_ent_emb)

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        idx_head_entity, idx_relation = x[:, 0], x[:, 1]
        encoded_ent_emb = self.encode(*self._training_edges(x[:, :2]))
        head_ent_emb = self.normalize_head_entity_embeddings(
            self.input_dp_ent_real(encoded_ent_emb.index_select(0, idx_head_entity)))
        rel_ent_emb = self.normalize_relation_embeddings(
            self.input_dp_rel_real(self.relation_embeddings(idx_relation)))
        return self.k_vs_all_score(head_ent_emb, rel_ent_emb, encoded_ent_emb)


class RGCN(RelationalGNNEncoder):
    """Relational Graph Convolutional Network (R-GCN) knowledge graph encoder.

    Reference: Schlichtkrull, Kipf, Bloem, van den Berg, Titov, Welling,
    *Modeling Relational Data with Graph Convolutional Networks*, ESWC 2018.
    https://arxiv.org/abs/1703.06103
    """

    name = "RGCN"

    def _build_layers(self) -> torch.nn.ModuleList:
        num_bases = self.args.get("gnn_num_bases", None)
        return torch.nn.ModuleList([
            RGCNConv(
                in_channels=self.embedding_dim,
                out_channels=self.embedding_dim,
                num_relations=self.num_relations,
                num_bases=num_bases,
            )
            for _ in range(self.gnn_num_layers)
        ])

    def _apply_layer(self, layer_idx: int, layer: torch.nn.Module, x: torch.FloatTensor,
                     edge_index: torch.LongTensor, edge_type: torch.LongTensor) -> torch.FloatTensor:
        x = layer(x, edge_index, edge_type)
        if layer_idx < self.gnn_num_layers - 1:
            x = F.relu(x)
        return x


class GATv2(RelationalGNNEncoder):
    """GATv2 knowledge graph encoder, relation-aware via edge features.

    Reference: Brody, Alon, Yahav, *How Attentive are Graph Attention
    Networks?*, ICLR 2022 (GATv2). https://arxiv.org/abs/2105.14491

    Relation embeddings are passed as ``edge_attr`` to ``GATv2Conv``, so relation
    identity enters the attention coefficients. Attention heads are concatenated
    on hidden layers and averaged on the final layer.
    """

    name = "GATv2"

    def __init__(self, args: dict):
        super().__init__(args)
        self.gnn_relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.gnn_relation_embeddings.weight.data)

    def _build_layers(self) -> torch.nn.ModuleList:
        heads = int(self.args.get("gnn_attn_heads", 4))
        assert self.embedding_dim % heads == 0, (
            f"embedding_dim ({self.embedding_dim}) must be divisible by gnn_attn_heads ({heads}) "
            "so concatenated per-head outputs stay at embedding_dim."
        )
        per_head_dim = self.embedding_dim // heads
        layers = torch.nn.ModuleList()
        for layer_idx in range(self.gnn_num_layers):
            is_last_layer = layer_idx == self.gnn_num_layers - 1
            if is_last_layer:
                layers.append(GATv2Conv(
                    in_channels=self.embedding_dim,
                    out_channels=self.embedding_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=self.embedding_dim,
                    dropout=self.hidden_dropout_rate,
                ))
            else:
                layers.append(GATv2Conv(
                    in_channels=self.embedding_dim,
                    out_channels=per_head_dim,
                    heads=heads,
                    concat=True,
                    edge_dim=self.embedding_dim,
                    dropout=self.hidden_dropout_rate,
                ))
        return layers

    def _apply_layer(self, layer_idx: int, layer: torch.nn.Module, x: torch.FloatTensor,
                     edge_index: torch.LongTensor, edge_type: torch.LongTensor) -> torch.FloatTensor:
        edge_attr = self.gnn_relation_embeddings(edge_type)
        x = layer(x, edge_index, edge_attr=edge_attr)
        if layer_idx < self.gnn_num_layers - 1:
            x = F.elu(x)
        return x
