from ..types import torch
from .base_model import BaseKGE
from .octonion import octonion_mul


def conjugate(*, C, dim):
    first, rest = C.hsplit((C.size(dim=1) // dim,))
    return torch.hstack((first, rest.neg()))


def hermitian(*, C_1, C_2, dim):
    if dim == 1:
        return C_1.mul(C_2)
    else:
        a, b = C_1.hsplit(2)
        c, d = C_2.hsplit(2)
        x = hermitian(C_1=conjugate(C=a, dim=dim), C_2=c, dim=dim // 2)
        y = hermitian(C_1=b, C_2=d, dim=dim // 2)
        return x.sub(y)


def hermitianproduct(*, C_1, C_2, dim):
    a_s = C_1.size(0)
    b_s = C_2.size(0)
    a = torch.hstack(tuple(item.repeat_interleave(b_s, dim=0) for item in C_1.hsplit(16)))
    b = torch.hstack(tuple(item.repeat(a_s, 1) for item in C_2.hsplit(16)))
    return hermitian(C_1=a, C_2=b, dim=16).sum(dim=1).reshape((a_s, b_s))


def innerproduct(*, C_1, C_2):
    return C_1.mm(C_2.transpose(1, 0))


def o_mul(*, O_1, O_2):
    return torch.hstack(octonion_mul(O_1=O_1.hsplit(8), O_2=O_2.hsplit(8)))


def s_mul(*, S_1, S_2):
    a, b = S_1.hsplit(2)
    c, d = S_2.hsplit(2)
    return torch.hstack((
        o_mul(O_1=a, O_2=c).sub(o_mul(O_1=d, O_2=conjugate(C=b, dim=16))),
        o_mul(O_1=conjugate(C=a, dim=16), O_2=d).add(o_mul(O_1=c, O_2=b)),
    ))


class SedE(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        assert self.embedding_dim % 16 == 0, 'number of embedding dimensions should be divisible by 16'
        self.name = 'SedE'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        # Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)

        # Apply multiplication.
        e = s_mul(S_1=head_ent_emb, S_2=rel_ent_emb)

        # Inner product.
        #return innerproduct(C_1=e, C_2=tail_ent_emb)

        # Hermitian product.
        return hermitianproduct(C_1=e, C_2=tail_ent_emb, dim=16).sum(dim=1)

    def forward_k_vs_all(self, x: torch.Tensor):
        """
        Given a head entity and a relation (h,r), we compute scores for all entities.
        [score(h,r,x)|x \\in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """

        # Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)

        # Apply multiplication.
        e = s_mul(S_1=head_ent_emb, S_2=rel_ent_emb)

        # Inner product.
        #return innerproduct(C_1=e, C_2=self.entity_embeddings.weight)

        # Hermitian product.
        return hermitianproduct(C_1=e, C_2=self.entity_embeddings.weight, dim=16)
