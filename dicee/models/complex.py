from typing import Tuple

import torch

from .base_model import BaseKGE


class ConEx(BaseKGE):
    """ Convolutional ComplEx Knowledge Graph Embeddings"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'ConEx'
        # Convolution
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_of_output_channels,
                                      kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)
        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels
        self.fc1 = torch.nn.Linear(self.fc_num_input, self.embedding_dim)  # Hard compression.
        self.norm_fc1 = self.normalizer_class(self.embedding_dim)

        self.bn_conv2d = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    def residual_convolution(self, C_1: Tuple[torch.Tensor, torch.Tensor],
                             C_2: Tuple[torch.Tensor, torch.Tensor]) -> torch.FloatTensor:
        """
        Compute residual score of two complex-valued embeddings.
        :param C_1: a tuple of two pytorch tensors that corresponds complex-valued embeddings
        :param C_2: a tuple of two pytorch tensors that corresponds complex-valued embeddings
        :return:
        """
        emb_ent_real, emb_ent_imag_i = C_1
        emb_rel_real, emb_rel_imag_i = C_2
        # Think of x a n image of two complex numbers.
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim // 2),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim // 2),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim // 2),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim // 2)], 2)

        x = torch.nn.functional.relu(self.bn_conv2d(self.conv2d(x)))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = torch.nn.functional.relu(self.norm_fc1(self.fc1(x)))
        return torch.chunk(x, 2, dim=1)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b = C_3
        emb_tail_real, emb_tail_imag = torch.hsplit(self.entity_embeddings.weight, 2)
        emb_tail_real, emb_tail_imag = emb_tail_real.transpose(1, 0), emb_tail_imag.transpose(1, 0)
        # (4)
        real_real_real = torch.mm(a * emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.mm(a * emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.mm(b * emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.mm(b * emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        emb_tail_real, emb_tail_imag = torch.hsplit(tail_ent_emb, 2)

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b = C_3
        # (3) Compute hermitian inner product.
        real_real_real = (a * emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (a * emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (b * emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (b * emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_k_vs_sample(self, x: torch.Tensor, target_entity_idx: torch.Tensor):
        # @OTOD: Double check later.
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        # (3) Apply convolution operation on (2).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b = C_3

        # (batch size, num. selected entity, dimension)
        # tail_entity_emb = self.normalize_tail_entity_embeddings(self.entity_embeddings(target_entity_idx))
        tail_entity_emb = self.entity_embeddings(target_entity_idx)
        # complex vectors
        emb_tail_real, emb_tail_i = torch.tensor_split(tail_entity_emb, 2, dim=2)

        emb_tail_real = emb_tail_real.transpose(1, 2)
        emb_tail_i = emb_tail_i.transpose(1, 2)

        real_real_real = torch.bmm((a * emb_head_real * emb_rel_real).unsqueeze(1), emb_tail_real)
        real_imag_imag = torch.bmm((a * emb_head_real * emb_rel_imag).unsqueeze(1), emb_tail_i)
        imag_real_imag = torch.bmm((b * emb_head_imag * emb_rel_real).unsqueeze(1), emb_tail_i)
        imag_imag_real = torch.bmm((b * emb_head_imag * emb_rel_imag).unsqueeze(1), emb_tail_real)
        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        return score.squeeze(1)


class AConEx(BaseKGE):
    """ Additive Convolutional ComplEx Knowledge Graph Embeddings """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'AConEx'
        # Convolution
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_of_output_channels,
                                      kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)
        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels
        self.fc1 = torch.nn.Linear(self.fc_num_input, self.embedding_dim + self.embedding_dim)  # Hard compression.
        self.norm_fc1 = self.normalizer_class(self.embedding_dim + self.embedding_dim)

        self.bn_conv2d = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    def residual_convolution(self, C_1: Tuple[torch.Tensor, torch.Tensor],
                             C_2: Tuple[torch.Tensor, torch.Tensor]) -> torch.FloatTensor:
        """
        Compute residual score of two complex-valued embeddings.
        :param C_1: a tuple of two pytorch tensors that corresponds complex-valued embeddings
        :param C_2: a tuple of two pytorch tensors that corresponds complex-valued embeddings
        :return:
        """
        emb_ent_real, emb_ent_imag_i = C_1
        emb_rel_real, emb_rel_imag_i = C_2
        # (N,C,H,W) : A single channel 2D image.
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim // 2),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim // 2),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim // 2),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim // 2)], 2)

        x = torch.nn.functional.relu(self.bn_conv2d(self.conv2d(x)))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = torch.nn.functional.relu(self.norm_fc1(self.fc1(x)))
        #
        return torch.chunk(x, 4, dim=1)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        # (3) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b, c, d = C_3
        # (4) Retrieve tail entity embeddings.
        emb_tail_real, emb_tail_imag = torch.hsplit(self.entity_embeddings.weight, 2)
        # (5) Transpose (4).
        emb_tail_real, emb_tail_imag = emb_tail_real.transpose(1, 0), emb_tail_imag.transpose(1, 0)
        # (6) Hermitian inner product with additive Conv2D connection.
        real_real_real = torch.mm(a + emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.mm(b + emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.mm(c + emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.mm(d + emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        emb_tail_real, emb_tail_imag = torch.hsplit(tail_ent_emb, 2)
        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b, c, d = C_3
        # (3) Hermitian inner product with additive Conv2D connection.
        real_real_real = (a + emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (b + emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (c + emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (d + emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_k_vs_sample(self, x: torch.Tensor, target_entity_idx: torch.Tensor):
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        # (3) Apply convolution operation on (2).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_imag),
                                        C_2=(emb_rel_real, emb_rel_imag))
        a, b, c, d = C_3

        # (4) Retrieve selected tail entity embeddings
        tail_entity_emb = self.normalize_tail_entity_embeddings(self.entity_embeddings(target_entity_idx))
        # (5) Split (4) into real and imaginary parts.
        emb_tail_real, emb_tail_i = torch.tensor_split(tail_entity_emb, 2, dim=2)
        # (6) Transpose (5)
        emb_tail_real = emb_tail_real.transpose(1, 2)
        emb_tail_i = emb_tail_i.transpose(1, 2)
        # (7) Hermitian inner product with additive Conv2D connection
        # (7.1) Elementwise multiply (2) according to the Hermitian Inner Product order
        # (7.2) Additive connection: Add (3) into (7.1)
        # (7.3) Batch matrix multiplication (7.2) and tail entity embeddings.
        # https://pytorch.org/docs/stable/generated/torch.bmm.html
        # input.shape (N, 1, D), mat2.shape (N,D,1)
        real_real_real = torch.bmm((a + emb_head_real * emb_rel_real).unsqueeze(1), emb_tail_real)
        real_imag_imag = torch.bmm((b + emb_head_real * emb_rel_imag).unsqueeze(1), emb_tail_i)
        imag_real_imag = torch.bmm((c + emb_head_imag * emb_rel_real).unsqueeze(1), emb_tail_i)
        imag_imag_real = torch.bmm((d + emb_head_imag * emb_rel_imag).unsqueeze(1), emb_tail_real)
        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        # (N,1,1) => (N,1).
        return score.squeeze(1)


class ComplEx(BaseKGE):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'ComplEx'

    @staticmethod
    def score(head_ent_emb: torch.FloatTensor, rel_ent_emb: torch.FloatTensor, tail_ent_emb: torch.FloatTensor):
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        emb_tail_real, emb_tail_imag = torch.hsplit(tail_ent_emb, 2)
        # (3) Compute hermitian inner product.
        real_real_real = (emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    @staticmethod
    def k_vs_all_score(emb_h: torch.FloatTensor, emb_r: torch.FloatTensor, emb_E: torch.FloatTensor):
        """

        Parameters
        ----------
        emb_h
        emb_r
        emb_E

        Returns
        -------

        """
        emb_head_real, emb_head_imag = torch.hsplit(emb_h, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(emb_r, 2)
        # (3) Transpose Entity embedding matrix to perform matrix multiplications in Hermitian Product.
        emb_tail_real, emb_tail_imag = torch.hsplit(emb_E, 2)
        emb_tail_real, emb_tail_imag = emb_tail_real.transpose(1, 0), emb_tail_imag.transpose(1, 0)
        # (4) Compute hermitian inner product on embedding vectors.
        real_real_real = torch.mm(emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.mm(emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.mm(emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.mm(emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        return self.k_vs_all_score(head_ent_emb,rel_ent_emb,self.entity_embeddings.weight)

    def forward_k_vs_sample(self, x: torch.LongTensor, target_entity_idx: torch.LongTensor):
        # (b,2d), (b,2d)
        emb_h, emb_r = self.get_head_relation_representation(x)
        # (b,k,2d)
        emb_T = self.entity_embeddings(target_entity_idx)
        # (b,d), (b,d)
        emb_head_real, emb_head_imag = torch.hsplit(emb_h, 2)
        # (b,d), (b,d)
        emb_rel_real, emb_rel_imag = torch.hsplit(emb_r, 2)
        # (b,k,d), (b,k,d)
        emb_tail_real, emb_tail_imag = torch.split(emb_T, self.embedding_dim // 2, dim=-1)
        # Compute hermitian inner product on embedding vectors.
        real_real_real = torch.einsum("bd, bkd -> bk",emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.einsum("bd, bkd -> bk",emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.einsum("bd, bkd -> bk",emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.einsum("bd, bkd -> bk",emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real


class RotatE(BaseKGE):
    """RotatE: knowledge graph embedding by relational rotation in complex space.

    Represents each entity as a complex vector in ℂ^(d/2), obtained by
    splitting its ``d``-dimensional embedding into a real half and an
    imaginary half.  Each relation is represented by ``d/2`` phase angles
    θ that define a unit-modulus rotation ``r_i = e^{iθ_i}``; since a phase
    angle needs only one real number, the relation embedding table is
    reinitialised at half of the entity embedding size.  A true triple
    ``(h, r, t)`` should satisfy ``h ∘ r ≈ t`` under element-wise complex
    multiplication, giving the score::

        f(h, r, t) = margin - ||h ∘ r - t||_2

    Unlike TransE, RotatE can model symmetric, antisymmetric, inverse, and
    composition relation patterns.

    References
    ----------
    Sun et al., *RotatE: Knowledge Graph Embedding by Relational Rotation in
    Complex Space*, ICLR 2019.  https://arxiv.org/abs/1902.10197
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'RotatE'
        self.margin = 6.0

        if self.embedding_dim % 2 != 0:
            raise ValueError(
                f"RotatE requires an even embedding_dim (got {self.embedding_dim}). "
                "Entities are split evenly into real and imaginary halves.")
        self.half_dim = self.embedding_dim // 2

        # Relations only need d/2 phase angles, so the full-size relation
        # embedding table created by BaseKGE is reinitialised at half width.
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.half_dim)
        self.param_init(self.relation_embeddings.weight.data)

        # Rebuild relation normalisation to match the halved embedding width.
        normalizer = self.args.get("normalization")
        if normalizer == "LayerNorm":
            self.normalize_relation_embeddings = torch.nn.LayerNorm(self.half_dim)
        elif normalizer == "BatchNorm1d":
            self.normalize_relation_embeddings = torch.nn.BatchNorm1d(self.half_dim, affine=False)

    def _rotate(self, ent_emb: torch.FloatTensor, phase: torch.FloatTensor) -> torch.FloatTensor:
        """Apply a complex rotation ``ent ∘ e^{i·phase}`` and return the result as a real vector.

        Parameters
        ----------
        ent_emb : torch.FloatTensor
            Shape ``(batch_size, embedding_dim)`` -- ``[real (d/2) | imag (d/2)]``.
        phase : torch.FloatTensor
            Shape ``(batch_size, d/2)`` relation phase angles θ.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size, embedding_dim)`` rotated ``[real | imag]`` vector.
        """
        ent_re, ent_im = ent_emb[:, :self.half_dim], ent_emb[:, self.half_dim:]
        rel_re, rel_im = torch.cos(phase), torch.sin(phase)
        rot_re = ent_re * rel_re - ent_im * rel_im
        rot_im = ent_re * rel_im + ent_im * rel_re
        return torch.cat((rot_re, rot_im), dim=1)

    def score(self, head_ent_emb: torch.FloatTensor, rel_ent_emb: torch.FloatTensor,
              tail_ent_emb: torch.FloatTensor) -> torch.FloatTensor:
        """Score a batch of triples using the RotatE margin-distance formula.

        Parameters
        ----------
        head_ent_emb, tail_ent_emb : torch.FloatTensor
            Each has shape ``(batch_size, embedding_dim)``.
        rel_ent_emb : torch.FloatTensor
            Shape ``(batch_size, d/2)`` relation phase angles θ.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size,)`` scores equal to
            ``margin - ||h ∘ r - t||_2``.
        """
        hr = self._rotate(head_ent_emb, rel_ent_emb)
        return self.margin - torch.nn.functional.pairwise_distance(hr, tail_ent_emb, p=2)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        """KvsAll forward pass: score head/relation against all entities.

        Computes ``margin - ||h ∘ r - e||_2`` for every entity embedding *e*.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, 2)`` integer tensor ``[head_idx, relation_idx]``.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size, num_entities)`` score matrix.
        """
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        hr = self._rotate(emb_head_real, emb_rel_real)
        distance = torch.nn.functional.pairwise_distance(torch.unsqueeze(hr, 1),
                                                          self.entity_embeddings.weight, p=2)
        return self.margin - distance

    def forward_k_vs_sample(self, x: torch.Tensor, target_entity_idx: torch.Tensor) -> torch.FloatTensor:
        """KvsSample forward pass: score head/relation against a sampled entity subset.

        Computes ``margin - ||h ∘ r - e||_2`` for each of the *k* sampled
        entities *e*.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, 2)`` integer tensor ``[head_idx, relation_idx]``.
        target_entity_idx : torch.Tensor
            Shape ``(batch_size, k)`` indices of the *k* target entities per sample.

        Returns
        -------
        torch.FloatTensor
            Shape ``(batch_size, k)`` score matrix.
        """
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        hr = self._rotate(emb_head_real, emb_rel_real)  # (B, d)
        emb_tail = self.entity_embeddings(target_entity_idx)  # (B, k, d)
        distance = torch.nn.functional.pairwise_distance(hr.unsqueeze(1), emb_tail, p=2)
        return self.margin - distance

