import torch
from .base_model import *
import numpy as np
from .static_funcs import quaternion_mul


class AdaptE(BaseKGE):
    """ AdaptE: A linear combination of DistMult, ComplEx and QMult """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'AdaptE'
        try:
            assert self.embedding_dim % 4 == 0
        except AssertionError:
            print('AdaptE embedding size must be dividable by 4')
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        xavier_normal_(self.entity_embeddings.weight.data), xavier_normal_(self.relation_embeddings.weight.data)

        self.input_dp_ent_real = torch.nn.Dropout(self.input_dropout_rate)
        self.input_dp_rel_real = torch.nn.Dropout(self.input_dropout_rate)

        self.normalize_head_entity_embeddings = self.normalizer_class(self.embedding_dim)
        self.normalize_relation_embeddings = self.normalizer_class(self.embedding_dim)
        self.normalize_tail_entity_embeddings = self.normalizer_class(self.embedding_dim)

        self.losses = []
        # If the current loss is not better than 60% of the previous interval loss
        # Upgrade.
        self.moving_average_interval = 10
        self.decision_criterion = .8
        self.mode = 0
        self.moving_average = 0

    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:

        # (1) Retrieve embeddings & Apply Dropout & Normalization
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Compute DistMult score on (1)
        score = self.compute_real_score(head_ent_emb, rel_ent_emb, tail_ent_emb)
        if self.mode >= 1:
            # (3) Compute ComplEx score on (1) and add it to (2)
            score += self.compute_complex_score(head_ent_emb, rel_ent_emb, tail_ent_emb)
        if self.mode >= 2:
            # (4) Compute QMult score on (1) and add it to (2)
            score += self.compute_quaternion_score(head_ent_emb, rel_ent_emb, tail_ent_emb)
        # (5) Average (2)
        if self.mode == 0:
            return score
        elif self.mode == 1:
            return score / 2
        elif self.mode == 2:
            return score / 3
        else:
            raise KeyError

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
        emb_head_real, emb_rel_real = self.get_head_relation_representation(x)
        score = torch.mm(self.hidden_dropout(self.hidden_normalizer(emb_head_real * emb_rel_real)),
                         self.entity_embeddings.weight.transpose(1, 0))
        if self.mode >= 1:
            # (2) Split (1) into real and imaginary parts.
            emb_head_real, emb_head_imag = torch.hsplit(emb_head_real, 2)
            emb_rel_real, emb_rel_imag = torch.hsplit(emb_rel_real, 2)
            # (3) Transpose Entity embedding matrix to perform matrix multiplications in Hermitian Product.
            emb_tail_real, emb_tail_imag = torch.hsplit(self.entity_embeddings.weight, 2)
            emb_tail_real, emb_tail_imag = emb_tail_real.transpose(1, 0), emb_tail_imag.transpose(1, 0)
            # (4) Compute hermitian inner product on embedding vectors.
            real_real_real = torch.mm(emb_head_real * emb_rel_real, emb_tail_real)
            real_imag_imag = torch.mm(emb_head_real * emb_rel_imag, emb_tail_imag)
            imag_real_imag = torch.mm(emb_head_imag * emb_rel_real, emb_tail_imag)
            imag_imag_real = torch.mm(emb_head_imag * emb_rel_imag, emb_tail_real)
            score += real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        if self.mode >= 2:
            # (2) Split (1) into real and imaginary parts.
            emb_head_real, emb_head_i, emb_head_j, emb_head_k = torch.hsplit(emb_head_real, 4)
            emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k = torch.hsplit(emb_rel_real, 4)
            r_val, i_val, j_val, k_val = quaternion_mul(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                                        Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))

            emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(self.entity_embeddings.weight, 4)
            emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = emb_tail_real.transpose(1, 0), emb_tail_i.transpose(1,
                                                                                                                    0), emb_tail_j.transpose(
                1, 0), emb_tail_k.transpose(1, 0)

            # (3)
            # (3.1) Dropout on (2)-result of quaternion multiplication.
            # (3.2) Inner product
            real_score = torch.mm(r_val, emb_tail_real)
            i_score = torch.mm(i_val, emb_tail_i)
            j_score = torch.mm(j_val, emb_tail_j)
            k_score = torch.mm(k_val, emb_tail_k)

            score += real_score + i_score + j_score + k_score
        # (5) Average (2)
        if self.mode == 0:
            return score
        elif self.mode == 1:
            return score / 2
        elif self.mode == 2:
            return score / 3
        else:
            raise KeyError

    def training_epoch_end(self, training_step_outputs):
        # (1) Store Epoch Loss.
        epoch_loss = float(training_step_outputs[0]['loss'].detach())
        self.losses.append(epoch_loss)
        # (2) Check whether we have enough epoch losses to compute moving average.
        if len(self.losses) % self.moving_average_interval == 0:
            # (2.1) Compute the average loss of epoch losses.
            avg_loss_in_last_epochs = sum(self.losses) / len(self.losses)
            # (2.2) Is the current epoch loss less than the current moving average of losses.
            tendency_of_decreasing_loss = avg_loss_in_last_epochs > epoch_loss
            # (2.3) Remove the oldest epoch loss saved.
            self.losses.pop(0)
            # (2.4) Check whether the moving average of epoch losses tends to decrease
            if tendency_of_decreasing_loss:
                # The loss is decreasing
                pass
            else:
                # (2.5) Stagnation detected.
                self.losses.clear()
                if self.mode == 0:
                    print('\nIncrease the mode to complex numbers')
                    self.mode += 1
                elif self.mode == 1:
                    print('\nincrease the mode to quaternions numbers')
                    self.mode += 1
                else:
                    # We may consider increasing the number of params
                    pass

    @staticmethod
    def compute_real_score(head, relation, tail):
        return (head * relation * tail).sum(dim=1)

    @staticmethod
    def compute_complex_score(head, relation, tail):
        emb_head_real, emb_head_imag = torch.hsplit(head, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(relation, 2)
        emb_tail_real, emb_tail_imag = torch.hsplit(tail, 2)

        # (3) Compute hermitian inner product.
        real_real_real = (emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    @staticmethod
    def compute_quaternion_score(head, relation, tail):
        # (5) Split (1) into real and 3 imaginary parts.
        r_val, i_val, j_val, k_val = quaternion_mul(Q_1=torch.hsplit(head, 4), Q_2=torch.hsplit(relation, 4))
        emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(tail, 4)
        real_score = (r_val * emb_tail_real).sum(dim=1)
        i_score = (i_val * emb_tail_i).sum(dim=1)
        j_score = (j_val * emb_tail_j).sum(dim=1)
        k_score = (k_val * emb_tail_k).sum(dim=1)
        return real_score + i_score + j_score + k_score
