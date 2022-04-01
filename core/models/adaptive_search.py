import torch
from .base_model import *
import numpy as np
from .static_funcs import quaternion_mul


def compute_quaternion_score(head, relation, tail):
    # (5) Split (1) into real and 3 imaginary parts.
    r_val, i_val, j_val, k_val = quaternion_mul(Q_1=torch.hsplit(head, 4),
                                                Q_2=torch.hsplit(relation, 4))
    emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k = torch.hsplit(tail, 4)
    real_score = (r_val * emb_tail_real).sum(dim=1)
    i_score = (i_val * emb_tail_i).sum(dim=1)
    j_score = (j_val * emb_tail_j).sum(dim=1)
    k_score = (k_val * emb_tail_k).sum(dim=1)
    return real_score + i_score + j_score + k_score


class AdaptE(BaseKGE):
    """ AdaptE: A linear combination of DistMult, ComplEx and QMult """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'AdaptE'
        try:
            assert self.embedding_dim % 4 == 0
        except AssertionError:
            print('AdaptE embedding size must be dividable by 4')
        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)

        self.norm_ent = self.normalizer_class(self.embedding_dim)
        self.norm_rel = self.normalizer_class(self.embedding_dim)
        self.norm_tail = self.normalizer_class(self.embedding_dim)

        self.losses = []
        # If the current loss is not better than 60% of the previous interval loss
        # Upgrade.
        self.moving_average_interval = 10
        self.decision_criterion = .8
        self.mode = 0
        self.moving_average = 0

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.emb_ent_real.weight.data.data.detach(), self.emb_rel_real.weight.data.detach()


    def forward_triples(self, x: torch.Tensor) -> torch.Tensor:
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e2_idx: torch.Tensor
        e1_idx, rel_idx, e2_idx = x[:, 0], x[:, 1], x[:, 2]
        head_ent_emb = self.norm_ent(self.emb_ent_real(e1_idx))
        rel_ent_emb = self.norm_rel(self.emb_rel_real(rel_idx))
        tail_ent_emb = self.norm_tail(self.emb_ent_real(e2_idx))
        # (1) real value.
        score = self.compute_real_score(head_ent_emb, rel_ent_emb, tail_ent_emb)

        if self.mode >= 1:
            # (2) Split (1) into real and imaginary parts.
            score += self.compute_complex_score(head_ent_emb, rel_ent_emb, tail_ent_emb)
        if self.mode >= 2:
            score += compute_quaternion_score(head_ent_emb, rel_ent_emb, tail_ent_emb)

        # Averaging helped on UMLS
        if self.mode == 0:
            return score
        elif self.mode == 1:
            return score / 2
        elif self.mode == 2:
            return score / 3
        else:
            raise KeyError

    def training_epoch_end(self, training_step_outputs):
        epoch_loss = float(training_step_outputs[0]['loss'].detach())
        self.losses.append(epoch_loss)
        if len(self.losses) % self.moving_average_interval == 0:
            # (1) Compute the average loss of previous epochs
            avg_loss_in_last_epochs = sum(self.losses) / len(self.losses)
            # (2) Is the current loss less than the average losses
            tendency_of_decreasing_loss = avg_loss_in_last_epochs > epoch_loss
            # Remove the oldest epoch loss saved.
            self.losses.pop(0)
            if tendency_of_decreasing_loss:
                """ The loss is decreasing """
            else:
                self.losses.clear()
                if self.mode == 0:
                    print('\nIncrease the mode to complex numbers')
                    self.mode += 1
                elif self.mode == 1:
                    print('\nincrease the mode to quaternions numbers')
                    self.mode += 1
                else:
                    pass

        """

                    # Either increase the embedding size or the multiplication
                    print('\nDouble the embedding size') 
                    # Leads to inferious results
                    x = nn.Embedding(self.num_entities, self.add_dim_size)
                    xavier_normal_(x.weight.data)
                    self.emb_ent_real.weight = nn.Parameter(
                        torch.cat((self.emb_ent_real.weight.detach(), x.weight.detach()), dim=1).data,
                        requires_grad=True)
                    x = nn.Embedding(self.num_relations, self.add_dim_size)
                    xavier_normal_(x.weight.data)
                    self.emb_rel_real.weight = nn.Parameter(
                        torch.cat((self.emb_rel_real.weight.detach(), x.weight.detach()), dim=1).data,
                        requires_grad=True)
                    del x
                    self.current_embedding_dim += self.add_dim_size
                    """

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
