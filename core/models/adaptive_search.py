from .base_model import *
import numpy as np


class AdaptE(BaseKGE):
    """
    Adaptive KGE
    from real to quaternions/octonions
    from real to multi-vectors
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = 'AdaptE'

        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)
        xavier_normal_(self.emb_ent_real.weight.data), xavier_normal_(self.emb_rel_real.weight.data)

        self.bn_ent_real = torch.nn.LayerNorm(self.embedding_dim)
        self.bn_rel_real = torch.nn.LayerNorm(self.embedding_dim)

        self.bn_hidden_real = torch.nn.LayerNorm(self.embedding_dim)

        self.bn_hidden_a, self.bn_hidden_b, self.bn_hidden_c, self.bn_hidden_d = None, None, None, None
        self.bn_hidden_aa, self.bn_hidden_bb, self.bn_hidden_cc, self.bn_hidden_dd = None, None, None, None

        self.losses = []
        # If the current loss is not better than 60% of the previous interval loss
        # Upgrade.
        self.moving_average_interval = 3
        self.decision_criterion = .6
        self.mode = 0
        self.moving_average = 0

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.emb_ent_real.weight.data.data.detach(), self.emb_rel_real.weight.data.detach()

    def forward_k_vs_all(self, x: torch.Tensor):
        e1_idx: torch.Tensor
        rel_idx: torch.Tensor
        e1_idx, rel_idx = x[:, 0], x[:, 1]
        # (1) Retrieve head entity and relation embeddings
        emb_head_real = self.bn_ent_real(self.emb_ent_real(e1_idx))
        emb_rel_real = self.bn_rel_real(self.emb_rel_real(rel_idx))
        # (2) Compute real score
        score = torch.mm(self.bn_hidden_real(emb_head_real * emb_rel_real), self.emb_ent_real.weight.transpose(1, 0))
        if self.mode >= 1:
            # (2) Split (1) into real and imaginary parts
            emb_head_real, emb_head_imag = torch.hsplit(emb_head_real, 2)
            emb_rel_real, emb_rel_imag = torch.hsplit(emb_rel_real, 2)
            # (3) Compute hermitian inner product.
            all_entity_emb_real, all_entity_emb_imag = torch.hsplit(self.emb_ent_real.weight, 2)
            real_real_real = torch.mm(self.bn_hidden_a(emb_head_real * emb_rel_real),
                                      all_entity_emb_real.transpose(1, 0))
            real_imag_imag = torch.mm(self.bn_hidden_b(emb_head_real * emb_rel_imag),
                                      all_entity_emb_imag.transpose(1, 0))
            imag_real_imag = torch.mm(self.bn_hidden_c(emb_head_imag * emb_rel_real),
                                      all_entity_emb_imag.transpose(1, 0))
            imag_imag_real = torch.mm(self.bn_hidden_d(emb_head_imag * emb_rel_imag),
                                      all_entity_emb_real.transpose(1, 0))
            # (4)
            score += real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

        if self.mode >= 2:
            # (5) Split (1) into real and 3 imaginary parts.
            r_val, i_val, j_val, k_val = quaternion_mul(Q_1=torch.hsplit(emb_head_real, 4),
                                                        Q_2=torch.hsplit(emb_rel_real, 4))
            all_entity_emb_real, all_entity_emb_i, all_entity_emb_j, all_entity_emb_k = torch.hsplit(
                self.emb_ent_real.weight, 4)
            real_score = torch.mm(self.bn_hidden_aa(r_val), all_entity_emb_real.transpose(1, 0))
            i_score = torch.mm(self.bn_hidden_bb(i_val), all_entity_emb_i.transpose(1, 0))
            j_score = torch.mm(self.bn_hidden_cc(j_val), all_entity_emb_j.transpose(1, 0))
            k_score = torch.mm(self.bn_hidden_dd(k_val), all_entity_emb_k.transpose(1, 0))
            score += real_score + i_score + j_score + k_score

        return score

    def training_epoch_end(self, training_step_outputs):
        epoch_loss = float(training_step_outputs[0]['loss'].detach())
        self.losses.append(epoch_loss)
        if len(self.losses) % self.moving_average_interval == 0:
            self.losses = np.array(self.losses)
            avg_loss_in_last_epochs = self.losses.mean()
            tendency_of_decreasing_loss = (avg_loss_in_last_epochs > epoch_loss).mean()

            self.losses = []
            if tendency_of_decreasing_loss > self.decision_criterion:
                # current loss is lower than 60% of the previous few epochs:
                pass
            else:
                if self.mode == 0:
                    print('\nincrease the mode to complex numbers')
                    self.mode += 1
                    self.bn_hidden_a = torch.nn.LayerNorm(self.embedding_dim // 2)
                    self.bn_hidden_b = torch.nn.LayerNorm(self.embedding_dim // 2)
                    self.bn_hidden_c = torch.nn.LayerNorm(self.embedding_dim // 2)
                    self.bn_hidden_d = torch.nn.LayerNorm(self.embedding_dim // 2)
                    print('####')
                elif self.mode == 1:
                    print('\nincrease the mode to quaternions numbers')
                    self.mode += 1
                    self.bn_hidden_aa = torch.nn.LayerNorm(self.embedding_dim // 4)
                    self.bn_hidden_bb = torch.nn.LayerNorm(self.embedding_dim // 4)
                    self.bn_hidden_cc = torch.nn.LayerNorm(self.embedding_dim // 4)
                    self.bn_hidden_dd = torch.nn.LayerNorm(self.embedding_dim // 4)

                    print('####')

                else:
                    pass

        """
        if self.current_embedding_dim + self.add_dim_size < self.embedding_dim:
            epoch_loss = float(training_step_outputs[0]['loss'].detach())
            self.losses.append(epoch_loss)
            if len(self.losses) % self.moving_average_interval == 0:
                moving_average = sum(self.losses) / len(self.losses)
                self.losses.clear()
                diff = abs(moving_average - epoch_loss)

                if diff > epoch_loss * .1:
                    # do nothing
                    pass
                else:
        """
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
