from typing import Tuple
import torch
from .base_model import BaseKGE


class ConEx(BaseKGE):
    """
    ConEx (Convolutional ComplEx) is a Knowledge Graph Embedding model that extends ComplEx embeddings with convolutional layers.
    It integrates convolutional neural networks into the embedding process to capture complex patterns in the data.

    Parameters
    ----------
    args : dict
        A dictionary of arguments containing hyperparameters and settings for the model, such as embedding dimensions,
        kernel size, number of output channels, and dropout rates.

    Attributes
    ----------
    name : str
        The name identifier for the ConEx model.
    conv2d : torch.nn.Conv2d
        A 2D convolutional layer used for processing complex-valued embeddings.
    fc1 : torch.nn.Linear
        A fully connected linear layer for compressing the output of the convolutional layer.
    norm_fc1 : Normalizer
        Normalization layer applied after the fully connected layer.
    bn_conv2d : torch.nn.BatchNorm2d
        Batch normalization layer applied after the convolutional operation.
    feature_map_dropout : torch.nn.Dropout2d
        Dropout layer applied to the output of the convolutional layer.

    Methods
    -------
    residual_convolution(C_1: Tuple[torch.Tensor, torch.Tensor], C_2: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
        Performs a residual convolution operation on two complex-valued embeddings.
    forward_k_vs_all(x: torch.Tensor) -> torch.FloatTensor
        Computes scores in a K-vs-All setting using convolutional operations on embeddings.
    forward_triples(x: torch.Tensor) -> torch.FloatTensor
        Computes scores for a batch of triples using convolutional operations.
    forward_k_vs_sample(x: torch.Tensor, target_entity_idx: torch.Tensor) -> torch.Tensor
        Computes scores against a sampled subset of entities using convolutional operations.

    Notes
    -----
    ConEx combines complex-valued embeddings with convolutional neural networks to capture intricate patterns and interactions
    in the knowledge graph, potentially leading to improved performance on tasks like link prediction.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "ConEx"
        # Convolution
        self.conv2d = torch.nn.Conv2d(
            in_channels=1,
            out_channels=self.num_of_output_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=1,
            padding=1,
            bias=True,
        )
        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels
        self.fc1 = torch.nn.Linear(
            self.fc_num_input, self.embedding_dim
        )  # Hard compression.
        self.norm_fc1 = self.normalizer_class(self.embedding_dim)

        self.bn_conv2d = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    def residual_convolution(
        self,
        C_1: Tuple[torch.Tensor, torch.Tensor],
        C_2: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Computes the residual score of two complex-valued embeddings by applying convolutional operations.
        This method is a key component of the ConEx model, combining complex embeddings with convolutional neural networks.

        Parameters
        ----------
        C_1 : Tuple[torch.Tensor, torch.Tensor]
            A tuple consisting of two PyTorch tensors representing the real and imaginary components of the first complex-valued embedding.
        C_2 : Tuple[torch.Tensor, torch.Tensor]
            A tuple consisting of two PyTorch tensors representing the real and imaginary components of the second complex-valued embedding.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor]
            A tuple of two tensors, representing the real and imaginary parts of the convolutionally transformed embeddings.

        Notes
        -----
        The method involves concatenating the real and imaginary components of the embeddings, applying a 2D convolution,
        followed by batch normalization, ReLU activation, dropout, and a fully connected layer. This process is intended to
        capture complex interactions between the embeddings in a convolutional manner.
        """
        emb_ent_real, emb_ent_imag_i = C_1
        emb_rel_real, emb_rel_imag_i = C_2
        # Think of x a n image of two complex numbers.
        x = torch.cat(
            [
                emb_ent_real.view(-1, 1, 1, self.embedding_dim // 2),
                emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim // 2),
                emb_rel_real.view(-1, 1, 1, self.embedding_dim // 2),
                emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim // 2),
            ],
            2,
        )

        x = torch.nn.functional.relu(self.bn_conv2d(self.conv2d(x)))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = torch.nn.functional.relu(self.norm_fc1(self.fc1(x)))
        return torch.chunk(x, 2, dim=1)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Computes scores in a K-vs-All setting using convolutional operations on complex-valued embeddings.
        This method is used for evaluating the performance of the model by computing scores for each head entity
        and relation pair against all possible tail entities.

        Parameters
        ----------
        x : torch.Tensor
            A tensor representing a batch of head entities and relations. Expected tensor shape: (n, 2),
            where 'n' is the batch size and '2' represents head entity and relation pairs.

        Returns
        -------
        torch.FloatTensor
            A tensor containing the scores for each head entity and relation pair against all possible tail entities.
            Tensor shape: (n, |E|), where '|E|' is the number of entities in the knowledge graph.

        Notes
        -----
        The method retrieves embeddings for head entities and relations, splits them into real and imaginary parts,
        and applies a convolution operation. It then computes the Hermitian product of the transformed embeddings
        with all tail entity embeddings to generate scores. This approach allows for capturing complex relational patterns
        in the knowledge graph.
        """
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(
            C_1=(emb_head_real, emb_head_imag), C_2=(emb_rel_real, emb_rel_imag)
        )
        a, b = C_3
        emb_tail_real, emb_tail_imag = torch.hsplit(self.entity_embeddings.weight, 2)
        emb_tail_real, emb_tail_imag = emb_tail_real.transpose(
            1, 0
        ), emb_tail_imag.transpose(1, 0)
        # (4)
        real_real_real = torch.mm(a * emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.mm(a * emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.mm(b * emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.mm(b * emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Computes scores for a batch of triples using convolutional operations on complex-valued embeddings.
        This method is crucial for evaluating the performance of the model on individual triples in the
        knowledge graph.

        Parameters
        ----------
        x : torch.Tensor
            A tensor representing a batch of triples. Each triple consists of indices for a head entity,
            a relation, and a tail entity. Expected tensor shape: (n, 3), where 'n' is the number of triples.

        Returns
        -------
        torch.FloatTensor
            A tensor containing the scores for each triple in the batch. Tensor shape: (n,), where 'n'
            is the number of triples.

        Notes
        -----
        The method retrieves embeddings for head entities, relations, and tail entities, and splits them
        into real and imaginary parts. It then applies a convolution operation on these embeddings and
        computes the Hermitian inner product, which involves a combination of real and imaginary parts
        of the embeddings. This process is designed to capture complex relational patterns and interactions
        within the knowledge graph, leveraging the power of convolutional neural networks.
        """
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        emb_tail_real, emb_tail_imag = torch.hsplit(tail_ent_emb, 2)

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(
            C_1=(emb_head_real, emb_head_imag), C_2=(emb_rel_real, emb_rel_imag)
        )
        a, b = C_3
        # (3) Compute hermitian inner product.
        real_real_real = (a * emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (a * emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (b * emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (b * emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_k_vs_sample(
        self, x: torch.Tensor, target_entity_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes scores against a sampled subset of entities using convolutional operations
        on complex-valued embeddings. This method is particularly useful for large knowledge graphs
        where computing scores against all entities is computationally expensive.

        Parameters
        ----------
        x : torch.Tensor
            A tensor representing a batch of head entities and relations. Expected tensor shape:
            (batch_size, 2), where 'batch_size' is the number of head entity and relation pairs.
        target_entity_idx : torch.Tensor
            A tensor of target entity indices for sampling. Tensor shape:
            (batch_size, num_selected_entities).

        Returns
        -------
        torch.Tensor
            A tensor containing the scores for each head entity and relation pair against the sampled
            subset of tail entities. Tensor shape: (batch_size, num_selected_entities).

        Notes
        -----
        The method first retrieves and processes the embeddings for head entities and relations. It then
        applies a convolution operation and computes the Hermitian inner product with the embeddings of
        the sampled tail entities. This process enables capturing complex relational patterns in a
        computationally efficient manner.
        """
        # @TODO: Double check later.
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        # (3) Apply convolution operation on (2).
        C_3 = self.residual_convolution(
            C_1=(emb_head_real, emb_head_imag), C_2=(emb_rel_real, emb_rel_imag)
        )
        a, b = C_3

        # (batch size, num. selected entity, dimension)
        # tail_entity_emb = self.normalize_tail_entity_embeddings(self.entity_embeddings(target_entity_idx))
        tail_entity_emb = self.entity_embeddings(target_entity_idx)
        # complex vectors
        emb_tail_real, emb_tail_i = torch.tensor_split(tail_entity_emb, 2, dim=2)

        emb_tail_real = emb_tail_real.transpose(1, 2)
        emb_tail_i = emb_tail_i.transpose(1, 2)

        real_real_real = torch.bmm(
            (a * emb_head_real * emb_rel_real).unsqueeze(1), emb_tail_real
        )
        real_imag_imag = torch.bmm(
            (a * emb_head_real * emb_rel_imag).unsqueeze(1), emb_tail_i
        )
        imag_real_imag = torch.bmm(
            (b * emb_head_imag * emb_rel_real).unsqueeze(1), emb_tail_i
        )
        imag_imag_real = torch.bmm(
            (b * emb_head_imag * emb_rel_imag).unsqueeze(1), emb_tail_real
        )
        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        return score.squeeze(1)


class AConEx(BaseKGE):
    """
    AConEx (Additive Convolutional ComplEx) extends the ConEx model by incorporating
    additive connections in the convolutional operations. This model integrates
    convolutional neural networks with complex-valued embeddings, emphasizing
    additive feature interactions for knowledge graph embeddings.

    Parameters
    ----------
    args : dict
        A dictionary of arguments containing hyperparameters and settings for the model,
        such as embedding dimensions, kernel size, number of output channels, and dropout rates.

    Attributes
    ----------
    name : str
        The name identifier for the AConEx model.
    conv2d : torch.nn.Conv2d
        A 2D convolutional layer used for processing complex-valued embeddings.
    fc_num_input : int
        The number of input features for the fully connected layer.
    fc1 : torch.nn.Linear
        A fully connected linear layer for compressing the output of the
        convolutional layer.
    norm_fc1 : Normalizer
        Normalization layer applied after the fully connected layer.
    bn_conv2d : torch.nn.BatchNorm2d
        Batch normalization layer applied after the convolutional operation.
    feature_map_dropout : torch.nn.Dropout2d
        Dropout layer applied to the output of the convolutional layer.

    Methods
    -------
    residual_convolution(C_1: Tuple[torch.Tensor, torch.Tensor],
                         C_2: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Performs a residual convolution operation on two complex-valued embeddings.
    forward_k_vs_all(x: torch.Tensor) -> torch.FloatTensor
        Computes scores in a K-vs-All setting using convolutional operations on embeddings.
    forward_triples(x: torch.Tensor) -> torch.FloatTensor
        Computes scores for a batch of triples using convolutional operations.
    forward_k_vs_sample(x: torch.Tensor, target_entity_idx: torch.Tensor)
        Computes scores against a sampled subset of entities using convolutional operations.

    Notes
    -----
    AConEx aims to enhance the modeling capabilities of knowledge graph embeddings
    by adding more complex interaction patterns through convolutional layers, potentially
    improving performance on tasks like link prediction.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "AConEx"
        # Convolution
        self.conv2d = torch.nn.Conv2d(
            in_channels=1,
            out_channels=self.num_of_output_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=1,
            padding=1,
            bias=True,
        )
        self.fc_num_input = self.embedding_dim * 2 * self.num_of_output_channels
        self.fc1 = torch.nn.Linear(
            self.fc_num_input, self.embedding_dim + self.embedding_dim
        )  # Hard compression.
        self.norm_fc1 = self.normalizer_class(self.embedding_dim + self.embedding_dim)

        self.bn_conv2d = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate)

    def residual_convolution(
        self,
        C_1: Tuple[torch.Tensor, torch.Tensor],
        C_2: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Computes the residual convolution of two complex-valued embeddings. This method
        is a core part of the AConEx model, applying convolutional neural network techniques
        to complex-valued embeddings to capture intricate relationships in the data.

        Parameters
        ----------
        C_1 : Tuple[torch.Tensor, torch.Tensor]
            A tuple of two PyTorch tensors representing the real and imaginary components
            of the first complex-valued embedding.
        C_2 : Tuple[torch.Tensor, torch.Tensor]
            A tuple of two PyTorch tensors representing the real and imaginary components
            of the second complex-valued embedding.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
            A tuple of four tensors, each representing a component of the convolutionally
            transformed embeddings. These components correspond to the modified real
            and imaginary parts of the input embeddings.

        Notes
        -----
        The method concatenates the real and imaginary components of the embeddings and
        applies a 2D convolution, followed by batch normalization, ReLU activation, dropout,
        and a fully connected layer. This convolutional process is designed to enhance
        the model's ability to capture complex patterns in knowledge graph embeddings.
        """
        emb_ent_real, emb_ent_imag_i = C_1
        emb_rel_real, emb_rel_imag_i = C_2
        # (N,C,H,W) : A single channel 2D image.
        x = torch.cat(
            [
                emb_ent_real.view(-1, 1, 1, self.embedding_dim // 2),
                emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim // 2),
                emb_rel_real.view(-1, 1, 1, self.embedding_dim // 2),
                emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim // 2),
            ],
            2,
        )

        x = torch.nn.functional.relu(self.bn_conv2d(self.conv2d(x)))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = torch.nn.functional.relu(self.norm_fc1(self.fc1(x)))
        #
        return torch.chunk(x, 4, dim=1)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Computes scores in a K-vs-All setting using convolutional and additive operations on
        complex-valued embeddings. This method evaluates the performance of the model by computing
        scores for each head entity and relation pair against all possible tail entities.

        Parameters
        ----------
        x : torch.Tensor
            A tensor representing a batch of head entities and relations. Expected tensor shape:
            (batch_size, 2), where 'batch_size' is the number of head entity and relation pairs.

        Returns
        -------
        torch.FloatTensor
            A tensor containing the scores for each head entity and relation pair against all possible
            tail entities. Tensor shape: (batch_size, |E|), where '|E|' is the number of entities
            in the knowledge graph.

        Notes
        -----
        The method first retrieves embeddings for head entities and relations, splits them into real
        and imaginary parts, and applies a convolutional operation. It then computes the Hermitian
        inner product with all tail entity embeddings, using an additive approach that combines the
        convolutional results with the original embeddings. This technique aims to capture complex
        relational patterns in the knowledge graph.
        """
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        # (3) Apply convolution operation on (1).
        C_3 = self.residual_convolution(
            C_1=(emb_head_real, emb_head_imag), C_2=(emb_rel_real, emb_rel_imag)
        )
        a, b, c, d = C_3
        # (4) Retrieve tail entity embeddings.
        emb_tail_real, emb_tail_imag = torch.hsplit(self.entity_embeddings.weight, 2)
        # (5) Transpose (4).
        emb_tail_real, emb_tail_imag = emb_tail_real.transpose(
            1, 0
        ), emb_tail_imag.transpose(1, 0)
        # (6) Hermitian inner product with additive Conv2D connection.
        real_real_real = torch.mm(a + emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.mm(b + emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.mm(c + emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.mm(d + emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_triples(self, x: torch.Tensor) -> torch.FloatTensor:
        """
        Computes scores for a batch of triples using convolutional operations and additive connections
        on complex-valued embeddings. This method is key for evaluating the model's performance on
        individual triples within the knowledge graph.

        Parameters
        ----------
        x : torch.Tensor
            A tensor representing a batch of triples. Each triple consists of indices for a head entity,
            a relation, and a tail entity. Expected tensor shape: (n, 3), where 'n' is the number of triples.

        Returns
        -------
        torch.FloatTensor
            A tensor containing the scores for each triple in the batch. Tensor shape: (n,), where 'n'
            is the number of triples.

        Notes
        -----
        The method retrieves embeddings for head entities, relations, and tail entities, and splits them
        into real and imaginary parts. It then applies a convolution operation on these embeddings and
        computes the Hermitian inner product, enhanced with an additive connection. This approach allows
        the model to capture complex relational patterns within the knowledge graph, potentially improving
        prediction accuracy and interpretability.
        """
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        emb_tail_real, emb_tail_imag = torch.hsplit(tail_ent_emb, 2)
        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(
            C_1=(emb_head_real, emb_head_imag), C_2=(emb_rel_real, emb_rel_imag)
        )
        a, b, c, d = C_3
        # (3) Hermitian inner product with additive Conv2D connection.
        real_real_real = (a + emb_head_real * emb_rel_real * emb_tail_real).sum(dim=1)
        real_imag_imag = (b + emb_head_real * emb_rel_imag * emb_tail_imag).sum(dim=1)
        imag_real_imag = (c + emb_head_imag * emb_rel_real * emb_tail_imag).sum(dim=1)
        imag_imag_real = (d + emb_head_imag * emb_rel_imag * emb_tail_real).sum(dim=1)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_k_vs_sample(
        self, x: torch.Tensor, target_entity_idx: torch.Tensor
    ) -> torch.FloatTensor:
        """
        Computes scores for a batch of samples (entity pairs) given a batch of queries. This method is used
        to predict the scores for different tail entities for a set of query triples.

        Parameters
        ----------
        x : torch.Tensor
            A tensor representing a batch of query triples. Each triple consists of indices for a head entity,
            a relation, and a dummy tail entity (used for scoring). Expected tensor shape: (n, 3), where 'n' is
            the number of query triples.

        target_entity_idx : torch.Tensor
            A tensor containing the indices of the target tail entities for which scores are to be predicted.
            Expected tensor shape: (n, m), where 'n' is the number of queries and 'm' is the number of target
            entities.

        Returns
        -------
        torch.FloatTensor
            A tensor containing the scores for each query-triple and target-entity pair. Tensor shape: (n, m),
            where 'n' is the number of queries and 'm' is the number of target entities.

        Notes
        -----
        This method retrieves embeddings for the head entities and relations in the query triples, splits them
        into real and imaginary parts, and applies convolutional operations with additive connections to capture
        complex patterns. It also retrieves embeddings for the target tail entities and computes Hermitian inner
        products to obtain scores, allowing the model to rank the tail entities based on their relevance to the queries.
        """
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        # (2) Split (1) into real and imaginary parts.
        emb_head_real, emb_head_imag = torch.hsplit(head_ent_emb, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(rel_ent_emb, 2)
        # (3) Apply convolution operation on (2).
        C_3 = self.residual_convolution(
            C_1=(emb_head_real, emb_head_imag), C_2=(emb_rel_real, emb_rel_imag)
        )
        a, b, c, d = C_3

        # (4) Retrieve selected tail entity embeddings
        tail_entity_emb = self.normalize_tail_entity_embeddings(
            self.entity_embeddings(target_entity_idx)
        )
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
        real_real_real = torch.bmm(
            (a + emb_head_real * emb_rel_real).unsqueeze(1), emb_tail_real
        )
        real_imag_imag = torch.bmm(
            (b + emb_head_real * emb_rel_imag).unsqueeze(1), emb_tail_i
        )
        imag_real_imag = torch.bmm(
            (c + emb_head_imag * emb_rel_real).unsqueeze(1), emb_tail_i
        )
        imag_imag_real = torch.bmm(
            (d + emb_head_imag * emb_rel_imag).unsqueeze(1), emb_tail_real
        )
        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        # (N,1,1) => (N,1).
        return score.squeeze(1)


class ComplEx(BaseKGE):
    """
    ComplEx (Complex Embeddings for Knowledge Graphs) is a model that extends
    the base knowledge graph embedding approach by using complex-valued embeddings.
    It emphasizes the interaction of real and imaginary components of embeddings
    to capture the asymmetric relationships often found in knowledge graphs.

    Parameters
    ----------
    args : dict
        A dictionary of arguments containing hyperparameters and settings for the model,
        such as embedding dimensions, learning rate, and regularization methods.

    Attributes
    ----------
    name : str
        The name identifier for the ComplEx model.

    Methods
    -------
    score(head_ent_emb: torch.FloatTensor, rel_ent_emb: torch.FloatTensor,
          tail_ent_emb: torch.FloatTensor) -> torch.FloatTensor
        Computes the score of a triple using the ComplEx scoring function.

    k_vs_all_score(emb_h: torch.FloatTensor, emb_r: torch.FloatTensor,
                   emb_E: torch.FloatTensor) -> torch.FloatTensor
        Computes scores in a K-vs-All setting using complex-valued embeddings.

    forward_k_vs_all(x: torch.LongTensor) -> torch.FloatTensor
        Performs a forward pass for K-vs-All scoring, returning scores for all entities.

    Notes
    -----
    ComplEx is particularly suited for modeling asymmetric relations and has been
    shown to perform well on various knowledge graph benchmarks. The use of complex
    numbers allows the model to encode additional information compared to real-valued models.
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "ComplEx"

    @staticmethod
    def score(
        head_ent_emb: torch.FloatTensor,
        rel_ent_emb: torch.FloatTensor,
        tail_ent_emb: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute the scoring function for a given triple using complex-valued embeddings.

        Parameters
        ----------
        head_ent_emb : torch.FloatTensor
            The complex embedding of the head entity.
        rel_ent_emb : torch.FloatTensor
            The complex embedding of the relation.
        tail_ent_emb : torch.FloatTensor
            The complex embedding of the tail entity.

        Returns
        -------
        torch.FloatTensor
            The score of the triple calculated using the Hermitian dot product of complex embeddings.

        Notes
        -----
        The scoring function exploits the complex vector space to model the interactions
        between entities and relations. It involves element-wise multiplication and
        summation of real and imaginary parts.
        """
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
    def k_vs_all_score(
        emb_h: torch.FloatTensor, emb_r: torch.FloatTensor, emb_E: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Compute scores for a head entity and relation against all entities in a K-vs-All scenario.

        Parameters
        ----------
        emb_h : torch.FloatTensor
            The complex embedding of the head entity.
        emb_r : torch.FloatTensor
            The complex embedding of the relation.
        emb_E : torch.FloatTensor
            The complex embeddings of all possible tail entities.

        Returns
        -------
        torch.FloatTensor
            Scores for all possible triples formed with the given head entity and relation.

        Notes
        -----
        This method is useful for tasks like link prediction where the model predicts
        the likelihood of a relation between a given entity pair.
        """
        emb_head_real, emb_head_imag = torch.hsplit(emb_h, 2)
        emb_rel_real, emb_rel_imag = torch.hsplit(emb_r, 2)
        # (3) Transpose Entity embedding matrix to perform matrix multiplications in Hermitian Product.
        emb_tail_real, emb_tail_imag = torch.hsplit(emb_E, 2)
        emb_tail_real, emb_tail_imag = emb_tail_real.transpose(
            1, 0
        ), emb_tail_imag.transpose(1, 0)
        # (4) Compute hermitian inner product on embedding vectors.
        real_real_real = torch.mm(emb_head_real * emb_rel_real, emb_tail_real)
        real_imag_imag = torch.mm(emb_head_real * emb_rel_imag, emb_tail_imag)
        imag_real_imag = torch.mm(emb_head_imag * emb_rel_real, emb_tail_imag)
        imag_imag_real = torch.mm(emb_head_imag * emb_rel_imag, emb_tail_real)
        return real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

    def forward_k_vs_all(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Perform a forward pass for K-vs-all scoring using complex-valued embeddings.

        Parameters
        ----------
        x : torch.LongTensor
            Tensor containing indices for head entities and relations.

        Returns
        -------
        torch.FloatTensor
            Scores for all triples formed with the given head entities and relations against all entities.

        Notes
        -----
        This method is typically used in training and evaluation of the model in a
        link prediction setting, where the goal is to rank all possible tail entities
        for a given head entity and relation.
        """
        # (1) Retrieve embeddings & Apply Dropout & Normalization.
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
        return self.k_vs_all_score(
            head_ent_emb, rel_ent_emb, self.entity_embeddings.weight
        )
