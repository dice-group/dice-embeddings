from typing import Callable, List, Tuple
from .base_model import BaseKGE
import torch
import numpy as np

class FMult(BaseKGE):
    """
    FMult is a model for learning neural networks on knowledge graphs. It extends
    the base knowledge graph embedding model by integrating neural network computations
    with entity and relation embeddings. The model is designed to work with complex
    embeddings and utilizes a neural network-based approach for embedding interactions.

    Parameters
    ----------
    args : dict
        A dictionary of arguments containing hyperparameters and settings for the model,
        such as embedding dimensions and other model-specific parameters.

    Attributes
    ----------
    name : str
        The name identifier for the FMult model.
    entity_embeddings : torch.nn.Embedding
        Embedding layer for entities in the knowledge graph.
    relation_embeddings : torch.nn.Embedding
        Embedding layer for relations in the knowledge graph.
    k : int
        Dimension size for reshaping weights in neural network layers.
    num_sample : int
        The number of samples to consider in the model computations.
    gamma : torch.Tensor
        Randomly initialized weights for the neural network layers.
    roots : torch.Tensor
        Precomputed roots for Legendre polynomials.
    weights : torch.Tensor
        Precomputed weights for Legendre polynomials.

    Methods
    -------
    compute_func(weights: torch.FloatTensor, x: torch.Tensor) -> torch.FloatTensor
        Computes the output of a two-layer neural network for given weights and input.

    chain_func(weights: torch.FloatTensor, x: torch.Tensor) -> torch.Tensor
        Chains two linear neural network layers for a given input.

    forward_triples(idx_triple: torch.Tensor) -> torch.Tensor
        Performs a forward pass for a batch of triples and computes the embedding interactions.
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "FMult"
        self.entity_embeddings = torch.nn.Embedding(
            self.num_entities, self.embedding_dim
        )
        self.relation_embeddings = torch.nn.Embedding(
            self.num_relations, self.embedding_dim
        )
        self.param_init(self.entity_embeddings.weight.data), self.param_init(
            self.relation_embeddings.weight.data
        )
        self.k = int(np.sqrt(self.embedding_dim // 2))
        self.num_sample = 50
        # self.gamma = torch.rand(self.k, self.num_sample) [0,1) uniform=> worse results
        self.gamma = torch.randn(self.k, self.num_sample)  # N(0,1)
        # Lazy import
        from scipy.special import roots_legendre
        from scipy.special import roots_legendre

        roots, weights = roots_legendre(self.num_sample)
        self.roots = (
            torch.from_numpy(roots).repeat(self.k, 1).float()
        )  # shape self.k by self.n
        self.weights = (
            torch.from_numpy(weights).reshape(1, -1).float()
        )  # shape 1 by self.n

    def compute_func(
        self, weights: torch.FloatTensor, x: torch.Tensor
    ) -> torch.FloatTensor:
        """
        Compute the output of a two-layer neural network.

        Parameters
        ----------
        weights : torch.FloatTensor
            The weights of the neural network, split into two sets for two layers.
        x : torch.Tensor
            The input tensor for the neural network.

        Returns
        -------
        torch.FloatTensor
            The output tensor after passing through the two-layer neural network.
        """
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.view(n, self.k, self.k)
        w2 = w2.view(n, self.k, self.k)
        # (2) Forward Pass
        out1 = torch.tanh(w1 @ x)  # torch.sigmoid => worse results
        out2 = w2 @ out1
        return out2  # no non-linearity => better results

    def chain_func(self, weights: torch.FloatTensor, x: torch.Tensor) -> torch.Tensor:
        """
        Chain two linear layers of a neural network for given weights and input.

        Parameters
        ----------
        weights : torch.FloatTensor
            The weights of the neural network, split into two sets for two layers.
        x : torch.Tensor
            The input tensor for the neural network.

        Returns
        -------
        torch.Tensor
            The output tensor after chaining the two linear layers.
        """
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.view(n, self.k, self.k)
        w2 = w2.view(n, self.k, self.k)
        # (2) Perform the forward pass
        out1 = torch.tanh(torch.bmm(w1, x))
        out2 = torch.bmm(w2, out1)
        return out2

    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of triples to compute embedding interactions.

        Parameters
        ----------
        idx_triple : torch.Tensor
            Tensor containing indices of triples.

        Returns
        -------
        torch.Tensor
            The computed scores for the batch of triples.
        """
        # (1) Retrieve embeddings: batch, \mathbb R^d
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(
            idx_triple
        )
        # (2) Compute NNs on \Gamma
        # Logits via FDistMult...
        # h_x = self.compute_func(head_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        # r_x = self.compute_func(rel_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        # t_x = self.compute_func(tail_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        # out = h_x * r_x * t_x  # batch, \mathbb{R}^k, |gamma|
        # (2) Compute NNs on \Gamma
        self.gamma = self.gamma.to(head_ent_emb.device)

        h_x = self.compute_func(
            head_ent_emb, x=self.gamma
        )  # batch, \mathbb{R}^k, |\Gamma|
        t_x = self.compute_func(
            tail_ent_emb, x=self.gamma
        )  # batch, \mathbb{R}^k, |\Gamma|
        r_h_x = self.chain_func(
            weights=rel_ent_emb, x=h_x
        )  # batch, \mathbb{R}^k, |\Gamma|
        # (3) Compute |\Gamma| predictions
        out = torch.sum(r_h_x * t_x, dim=1)  # batch, |gamma| #
        # (4) Average (3) over \Gamma
        out = torch.mean(out, dim=1)  # batch
        return out


class GFMult(BaseKGE):
    """
    GFMult (Graph Function Multiplication) extends the base knowledge graph embedding
    model by integrating neural network computations with entity and relation embeddings.
    This model is designed to leverage the strengths of neural networks in capturing
    complex interactions within knowledge graphs.

    Parameters
    ----------
    args : dict
        A dictionary of arguments containing hyperparameters and settings for the model,
        such as embedding dimensions, learning rate, and other model-specific parameters.

    Attributes
    ----------
    name : str
        The name identifier for the GFMult model.
    entity_embeddings : torch.nn.Embedding
        Embedding layer for entities in the knowledge graph.
    relation_embeddings : torch.nn.Embedding
        Embedding layer for relations in the knowledge graph.
    k : int
        The dimension size for reshaping weights in neural network layers.
    num_sample : int
        The number of samples to use in the model computations.
    roots : torch.Tensor
        Precomputed roots for Legendre polynomials, repeated for each dimension.
    weights : torch.Tensor
        Precomputed weights for Legendre polynomials.

    Methods
    -------
    compute_func(weights: torch.FloatTensor, x: torch.Tensor) -> torch.FloatTensor
        Computes the output of a two-layer neural network for given weights and input.

    chain_func(weights: torch.FloatTensor, x: torch.Tensor) -> torch.Tensor
        Chains two linear neural network layers for a given input.

    forward_triples(idx_triple: torch.Tensor) -> torch.Tensor
        Performs a forward pass for a batch of triples and computes the embedding interactions.
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "GFMult"
        self.entity_embeddings = torch.nn.Embedding(
            self.num_entities, self.embedding_dim
        )
        self.relation_embeddings = torch.nn.Embedding(
            self.num_relations, self.embedding_dim
        )
        self.param_init(self.entity_embeddings.weight.data), self.param_init(
            self.relation_embeddings.weight.data
        )
        self.k = int(np.sqrt(self.embedding_dim // 2))
        self.num_sample = 250
        roots, weights = roots_legendre(self.num_sample)
        self.roots = (
            torch.from_numpy(roots).repeat(self.k, 1).float()
        )  # shape self.k by self.n
        self.weights = (
            torch.from_numpy(weights).reshape(1, -1).float()
        )  # shape 1 by self.n

    def compute_func(
        self, weights: torch.FloatTensor, x: torch.Tensor
    ) -> torch.FloatTensor:
        """
        Compute the output of a two-layer neural network.

        Parameters
        ----------
        weights : torch.FloatTensor
            The weights of the neural network, split into two sets for two layers.
        x : torch.Tensor
            The input tensor for the neural network.

        Returns
        -------
        torch.FloatTensor
            The output tensor after passing through the two-layer neural network.
        """
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.view(n, self.k, self.k)
        w2 = w2.view(n, self.k, self.k)
        # (2) Forward Pass
        out1 = torch.tanh(w1 @ x)  # torch.sigmoid => worse results
        out2 = w2 @ out1
        return out2  # no non-linearity => better results

    def chain_func(self, weights: torch.FloatTensor, x: torch.Tensor) -> torch.Tensor:
        """
        Chain two linear layers of a neural network for given weights and input.

        Parameters
        ----------
        weights : torch.FloatTensor
            The weights of the neural network, split into two sets for two layers.
        x : torch.Tensor
            The input tensor for the neural network.

        Returns
        -------
        torch.Tensor
            The output tensor after chaining the two linear layers.
        """
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.view(n, self.k, self.k)
        w2 = w2.view(n, self.k, self.k)
        # (2) Perform the forward pass
        out1 = torch.tanh(torch.bmm(w1, x))
        out2 = torch.bmm(w2, out1)
        return out2

    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of triples to compute embedding interactions.

        Parameters
        ----------
        idx_triple : torch.Tensor
            Tensor containing indices of triples.

        Returns
        -------
        torch.Tensor
            The computed scores for the batch of triples.
        """
        # (1) Retrieve embeddings: batch, \mathbb R^d
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(
            idx_triple
        )
        # (2) Compute NNs on \Gamma
        self.roots = self.roots.to(head_ent_emb.device)
        self.weights = self.weights.to(head_ent_emb.device)

        h_x = self.compute_func(
            head_ent_emb, x=self.roots
        )  # batch, \mathbb{R}^k, |\Gamma|
        t_x = self.compute_func(
            tail_ent_emb, x=self.roots
        )  # batch, \mathbb{R}^k, |\Gamma|
        r_h_x = self.chain_func(
            weights=rel_ent_emb, x=h_x
        )  # batch, \mathbb{R}^k, |\Gamma|
        # (3) Compute |\Gamma| predictions.
        out = torch.sum(r_h_x * t_x, dim=1) * self.weights  # batch, |gamma| #
        # (4) Average (3) over \Gamma
        out = torch.mean(out, dim=1)  # batch
        return out


class FMult2(BaseKGE):
    """
    FMult2 is a model for learning neural networks on knowledge graphs, offering
    enhanced capabilities for capturing complex interactions in the graph. It extends
    the base knowledge graph embedding model by integrating multi-layer neural network
    computations with entity and relation embeddings.

    Parameters
    ----------
    args : dict
        A dictionary of arguments containing hyperparameters and settings for the model,
        such as embedding dimensions, learning rate, number of layers, and other model-specific parameters.

    Attributes
    ----------
    name : str
        The name identifier for the FMult2 model.
    n_layers : int
        Number of layers in the neural network.
    k : int
        Dimension size for reshaping weights in neural network layers.
    n : int
        The number of discrete points for computations.
    a : float
        Lower bound of the range for discrete points.
    b : float
        Upper bound of the range for discrete points.
    score_func : str
        The scoring function used in the model.
    discrete_points : torch.Tensor
        Tensor of discrete points used in the computations.
    entity_embeddings : torch.nn.Embedding
        Embedding layer for entities in the knowledge graph.
    relation_embeddings : torch.nn.Embedding
        Embedding layer for relations in the knowledge graph.

    Methods
    -------
    build_func(Vec: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]
        Constructs a multi-layer neural network from a vector representation.

    build_chain_funcs(list_Vec: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]
        Builds chained functions from a list of vector representations.

    compute_func(W: List[torch.Tensor], b: torch.Tensor, x: torch.Tensor) -> torch.FloatTensor
        Computes the output of a multi-layer neural network.

    function(list_W: List[List[torch.Tensor]], list_b: List[torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]
        Defines a function for neural network computation based on weights and biases.

    trapezoid(list_W: List[List[torch.Tensor]], list_b: List[torch.Tensor]) -> torch.Tensor
        Applies the trapezoidal rule for integration on the function output.

    forward_triples(idx_triple: torch.Tensor) -> torch.Tensor
        Performs a forward pass for a batch of triples and computes the embedding interactions.
    """

    def __init__(self, args: dict):
        super().__init__(args)
        self.name = "FMult2"
        self.n_layers = 3
        tuned_embedding_dim = False
        while int(np.sqrt((self.embedding_dim - 1) / self.n_layers)) != np.sqrt(
            (self.embedding_dim - 1) / self.n_layers
        ):
            self.embedding_dim += 1
            tuned_embedding_dim = True
        if tuned_embedding_dim:
            print(
                f"\n\n*****Embedding dimension reset to {self.embedding_dim} to fit model architecture!*****\n"
            )
        self.k = int(np.sqrt((self.embedding_dim - 1) // self.n_layers))
        self.n = 50
        self.a, self.b = -1.0, 1.0
        # self.score_func = "vtp" # "vector triple product"
        # self.score_func = "trilinear"
        self.score_func = "compositional"
        # self.score_func = "full-compositional"
        # self.discrete_points = torch.linspace(self.a, self.b, steps=self.n)
        self.discrete_points = torch.linspace(self.a, self.b, steps=self.n).repeat(
            self.k, 1
        )

        self.entity_embeddings = torch.nn.Embedding(
            self.num_entities, self.embedding_dim
        )
        self.relation_embeddings = torch.nn.Embedding(
            self.num_relations, self.embedding_dim
        )
        self.param_init(self.entity_embeddings.weight.data), self.param_init(
            self.relation_embeddings.weight.data
        )

    def build_func(self, Vec: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Constructs a multi-layer neural network from a vector representation.

        Parameters
        ----------
        Vec : torch.Tensor
            The vector representation from which the neural network is constructed.

        Returns
        -------
        Tuple[List[torch.Tensor], torch.Tensor]
            A tuple containing the list of weight matrices for each layer and the bias vector.
        """
        n = len(Vec)
        # (1) Construct self.n_layers layered neural network
        W = list(torch.hsplit(Vec[:, :-1], self.n_layers))
        # (2) Reshape weights of the layers
        for i, w in enumerate(W):
            W[i] = w.reshape(n, self.k, self.k)
        return W, Vec[:, -1]

    def build_chain_funcs(
        self, list_Vec: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Builds chained functions from a list of vector representations. This method
        constructs a sequence of neural network layers and their corresponding biases
        based on the provided vector representations.

        Each vector representation in the list is first transformed into a set of weights
        and biases for a neural network layer using the `build_func` method. The method
        then computes a chained multiplication of these weights, adjusted by biases,
        to form a composite neural network function.

        Parameters
        ----------
        list_Vec : List[torch.Tensor]
            A list of vector representations, each corresponding to a set of parameters
            for constructing a neural network layer.

        Returns
        -------
        Tuple[List[torch.Tensor], torch.Tensor]
            A tuple where the first element is a list of weight tensors for each layer of
            the composite neural network, and the second element is the bias tensor for
            the last layer in the list.

        Notes
        -----
        This method is specifically designed to work with the neural network architecture
        defined in the FMult2 model. It assumes that each vector in `list_Vec` can be
        decomposed into weights and biases suitable for a layer in a neural network.
        """
        list_W = []
        list_b = []
        for Vec in list_Vec:
            W_, b = self.build_func(Vec)
            list_W.append(W_)
            list_b.append(b)

        W = list_W[-1][1:]
        for i in range(len(list_W) - 1):
            for j, w in enumerate(list_W[i]):
                if i == 0 and j == 0:
                    W_temp = w
                else:
                    W_temp = w @ W_temp
            W_temp = W_temp + list_b[i].reshape(-1, 1, 1)
        W_temp = list_W[-1][0] @ W_temp / ((len(list_Vec) - 1) * w.shape[1])
        W.insert(0, W_temp)
        return W, list_b[-1]

    def compute_func(
        self, W: List[torch.Tensor], b: torch.Tensor, x: torch.Tensor
    ) -> torch.FloatTensor:
        """
        Computes the output of a multi-layer neural network defined by the given weights and bias.

        This method sequentially applies a series of matrix multiplications and non-linear
        transformations to an input tensor `x`, using the provided weights `W`. The method
        alternates between applying a non-linear function (tanh) and a linear transformation
        to the intermediate outputs. The final output is adjusted with a bias term `b`.

        Parameters
        ----------
        W : List[torch.Tensor]
            A list of weight tensors for each layer in the neural network. Each tensor
            in the list represents the weights of a layer.
        b : torch.Tensor
            The bias tensor to be added to the output of the final layer.
        x : torch.Tensor
            The input tensor to be processed by the neural network.

        Returns
        -------
        torch.FloatTensor
            The output tensor after processing by the multi-layer neural network.

        Notes
        -----
        The method assumes an odd-indexed layer applies a non-linearity (tanh), while
        even-indexed layers apply linear transformations. This design choice is based on
        empirical observations for better performance in the context of the FMult2 model.
        """
        out = W[0] @ x
        for i, w in enumerate(W[1:]):
            if i % 2 == 0:  # no non-linearity => better results
                out = out + torch.tanh(w @ out)
            else:
                out = out + w @ out
        return out + b.reshape(-1, 1, 1)

    def function(
        self, list_W: List[List[torch.Tensor]], list_b: List[torch.Tensor]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Defines a function that computes the output of a composite neural network.
        This higher-order function returns a callable that applies a sequence of
        transformations defined by the provided weights and biases.

        The returned function (`f`) takes an input tensor `x` and applies a series of
        neural network computations on it. If only one set of weights and biases is provided,
        it directly computes the output using `compute_func`. Otherwise, it sequentially
        multiplies the outputs of multiple calls to `compute_func`, each using a different
        set of weights and biases from `list_W` and `list_b`.

        Parameters
        ----------
        list_W : List[List[torch.Tensor]]
            A list where each element is a list of weight tensors for a neural network.
        list_b : List[torch.Tensor]
            A list of bias tensors corresponding to each set of weights in `list_W`.

        Returns
        -------
        Callable[[torch.Tensor], torch.Tensor]
            A function that takes an input tensor and returns the output of the composite
            neural network.

        Notes
        -----
        This method is part of the FMult2 model's approach to construct complex scoring
        functions for knowledge graph embeddings. The flexibility in combining multiple
        neural network layers enables capturing intricate patterns in the data.
        """

        def f(x: torch.Tensor) -> torch.Tensor:
            """
            Applies a sequence of neural network transformations to the input tensor `x`.

            If only one set of weights and biases is provided in `list_W` and `list_b`,
            `f` applies a single neural network transformation using the `compute_func` method.
            If multiple sets of weights and biases are provided, `f` sequentially multiplies
            the outputs of `compute_func` applied with each set of weights and biases.
            This creates a composite function from multiple neural network layers.

            Parameters
            ----------
            x : torch.Tensor
                The input tensor to be processed by the neural network layers.

            Returns
            -------
            torch.Tensor
                The output tensor after processing by the composite neural network.

            Notes
            -----
            This function is designed to work within the `function` method of the FMult2 class.
            It leverages the `compute_func` method for each layer's computation and combines
            these layers in a multiplicative fashion to enhance the modeling capability of
            the network, especially in the context of knowledge graph embeddings.
            """
            if len(list_W) == 1:
                return self.compute_func(list_W[0], list_b[0], x)
            score = self.compute_func(list_W[0], list_b[0], x)
            for W, b in zip(list_W[1:], list_b[1:]):
                score = score * self.compute_func(W, b, x)
            return score

        return f

    def trapezoid(
        self, list_W: List[List[torch.Tensor]], list_b: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the integral of the output of a composite neural network function over a
        range of discrete points using the trapezoidal rule.

        This method first constructs a composite neural network function using the `function`
        method with the provided weights `list_W` and biases `list_b`. It then evaluates this
        function at a series of discrete points (`self.discrete_points`) and applies the
        trapezoidal rule to approximate the integral of the function over these points. The
        sum of the integral approximations across all dimensions is returned.

        Parameters
        ----------
        list_W : List[List[torch.Tensor]]
            A list where each element is a list of weight tensors for a neural network.
        list_b : List[torch.Tensor]
            A list of bias tensors corresponding to each set of weights in `list_W`.

        Returns
        -------
        torch.Tensor
            The sum of the integral of the composite function's output over the range
            of discrete points, computed using the trapezoidal rule.

        Notes
        -----
        The trapezoidal rule is a numerical method to approximate definite integrals.
        In the context of the FMult2 model, this method is used to integrate the output
        of the neural network over a range of inputs, which is crucial for certain types
        of calculations in knowledge graph embeddings.
        """
        return torch.trapezoid(
            self.function(list_W, list_b)(self.discrete_points),
            x=self.discrete_points,
            dim=-1,
        ).sum(dim=-1)

    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of triples to compute embedding interactions.

        Parameters
        ----------
        idx_triple : torch.Tensor
            Tensor containing indices of triples.

        Returns
        -------
        torch.Tensor
            The computed scores for the batch of triples.
        """
        # (1) Retrieve embeddings: batch, \mathbb R^d
        head_ent_emb, rel_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
        if self.discrete_points.device != head_ent_emb.device:
            self.discrete_points = self.discrete_points.to(head_ent_emb.device)
        if self.score_func == "vtp":
            h_W, h_b = self.build_func(head_ent_emb)
            r_W, r_b = self.build_func(rel_emb)
            t_W, t_b = self.build_func(tail_ent_emb)
            out = -self.trapezoid([t_W], [t_b]) * self.trapezoid(
                [h_W, r_W], [h_b, r_b]
            ) + self.trapezoid([r_W], [r_b]) * self.trapezoid([t_W, h_W], [t_b, h_b])
        elif self.score_func == "compositional":
            t_W, t_b = self.build_func(tail_ent_emb)
            chain_W, chain_b = self.build_chain_funcs([head_ent_emb, rel_emb])
            out = self.trapezoid([chain_W, t_W], [chain_b, t_b])
        elif self.score_func == "full-compositional":
            chain_W, chain_b = self.build_chain_funcs(
                [head_ent_emb, rel_emb, tail_ent_emb]
            )
            out = self.trapezoid([chain_W], [chain_b])
        elif self.score_func == "trilinear":
            h_W, h_b = self.build_func(head_ent_emb)
            r_W, r_b = self.build_func(rel_emb)
            t_W, t_b = self.build_func(tail_ent_emb)
            out = self.trapezoid([h_W, r_W, t_W], [h_b, r_b, t_b])
        return out
    

class LFMult1(BaseKGE): 

    '''Embedding with trigonometric functions. We represent all entities and relations in the complex number space as:
      f(x) = \sum_{k=0}^{k=d-1}wk e^{kix}. and use the three differents scoring function as in the paper to evaluate the score'''
    
    def __init__(self,args):
        super().__init__(args)
        self.name = 'LFMult1'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)

    def forward_triples(self, idx_triple): # idx_triplet = (h_idx, r_idx, t_idx) #change this to the forward_triples

        head_ent_emb, rel_emb, tail_ent_emb = self.get_triple_representation(idx_triple)

        score = self.vtp_score(head_ent_emb,rel_emb,tail_ent_emb)
    
        return score

    def tri_score(self,h,r,t):

        i_range, j_range, k_range = torch.meshgrid(torch.arange(self.embedding_dim),torch.arange(self.embedding_dim),torch.arange(self.embedding_dim))
        eps = 10**-6   #for stability reason
        cond = i_range + j_range == k_range

        s1 = torch.sum(torch.where(~cond, torch.zeros_like(~cond),  h[:, i_range] * r[:, j_range] * t[:, k_range]),dim=(-3,-2,-1)) # sum on i+j = k

        s2 = torch.sum(torch.where(cond, torch.zeros_like(cond), torch.sin(i_range + j_range - k_range) \
                                * h[:, i_range] * r[:, j_range] * t[:, k_range] /(eps+i_range + j_range - k_range)),dim=(-3,-2,-1))# sum on i+j != k
        s = s1 + s2 # combine the two sums.
        return s
    
    def vtp_score(self,h,r,t):

        i_range, j_range = torch.meshgrid(torch.arange(self.embedding_dim),torch.arange(self.embedding_dim))
        eps = 10**-6   #for stability reason
        cond = i_range == j_range

        p1 = torch.sum(torch.where(cond, torch.zeros_like(cond), torch.sin(i_range - j_range) \
                                * h[:, i_range] * t[:, j_range] /(eps+i_range - j_range)),dim=(-3,-2,-1)) \
                                    + torch.sum(h[:, i_range] * t[:, i_range],dim=(-3,-2,-1))# sum on i != j
        i_1 = torch.arange(1,self.embedding_dim)
        p2 = torch.sum(r[:, i_1] * torch.sin(i_1)/(i_1) ,dim=-1) + r[:,0]
        
        s1 = p1*p2

        p3 = torch.sum(torch.where(cond, torch.zeros_like(cond), torch.sin(i_range - j_range) \
                                * r[:, i_range] * t[:, j_range] /(eps+i_range - j_range)),dim=(-3,-2,-1)) \
                                    + torch.sum(r[:, i_range] * t[:, i_range],dim=(-3,-2,-1))# sum on i != j
        
        p4 = torch.sum(h[:, i_1] * torch.sin(i_1)/(i_1) ,dim=-1) + h[:,0]
        s2 = p3*p4


        s = s1 - s2 # combine the two sums.
        return s
    
class LFMult(BaseKGE): 

    '''Embedding with polynomial functions. We represent all entities and relations in the polynomial space as:
      f(x) = \sum_{i=0}^{d-1} a_k x^{i%d} and use the three differents scoring function as in the paper to evaluate the score.
      We also consider combining with Neural Networks.'''
    
    def __init__(self,args):
        super().__init__(args)
        self.name = 'LFMult'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.degree = self.args.get("degree",0)
        self.m = int(self.embedding_dim/(1+self.degree))
        self.x_values = torch.linspace(0, 1, 100)

    def forward_triples(self, idx_triple): # idx_triplet = (h_idx, r_idx, t_idx) #change this to the forward_triples

        head_ent_emb, rel_emb, tail_ent_emb = self.get_triple_representation(idx_triple)

        coeff_head, coeff_rel, coeff_tail = self.construct_multi_coeff(head_ent_emb), self.construct_multi_coeff(rel_emb), self.construct_multi_coeff(tail_ent_emb)

        ###### polynomial score with trilinear scoring

        # score = self.tri_score(coeff_head,coeff_rel,coeff_tail)
        
        # score = score.reshape(-1,self.m).sum(dim=1)
    

        ##### polynomial score with NN

        score = torch.trapezoid(self.poly_NN(self.x_values,coeff_head, coeff_rel, coeff_tail),self.x_values)
        # score = integral_value.reshape(1,-1).squeeze(0)
       
        
        return score 
    
    def construct_multi_coeff(self, x):

        coeffs = torch.hsplit(x,self.degree + 1)
        coeffs = torch.stack(coeffs,dim=1)

        return coeffs.transpose(1,2)
    


    def poly_NN(self, x, coefh, coefr, coeft):

        ''' Constructing a 2 layers NN to represent the embeddings. 
         h = \sigma(wh^T x + bh ),  r = \sigma(wr^T x + br ),  t = \sigma(wt^T x + bt )'''

        wh, bh = coefh[:, :self.m,0], coefh[:, :self.m,1]
        wr, br = coefr[:, :self.m,0], coefr[:, :self.m,1]
        wt, bt = coeft[:, :self.m,0], coeft[:, :self.m,1]

        h_emb = self.linear(x,wh,bh).reshape(-1,self.m,x.size(0))
        r_emb = self.linear(x,wr,br).reshape(-1,self.m,x.size(0))
        t_emb = self.linear(x,wt,bt).reshape(-1,self.m,x.size(0))

        return self.scalar_batch_NN(h_emb, r_emb, t_emb)#(linear(x,wh,bh)*linear(x,wr,br)*linear(x,wt,bt))
    
    def linear(self,x,w,b):
        return torch.tanh((w.reshape(-1,1)*x.unsqueeze(0) + b.reshape(-1,1)))
    


    
    def scalar_batch_NN(self, a, b, c):

        '''element wise multiplication between a,b and c:
        Inputs : a, b, c ====> torch.tensor of size batch_size x m x d
        Output : a tensor of size batch_size x d'''

        a_reshaped = a.transpose(1, 2).reshape(-1, a.size(-1), a.size(-2))
        b_reshaped = b.transpose(1, 2).reshape(-1, b.size(-1), b.size(-2))
        c_reshaped = c.transpose(1, 2).reshape(-1, c.size(-1), c.size(-2))

        mul_result = a_reshaped * b_reshaped * c_reshaped
        
        return mul_result.sum(dim=-1)




    def tri_score(self, coeff_h, coeff_r, coeff_t):

        '''this part implement the trilinear scoring techniques: 

        score(h,r,t) = \int_{0}{1} h(x)r(x)t(x) dx = \sum_{i,j,k = 0}^{d-1} \dfrac{a_i*b_j*c_k}{1+(i+j+k)%d} 

        1. generate the range for i,j and k from [0 d-1]

        2. perform
        \dfrac{a_i*b_j*c_k}{1+(i+j+k)%d} in parallel for every batch

        3. take the sum over each batch

        '''

        i_range, j_range, k_range = torch.meshgrid(torch.arange(self.degree+1),torch.arange(self.degree+1),torch.arange(self.degree+1))

        if self.degree == 0:
            terms = 1 / (1 + i_range + j_range + k_range) #%self.degree
        else:
            terms = 1 / (1 + (i_range + j_range + k_range)%self.degree) #%self.degree




        weighted_terms = terms.unsqueeze(0)*coeff_h.reshape(-1, 1, self.degree+1, 1) *coeff_r.reshape(-1, self.degree+1, 1, 1) * coeff_t.reshape(-1, 1, 1,self.degree+1)
        
        result = torch.sum(weighted_terms, dim=[-3,-2,-1])

        return result
    
    def vtp_score(self, h, r, t):
            
        '''this part implement the vector triple product scoring techniques: 

        score(h,r,t) = \int_{0}{1} h(x)r(x)t(x) dx = \sum_{i,j,k = 0}^{d-1} \dfrac{a_i*c_j*b_k - b_i*c_j*a_k}{(1+(i+j)%d)(1+k)} 

        1. generate the range for i,j and k from [0 d-1]

        2. Compute the first and second terms of the sum

        3.  Multiply with then denominator and take the sum
        
        4. take the sum over each batch
        
        '''
            
        i_range, j_range, k_range = torch.meshgrid(torch.arange(self.embedding_dim),torch.arange(self.embedding_dim),torch.arange(self.embedding_dim))

        # terms = 1 / (1 + (i_range + j_range)%self.embedding_dim) / (1+ k_range) # with modulo

        terms = 1 / (1 + i_range + j_range) / (1+ k_range)   #without dthe modulo


        terms1 = h.view(-1, 1, self.embedding_dim, 1) * t.view(-1, self.embedding_dim, 1, 1) * r.view(-1, 1, 1,self.embedding_dim)
        terms2 = r.view(-1, 1, self.embedding_dim, 1) * t.view(-1, self.embedding_dim, 1, 1) * h.view(-1, 1, 1,self.embedding_dim)

        weighted_terms = terms * (terms1-terms2)
        
        result = torch.sum(weighted_terms, dim=[-3,-2,-1])

        return result

    def comp_func(self,h,r,t): 
        '''this part implement the function composition scoring techniques: i.e. score = <hor, t>'''

        degree = torch.arange(self.embedding_dim, dtype=torch.float32)

        r_emb = self.polynomial(r,self.x_values,degree) 

        t_emb = self.polynomial(t,self.x_values,degree) 

        hor = self.pop(h,r_emb,degree) 
        
        score = torch.trapz(hor*t_emb , self.x_values) #Computing the score with the trapezoid method

        return score

    def polynomial(self,coeff,x,degree):
        '''This function takes a matrix tensor of coefficients (coeff), a tensor vector of points x  and range of integer [0,1,...d]
            and return a vector tensor (coeff[0][0] + coeff[0][1]x +...+ coeff[0][d]x^d,
                                coeff[1][0] + coeff[1][1]x +...+ coeff[1][d]x^d)
                                        ....'''

        x_powers = x.unsqueeze(1) ** degree

        vect = torch.matmul(coeff,x_powers.T)

        return vect

        
    def pop(self,coeff,x,degree):
        '''This function allow us to evaluate the composition of two polynomes without for loops :) 
        it takes a matrix tensor of coefficients (coeff), a matrix tensor of points x  and range of integer [0,1,...d]
            and return a tensor (coeff[0][0] + coeff[0][1]x +...+ coeff[0][d]x^d,
                                coeff[1][0] + coeff[1][1]x +...+ coeff[1][d]x^d)
                                        ....'''
        x_powers = x.unsqueeze(2) ** degree

        Mat = (coeff.unsqueeze(1)*x_powers).sum(dim=-1)

        return Mat