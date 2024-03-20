import datetime
import time
from typing import OrderedDict, Tuple
import numpy as np
import torch

import dicee.models.base_model
from .static_funcs import save_checkpoint_model, save_pickle
from .abstracts import AbstractCallback
import pandas as pd
from lightning import *


class AccumulateEpochLossCallback(AbstractCallback):
    """
    A callback to accumulate and save epoch losses to a CSV file at the end of training.

    This callback listens to the end of the training process and saves the accumulated
    epoch losses stored in the model's loss history to a CSV file. The file is saved
    in the specified directory.

    Parameters
    ----------
    path : str
        The directory path where the epoch loss CSV file will be saved.

    Attributes
    ----------
    path : str
        Stores the provided directory path for later use in saving the epoch losses.
    """

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def on_fit_end(self, trainer: Trainer, model: torch.nn.Module) -> None:
        """
        Invoked at the end of the training process to save the epoch losses.

        This method is called automatically by the training loop at the end of training.
        It retrieves the loss history from the model and saves it as a CSV file in the
        specified directory.

        Parameters
        ----------
        trainer : Trainer
            The trainer instance conducting the training process. Not used in this method,
            but required for compatibility with the callback interface.
        model : torch.nn.Module
            The model being trained. This model should have a `loss_history` attribute
            containing the losses of each epoch.

        Returns
        -------
        None
        """
        pd.DataFrame(model.loss_history, columns=["EpochLoss"]).to_csv(
            f"{self.path}/epoch_losses.csv"
        )


class PrintCallback(AbstractCallback):
    """
    A callback that prints the start time of training and its total runtime upon completion.

    This callback demonstrates a simple usage of the PyTorch Lightning callback system,
    printing a message when the training starts and another when it ends, showing how
    long the training took.
    """

    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        Invoked at the start of the fit process.

        Prints a message indicating that the training is starting, along with the current date and time.

        Parameters
        ----------
        trainer : Trainer
            The trainer instance conducting the training process.
        pl_module : LightningModule
            The LightningModule instance being trained.

        Returns
        -------
        None
        """
        # print(pl_module)
        # print(pl_module.summarize())
        # print(pl_module.selected_optimizer)
        print(f"\nTraining is starting {datetime.datetime.now()}...")

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        """
        Invoked at the end of the fit process.

        Calculates and prints the total training time in an appropriate time unit (seconds, minutes, or hours).

        Parameters
        ----------
        trainer : Trainer
            The trainer instance conducting the training process.
        pl_module : LightningModule
            The LightningModule instance that was trained.

        Returns
        -------
        None
        """
        training_time = time.time() - self.start_time
        if 60 > training_time:
            message = f"{training_time:.3f} seconds."
        elif 60 * 60 > training_time > 60:
            message = f"{training_time / 60:.3f} minutes."
        elif training_time > 60 * 60:
            message = f"{training_time / (60 * 60):.3f} hours."
        else:
            message = f"{training_time:.3f} seconds."
        print(f"Training Runtime: {message}\n")

    def on_train_batch_end(self, *args, **kwargs):
        """
        Dummy method for handling the end of a training batch. Implemented as a placeholder.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None
        """
        return

    def on_train_epoch_end(self, *args, **kwargs):
        """
        Dummy method for handling the end of a training epoch. Implemented as a placeholder.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None
        """
        return


class KGESaveCallback(AbstractCallback):
    """
    A callback to save the model periodically during training.

    This callback is intended to periodically save the current state of the model during training,
    allowing for checkpointing and potential recovery of intermediate states.

    Parameters
    ----------
    every_x_epoch : int
        Interval between epochs to save the model. The model will be saved every 'every_x_epoch' epochs.
    max_epochs : int
        The maximum number of epochs for the training. Used to calculate the default saving interval if 'every_x_epoch' is not provided.
    path : str
        The directory path where the model checkpoints will be saved.

    Attributes
    ----------
    epoch_counter : int
        A counter to keep track of the current epoch.

    Methods
    -------
    on_epoch_end(model, trainer, **kwargs)
        Saves the model at specified intervals.
    """

    def __init__(self, every_x_epoch: int, max_epochs: int, path: str):
        super().__init__()
        self.every_x_epoch = every_x_epoch
        self.max_epochs = max_epochs
        self.epoch_counter = 0
        self.path = path
        if self.every_x_epoch is None:
            self.every_x_epoch = max(self.max_epochs // 2, 1)

    def on_train_batch_end(self, *args, **kwargs):
        return

    def on_fit_start(self, trainer, pl_module):
        pass

    def on_train_epoch_end(self, *args, **kwargs):
        pass

    def on_fit_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, model: LightningModule, trainer: Trainer, **kwargs):
        """
        Invoked at the end of each epoch to potentially save the model.

        Checks if the current epoch matches the saving criteria. If so, the model's state is saved as a checkpoint.

        Parameters
        ----------
        model : LightningModule
            The model being trained.
        trainer : Trainer
            The trainer instance conducting the training process.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None
        """
        if self.epoch_counter % self.every_x_epoch == 0 and self.epoch_counter > 1:
            print(f"\nStoring model {self.epoch_counter}...")
            save_checkpoint_model(
                model,
                path=self.path + f"/model_at_{str(self.epoch_counter)}_"
                f"epoch_{str(str(datetime.datetime.now()))}.pt",
            )
        self.epoch_counter += 1


class PseudoLabellingCallback(AbstractCallback):
    """
    A callback for implementing pseudo-labelling during training.

    Pseudo-labelling is a semi-supervised learning technique that uses the model's predictions
    on unlabeled data as labels for retraining the model. This callback generates pseudo-labels
    for a batch of randomly created or selected unlabeled data and adds them to the training set.

    Parameters
    ----------
    data_module : LightningDataModule
        The data module that provides data loaders for the training process.
    kg : KnowledgeGraph
        The knowledge graph object that contains information about the entities, relations, and the unlabeled set.
    batch_size : int
        The size of the batch to generate or select for pseudo-labelling.

    Attributes
    ----------
    num_of_epochs : int
        Tracks the number of epochs that have been processed.
    unlabelled_size : int
        The size of the unlabeled dataset in the knowledge graph.
    """

    def __init__(self, data_module: LightningDataModule, kg, batch_size: int):
        super().__init__()
        self.data_module = data_module
        self.kg = kg
        self.num_of_epochs = 0
        self.unlabelled_size = len(self.kg.unlabelled_set)
        self.batch_size = batch_size

    def create_random_data(self) -> torch.Tensor:
        """
        Generates a batch of random triples (head entity, relation, tail entity).

        Returns
        -------
        torch.Tensor
            A batch of randomly generated triples.
        """
        entities = torch.randint(
            low=0, high=self.kg.num_entities, size=(self.batch_size, 2)
        )
        relations = torch.randint(
            low=0, high=self.kg.num_relations, size=(self.batch_size,)
        )
        # unlabelled triples
        return torch.stack((entities[:, 0], relations, entities[:, 1]), dim=1)

    def on_epoch_end(self, trainer: Trainer, model: LightningModule) -> None:
        """
        Invoked at the end of each epoch to perform pseudo-labelling.

        Generates or selects a batch of unlabeled data, uses the model to predict pseudo-labels,
        and adds the selected triples with high-confidence pseudo-labels to the training set.

        Parameters
        ----------
        trainer : Trainer
            The trainer instance conducting the training process.
        model : LightningModule
            The model being trained.

        Returns
        -------
        None
        """
        # Create random triples
        # if trainer.current_epoch < 10:
        #    return None
        # Increase it size, Now we increase it.
        model.eval()
        with torch.no_grad():
            # (1) Create random triples
            # unlabelled_input_batch = self.create_random_data()
            # (2) or use unlabelled batch
            unlabelled_input_batch = self.kg.unlabelled_set[
                torch.randint(low=0, high=self.unlabelled_size, size=(self.batch_size,))
            ]
            # (2) Predict unlabelled batch, and use prediction as pseudo-labels
            pseudo_label = torch.sigmoid(model(unlabelled_input_batch))
            selected_triples = unlabelled_input_batch[pseudo_label >= 0.90]
        if len(selected_triples) > 0:
            # Update dataset
            self.data_module.train_set_idx = np.concatenate(
                (self.data_module.train_set_idx, selected_triples.detach().numpy()),
                axis=0,
            )
            trainer.train_dataloader = self.data_module.train_dataloader()
            print(
                f"\tEpoch:{trainer.current_epoch}: Pseudo-labelling\t |D|= {len(self.data_module.train_set_idx)}"
            )
        model.train()


def estimate_q(eps) -> float:
    """
    Estimate the rate of convergence, q, from a sequence of errors.

    Parameters
    ----------
    eps : array-like
        A sequence of errors (epsilons) from which the rate of convergence is to be estimated.
        It's expected that `eps` represents a decreasing sequence of errors as the approximation
        improves, typically from an iterative numerical method.

    Returns
    -------
    float
        The estimated rate of convergence, q.

    Notes
    -----
    The function estimates the rate of convergence by fitting a line to the logarithm of the
    absolute difference of the logarithm of the errors. The slope of this line corresponds to
    the logarithm of the rate of convergence, q. This method assumes exponential convergence,
    where the error decreases as a power of the number of iterations.

    Examples
    --------
    >>> eps = [1/2**n for n in range(1, 6)]
    >>> q = estimate_q(eps)
    >>> print(q)
    2.0

    This indicates a quadratic convergence rate, as expected for the given sequence of errors
    that halve at each step.
    """
    x = np.arange(len(eps) - 1)
    y = np.log(np.abs(np.diff(np.log(eps))))
    line = np.polyfit(x, y, 1)  # fit degree 1 polynomial
    q = np.exp(line[0])  # find q
    return q


def compute_convergence(seq, i: int) -> float:
    """
    Compute the convergence rate of the last `i` elements in a sequence.

    Parameters
    ----------
    seq : array-like
        The sequence of numeric values for which the convergence rate is to be computed.
    i : int
        The number of elements from the end of `seq` to use for computing the convergence rate.

    Returns
    -------
    float
        The estimated rate of convergence over the last `i` elements of `seq`.

    Raises
    ------
    AssertionError
        If `i` is not less than or equal to the length of `seq` or if `i` is not greater than 0.

    Notes
    -----
    This function wraps the `estimate_q` function to specifically evaluate the convergence rate
    of a subsection of a given sequence. It modifies the sequence to fit the model of `estimate_q`
    by dividing each element by its index (adjusted for Python's 0-indexing), which normalizes
    the sequence in preparation for estimating the convergence rate.

    Examples
    --------
    >>> seq = np.array([1/2**n for n in range(10)])
    >>> compute_convergence(seq, 5)
    2.0

    Here, `compute_convergence` estimates the rate of convergence using the last 5 elements
    of a sequence exhibiting quadratic convergence. The function should return a value close
    to 2.0, indicating quadratic convergence.
    """
    assert len(seq) >= i > 0
    return estimate_q(seq[-i:] / (np.arange(i) + 1))

class ASWA(AbstractCallback):
    """
    Implements the Adaptive Stochastic Weight Averaging (ASWA) technique.
    This technique keeps track of validation performance and updates the ensemble model accordingly.

    Parameters
    ----------
    num_epochs : int
        The total number of epochs to train the model.
    path : str
        Path where the model and intermediate results will be saved.

    Attributes
    ----------
    initial_eval_setting : None or str
        Initial evaluation setting, used to restore the original evaluation mode of the model after ASWA is applied.
    alphas : list of float
        Weights for each model state in the ensemble.
    val_aswa : float
        Validation performance (MRR) of the current ASWA model.

    Methods
    -------
    on_fit_end(trainer, model) -> None:
        Applies the ASWA technique at the end of training.
    compute_mrr(trainer, model) -> float:
        Computes the Mean Reciprocal Rank (MRR) on the validation dataset.
    get_aswa_state_dict(model) -> OrderedDict:
        Retrieves the state dictionary for the ASWA model.
    decide(running_model_state_dict, ensemble_state_dict, val_running_model, mrr_updated_ensemble_model) -> None:
        Decides whether to update ASWA based on validation performance.
    on_train_epoch_end(trainer, model) -> None:
        Performs the ASWA update process at the end of each training epoch.
    """

    def __init__(self, num_epochs:int, path: str):
        super().__init__(
            num_epochs, path, epoch_to_start=None, last_percent_to_consider=None
        )
        self.initial_eval_setting = None
        self.epoch_count=0
        self.alphas = []
        self.val_aswa = -1

    def on_fit_end(self, trainer: Trainer, model: torch.nn.Module) -> None:
        """
        Called at the end of the fit process to apply the ASWA technique.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer instance.
        model : torch.nn.Module
            The model being trained.
        """
        # super().on_fit_end(trainer, model)
        if self.initial_eval_setting:
            # ADD this info back
            trainer.evaluator.args.eval_model = self.initial_eval_setting

        param_ensemble = torch.load(f"{self.path}/aswa.pt", torch.device("cpu"))
        model.load_state_dict(param_ensemble)

    @staticmethod
    def compute_mrr(trainer: Trainer, model: torch.nn.Module) -> float:
        """
        Computes the Mean Reciprocal Rank (MRR) for the model on the validation dataset.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer instance.
        model : torch.nn.Module
            The model for which MRR will be computed.

        Returns
        -------
        float
            The MRR score of the model on the validation dataset.
        """
        # (2) Enable eval mode.
        model.eval()
        # (3) MRR performance on the validation data of running model.
        device_name = model.device
        model.to("cpu")
        last_val_mrr_running_model = trainer.evaluator.eval(
            dataset=trainer.dataset,
            trained_model=model,
            form_of_labelling=trainer.form_of_labelling,
            during_training=True,
        )["Val"]["MRR"]
        model.to(device_name)
        # (4) Enable train mode.
        model.train()
        return last_val_mrr_running_model

    def get_aswa_state_dict(self, model: torch.nn.Module) -> OrderedDict:
        """
        Retrieves the state dictionary for the ASWA model.

        Parameters
        ----------
        model : torch.nn.Module
            The current model from which the ASWA state will be derived.

        Returns
        -------
        OrderedDict
            The state dictionary of the ASWA model.
        """
        # (2) Question: Soft update or Rejection?!
        ensemble_state_dict = torch.load(
            f"{self.path}/aswa.pt", torch.device(model.device)
        )
        # Perform provision parameter update.
        with torch.no_grad():
            for k, parameters in model.state_dict().items():
                if parameters.dtype == torch.float:
                    ensemble_state_dict[k] = (
                        ensemble_state_dict[k] * sum(self.alphas) + parameters
                    ) / (1 + sum(self.alphas))
        return ensemble_state_dict

    def decide(
        self,
        running_model_state_dict: OrderedDict,
        ensemble_state_dict: OrderedDict,
        val_running_model: float,
        mrr_updated_ensemble_model: float,
    ) -> bool:
        """
        Decides whether to update the ASWA model based on the validation performance.

        Parameters
        ----------
        running_model_state_dict : OrderedDict
            The state dictionary of the current running model.
        ensemble_state_dict : OrderedDict
            The state dictionary of the current ASWA model.
        val_running_model : float
            The validation performance (MRR) of the running model.
        mrr_updated_ensemble_model : float
            The validation performance (MRR) of the updated ASWA model.

        Returns
        -------
        bool
            The boolean flag to determine the updation of the ASWA model.
        """
        # (1) HARD UPDATE:
        # If the validation performance of the running model is greater than
        # the validation performance of updated ASWA and
        # the validation performance of ASWA
        if (
            val_running_model > mrr_updated_ensemble_model
            and val_running_model > self.val_aswa
        ):
            """Hard Update"""
            # (1.1) Save the running model as ASWA
            torch.save(running_model_state_dict, f=f"{self.path}/aswa.pt")
            # (2.1) Resect alphas/ensemble weights
            self.alphas.clear()
            # (2.2) Store the validation performance of ASWA
            self.val_aswa = val_running_model
            return True

        # (2) SOFT UPDATE:
        # If the validation performance of the running model is less  than
        # the validation performance of updated ASWA
        if mrr_updated_ensemble_model > self.val_aswa:
            """Soft update"""
            self.val_aswa = mrr_updated_ensemble_model
            torch.save(ensemble_state_dict, f=f"{self.path}/aswa.pt")
            self.alphas.append(1.0)
            return True
        # (3) Rejection:
        if self.val_aswa > mrr_updated_ensemble_model:
            """Ignore"""
            self.alphas.append(0)
            return True

    def on_train_epoch_end(self, trainer: Trainer, model: torch.nn.Module):
        """
        Called at the end of each training epoch to possibly update the ASWA model.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer instance.
        model : torch.nn.Module
            The model being trained.
        """
        # (1) Increment epoch counter
        self.epoch_count += 1
        # (2) Save the given eval setting if it is not saved.
        if self.initial_eval_setting is None:
            self.initial_eval_setting = trainer.evaluator.args.eval_model
            trainer.evaluator.args.eval_model = "val"
        # (3) Compute MRR of the running model.
        val_running_model = self.compute_mrr(trainer, model)

        # (4) Initialize ASWA if it is not initialized.
        if self.val_aswa == -1:
            torch.save(model.state_dict(), f=f"{self.path}/aswa.pt")
            self.alphas.append(1.0)
            self.val_aswa = val_running_model
            return True
        else:
            # (5) Load ASWA ensemble parameters.
            ensemble_state_dict = self.get_aswa_state_dict(model)
            # (6) Initialize ASWA ensemble with (5).
            ensemble = type(model)(model.args)
            ensemble.load_state_dict(ensemble_state_dict)
            # (7) Evaluate (6) on the validation data, i.e., perform the lookahead operation.
            mrr_updated_ensemble_model = trainer.evaluator.eval(
                dataset=trainer.dataset,
                trained_model=ensemble,
                form_of_labelling=trainer.form_of_labelling,
                during_training=True,
            )["Val"]["MRR"]
            # print(f"| MRR Running {val_running_model:.4f} | MRR ASWA: {self.val_aswa:.4f} |ASWA|:{sum(self.alphas)}")
            # (8) Decide whether ASWA should be updated via the current running model.
            self.decide(
                model.state_dict(),
                ensemble_state_dict,
                val_running_model,
                mrr_updated_ensemble_model,
            )


class Eval(AbstractCallback):
    """
    Callback for evaluating the model at certain epochs during training and logging the results.

    Parameters
    ----------
    path : str
        Path where evaluation reports will be saved.
    epoch_ratio : int, optional
        Interval of epochs after which the evaluation will be performed. Default is 1, meaning evaluation after every epoch.

    Attributes
    ----------
    reports : list of dict
        List of evaluation reports generated after each evaluation.
    epoch_counter : int
        Counter for keeping track of the number of epochs passed.

    Methods
    -------
    on_fit_end(trainer, model) -> None:
        Saves the evaluation reports to a file and optionally generates plots for training and validation MRR.
    on_train_epoch_end(trainer, model) -> None:
        Evaluates the model if the current epoch matches the specified epoch ratio and appends the report to `reports`.
    """

    def __init__(self, path, epoch_ratio: int = None):
        super().__init__()
        self.path = path
        self.reports = []
        self.epoch_ratio = epoch_ratio if epoch_ratio is not None else 1
        self.epoch_counter = 0

    def on_fit_start(self, trainer, model):
        pass

    def on_fit_end(self, trainer: Trainer, model : torch.nn.Module) -> None:
        """
        Called at the end of the fit process. Saves the collected evaluation reports to a file.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer instance.
        model : torch.nn.Module
            The model being trained.
        """
        save_pickle(
            data=self.reports,
            file_path=trainer.attributes.full_storage_path + "/evals_per_epoch",
        )
        """

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
        for (p,q), mrr in pairs_to_train_mrr.items():
            ax1.plot(mrr, label=f'{p},{q}')
        ax1.set_ylabel('Train MRR')

        for (p,q), mrr in pairs_to_val_mrr.items():
            ax2.plot(mrr, label=f'{p},{q}')
        ax2.set_ylabel('Val MRR')

        plt.legend()
        plt.xlabel('Epochs')
        plt.savefig('{full_storage_path}train_val_mrr.pdf')
        plt.show()
        """

    def on_train_epoch_end(self, trainer: Trainer, model : torch.nn.Module) -> None:
        """
        Called at the end of each training epoch. Performs evaluation if the current epoch matches the epoch_ratio.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer instance.
        model : torch.nn.Module
            The model being trained.
        """
        self.epoch_counter += 1
        if self.epoch_counter % self.epoch_ratio == 0:
            model.eval()
            report = trainer.evaluator.eval(
                dataset=trainer.dataset,
                trained_model=model,
                form_of_labelling=trainer.form_of_labelling,
                during_training=True,
            )
            model.train()
            self.reports.append(report)

    def on_train_batch_end(self, *args, **kwargs) -> None:
        """
        Called at the end of each training batch. This method is not implemented in this callback.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        return


class KronE(AbstractCallback):
    """
    Callback for augmenting triple representations with Kronecker product embeddings during training.

    Methods
    -------
    batch_kronecker_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        Computes the Kronecker product of two tensors with batch dimensions.
    get_kronecker_triple_representation(indexed_triple: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Augments triple representations with Kronecker product embeddings.
    on_fit_start(trainer, model) -> None:
        Overrides the model's method to get triple representations with a method that includes Kronecker product embeddings.
    """

    def __init__(self):
        super().__init__()
        self.f = None

    @staticmethod
    def batch_kronecker_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Computes the Kronecker product of two tensors `a` and `b` with batch dimensions.

        Parameters
        ----------
        a : torch.Tensor
            The first tensor with batch dimensions.
        b : torch.Tensor
            The second tensor with batch dimensions.

        Returns
        -------
        torch.Tensor
            The Kronecker product of `a` and `b`.
        """

        a, b = a.unsqueeze(1), b.unsqueeze(1)

        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        res = res.reshape(siz0 + siz1)
        return res.flatten(1)

    def get_kronecker_triple_representation(
        self, indexed_triple: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Augments triple representations with Kronecker product embeddings.

        Parameters
        ----------
        indexed_triple : torch.LongTensor
            Indexed triple representations.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Augmented head entity, relation, and tail entity embeddings.
        """
        n, d = indexed_triple.shape
        assert d == 3
        # Get the embeddings
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.f(indexed_triple)

        head_ent_kron_emb = self.batch_kronecker_product(*torch.hsplit(head_ent_emb, 2))
        rel_ent_kron_emb = self.batch_kronecker_product(*torch.hsplit(rel_ent_emb, 2))
        tail_ent_kron_emb = self.batch_kronecker_product(*torch.hsplit(tail_ent_emb, 2))

        return (
            torch.cat((head_ent_emb, head_ent_kron_emb), dim=1),
            torch.cat((rel_ent_emb, rel_ent_kron_emb), dim=1),
            torch.cat((tail_ent_emb, tail_ent_kron_emb), dim=1),
        )

    def on_fit_start(self, trainer: Trainer, model : torch.nn.Module) -> None:
        """
        Overrides the model's method to get triple representations with a method that includes Kronecker product embeddings.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer instance.
        model : torch.nn.Module
            The model being trained.
        """
        if isinstance(
            model.normalize_head_entity_embeddings,
            dicee.models.base_model.IdentityClass,
        ):
            self.f = model.get_triple_representation
            model.get_triple_representation = self.get_kronecker_triple_representation

        else:
            raise NotImplementedError("Normalizer should be reinitialized")


class Perturb(AbstractCallback):
    """
    Implements a three-level perturbation technique for knowledge graph embedding models during training.
    The perturbations can be applied at the input, parameter, or output levels.

    Attributes
    ----------
    level : str
        The perturbation level. Must be one of {"input", "param", "out"}.
    ratio : float
        The ratio of the mini-batch data points to be perturbed, between [0, 1].
    method : str, optional
        The method used for perturbation.
    scaler : float, optional
        The scaler factor used for perturbation.
    frequency : int, optional
        The frequency of perturbation, e.g., per epoch or per mini-batch.

    Methods
    -------
    on_train_batch_start(trainer, model, batch, batch_idx):
        Applies perturbation to the batch data points before the training batch starts.
    """

    def __init__(
        self,
        level: str = "input",
        ratio: float = 0.0,
        method: str = None,
        scaler: float = None,
        frequency=None,
    ):
        super().__init__()

        assert level in {"input", "param", "out"}
        assert ratio >= 0.0
        self.level = level
        self.ratio = ratio
        self.method = method
        self.scaler = scaler
        self.frequency = frequency  # per epoch, per mini-batch ?

    def on_train_batch_start(self, trainer: Trainer, model: torch.nn.Module, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Applies perturbation to the batch data points before the training batch starts.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning trainer instance.
        model : torch.nn.Module
            The model being trained.
        batch : torch.Tensor
            The current mini-batch of data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        None
        """
        # Modifications should be in-place
        # (1) Extract the input and output data points in a given batch.
        x, y = batch
        n, _ = x.shape
        assert n > 0
        # (2) Compute the number of perturbed data points.
        num_of_perturbed_data = int(n * self.ratio)
        if num_of_perturbed_data == 0:
            return None
        # (3) Detect the device on which data points reside
        device = x.get_device()
        if device == -1:
            device = "cpu"
        # (4) Sample random integers from 0 to n without replacement and take num_of_perturbed_data of tem
        random_indices = torch.randperm(n, device=device)[:num_of_perturbed_data]
        # (5) Apply perturbation depending on the level.

        # (5.1) Apply Input level perturbation.
        if self.level == "input":
            if torch.rand(1) > 0.5:
                # (5.1.1) Perturb input via heads: Sample random indices for heads.
                perturbation = torch.randint(
                    low=0,
                    high=model.num_entities,
                    size=(num_of_perturbed_data,),
                    device=device,
                )
                # Replace the head entities with (5.1.1) on given randomly selected data points in a mini-batch.
                x[random_indices] = torch.column_stack(
                    (perturbation, x[:, 1][random_indices])
                )
            else:
                # (5.1.2) Perturb input via relations : Sample random indices for relations.
                perturbation = torch.randint(
                    low=0,
                    high=model.num_relations,
                    size=(num_of_perturbed_data,),
                    device=device,
                )
                # Replace the relations with (5.1.2) on given randomly selected data points in a mini-batch.
                x[random_indices] = torch.column_stack(
                    (x[:, 0][random_indices], perturbation)
                )
        # (5.2) Apply Parameter level perturbation.
        elif self.level == "param":
            h, r = torch.hsplit(x, 2)
            # (5.2.1) Apply Gaussian Perturbation
            if self.method == "GN":
                if torch.rand(1) > 0.0:
                    # (5.2.1.1) Apply Gaussian Perturbation on heads.
                    h_selected = h[random_indices]
                    with torch.no_grad():
                        model.entity_embeddings.weight[h_selected] += torch.normal(
                            mean=0,
                            std=self.scaler,
                            size=model.entity_embeddings.weight[h_selected].shape,
                            device=model.device,
                        )
                else:
                    # (5.2.1.2) Apply Gaussian Perturbation on relations.
                    r_selected = r[random_indices]
                    with torch.no_grad():
                        model.relation_embeddings.weight[r_selected] += torch.normal(
                            mean=0,
                            std=self.scaler,
                            size=model.entity_embeddings.weight[r_selected].shape,
                            device=model.device,
                        )
            # (5.2.2) Apply Random Perturbation
            elif self.method == "RN":
                if torch.rand(1) > 0.0:
                    # (5.2.2.1) Apply Random Perturbation on heads.
                    h_selected = h[random_indices]
                    with torch.no_grad():
                        model.entity_embeddings.weight[h_selected] += (
                            torch.rand(
                                size=model.entity_embeddings.weight[h_selected].shape,
                                device=model.device,
                            )
                            * self.scaler
                        )
                else:
                    # (5.2.2.2) Apply Random Perturbation on relations.
                    r_selected = r[random_indices]
                    with torch.no_grad():
                        model.relation_embeddings.weight[r_selected] += (
                            torch.rand(
                                size=model.entity_embeddings.weight[r_selected].shape,
                                device=model.device,
                            )
                            * self.scaler
                        )
            else:
                raise RuntimeError(f"--method is given as {self.method}!")
        elif self.level == "out":
            # (5.3) Apply output level perturbation.
            if self.method == "Soft":
                # (5.3) Output level soft perturbation resembles label smoothing.
                # (5.3.1) Compute the perturbation rate.
                perturb = torch.rand(1, device=model.device) * self.scaler
                # https://pytorch.org/docs/stable/generated/torch.where.html
                # 1.0 => 1.0 - perturb
                # 0.0 => perturb
                # (5.3.2) Reduces 1s and increases 0s via (5.2.1)
                batch[1][random_indices] = torch.where(
                    batch[1][random_indices] == 1.0, 1.0 - perturb, perturb
                )
            elif self.method == "Hard":
                # (5.3) Output level hard perturbation flips 1s to 0 and 0s to 1s.
                batch[1][random_indices] = torch.where(
                    batch[1][random_indices] == 1.0, 0.0, 1.0
                )
            else:
                raise NotImplementedError(f"{self.level}")
        else:
            raise RuntimeError(f"--level is given as {self.level}!")
