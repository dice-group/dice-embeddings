import datetime
import time
import numpy as np
import torch
import os
import json
import copy

import dicee.models.base_model
from dicee.models.ensemble import EnsembleKGE
from .static_funcs import save_checkpoint_model, save_pickle
from .abstracts import AbstractCallback
import pandas as pd
from collections import defaultdict
import math
from torch.optim.lr_scheduler import LambdaLR
from torch._dynamo.eval_frame import OptimizedModule
from .eval_static_funcs import evaluate_ensemble_link_prediction_performance
from pytorch_lightning.utilities import rank_zero_only


class AccumulateEpochLossCallback(AbstractCallback):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def on_fit_end(self, trainer, model) -> None:
        """
        Store epoch loss


        Parameter
        ---------
        trainer:

        model:

        Returns
        ---------
        None
        """
        pd.DataFrame(model.loss_history, columns=['EpochLoss']).to_csv(f'{self.path}/epoch_losses.csv')


class PrintCallback(AbstractCallback):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def on_fit_start(self, trainer, pl_module):
        # print(pl_module)
        # print(pl_module.summarize())
        # print(pl_module.selected_optimizer)
        print(f"\nTraining is starting {datetime.datetime.now()}...")

    def on_fit_end(self, trainer, pl_module):
        training_time = time.time() - self.start_time
        if 60 > training_time:
            message = f'{training_time:.3f} seconds.'
        elif 60 * 60 > training_time > 60:
            message = f'{training_time / 60:.3f} minutes.'
        elif training_time > 60 * 60:
            message = f'{training_time / (60 * 60):.3f} hours.'
        else:
            message = f'{training_time:.3f} seconds.'
        print(f"Training Runtime: {message}\n")

    def on_train_batch_end(self, *args, **kwargs):
        return

    def on_train_epoch_end(self, *args, **kwargs):
        return


class KGESaveCallback(AbstractCallback):
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

    def on_epoch_end(self, model, trainer, **kwargs):
        if self.epoch_counter % self.every_x_epoch == 0 and self.epoch_counter > 1:
            print(f'\nStoring model {self.epoch_counter}...')
            save_checkpoint_model(model,
                                  path=self.path + f'/model_at_{str(self.epoch_counter)}_'
                                                   f'epoch_{str(str(datetime.datetime.now()))}.pt')
        self.epoch_counter += 1


class PseudoLabellingCallback(AbstractCallback):
    def __init__(self, data_module, kg, batch_size):
        super().__init__()
        self.data_module = data_module
        self.kg = kg
        self.num_of_epochs = 0
        self.unlabelled_size = len(self.kg.unlabelled_set)
        self.batch_size = batch_size

    def create_random_data(self):
        entities = torch.randint(low=0, high=self.kg.num_entities, size=(self.batch_size, 2))
        relations = torch.randint(low=0, high=self.kg.num_relations, size=(self.batch_size,))
        # unlabelled triples
        return torch.stack((entities[:, 0], relations, entities[:, 1]), dim=1)

    def on_epoch_end(self, trainer, model):
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
                torch.randint(low=0, high=self.unlabelled_size, size=(self.batch_size,))]
            # (2) Predict unlabelled batch, and use prediction as pseudo-labels
            pseudo_label = torch.sigmoid(model(unlabelled_input_batch))
            selected_triples = unlabelled_input_batch[pseudo_label >= .90]
        if len(selected_triples) > 0:
            # Update dataset
            self.data_module.train_set_idx = np.concatenate(
                (self.data_module.train_set_idx, selected_triples.detach().numpy()),
                axis=0)
            trainer.train_dataloader = self.data_module.train_dataloader()
            print(f'\tEpoch:{trainer.current_epoch}: Pseudo-labelling\t |D|= {len(self.data_module.train_set_idx)}')
        model.train()


def estimate_q(eps):
    """ estimate rate of convergence q from sequence esp"""
    x = np.arange(len(eps) - 1)
    y = np.log(np.abs(np.diff(np.log(eps))))
    line = np.polyfit(x, y, 1)  # fit degree 1 polynomial
    q = np.exp(line[0])  # find q
    return q


def compute_convergence(seq, i):
    assert len(seq) >= i > 0
    return estimate_q(seq[-i:] / (np.arange(i) + 1))


class ASWA(AbstractCallback):
    """ Adaptive stochastic weight averaging
        ASWE keeps track of the validation performance and update s the ensemble model accordingly.
        """

    def __init__(self, num_epochs, path):
        super().__init__()
        self.path=path
        self.num_epochs=num_epochs
        self.initial_eval_setting = None
        self.epoch_count=0
        self.alphas = []
        self.val_aswa = -1

    @rank_zero_only
    def on_fit_end(self, trainer, model):
        # super().on_fit_end(trainer, model)
        if self.initial_eval_setting:
            # ADD this info back
            trainer.evaluator.args.eval_model = self.initial_eval_setting
        
        param_ensemble = torch.load(f"{self.path}/aswa.pt", torch.device("cpu"))
        model.load_state_dict(param_ensemble)

    @staticmethod
    def compute_mrr(trainer, model) -> float:
        # (2) Enable eval mode.
        eval_model = copy.deepcopy(model)
        eval_model.eval()
        # (3) MRR performance on the validation data of running model.
        eval_model.to("cpu")
        last_val_mrr_running_model = trainer.evaluator.eval(dataset=trainer.dataset,
                                                            trained_model=eval_model,
                                                            form_of_labelling=trainer.form_of_labelling,
                                                            during_training=True)["Val"]["MRR"]
        del eval_model
        return last_val_mrr_running_model

    def get_aswa_state_dict(self, model):
        # (2) Question: Soft update or Rejection?!
        ensemble_state_dict = torch.load(f"{self.path}/aswa.pt", torch.device(next(model.parameters()).device))
        # Perform provision parameter update.
        with torch.no_grad():
            for k, parameters in model.state_dict().items():
                if parameters.dtype == torch.float:
                    ensemble_state_dict[k] = (ensemble_state_dict[k] * sum(self.alphas) + parameters) / (1 + sum(self.alphas))
        return ensemble_state_dict

    def decide(self, running_model_state_dict, ensemble_state_dict, val_running_model, mrr_updated_ensemble_model):
        """
        Perform Hard Update, software or rejection

        Parameters
        ----------
        running_model_state_dict
        ensemble_state_dict
        val_running_model
        mrr_updated_ensemble_model

        Returns
        -------

        """
        # (1) HARD UPDATE:
        # If the validation performance of the running model is greater than
        # the validation performance of updated ASWA and
        # the validation performance of ASWA
        if val_running_model > mrr_updated_ensemble_model and val_running_model > self.val_aswa:
            """Hard Update """
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
            """ Ignore """
            self.alphas.append(0)
            return True

    @rank_zero_only
    def on_train_epoch_end(self, trainer, model):
        # if (trainer.global_rank == trainer.local_rank == 0) is False:
        #     return None
        if isinstance(model, OptimizedModule):
            model = model._orig_mod
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
            mrr_updated_ensemble_model = trainer.evaluator.eval(dataset=trainer.dataset,
                                                                trained_model=ensemble,
                                                                form_of_labelling=trainer.form_of_labelling,
                                                                during_training=True)["Val"]["MRR"]
            # print(f"| MRR Running {val_running_model:.4f} | MRR ASWA: {self.val_aswa:.4f} |ASWA|:{sum(self.alphas)}")
            # (8) Decide whether ASWA should be updated via the current running model.
            self.decide(model.state_dict(), ensemble_state_dict, val_running_model, mrr_updated_ensemble_model)

class Eval(AbstractCallback):
    def __init__(self, path, epoch_ratio: int = None):
        super().__init__()
        self.path = path
        self.reports = []
        self.epoch_ratio = epoch_ratio if epoch_ratio is not None else 1
        self.epoch_counter = 0

    def on_fit_start(self, trainer, model):
        pass

    def on_fit_end(self, trainer, model):
        save_pickle(data=self.reports, file_path=trainer.attributes.full_storage_path + '/evals_per_epoch')
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

    def on_train_epoch_end(self, trainer, model):
        self.epoch_counter += 1
        if self.epoch_counter % self.epoch_ratio == 0:
            model.eval()
            report = trainer.evaluator.eval(dataset=trainer.dataset, trained_model=model,
                                            form_of_labelling=trainer.form_of_labelling, during_training=True)
            model.train()
            self.reports.append(report)

    def on_train_batch_end(self, *args, **kwargs):
        return


class KronE(AbstractCallback):
    def __init__(self):
        super().__init__()
        self.f = None

    @staticmethod
    def batch_kronecker_product(a, b):
        """
        Kronecker product of matrices a and b with leading batch dimensions.
        Batch dimensions are broadcast. The number of them mush
        :type a: torch.Tensor
        :type b: torch.Tensor
        :rtype: torch.Tensor
        """

        a, b = a.unsqueeze(1), b.unsqueeze(1)

        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        res = res.reshape(siz0 + siz1)
        return res.flatten(1)

    def get_kronecker_triple_representation(self, indexed_triple: torch.LongTensor):
        """
        Get kronecker embeddings
        """
        n, d = indexed_triple.shape
        assert d == 3
        # Get the embeddings
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.f(indexed_triple)

        head_ent_kron_emb = self.batch_kronecker_product(*torch.hsplit(head_ent_emb, 2))
        rel_ent_kron_emb = self.batch_kronecker_product(*torch.hsplit(rel_ent_emb, 2))
        tail_ent_kron_emb = self.batch_kronecker_product(*torch.hsplit(tail_ent_emb, 2))

        return torch.cat((head_ent_emb, head_ent_kron_emb), dim=1), \
            torch.cat((rel_ent_emb, rel_ent_kron_emb), dim=1), \
            torch.cat((tail_ent_emb, tail_ent_kron_emb), dim=1)

    def on_fit_start(self, trainer, model):
        if isinstance(model.normalize_head_entity_embeddings, dicee.models.base_model.IdentityClass):
            self.f = model.get_triple_representation
            model.get_triple_representation = self.get_kronecker_triple_representation

        else:
            raise NotImplementedError('Normalizer should be reinitialized')


class Perturb(AbstractCallback):
    """ A callback for a three-Level Perturbation

    Input Perturbation: During training an input x is perturbed by randomly replacing its element.
    In the context of knowledge graph embedding models, x can denote a triple, a tuple of an entity and a relation,
    or a tuple of two entities.
    A perturbation means that a component of x is randomly replaced by an entity or a relation.

    Parameter Perturbation:

    Output Perturbation:
    """

    def __init__(self, level: str = "input", ratio: float = 0.0, method: str = None, scaler: float = None,
                 frequency=None):
        """
        level in {input, param, output}
        ratio:float btw [0,1] a percentage of mini-batch data point to be perturbed.
        method = ?
        """
        super().__init__()

        assert level in {"input", "param", "out"}
        assert ratio >= 0.0
        self.level = level
        self.ratio = ratio
        self.method = method
        self.scaler = scaler
        self.frequency = frequency  # per epoch, per mini-batch ?

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
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
                perturbation = torch.randint(low=0, high=model.num_entities,
                                             size=(num_of_perturbed_data,),
                                             device=device)
                # Replace the head entities with (5.1.1) on given randomly selected data points in a mini-batch.
                x[random_indices] = torch.column_stack((perturbation, x[:, 1][random_indices]))
            else:
                # (5.1.2) Perturb input via relations : Sample random indices for relations.
                perturbation = torch.randint(low=0, high=model.num_relations,
                                             size=(num_of_perturbed_data,),
                                             device=device)
                # Replace the relations with (5.1.2) on given randomly selected data points in a mini-batch.
                x[random_indices] = torch.column_stack(
                    (x[:, 0][random_indices], perturbation))
        # (5.2) Apply Parameter level perturbation.
        elif self.level == "param":
            h, r = torch.hsplit(x, 2)
            # (5.2.1) Apply Gaussian Perturbation
            if self.method == "GN":
                if torch.rand(1) > 0.0:
                    # (5.2.1.1) Apply Gaussian Perturbation on heads.
                    h_selected = h[random_indices]
                    with torch.no_grad():
                        model.entity_embeddings.weight[h_selected] += torch.normal(mean=0, std=self.scaler,
                                                                                   size=model.entity_embeddings.weight[
                                                                                       h_selected].shape,
                                                                                   device=model.device)
                else:
                    # (5.2.1.2) Apply Gaussian Perturbation on relations.
                    r_selected = r[random_indices]
                    with (torch.no_grad()):
                        model.relation_embeddings.weight[r_selected] += torch.normal(mean=0, std=self.scaler,
                                                                                     size=
                                                                                     model.entity_embeddings.weight[
                                                                                         r_selected].shape,
                                                                                     device=model.device)
            # (5.2.2) Apply Random Perturbation
            elif self.method == "RN":
                if torch.rand(1) > 0.0:
                    # (5.2.2.1) Apply Random Perturbation on heads.
                    h_selected = h[random_indices]
                    with torch.no_grad():
                        model.entity_embeddings.weight[h_selected] += torch.rand(
                            size=model.entity_embeddings.weight[h_selected].shape, device=model.device) * self.scaler
                else:
                    # (5.2.2.2) Apply Random Perturbation on relations.
                    r_selected = r[random_indices]
                    with torch.no_grad():
                        model.relation_embeddings.weight[r_selected] += torch.rand(
                            size=model.entity_embeddings.weight[r_selected].shape, device=model.device) * self.scaler
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
                batch[1][random_indices] = torch.where(batch[1][random_indices] == 1.0, 1.0 - perturb, perturb)
            elif self.method == "Hard":
                # (5.3) Output level hard perturbation flips 1s to 0 and 0s to 1s.
                batch[1][random_indices] = torch.where(batch[1][random_indices] == 1.0, 0.0, 1.0)
            else:
                raise NotImplementedError(f"{self.level}")
        else:
            raise RuntimeError(f"--level is given as {self.level}!")

class PeriodicEvalCallback(AbstractCallback):
    """
    Callback to periodically evaluate the model and optionally save checkpoints during training.

    Evaluates at regular intervals (every N epochs) or at explicitly specified epochs.
    Stores evaluation reports and model checkpoints.
    """

    def __init__(self, experiment_path: str, max_epochs: int,
                 eval_every_n_epoch: int = 0, eval_at_epochs: list = None,
                 save_model_every_n_epoch: bool = True, n_epochs_eval_model: str = "val_test"):
        """
        Initialize the PeriodicEvalCallback.

        Parameters
        ----------
        experiment_path : str
            Directory where evaluation reports and model checkpoints will be saved.
        max_epochs : int
            Maximum number of training epochs.
        eval_every_n_epoch : int, optional
            Evaluate every N epochs. Ignored if eval_at_epochs is provided.
        eval_at_epochs : list, optional
            List of specific epochs at which to evaluate. If provided and non-empty, overrides eval_every_n_epoch.
        save_model_every_n_epoch : bool, optional
            Whether to save model checkpoints at each evaluation epoch.
        n_epochs_eval_model : str, optional
            Evaluation mode for N epochs. Default is "val_test".
        """
        super().__init__()
        self.experiment_dir = experiment_path
        self.max_epochs = max_epochs
        self.epoch_counter = 0
        self.save_model_every_n_epoch = save_model_every_n_epoch
        self.reports = defaultdict(dict)
        self.n_epochs_eval_model = n_epochs_eval_model
        self.default_eval_model =  None

        # Determine evaluation epochs: combine explicit list and interval if provided
        eval_epochs_set = set()
        if eval_at_epochs and len(eval_at_epochs) > 0:
            eval_epochs_set.update(eval_at_epochs)
        if eval_every_n_epoch > 0:
            eval_epochs_set.update(range(eval_every_n_epoch, max_epochs + 1, eval_every_n_epoch))
        self.eval_epochs = eval_epochs_set

        # Prepare directory for model checkpoints if needed
        if self.save_model_every_n_epoch:
            self.n_epochs_storage_path = os.path.join(self.experiment_dir, 'models_n_epochs')
            os.makedirs(self.n_epochs_storage_path, exist_ok=True)

    @rank_zero_only
    def on_fit_end(self, trainer, model):
        """ Called at the end of training. Saves final evaluation report."""
        report_path = os.path.join(self.experiment_dir, 'eval_report_n_epochs.json')
        with open(report_path, 'w') as f:
            json.dump(self.reports, f, indent=4)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, model):
        """
        Called at the end of each training epoch. Performs evaluation and checkpointing if scheduled.
        """
        self.epoch_counter += 1

        # Check if current epoch is scheduled for evaluation
        if self.epoch_counter not in self.eval_epochs:
            return

        # Store default evaluation mode once
        if self.default_eval_model is None:
            self.default_eval_model = trainer.evaluator.args.eval_model

        # Skip evaluation if default model already covers all eval splits and it's the final epoch
        if self.epoch_counter == self.max_epochs:
            default_splits = set(self.default_eval_model.split('_'))
            required_splits = set(self.n_epochs_eval_model.split('_'))
            if required_splits.issubset(default_splits):
                return

        # Set evaluation mode for this scheduled epoch
        trainer.evaluator.args.eval_model = self.n_epochs_eval_model

        # Prepare evaluation model
        eval_model = None

        if model.args.get("swa"):
            eval_model = copy.deepcopy(trainer.swa_model)

        elif model.args.get("adaptive_swa"):
            # Load ASWA weights and apply to a deepcopy of the model
            aswa_path = os.path.join(self.experiment_dir, "aswa.pt")
            aswa_ensemble_params = torch.load(aswa_path, map_location="cpu")
            
            # Clone model and apply ASWA weights
            if isinstance(model, OptimizedModule):
                eval_model = type(model._orig_mod)(model.args)
            else:
                eval_model = type(model)(model.args)
            eval_model.load_state_dict(aswa_ensemble_params)

        else:
            eval_model = copy.deepcopy(model)

        eval_model.to('cpu')
        eval_model.eval()
        
        report = trainer.evaluator.eval(dataset=trainer.dataset,
                trained_model=eval_model,
                form_of_labelling=trainer.form_of_labelling,
                during_training=True)


        # Restore evaluation mode
        trainer.evaluator.args.eval_model = self.default_eval_model

        # Store evaluation report
        self.reports[f'epoch_{self.epoch_counter}_eval'] = report

        # Save model checkpoint if needed
        if self.save_model_every_n_epoch:
            save_path = os.path.join(self.n_epochs_storage_path, f'model_at_epoch_{self.epoch_counter}.pt')
            torch.save(eval_model.state_dict(), save_path)

        # Free memory only if eval_model is a separate instance (ASWA case)
        # if model.args.get("adaptive_swa") and eval_model is not model:
        #     del eval_model
        del eval_model
        torch.cuda.empty_cache()
class LRScheduler(AbstractCallback):
    """
    Callback for managing learning rate scheduling and model snapshots.

    Supports cosine annealing ("cca"), MMCCLR ("mmcclr"), and their deferred (warmup) variants:
    - "deferred_cca"
    - "deferred_mmcclr"

    At the end of each learning rate cycle, the model can optionally be saved as a snapshot.
    """
    def __init__(
        self,
        adaptive_lr_config: dict,
        total_epochs: int,
        experiment_dir: str,
        eta_max: float = 0.1,
        snapshot_dir: str = "snapshots",
    ):
        """
        Initialize the LR scheduler callback.

        Args:
            adaptive_lr_config (dict): Configuration dictionary containing LR scheduling parameters.
                Can include: scheduler_name, lr_min, num_cycles, weighted_ensemble, n_snapshots
            total_epochs (int): Total number of training epochs (args.num_epochs)
            experiment_dir (str): Directory for the experiment, used as base for snapshots.
            eta_max (float, optional): Maximum learning rate at the start of each cycle.
                passed from `args.lr`. Default is 0.1.
            snapshot_dir (str, optional): Subdirectory inside experiment_dir where snapshots will be saved. Default is "snapshots".
        """
        # Validate and set defaults for configuration
        self._validate_and_set_config(adaptive_lr_config, eta_max)
        
        self.total_epochs = total_epochs
        self.experiment_dir = experiment_dir
        self.snapshot_dir = os.path.join(experiment_dir, snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)

        assert self.eta_max > self.eta_min, \
            f"Max Learning Rate ({self.eta_max}) must be greater than Min Learning Rate ({self.eta_min})"

        # Calculate warmup epochs only for deferred schedulers
        if self.scheduler_name.startswith("deferred"):
            # Use formula: defer for (n_cycles - n_snapshots) cycles
            deferred_cycles = self.n_cycles - self.n_snapshots
            self.warmup_epochs = int(deferred_cycles / self.n_cycles * self.total_epochs)
        else:
            # Non-deferred schedulers don't use warmup
            self.warmup_epochs = 0

        # Placeholders to be set during training
        self.batches_per_epoch = None
        self.total_steps = None
        self.cycle_length = None
        self.warmup_steps = None
        self.lr_lambda = None
        self.scheduler = None
        self.step_count = 0

        self.snapshot_loss = defaultdict(float)

    def _validate_and_set_config(self, config: dict, eta_max: float):
        """
        Validate the adaptive_lr_config and set default values for missing parameters.
        """
        # Default configuration
        defaults = {
            "scheduler_name": "cca",
            "lr_min": 0.01,
            "num_cycles": 10,
            "weighted_ensemble": True,
            "n_snapshots": 5
        }
        
        # Validate config is a dictionary
        if not isinstance(config, dict):
            raise ValueError("adaptive_lr_config must be a dictionary")
        
        # Validate scheduler_name
        if "scheduler_name" in config:
            valid_schedulers = ["cca", "mmcclr", "deferred_cca", "deferred_mmcclr"]
            if config["scheduler_name"] not in valid_schedulers:
                raise ValueError(f"Invalid scheduler_name '{config['scheduler_name']}'. "
                               f"Must be one of: {valid_schedulers}")
        
        # Validate lr_min
        if "lr_min" in config:
            lr_min = config["lr_min"]
            if not isinstance(lr_min, (int, float)) or lr_min <= 0:
                raise ValueError(f"lr_min must be a positive number, got: {lr_min}")
            if lr_min >= eta_max:
                raise ValueError(f"lr_min ({lr_min}) must be less than eta_max ({eta_max})")
        
        # Validate num_cycles
        if "num_cycles" in config:
            num_cycles = config["num_cycles"]
            if not isinstance(num_cycles, (int, float)) or num_cycles <= 0:
                raise ValueError(f"num_cycles must be a positive number, got: {num_cycles}")
        
        # Validate n_snapshots
        if "n_snapshots" in config:
            n_snapshots = config["n_snapshots"]
            if not isinstance(n_snapshots, int) or n_snapshots <= 0:
                raise ValueError(f"n_snapshots must be a positive integer, got: {n_snapshots}")
        
        # Validate weighted_ensemble
        if "weighted_ensemble" in config:
            weighted_ensemble = config["weighted_ensemble"]
            if not isinstance(weighted_ensemble, bool):
                raise ValueError(f"weighted_ensemble must be a boolean, got: {weighted_ensemble}")
        
        # Set attributes with defaults for missing values
        self.scheduler_name = config.get("scheduler_name", defaults["scheduler_name"]).lower()
        self.eta_min = config.get("lr_min", defaults["lr_min"])
        self.n_cycles = config.get("num_cycles", defaults["num_cycles"])
        self.weighted_ensemble = config.get("weighted_ensemble", defaults["weighted_ensemble"])
        self.n_snapshots = config.get("n_snapshots", defaults["n_snapshots"])
        self.eta_max = eta_max

        assert self.n_snapshots <= self.n_cycles, \
            f"n_snapshots ({self.n_snapshots}) must be less than or equal to num_cycles ({self.n_cycles})"
        
        print(f"LRScheduler initialized with config: {config}")
        print(f"Using: scheduler_name={self.scheduler_name}, eta_min={self.eta_min}, "
              f"n_cycles={self.n_cycles}, weighted_ensemble={self.weighted_ensemble}, "
              f"n_snapshots={self.n_snapshots}")


    def _initialize_training_params(self, num_training_batches):
        """Set batches per epoch, total steps, cycle length, and warmup steps."""
        self.batches_per_epoch = num_training_batches
        self.total_steps = self.total_epochs * self.batches_per_epoch
        self.cycle_length = self.total_steps // self.n_cycles
        # Ensure cycle length is at least 1 to avoid division by zero
        if self.cycle_length < 1:
            raise ValueError(f"Cycle length ({self.cycle_length}) must be at least 1. "
                             f"Total steps: {self.total_steps}, n_cycles: {self.n_cycles}")
        assert self.total_steps > self.n_cycles, \
            f"Total steps ({self.total_steps}) must be greater than Total Cycles ({self.n_cycles})."
        
        # Calculate warmup steps based on warmup epochs
        if self.warmup_epochs > 0:
            self.warmup_steps = int(self.warmup_epochs * self.batches_per_epoch)
            if self.warmup_steps >= self.total_steps:
                raise ValueError(f"Warmup steps ({self.warmup_steps}) must be less than total steps ({self.total_steps}).")


    def _get_lr_schedule(self):
        
        def cosine_annealing(step):
            cycle_length = math.ceil(self.total_steps / self.n_cycles)
            cycle_step = step % cycle_length
            # Return multiplier: cosine annealing between eta_min/base_lr and eta_max/base_lr
            # Assuming base_lr is eta_max, we scale between eta_min/eta_max and 1.0
            cosine_factor = 0.5 * (1 + np.cos(np.pi * cycle_step / cycle_length))
            min_multiplier = self.eta_min / self.eta_max
            return min_multiplier + (1.0 - min_multiplier) * cosine_factor

        def mmcclr(step):
            # Convert step to epoch-based calculation
            current_epoch = step // self.batches_per_epoch
            cycle_length_epochs = self.total_epochs // self.n_cycles
            cycle_step = current_epoch % cycle_length_epochs
            # Return multiplier: cosine annealing between eta_min/base_lr and eta_max/base_lr
            # Assuming base_lr is eta_max, we scale between eta_min/eta_max and 1.0
            cosine_factor = 0.5 * (1 + np.cos(np.pi * cycle_step / cycle_length_epochs))
            min_multiplier = self.eta_min / self.eta_max
            return min_multiplier + (1.0 - min_multiplier) * cosine_factor

        def deferred(base_schedule):
            # Warmup returns 1.0; afterwards use base schedule shifted by warmup steps
            return lambda step: 1.0 if step < self.warmup_steps else base_schedule(step - self.warmup_steps)

        sched_map = {
            "cca": cosine_annealing,
            "mmcclr": mmcclr,
            "deferred_cca": deferred(cosine_annealing),
            "deferred_mmcclr": deferred(mmcclr),
        }

        if self.scheduler_name not in sched_map:
            raise ValueError(f"Unknown scheduler name: {self.scheduler_name}")

        return sched_map[self.scheduler_name]
    
    def _calculate_snap_weights(self):
        """
        Calculate weights for model snapshots based on their loss values.
        The weight for each snapshot is inversely proportional to its loss.
        """
        # Get losses in the order of the model names you intend to use in your ensemble:
        model_names = list(self.snapshot_loss.keys())
        losses = np.array([self.snapshot_loss[name] for name in model_names])

        min_loss = losses.min()
        max_loss = losses.max()

        # SnapE weights: (max+min) - model_loss
        raw_weights = (max_loss + min_loss) - losses

        # Clip to avoid negative weights
        raw_weights = np.clip(raw_weights, a_min=0, a_max=None)

        # Normalize weights to sum to 1
        if raw_weights.sum() > 0:
            weights = raw_weights / raw_weights.sum()
        else:
            weights = np.ones_like(raw_weights) / len(raw_weights)
        
        self.snapshot_weights = dict(zip(model_names, weights))

    def on_train_start(self, trainer, model):
        """Initialize training parameters and LR scheduler at start of training."""
        self._initialize_training_params(trainer.num_training_batches)
        self.lr_lambda = self._get_lr_schedule()
        self.scheduler = LambdaLR(trainer.optimizers[0], lr_lambda=self.lr_lambda)
        self.step_count = 0

        print(f"Using learning rate scheduler: {self.scheduler_name}")

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        """Step the LR scheduler and save model snapshot if needed after each batch."""
        self.scheduler.step()
        self.step_count += 1

        # Log the learning rate for this step
        # current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, "get_last_lr") else None

        if self._is_snapshot_step(self.step_count):
            snapshot_path = os.path.join(
                self.snapshot_dir, f"snapshot_epoch_{trainer.current_epoch}.pt"
            )
            torch.save(model.state_dict(), snapshot_path)
            self.snapshot_loss[os.path.basename(snapshot_path)] = outputs['loss'].item()  # Store loss at snapshot step

    def _is_snapshot_step(self, step):
        """
        Determine if the current step is a snapshot step.

        For deferred schedulers: Take n_snapshots evenly distributed in the active scheduling phase.
        For regular schedulers: Take snapshots at the end of each cycle.
        """
        if self.scheduler_name.startswith("deferred"):
            # Skip snapshots during warmup
            if step < self.warmup_steps:
                return False
            
            # Take n_snapshots evenly distributed in the remaining steps after warmup
            remaining_steps = self.total_steps - self.warmup_steps
            snapshot_interval = remaining_steps // self.n_snapshots
            steps_after_warmup = step - self.warmup_steps
            
            # Check if we're at a snapshot interval boundary
            return (steps_after_warmup + 1) % snapshot_interval == 0
        else:
            # For non-deferred schedulers, use cycle-based snapshots
            return (step + 1) % self.cycle_length == 0

    def on_fit_end(self, trainer, model):
        # Load all model snapshots from the snapshot directory
        self.ensemble_weights = None
        snapshot_files = sorted(
            [f for f in os.listdir(self.snapshot_dir) if f.endswith('.pt')]
        )
        self.model_snapshots = []
        for checkpoint in snapshot_files:
            checkpoint_path = os.path.join(self.snapshot_dir, checkpoint)
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model_copy = type(model)(model.args)
            model_copy.load_state_dict(state_dict)
            self.model_snapshots.append(model_copy)

        if self.snapshot_loss and self.weighted_ensemble:
            self._calculate_snap_weights()
            # 2. Build the weight list aligned to snapshot_files order:
            self.ensemble_weights = [self.snapshot_weights[fname] for fname in snapshot_files]
        
        
        ensemble_eval_report = evaluate_ensemble_link_prediction_performance(
            models=self.model_snapshots,
            triples=trainer.dataset.test_set,
            er_vocab=trainer.dataset.er_vocab.result(),
            weights=self.ensemble_weights,
            batch_size=trainer.num_training_batches,
            weighted_averaging=self.weighted_ensemble
            )
        # Prepare a single dictionary with LR scheduling info and nested ensemble eval report
        self.ensemble_eval_report = {
            "scheduler_name": self.scheduler_name,
            "total_epochs": self.total_epochs,
            "num_cycles": self.n_cycles,
            "warmup_epochs": self.warmup_epochs,
            "lr_max": self.eta_max,
            "lr_min": self.eta_min,
            "batches_per_epoch": self.batches_per_epoch,
            "total_steps": self.total_steps,
            "cycle_length": self.cycle_length,
            "warmup_steps": self.warmup_steps,
            "weighted_ensemble": self.weighted_ensemble,
            "n_snapshots": self.n_snapshots,
            "ensemble_eval_report": ensemble_eval_report,
            "snapshot_loss": self.snapshot_loss
        }

        ensemble_eval_report_path = os.path.join(self.experiment_dir, "ensemble_eval_report.json")
        # Write the dictionary to the JSON file
        with open(ensemble_eval_report_path, 'w', encoding='utf-8') as f:
            json.dump(self.ensemble_eval_report, f, indent=4, ensure_ascii=False)
        print(f"Ensemble Evaluations: Evaluate {model.name} on Test Set with an ensemble of {len(self.model_snapshots)} models: \n{ensemble_eval_report}")

class SWA(AbstractCallback):
    """Stochastic Weight Averaging callbacks."""

    def __init__(self, swa_start_epoch, swa_c_epochs:int=1, lr_init:float=0.1, swa_lr:float=0.05, max_epochs :int=None):
        super().__init__()
        """
        Initialize SWA callback.
        Parameters
        ----------
        swa_start_epoch: int
            The epoch at which to start SWA.
        swa_c_epochs: int
            The number of epochs to use for SWA.
        lr_init: float
            The initial learning rate.
        swa_lr: float
            The learning rate to use during SWA.
        max_epochs: int
            The maximum number of epochs. args.num_epochs
        """
        self.swa_start_epoch = swa_start_epoch
        self.swa_c_epochs = swa_c_epochs
        self.swa_lr = swa_lr
        self.lr_init = lr_init
        self.max_epochs = max_epochs
        self.swa_model = None
        self.swa_n = 0
        self.current_epoch = -1
        self._collected_models = []
    
    @staticmethod
    def moving_average(swa_model, running_model, alpha):
        """Update SWA model with moving average of current model."""
        with torch.no_grad():
            swa_model.to(next(running_model.parameters()).device)
            for swa_param, param in zip(swa_model.parameters(), running_model.parameters()):
                swa_param.data = (1.0 - alpha) * swa_param.data + alpha * param.data

    @rank_zero_only
    def on_fit_start(self, trainer, model):
        """Initialize SWA model(s) with same architecture as main model(s)."""
        
        # Handle OptimizedModule wrapper
        kge_model = model._orig_mod if isinstance(model, OptimizedModule) else model

        # Case 1: Ensemble of models
        if isinstance(kge_model, EnsembleKGE):
            self.swa_models = []
            for submodel in kge_model.models:  # assuming .models holds the ensemble
                swa_submodel = type(submodel)(submodel.args)
                swa_submodel.load_state_dict(submodel.state_dict())
                self.swa_models.append(swa_submodel)

            # Collect optimizers from the ensemble
            self.optimizers = list(getattr(kge_model, "optimizers", []))

        # Case 2: Single model
        else:
            self.swa_model = type(kge_model)(kge_model.args)
            self.swa_model.load_state_dict(kge_model.state_dict())

            self.optimizers = []

            # Single optimizer case
            if getattr(trainer, "optimizer", None) is not None:
                self.optimizers.append(trainer.optimizer)

            # Multiple optimizer(s) case
            optims_attr = getattr(trainer, "optimizers", None)
            if optims_attr is not None:
                if isinstance(optims_attr, list):
                    self.optimizers.extend(optims_attr)
                else:
                    self.optimizers.append(optims_attr)

    @rank_zero_only
    def on_train_epoch_start(self, trainer, model):
        """Update learning rate according to SWA schedule."""
        # Get current epoch - simplified with fallback
        if hasattr(trainer, 'current_epoch'):
            self.current_epoch = trainer.current_epoch
        else:
            self.current_epoch += 1
        if self.current_epoch < self.swa_start_epoch:
            return
        # Calculate learning rate using the schedule
        t = self.current_epoch / self.max_epochs
        lr_ratio = self.swa_lr / self.lr_init
        
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
            
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg["lr"] *= factor

    @rank_zero_only
    def on_train_epoch_end(self, trainer, model):
        """Apply SWA averaging if conditions are met."""
        if self.current_epoch < self.swa_start_epoch:
            return

        # Check if we should apply SWA
        if self.current_epoch >= self.swa_start_epoch and \
        (self.current_epoch - self.swa_start_epoch) % self.swa_c_epochs == 0:
            
            current_model = model._orig_mod if isinstance(model, OptimizedModule) else model

            if isinstance(current_model, EnsembleKGE):
                # Update each submodel and its SWA counterpart
                for submodel, swa_submodel in zip(current_model.models, self.swa_models):
                    self.moving_average(swa_submodel, submodel, 1.0 / (self.swa_n + 1))
            else:
                # Single model case
                self.moving_average(self.swa_model, current_model, 1.0 / (self.swa_n + 1))

            self.swa_n += 1
    
    @rank_zero_only
    def on_fit_end(self, trainer, model):
        """Replace main model with SWA model at the end of training."""
        if self.swa_model is not None and self.swa_n > 0:
            # Copy SWA weights back to main model directly
            model.load_state_dict(self.swa_model.state_dict())