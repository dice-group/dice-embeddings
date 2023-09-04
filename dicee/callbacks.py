import datetime
import time
import numpy as np
import torch

import dicee.models.base_model
from .static_funcs import save_checkpoint_model, exponential_function, save_pickle
from .abstracts import AbstractCallback, AbstractPPECallback
import pandas as pd


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
        print(pl_module)
        print(pl_module.summarize())
        print(pl_module.selected_optimizer)
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


class PPE(AbstractPPECallback):
    """ A callback for Polyak Parameter Ensemble Technique
        Maintains a running parameter average for all parameters requiring gradient signals
    """

    def __init__(self, num_epochs, path, last_percent_to_consider=None):
        super().__init__(num_epochs, path, last_percent_to_consider)
        self.alphas = np.ones(self.num_ensemble_coefficient) / self.num_ensemble_coefficient
        print("Equal Ensemble Coefficients:", self.alphas)


class FPPE(AbstractPPECallback):
    """
    import matplotlib.pyplot as plt
    import numpy as np
    def exponential_function(x: np.ndarray, lam: float, ascending_order=True) -> torch.FloatTensor:
        # A sequence in exponentially decreasing order
        result = np.exp(-lam * x) / np.sum(np.exp(-lam * x))
        assert 0.999 < sum(result) < 1.0001
        result = np.flip(result) if ascending_order else result
        return torch.tensor(result.tolist())

    N = 100
    equal_weights = np.ones(N) / N
    plt.plot(equal_weights, 'r', label="Equal")
    plt.plot(exponential_function(np.arange(N), lam=0.1,), 'c-', label="Exp. forgetful with 0.1")
    plt.plot(exponential_function(np.arange(N), lam=0.05), 'g-', label="Exp. forgetful with 0.05")
    plt.plot(exponential_function(np.arange(N), lam=0.025), 'b-', label="Exp. forgetful with 0.025")
    plt.plot(exponential_function(np.arange(N), lam=0.01), 'k-', label="Exp. forgetful with 0.01")
    plt.title('Ensemble coefficients')
    plt.xlabel('Epochs')
    plt.ylabel('Coefficients')
    plt.legend()
    plt.savefig('ensemble_coefficients.pdf')
    plt.show()
    """

    def __init__(self, num_epochs, path, last_percent_to_consider=None):
        super().__init__(num_epochs, path, last_percent_to_consider)
        lamb = 0.1
        self.alphas = exponential_function(np.arange(self.num_ensemble_coefficient), lam=lamb, ascending_order=True)
        print(f"Forgetful Ensemble Coefficients with lambda {lamb}:", self.alphas)


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
    def __init__(self, level: str, ratio: float, method: str = None):
        """
        level in {input, param, output}
        ratio in [0,1]
        method = ?
        """
        super().__init__()

        assert level in {"input"}
        assert ratio >= 0.0
        self.level = level
        self.ratio = ratio
        self.method = method

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        # Modifications should be in-place
        x, y = batch
        n, _ = x.shape
        num_of_perturbed_data = int(n * self.ratio)


        devicee=x.get_device()
        # Sample random integers from 0 to n without replacement and take k of tem
        random_indices = torch.randperm(n,device=devicee)[:num_of_perturbed_data]
        if self.level == "input":
            if torch.rand(1) > 0.5:
                # Perturb input via heads
                perturbation = torch.randint(low=0, high=model.num_entities, size=(num_of_perturbed_data,),device=devicee)
                x[random_indices] = torch.column_stack(
                    (perturbation, x[:, 1][random_indices]))
            else:
                # Perturb input via relations
                perturbation = torch.randint(low=0, high=model.num_relations, size=(num_of_perturbed_data,),device=devicee)
                x[random_indices] = torch.column_stack(
                    (x[:, 0][random_indices], perturbation))


class GN(AbstractCallback):
    '''
    Adding Gaussian Noise into Inputs/Parameters
    '''

    def __init__(self, std: float = 0.1, epoch_ratio: int = None):
        super().__init__()
        self.std = std
        self.epoch_ratio = epoch_ratio if epoch_ratio is not None else 1
        self.epoch_counter = 0

    def on_train_epoch_start(self, trainer, model):
        if self.epoch_counter % self.epoch_ratio == 0:
            with torch.no_grad():
                # Access the parameters
                for param in model.parameters():
                    noise_mat = torch.normal(mean=0, std=self.std, size=param.shape, device=model.device)
                    param.add_(noise_mat)
        self.epoch_counter += 1


class RN(AbstractCallback):
    """ Adding Uniform at Random Noise into Inputs/Parameters """

    def __init__(self, std: float = 0.1, epoch_ratio: int = None):
        super().__init__()
        self.std = std
        self.epoch_ratio = epoch_ratio if epoch_ratio is not None else 1
        self.epoch_counter = 0

    def on_train_epoch_start(self, trainer, model):
        if self.epoch_counter % self.epoch_ratio == 0:
            with torch.no_grad():
                # Access the parameters
                for param in model.parameters():
                    noise_mat = torch.rand(size=param.shape) * self.std
                    param.add_(noise_mat)
        self.epoch_counter += 1
