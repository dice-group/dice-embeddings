import torch
import torch.nn as nn
from torch._dynamo.eval_frame import OptimizedModule

from dicee.models.ensemble import EnsembleKGE
from .abstracts import AbstractCallback

class SWA(AbstractCallback):
    """Stochastic Weight Averaging callbacks."""

    def __init__(self, swa_start_epoch, swa_c_epochs:int=1, lr_init:float=0.1,
                  swa_lr:float=0.05, max_epochs :int=None):
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
    
    @staticmethod
    def moving_average(swa_model, running_model, alpha):
        """Update SWA model with moving average of current model.
        Math: 
        # SWA update:
        # θ_swa ← (1 - alpha) * θ_swa + alpha * θ
        # alpha = 1 / (n + 1), where n = number of models already averaged 
        # alpha is tracked via self.swa_n in code"""

        with torch.no_grad():
            swa_model.to(next(running_model.parameters()).device)
            for swa_param, param in zip(swa_model.parameters(), running_model.parameters()):
                swa_param.data = (1.0 - alpha) * swa_param.data + alpha * param.data

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

        if isinstance(model, EnsembleKGE):
            optimizers = getattr(model, "optimizers", [])
        elif hasattr(trainer, "optimizers") and trainer.optimizers:
                optimizers = trainer.optimizers if isinstance(trainer.optimizers, list) else [trainer.optimizers]
        elif hasattr(trainer, "optimizer") and trainer.optimizer is not None:
                optimizers = [trainer.optimizer]
        else:
            optimizers = None

        if optimizers is not None:
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= factor

    def on_train_epoch_end(self, trainer, model):
        """Apply SWA averaging if conditions are met."""
        if self.current_epoch < self.swa_start_epoch:
            return

        # Check if we should apply SWA
        if self.current_epoch >= self.swa_start_epoch and \
        (self.current_epoch - self.swa_start_epoch) % self.swa_c_epochs == 0:
            
            running_model = model._orig_mod if isinstance(model, OptimizedModule) else model
            
            if self.swa_model is None:
                # Case: EnsembleKGE
                if isinstance(running_model, EnsembleKGE):
                    self.swa_model = type(running_model)(running_model.models)
                    self.swa_model.load_state_dict(running_model.state_dict())
                
                else:
                    self.swa_model = type(running_model)(running_model.args)
                    self.swa_model.load_state_dict(running_model.state_dict())

            if isinstance(running_model, EnsembleKGE):
                # Update each submodel and its SWA counterpart
                for submodel, swa_submodel in zip(running_model.models, self.swa_model.models):
                    self.moving_average(swa_submodel, submodel, 1.0 / (self.swa_n + 1))
            else:
                # Single model case
                self.moving_average(self.swa_model, running_model, 1.0 / (self.swa_n + 1))

            self.swa_n += 1
    
        if model.args["eval_every_n_epochs"] > 0 or model.args["eval_at_epochs"] is not None:
            trainer.swa_model = self.swa_model
    
    def on_fit_end(self, trainer, model):
        """Replace main model with SWA model at the end of training."""
        if self.swa_model is not None and self.swa_n > 0:
            # Copy SWA weights back to main model directly
            model.load_state_dict(self.swa_model.state_dict())

class SWAG(AbstractCallback):
    """Stochastic Weight Averaging - Gaussian  (SWAG)."""

    def __init__(self, swa_start_epoch, swa_c_epochs:int=1,
                 lr_init:float=0.1, swa_lr:float=0.05,
                 max_epochs:int=None, max_num_models:int=20, var_clamp:float=1e-30):
        super().__init__()
        """
        Parameters
        ----------
        swa_start_epoch : int
            Epoch at which to start collecting weights.
        swa_c_epochs : int
            Interval of epochs between updates.
        lr_init : float
            Initial LR.
        swa_lr : float
            LR in SWA / GSWA phase.
        max_epochs : int
            Total number of epochs.
        max_num_models : int
            Number of models to keep for low-rank covariance approx.
        var_clamp : float
            Clamp low variance for stability.
        """
        self.swa_start_epoch = swa_start_epoch
        self.swa_c_epochs = swa_c_epochs
        self.swa_lr = swa_lr
        self.lr_init = lr_init
        self.max_epochs = max_epochs
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp

        # Stats for Gaussian averaging
        self.mean = None
        self.sq_mean = None
        self.deviations = []
        self.gswa_n = 0
        self.current_epoch = -1

    def _collect_stats(self, model):
        """Collect weights to update mean, sq_mean, and covariance deviations.
    
        Math:
        # Let θ_i be the model parameter vector at collection step i
        # gswa_n = number of models collected so far (0-based)
        
        # Update running mean:
        # μ_{n+1} = (n * μ_n + θ_{n+1}) / (n + 1)
        # This is a cumulative moving average of model weights
        
        # Update running squared mean:
        # μ2_{n+1} = (n * μ2_n + θ_{n+1}^2) / (n + 1)
        # This is used to compute variance: alpha^2 ≈ μ2 - μ^2
        
        # Compute deviation for low-rank covariance approximation:
        # dev_{n+1} = θ_{n+1} - μ_{n+1}
        # We store the last max_num_models deviations to approximate covariance
        """
        vec = nn.utils.parameters_to_vector(model.parameters()).detach().cpu()

        if self.mean is None:
            self.mean = vec.clone()
            self.sq_mean = vec.clone()**2
        else:
            self.mean = (self.mean * self.gswa_n + vec) / (self.gswa_n + 1)
            self.sq_mean = (self.sq_mean * self.gswa_n + vec**2) / (self.gswa_n + 1)

        # low-rank covariance info
        dev = (vec - self.mean).unsqueeze(1)
        self.deviations.append(dev)
        if len(self.deviations) > self.max_num_models:
            self.deviations.pop(0)

        self.gswa_n += 1

    def get_mean_and_var(self):
        """Return mean + variance (diagonal part)."""
        if self.mean is None:
            return None, None
        var = torch.clamp(self.sq_mean - self.mean**2, min=self.var_clamp)
        return self.mean, var

    def sample(self, base_model_fn, scale=0.5, device="cpu"):
        """Sample new model from SWAG posterior distribution.
        
        Math:
        # From SWAG, posterior is approximated as:
        # θ ~ N(mean, Σ)
        # where Σ ≈ diag(var) + (1/(K-1)) * D D^T
        #   - mean = running average of weights
        #   - var = elementwise variance (sq_mean - mean^2)
        #   - D = [dev_1, dev_2, ..., dev_K], deviations from mean (low-rank approx)
        #   - K = number of collected models

        # Sampling step:
        # 1. θ_diag = mean + scale * std ⊙ ε,  where ε ~ N(0, I)
        # 2. θ_lowrank = θ_diag + (D z) / sqrt(K-1), where z ~ N(0, I_K)
        # Final sample = θ_lowrank
        """
        if self.mean is None:
            raise RuntimeError("No SWAG stats collected yet.")

        # Mean and variance
        mean, var = self.get_mean_and_var()
        std = torch.sqrt(var)

        # Diagonal Gaussian perturbation
        sample_vec = mean + scale * std * torch.randn_like(std)

        # Low-rank covariance perturbation
        if self.deviations:
            D = torch.cat(self.deviations, dim=1)  # shape: [num_params, K]
            z = torch.randn(D.shape[1], device=D.device)  # random vector in K-dim space
            denom = max(1, len(self.deviations) - 1) ** 0.5  # normalization
            sample_vec += (D @ z) / denom

        # Build new model and load sampled weights
        m = base_model_fn().to(device)
        nn.utils.vector_to_parameters(sample_vec.to(device), m.parameters())
        return m

    def on_train_epoch_start(self, trainer, model):
        """Update LR schedule (same as SWA)."""
        if hasattr(trainer, 'current_epoch'):
            self.current_epoch = trainer.current_epoch
        else:
            self.current_epoch += 1
        if self.current_epoch < self.swa_start_epoch: 
            return

        # LR cosine-like schedule
        t = self.current_epoch / self.max_epochs
        lr_ratio = self.swa_lr / self.lr_init
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio

        # update trainer optimizers
        if hasattr(trainer, "optimizers") and trainer.optimizers:
            optimizers = trainer.optimizers if isinstance(trainer.optimizers, list) else [trainer.optimizers]
        elif hasattr(trainer, "optimizer") and trainer.optimizer is not None:
            optimizers = [trainer.optimizer]
        else:
            optimizers = []
        for optimizer in optimizers:
            for pg in optimizer.param_groups:
                pg['lr'] *= factor

    def on_train_epoch_end(self, trainer, model):
        """Collect Gaussian stats at the end of epochs after swa_start."""
        if self.current_epoch < self.swa_start_epoch:
            return
        if (self.current_epoch - self.swa_start_epoch) % self.swa_c_epochs != 0:
            return

        self._collect_stats(model)

    def on_fit_end(self, trainer, model):
        """Set model weights to the collected SWAG mean at the end of training."""
        if self.mean is not None:
            nn.utils.vector_to_parameters(
                self.mean.to(next(model.parameters()).device),
                model.parameters()
            )
        return model


class EMA(AbstractCallback):
    """Exponential Moving Average (EMA) callback."""

    def __init__(self, ema_start_epoch: int, decay: float = 0.999, max_epochs: int = None):
        """
        Parameters
        ----------
        ema_start_epoch : int
            Epoch to start EMA.
        decay : float
            EMA decay rate (typical: 0.99 - 0.9999)
            Math: θ_ema <- decay * θ_ema + (1 - decay) * θ
        max_epochs : int
            Maximum number of epochs.
        ( should be compatible with all trainers and multi-gpu setup )
        """
        super().__init__()
        self.ema_start_epoch = ema_start_epoch
        self.decay = decay
        self.max_epochs = max_epochs

        self.ema_model = None
        self.current_epoch = -1

    @staticmethod
    def ema_update(ema_model, running_model, decay: float):
        """Update EMA model with exponential moving average of current model.
        Math: 
        # EMA update:
        # θ_ema ← (1 - alpha) * θ_ema + alpha * θ
        # alpha = 1 - decay, where decay is the EMA smoothing factor (typical 0.99 - 0.999)
        # alpha controls how much of the current model θ contributes to the EMA
        # decay  is fixed  in code --> can be extended to sheduled
        """
        with torch.no_grad():
            ema_model.to(next(running_model.parameters()).device)
            for ema_param, param in zip(ema_model.parameters(), running_model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

    def on_train_epoch_start(self, trainer, model):
        """Track current epoch."""
        if hasattr(trainer, 'current_epoch'):
            self.current_epoch = trainer.current_epoch
        else:
            self.current_epoch += 1

    def on_train_epoch_end(self, trainer, model):
        """Update EMA if past start epoch."""
        if self.current_epoch < self.ema_start_epoch:
            return

        running_model = model._orig_mod if isinstance(model, OptimizedModule) else model

        if self.ema_model is None:
            # Initialize EMA model as a copy of running model
            if isinstance(running_model, EnsembleKGE):
                self.ema_model = type(running_model)(running_model.models)
                self.ema_model.load_state_dict(running_model.state_dict())
            else:
                self.ema_model = type(running_model)(running_model.args)
                self.ema_model.load_state_dict(running_model.state_dict())

        # Always use fixed decay since we start late
        decay_t = self.decay

        if isinstance(running_model, EnsembleKGE):
            for submodel, ema_submodel in zip(running_model.models, self.ema_model.models):
                self.ema_update(ema_submodel, submodel, decay_t)
        else:
            self.ema_update(self.ema_model, running_model, decay_t)

        # Make EMA model available for evaluation
        if model.args.get("eval_every_n_epochs", 0) > 0 or model.args.get("eval_at_epochs") is not None:
            trainer.ema_model = self.ema_model

    def on_fit_end(self, trainer, model):
        """Replace main model with EMA model at the end of training."""
        if self.ema_model is not None:
            model.load_state_dict(self.ema_model.state_dict())