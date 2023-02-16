import torch
import copy
import collections
import numpy as np

from . import utils as ut

class StochLineSearchBase(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0,
                 reset_option=0,
                 line_search_fn="armijo"):
        params = list(params)
        super().__init__(params, {})

        self.params = params
        self.c = c
        self.beta_b = beta_b
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.n_batches_per_epoch = n_batches_per_epoch
        self.line_search_fn = line_search_fn
        self.state['step'] = 0
        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.state['n_backtr'] = []
        self.budget = 0
        self.reset_option = reset_option
        self.new_epoch()


    def step(self, closure):
        # deterministic closure
        raise RuntimeError("This function should not be called")

    def line_search(self, step_size, params_current, grad_current, loss, closure_deterministic, grad_norm, non_parab_dec=None, precond=False):
        with torch.no_grad():

            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
            if grad_norm >= 1e-8 and loss.item() != 0:
                # check if condition is satisfied
                found = 0

                if non_parab_dec is not None:
                    suff_dec = non_parab_dec
                else:
                    suff_dec = grad_norm**2

                for e in range(100):
                    # try a prospective step
                    if precond:
                        self.try_sgd_precond_update(self.params, step_size, params_current, grad_current, momentum=0.)
                    else:
                        ut.try_sgd_update(self.params, step_size, params_current, grad_current)

                    # compute the loss at the next step; no need to compute gradients.
                    loss_next = closure_deterministic()
                    self.state['n_forwards'] += 1

                    if self.line_search_fn == "armijo":
                        found, step_size = self.check_armijo_conditions(step_size=step_size,
                                                                        loss=loss,
                                                                        suff_dec=suff_dec,
                                                                        loss_next=loss_next,
                                                                        c=self.c,
                                                                        beta_b=self.beta_b)
                    if found == 1:
                        break
                   
                # if line search exceeds max_epochs
                if found == 0:
                    step_size = torch.tensor(data=1e-6)
                    ut.try_sgd_update(self.params, 1e-6, params_current, grad_current)

                self.state['backtracks'] += e
                self.state['f_eval'].append(e)
                self.state['n_backtr'].append(e)

            else:
                print("Grad norm is {} and loss is {}".format(grad_norm, loss.item()))
                if loss.item() == 0:
                    self.state['numerical_error'] += 1
                if grad_norm == 0:
                    self.state["zero_steps"] += 1
                step_size = 0
                loss_next = closure_deterministic()

        return step_size, loss_next

    @staticmethod
    def check_armijo_conditions(step_size, loss, suff_dec, loss_next, c, beta_b):
        found = 0
        sufficient_decrease = step_size * c * suff_dec
        rhs = loss - sufficient_decrease
        break_condition = loss_next - rhs
        if break_condition <= 0:
            found = 1
        else:
            step_size = step_size * beta_b

        return found, step_size

    @staticmethod
    def reset_step(step_size, n_batches_per_epoch=None, gamma=None, reset_option=1, init_step_size=None):

        if reset_option == 0:
            pass
        elif reset_option == 1:
            step_size = step_size * gamma ** (1. / n_batches_per_epoch)
        elif reset_option == 11:
            step_size = min(step_size * gamma ** (1. / n_batches_per_epoch), 10)
        elif reset_option == 2:
            step_size = init_step_size
        else:
            raise ValueError("reset_option {} does not existing".format(reset_option))

        return step_size

    def save_state(self, step_size, loss, loss_next, grad_norm):
        if isinstance(step_size, torch.Tensor):
            step_size = step_size.item()
        self.state['step_size'] = step_size
        self.state['step'] += 1
        self.state['all_step_size'].append(step_size)
        self.state['all_losses'].append(loss.item())
        self.state['all_new_losses'].append(loss_next.item())
        self.state['n_batches'] += 1
        self.state['avg_step'] += step_size
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        self.state['grad_norm'].append(grad_norm)

    def new_epoch(self):
        self.state['avg_step'] = 0
        self.state['semi_last_step_size'] = 0
        self.state['all_step_size'] = []
        self.state['all_losses'] = []
        self.state['grad_norm'] = []
        self.state['all_new_losses'] = []
        self.state['f_eval'] = []
        self.state['backtracks'] = 0
        self.state['n_batches'] = 0
        self.state['zero_steps'] = 0
        self.state['numerical_error'] = 0

    @staticmethod
    def gather_flat_grad(self, params):
        views = []
        for p in params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    @staticmethod
    def flatten_vect(self, vect):
        views = []
        for p in vect:
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)
