import copy
import time
import torch
import numpy as np

from . import utils as ut
from .sls_base import StochLineSearchBase


class AdamSLS(StochLineSearchBase):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=0.00001,
                 c=0.1,
                 gamma=2.0,
                 beta=0.999,
                 momentum=0.9,
                 gv_option='per_param',
                 base_opt='adam',
                 pp_norm_method='pp_armijo',
                 mom_type='standard',
                 clip_grad=False,
                 beta_b=0.9,
                 beta_f=2.0,
                 reset_option=1,
                 line_search_fn="armijo"):
        params = list(params)
        super().__init__(params,
                         n_batches_per_epoch=n_batches_per_epoch,
                         init_step_size=init_step_size,
                         c=c,
                         beta_b=beta_b,
                         gamma=gamma,
                         reset_option=reset_option,
                         line_search_fn=line_search_fn)
        self.mom_type = mom_type
        self.pp_norm_method = pp_norm_method

        # sps stuff
        # self.adapt_flag = adapt_flag

        # sls stuff
        self.beta_f = beta_f
        self.beta_b = beta_b
        self.reset_option = reset_option

        # others
        self.params = params
        if self.mom_type == 'heavy_ball':
            self.params_prev = copy.deepcopy(params)

        self.momentum = momentum
        self.beta = beta
        # self.state['step_size'] = init_step_size

        self.clip_grad = clip_grad
        self.gv_option = gv_option
        self.base_opt = base_opt
        # self.step_size_method = step_size_method

        # gv options
        self.gv_option = gv_option
        if self.gv_option in ['scalar']:
            self.state['gv'] = 0.

        elif self.gv_option == 'per_param':
            self.state['gv'] = [torch.zeros(p.shape).to(p.device) for p in params]

            if self.base_opt in ['amsgrad', 'adam']:
                self.state['mv'] = [torch.zeros(p.shape).to(p.device) for p in params]

            if self.base_opt == 'amsgrad':
                self.state['gv_max'] = [torch.zeros(p.shape).to(p.device) for p in params]

    def step(self, closure):
        # deterministic closure
        seed = time.time()

        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()

        loss = closure_deterministic()
        loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.params, 0.25)
        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1
        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = ut.get_grad_list(self.params)
        grad_norm = ut.compute_grad_norm(grad_current)

        #  Gv options
        # =============
        if self.gv_option in ['scalar']:
            # update gv
            self.state['gv'] += (grad_norm.item()) ** 2

        elif self.gv_option == 'per_param':
            # update gv
            for i, g in enumerate(grad_current):
                if self.base_opt == 'adagrad':
                    self.state['gv'][i] += g ** 2

                elif self.base_opt == 'rmsprop':
                    self.state['gv'][i] = (1 - self.beta) * (g ** 2) + (self.beta) * self.state['gv'][i]

                elif self.base_opt in ['amsgrad', 'adam']:
                    self.state['gv'][i] = (1 - self.beta) * (g ** 2) + (self.beta) * self.state['gv'][i]
                    self.state['mv'][i] = (1 - self.momentum) * g + (self.momentum) * self.state['mv'][i]

                else:
                    raise ValueError('%s does not exist' % self.base_opt)

        pp_norm, pp_norms = self.get_pp_norm(grad_current=grad_current)
        step_size = self.reset_step(step_size=self.state.get('step_size') or self.init_step_size,
                                    n_batches_per_epoch=self.n_batches_per_epoch,
                                    gamma=self.gamma,
                                    reset_option=self.reset_option,
                                    init_step_size=self.init_step_size)

        # compute step size
        # =================
        if self.pp_norm_method == "just_pp":
            orig_step = pp_norm * self.init_step_size
            step_size, loss_next = self.line_search(orig_step, params_current, grad_current, loss,
                                                    closure_deterministic, grad_norm)
            ut.try_sgd_update(self.params, step_size, params_current, grad_current)
        else:
            step_size, loss_next = self.line_search(step_size, params_current, grad_current, loss,
                                                    closure_deterministic, grad_norm, non_parab_dec=pp_norm,
                                                    precond=True)
            self.try_sgd_precond_update(self.params, step_size, params_current, grad_current, momentum=self.momentum)

        self.save_state(step_size, loss, loss_next, grad_norm)

        # compute gv stats
        gv_max = 0.
        gv_min = np.inf
        gv_sum = 0
        gv_count = 0

        for i, gv in enumerate(self.state['gv']):
            gv_max = max(gv_max, gv.max().item())
            gv_min = min(gv_min, gv.min().item())
            gv_sum += gv.sum().item()
            gv_count += len(gv.view(-1))

        self.state['gv_stats'] = {'gv_max': gv_max, 'gv_min': gv_min, 'gv_mean': gv_sum / gv_count}

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('nans detected')

        return loss

    def get_pp_norm(self, grad_current):
        if self.pp_norm_method in ['pp_armijo', "just_pp"]:
            pp_norm = 0
            pp_norms = []
            for i, (g_i, gv_i) in enumerate(zip(grad_current, self.state['gv'])):
                if self.base_opt in ['diag_hessian', 'diag_ggn_ex', 'diag_ggn_mc']:
                    pv_i = 1. / (gv_i + 1e-8)  # computing 1 / diagonal for using in the preconditioner

                elif self.base_opt == 'adam':
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step'] + 1)
                    pv_i = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                elif self.base_opt == 'amsgrad':
                    self.state['gv_max'][i] = torch.max(gv_i, self.state['gv_max'][i])
                    gv_i_scaled = scale_vector(self.state['gv_max'][i], self.beta, self.state['step'] + 1)

                    pv_i = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                elif self.base_opt in ['adagrad', 'rmsprop']:
                    pv_i = 1. / (torch.sqrt(gv_i) + 1e-8)
                else:
                    raise ValueError('%s not found' % self.base_opt)
                if self.pp_norm_method == 'pp_armijo':
                    layer_norm = ((g_i ** 2) * pv_i).sum()
                elif self.pp_norm_method == "just_pp":
                    layer_norm = pv_i.sum()
                pp_norm += layer_norm
                pp_norms.append(layer_norm.item())

        elif self.pp_norm_method in ['pp_lipschitz']:
            pp_norm = 0

            for g_i in grad_current:
                if isinstance(g_i, float) and g_i == 0:
                    continue
                pp_norm += (g_i * (g_i + 1e-8)).sum()

        else:
            raise ValueError('%s does not exist' % self.pp_norm_method)

        return pp_norm, pp_norms

    @torch.no_grad()
    def try_sgd_precond_update(self, params, step_size, params_current, grad_current, momentum):
        if self.gv_option in ['scalar']:
            zipped = zip(params, params_current, grad_current, self.state['gv'])

            for p_next, p_current, g_current, gv_i in zipped:
                p_next.data = p_current - (step_size / torch.sqrt(gv_i)) * g_current

        elif self.gv_option == 'per_param':
            if self.base_opt == 'adam':
                zipped = zip(params, params_current, grad_current, self.state['gv'], self.state['mv'])
                for p_next, p_current, g_current, gv_i, mv_i in zipped:
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step'] + 1)
                    pv_list = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                    if momentum == 0. or self.mom_type == 'heavy_ball':
                        mv_i_scaled = g_current
                    elif self.mom_type == 'standard':
                        mv_i_scaled = scale_vector(mv_i, momentum, self.state['step'] + 1)

                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list * mv_i_scaled), alpha=- step_size)

            elif self.base_opt == 'amsgrad':
                zipped = zip(params, params_current, grad_current, self.state['gv'], self.state['mv'])

                for i, (p_next, p_current, g_current, gv_i, mv_i) in enumerate(zipped):
                    self.state['gv_max'][i] = torch.max(gv_i, self.state['gv_max'][i])
                    gv_i_scaled = scale_vector(self.state['gv_max'][i], self.beta, self.state['step'] + 1)
                    pv_list = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                    if momentum == 0. or self.mom_type == 'heavy_ball':
                        mv_i_scaled = g_current
                    elif self.mom_type == 'standard':
                        mv_i_scaled = scale_vector(mv_i, momentum, self.state['step'] + 1)
                    else:
                        raise ValueError('does not exist')

                    # p_next.data = p_current - step_size * (pv_list *  mv_i_scaled)
                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list * mv_i_scaled), alpha=- step_size)

            elif (self.base_opt in ['rmsprop', 'adagrad']):
                zipped = zip(params, params_current, grad_current, self.state['gv'])
                for p_next, p_current, g_current, gv_i in zipped:
                    pv_list = 1. / (torch.sqrt(gv_i) + 1e-8)
                    # p_next.data = p_current - step_size * (pv_list *  g_current)

                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list * g_current), alpha=- step_size)

            elif (self.base_opt in ['diag_hessian', 'diag_ggn_ex', 'diag_ggn_mc']):
                zipped = zip(params, params_current, grad_current, self.state['gv'])
                for p_next, p_current, g_current, gv_i in zipped:
                    pv_list = 1. / (gv_i + 1e-8)  # adding 1e-8 to avoid overflow.
                    # p_next.data = p_current - step_size * (pv_list *  g_current)

                    # need to do this variant of the update for LSTM memory problems.
                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list * g_current), alpha=- step_size)


            else:
                raise ValueError('%s does not exist' % self.base_opt)

        else:
            raise ValueError('%s does not exist' % self.gv_option)


def scale_vector(vector, alpha, step, eps=1e-8):
    scale = (1 - alpha ** (max(1, step)))
    return vector / scale
