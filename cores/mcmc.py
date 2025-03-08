import tqdm
import torch
import numpy as np
import torch.nn as nn
from cores.trajectory import Trajectory


class MCMCSampler(nn.Module):
    """
        Langevin Dynamics sampling method.
    """

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01, prior_solver='gaussian', prior_sigma_min=1e-2,
                 mc_algo='langevin', momentum=0.9):
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio
        self.prior_solver = prior_solver
        self.prior_sigma_min = prior_sigma_min
        self.mc_algo = mc_algo
        self.momentum = momentum

    def score_fn(self, x, x0hat, model, xt, operator, measurement, sigma):
        """
            compute/approximate the score function p(x_0 = x | x_t, y)
        """
        data_fitting_grad, data_fitting_loss = operator.gradient(x, measurement, return_loss=True)
        data_term = -data_fitting_grad / self.tau ** 2
        xt_term = (xt - x) / sigma ** 2
        prior_term = self.get_prior_score(x, x0hat, xt, model, sigma)
        return data_term + xt_term + prior_term, data_fitting_loss

    def get_prior_score(self, x, x0hat, xt, model, sigma):
        if self.prior_solver == 'score-min' or self.prior_solver == 'score-t' or self.prior_solver == 'gaussian':
            prior_score = self.prior_score
        elif self.prior_solver == 'exact':
            prior_score = model.score(x, self.prior_sigma_min).detach()
        else:
            raise NotImplementedError
        return prior_score

    def prepare_prior_score(self, x0hat, xt, model, sigma):
        if self.prior_solver == 'score-min':
            self.prior_score = model.score(x0hat, self.prior_sigma_min).detach()

        elif self.prior_solver == 'score-t':
            self.prior_score = model.score(xt, sigma).detach()

        elif self.prior_solver == 'gaussian':
            self.prior_score = (x0hat - xt).detach() / sigma ** 2

        elif self.prior_solver == 'exact':
            pass

        else:
            raise NotImplementedError

    def mc_prepare(self, x0hat, xt, model, operator, measurement, sigma):
        if self.mc_algo == 'hmc':
            self.velocity = torch.randn_like(x0hat)

    def mc_update(self, x, cur_score, lr, epsilon):
        if self.mc_algo == 'langevin':
            x_new = x + lr * cur_score + np.sqrt(2 * lr) * epsilon
        elif self.mc_algo == 'hmc':  # (damping) hamiltonian monte carlo
            step_size = np.sqrt(lr)
            self.velocity = self.momentum * self.velocity + step_size * cur_score + np.sqrt(2 * (1 - self.momentum)) * epsilon
            x_new = x + self.velocity * step_size
        else:
            raise NotImplementedError
        return x_new

    def sample_MH(self, xt, model, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
        if record:
            self.trajectory = Trajectory()
        
        lr = self.get_lr(ratio)
        x = x0hat.clone().detach()
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        for _ in pbar:
            # Gaussain proposal
            x_new = x + torch.randn_like(x) * np.sqrt(lr)
            # compute acceptance ratio
            loss_new = operator.loss(x_new, measurement)
            loss_old = operator.loss(x, measurement)
            log_data_ratio = (loss_old - loss_new) / (2 * self.tau ** 2)
            # print(log_data_ratio.shape)
            # print(log_data_ratio)
            # compute prior p(x_0 | x_t)
            prior_loss_new = (x_new - x0hat).pow(2).flatten(1).sum(-1) 
            prior_loss_old = (x - x0hat).pow(2).flatten(1).sum(-1)
            log_prior_ratio = (prior_loss_old - prior_loss_new) / (2 * sigma ** 2)
            # print(log_prior_ratio.shape)
            # print(log_prior_ratio)

            # compute acceptance probability
            log_accept_prob= log_data_ratio + log_prior_ratio
            accept = torch.rand_like(log_accept_prob).log() < log_accept_prob
            accept = accept.view(-1, *[1] * len(x.shape[1:]))
            # print(accept.float().mean())
            # update: accept new sample
            x = torch.where(accept, x_new, x)
        return x.detach()

    def sample(self, xt, model, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
        if self.mc_algo == 'MH':
            return self.sample_MH(xt, model, x0hat, operator, measurement, sigma, ratio, record, verbose)
        if record:
            self.trajectory = Trajectory()
        lr = self.get_lr(ratio)
        self.mc_prepare(x0hat, xt, model, operator, measurement, sigma)
        self.prepare_prior_score(x0hat, xt, model, sigma)

        x = x0hat.clone().detach()
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        for _ in pbar:
            # Langevin step: compute/approximate the score function p(x_0 = x | x_t, y)
            cur_score, fitting_loss = self.score_fn(x, x0hat, model, xt, operator, measurement, sigma)
            epsilon = torch.randn_like(x)

            # update
            x = self.mc_update(x, cur_score, lr, epsilon)

            # print('fitting loss:', '{:.4f}'.format(fitting_loss.sqrt().item()))
            # print('sigma:', '{:.4f}'.format(sigma), 'lr:', '{:.4f}'.format(lr))

            # early stopping with NaN
            if torch.isnan(x).any():
                return torch.zeros_like(x) 

            # record
            if record:
                self._record(x, epsilon, fitting_loss.sqrt())
        return x.detach()

    def _record(self, x, epsilon, loss):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xi', x)
        self.trajectory.add_tensor(f'epsilon', epsilon)
        self.trajectory.add_value(f'loss', loss)

    def get_lr(self, ratio):
        """
            Computes the learning rate based on the given ratio.
        """
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr
