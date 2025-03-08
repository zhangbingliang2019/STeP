import tqdm
import torch
import numpy as np
import torch.nn as nn
from cores.scheduler import get_diffusion_scheduler, DiffusionPFODE
from forward_operator.latent_wrapper import LatentWrapper
from cores.mcmc import MCMCSampler
from .base import register_sampler


class Trajectory(nn.Module):
    """
        Class for recording and storing trajectory data.
    """

    def __init__(self):
        super().__init__()
        self.tensor_data = {}
        self.value_data = {}
        self._compile = False

    def add_tensor(self, name, images):
        """
            Adds image data to the trajectory.

            Parameters:
                name (str): Name of the image data.
                images (torch.Tensor): Image tensor to add.
        """
        if name not in self.tensor_data:
            self.tensor_data[name] = []
        self.tensor_data[name].append(images.detach().cpu())

    def add_value(self, name, values):
        """
            Adds value data to the trajectory.

            Parameters:
                name (str): Name of the value data.
                values (any): Value to add.
        """
        if name not in self.value_data:
            self.value_data[name] = []
        self.value_data[name].append(values)

    def compile(self):
        """
            Compiles the recorded data into tensors.

            Returns:
                Trajectory: The compiled trajectory object.
        """
        if not self._compile:
            self._compile = True
            for name in self.tensor_data.keys():
                self.tensor_data[name] = torch.stack(self.tensor_data[name], dim=0)
            for name in self.value_data.keys():
                self.value_data[name] = torch.tensor(self.value_data[name])
        return self

    @classmethod
    def merge(cls, trajs):
        """
            Merge a list of compiled trajectories from different batches

            Returns:
                Trajectory: The merged and compiled trajectory object.
        """
        merged_traj = cls()
        for name in trajs[0].tensor_data.keys():
            merged_traj.tensor_data[name] = torch.cat([traj.tensor_data[name] for traj in trajs], dim=1)
        for name in trajs[0].value_data.keys():
            merged_traj.value_data[name] = trajs[0].value_data[name]
        return merged_traj


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
        # print('data range [x]', torch.min(x), torch.max(x), '[obs]', torch.min(measurement), torch.max(measurement))
        # print('data range [x0hat]', torch.min(x0hat), torch.max(x0hat), '[xt]', torch.min(xt), torch.max(xt))
        # print('error:', ((operator(x) - measurement) **2).flatten(1).sum(-1))
        data_fitting_grad, data_fitting_loss = operator.gradient(x, measurement, return_loss=True)
        data_term = -data_fitting_grad / self.tau ** 2
        xt_term = (xt - x) / sigma ** 2
        prior_term = self.get_prior_score(x, x0hat, xt, model, sigma)
        return data_term + xt_term + prior_term, data_fitting_loss

    def get_prior_score(self, x, x0hat, xt, model, sigma):
        if self.prior_solver == 'score-min' or self.prior_solver == 'score-t' or self.prior_solver == 'gaussian':
            prior_score = self.prior_score
        elif self.prior_solver == 'exact':
            prior_score = model.score(x, torch.tensor(self.prior_sigma_min).to(x.device)).detach()
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

            # print('fitting loss:', '{:.4f}'.format(fitting_loss.sqrt().item()))
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

    def optimize(self, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
        lr = self.get_lr(ratio)
        x = x0hat.clone().detach()
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        for _ in pbar:
            data_fitting_grad, data_fitting_loss = operator.gradient(x, measurement, return_loss=True)
            # print('fitting loss:', '{:.4f}'.format(data_fitting_loss.sqrt().item()))
            reg_grad = (x - x0hat.detach()) / sigma ** 2
            # gradient descent
            x = x - lr * (data_fitting_grad + reg_grad)
            if record:
                self._record(x, data_fitting_grad, data_fitting_loss)
        # print(data_fitting_loss)
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

    def summary(self):
        print('+' * 50)
        print('MCMC Sampler Summary')
        print('+' * 50)
        print(f"Prior Solver    : {self.prior_solver}")
        print(f"MCMC Algorithm  : {self.mc_algo}")
        print(f"Num Steps       : {self.num_steps}")
        print(f"Learning Rate   : {self.lr}")
        print(f"Tau             : {self.tau}")
        print(f"LR Min Ratio    : {self.lr_min_ratio}")
        print('+' * 50)


@register_sampler('latent_daps')
class LatentDAPS(nn.Module):
    """
        Implementation of latent decoupled annealing posterior sampling.
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config, latent_wrapper=True, batch_consistency=False):
        """
            Initializes the DAPS sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
                mcmc_sampler_config (dict): Configuration for MCMC sampler.
        """
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
                                                                             diffusion_scheduler_config)
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.mcmc_sampler = MCMCSampler(**mcmc_sampler_config)
        self.batch_consistency = batch_consistency
        self.latent_wrapper = latent_wrapper
    
    def get_randn_like(self, ref):
        # ref: (B, T, C, H, W)
        if self.batch_consistency:
            noise = torch.randn_like(ref[:, 0])
            noise = noise.unsqueeze(1).expand_as(ref)
        else:
            noise = torch.randn_like(ref)
        return noise

    def sample(self, model, z_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        if self.latent_wrapper:
            wrapped_operator = LatentWrapper(operator, model)
        else: 
            wrapped_operator = operator

        zt = z_start
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                pfode = DiffusionPFODE(model, diffusion_scheduler, 'euler')
                z0hat = pfode.sample(zt)
                x0hat = model.decode(z0hat)

            # 2. langevin dynamics
            if self.latent_wrapper:
                z0y = self.mcmc_sampler.sample(zt, model, z0hat, wrapped_operator, measurement, sigma, step / self.annealing_scheduler.num_steps)
                with torch.no_grad():
                    x0y = model.decode(z0y)
            else:
                x0y = self.mcmc_sampler.optimize(x0hat, operator, measurement, sigma, step / self.annealing_scheduler.num_steps)
                z0y = model.encode(x0y)

            # 3. forward diffusion
            if step != self.annealing_scheduler.num_steps - 1:
                zt = z0y + self.get_randn_like(z0y) * self.annealing_scheduler.sigma_steps[step + 1]
            else:
                zt = z0y

            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                xt = model.decode(zt)
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)
        xt = model.decode(zt)
        return xt

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """
            Checks and updates the configurations for the schedulers.
        """
        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        return annealing_scheduler_config, diffusion_scheduler_config

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = self.get_randn_like(ref) * self.annealing_scheduler.sigma_max
        return x_start

    def summary(self):
        print('Annealing Scheduler:')
        self.annealing_scheduler.summary()
        print('MCMC Sampler:')
        self.mcmc_sampler.summary()
