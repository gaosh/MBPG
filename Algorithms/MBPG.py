import collections
import copy

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage.misc import tensor_utils
from garage.np.algos import BatchPolopt
from .special import *
#from garage.torch.algos._utils import (_Default, compute_advantages, filter_valids,
 #                               make_optimizer, pad_to_last)
from ._utils import (compute_advantages, filter_valids, make_optimizer, pad_to_last, flatten_batch)
from tensorboardX import SummaryWriter
#from garage.torch.utils import flatten_batch
import time


def normalize_gradient(grad):
    n_grad = grad/grad.norm(p=2)
    #print(n_grad.norm(p=2))
    return n_grad
timestamp = time.time()
timestruct = time.localtime(timestamp)
#exp_time = time.strftime('%Y-%m-%d %H:%M:%S', timestruct)
def compute_weights(o_lh, c_lh, th=1.8):
    #mini_bs = o_lh.size()
    lth = th - 1
    #3lth = max(lth, 0)
    o_lh_sum = o_lh.sum(dim=1)
    c_lh_sum = c_lh.sum(dim=1)
    weigt = o_lh_sum - c_lh_sum
    weigt = torch.exp(weigt)
    weigt[weigt>=th] = th
    #weigt[weigt<lth] = lth
    print(weigt.max())
    return weigt.unsqueeze(1).expand_as(c_lh)

class MBPG_IM(BatchPolopt):

    def __init__(
            self,
            env_spec,
            env,
            policy,
            baseline,
            env_name,
            #optimizer=torch.optim.Adam,
            count = 0,
            policy_lr=1e-2,
            w = 10,
            c = 100,

            #hyperparameters for HAP
            n_timestep=1e6,
            th = 1.8,
            batch_size=50000,
            grad_factor = 10,
            max_path_length=500,
            num_train_per_epoch=1,
            discount=0.99,
            gae_lambda=1,
            center_adv=True,
            positive_adv=False,
            policy_ent_coeff=0.0,
            g_max = 0.05,
            use_softplus_entropy=False,
            stop_entropy_gradient=False,
            decay_learning_rate=True,
            star_version=True,
            entropy_method='no_entropy',
            log_dir='./log',
    ):
        self._env_spec = env_spec
        self.env = env
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._policy_ent_coeff = policy_ent_coeff
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._entropy_method = entropy_method
        self._eps = 1e-8

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)
        self._episode_reward_mean = collections.deque(maxlen=100)

        self.first_flag = True

        self.sv = star_version

        self.lr = policy_lr
        self.batch_size = batch_size
        self.n_timestep = n_timestep

        self.w = w
        self.c = c
        self.th = th
        self.storm_dict = {}
        self.grad_factor = grad_factor
        super().__init__(policy=policy,
                         baseline=baseline,
                         discount=discount,
                         max_path_length=max_path_length,
                         n_samples=num_train_per_epoch)

        self._policy = copy.deepcopy(self.policy)
        self._old_policy = copy.deepcopy(self.policy)
        #self._optimizer = STORM(self._policy.parameters())
        self.g_max = g_max
        self.a = torch.Tensor([0])

        self.decay_learning_rate = decay_learning_rate

        self.log_dir = log_dir + '/MBPG_IM_%s_%s_bs_%d_lr_%f' % (count, env_name,batch_size, self.lr)
        if self.sv:
            self.log_dir = self.log_dir + '_sv'
            self.steps = 0
        self.eta_t = policy_lr / ((self.w ) ** (1 / 3))
        #self.writer = SummaryWriter(log_dir)
        #self.file = open(log_dir + '.txt', 'a')
    @staticmethod
    def _check_entropy_configuration(entropy_method, center_adv,
                                     stop_entropy_gradient, policy_ent_coeff):
        if entropy_method not in ('max', 'regularized', 'no_entropy'):
            raise ValueError('Invalid entropy_method')

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
        if entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')

    def process_samples(self, itr, paths):
        """Process sample data based on the collected paths.
        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths
        Returns:
            dict: Processed sample data, with key
                * average_return: (float)
        """
        for path in paths:
            path['returns'] = tensor_utils.discount_cumsum(
                path['rewards'], self.discount)

        valids = [len(path['actions']) for path in paths]
        obs = torch.stack([
            pad_to_last(path['observations'],
                        total_length=self.max_path_length,
                        axis=0) for path in paths
        ])
        actions = torch.stack([
            pad_to_last(path['actions'],
                        total_length=self.max_path_length,
                        axis=0) for path in paths
        ])
        rewards = torch.stack([
            pad_to_last(path['rewards'], total_length=self.max_path_length)
            for path in paths
        ])

        #print(valids)

        return valids, obs, actions, rewards


    def storm_step(self, grads, old_grads, lr, lr_mutipler=1):
        G_p = grads.norm(p=2)

        if G_p < self.g_max:
            g_max = G_p.item()
        else:
            g_max = self.g_max
        if 'd_p_buffer' not in self.storm_dict:
            d_p_buffer = self.storm_dict['d_p_buffer'] = torch.zeros_like(grads)
        else:
            d_p_buffer = self.storm_dict['d_p_buffer']
            # d_p_buffer.copy_(d_p)
        if self.first_flag:
            d_p = grads
        else:
            o_g = old_grads

            d_p = grads + (1 - self.a.item()) * (d_p_buffer - o_g)
            # d_p = grads +

        self.storm_dict['d_p_buffer'].copy_(d_p)

        if 'sum_gradnorm_buffer' not in self.storm_dict:
            sum_gradnorm = self.storm_dict['sum_gradnorm_buffer'] = torch.zeros(1)
            sum_gradnorm.copy_(G_p.pow(2))
        #    self.params_state['sum_gradnorm_buffer'].copy_(sum_gradnorm)
        else:
            sum_gradnorm = self.storm_dict['sum_gradnorm_buffer']
            sum_gradnorm.add_(G_p.pow(2))
        self.storm_dict['sum_gradnorm_buffer'].copy_(sum_gradnorm)
        if self.sv:
            self.steps += 1
            eta_t = lr/  ((self.w + 0.03*self.steps) ** (1 / 3))
        else:
            eta_t = lr / ((self.w + 5*sum_gradnorm) ** (1 / 3))
            eta_t = eta_t.item()
        # print(eta_t)
        d_p = torch.clamp(d_p, -g_max, g_max)

        params = self._policy.get_param_values()

        params = params + lr_mutipler*eta_t * d_p

        params = params - 1e-4 * params

        self._policy.set_param_values(params)


        self.a = torch.min(torch.Tensor([1.0]), torch.Tensor([self.c * (eta_t ** 2)]))
        self.a = torch.max(self.a, torch.Tensor([0.3]))
        #print(self.a.item())
        # print(G_p.item())
        # print(self.c * (eta_t ** 2))
        # print(eta_t)
        self.eta_t = eta_t
        self.first_flag = False

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.
        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.
        Returns:
            float: The average return in last epoch cycle.
        """
        last_return = None
        runner.sampler.start_worker()


        j = 0
        file = open(self.log_dir + '.txt', 'a')
        while (self.batch_size < self.n_timestep - j):
            # change sample policy
            self.policy = self._policy
            paths = runner.obtain_samples(j)
            sample_data = self.process_samples(j, paths)

            j += sum([len(path["rewards"]) for path in paths])
            avg_returns = np.mean([sum(p["rewards"]) for p in paths])
            print("timesteps: " + str(j) + " average return: ", avg_returns)
            file.write("%d %.4f %.4f\n" % (j, avg_returns, self.eta_t))
            self.storm_optimization(j, sample_data, paths)

        file.close()
        runner.sampler.shutdown_worker()
        return last_return
    def train_once(self, itr, paths):
        pass

    def _compute_loss(self, paths, valids, obs, actions, rewards, policy, current_lh=None):
        #policy_entropies = self._compute_policy_entropy(obs)

        baselines = torch.stack([
            pad_to_last(self._get_baselines(path),
                        total_length=self.max_path_length) for path in paths
        ])


        advantages = compute_advantages(self.discount, self._gae_lambda,
                                        self.max_path_length, baselines,
                                        rewards)



        if self._center_adv:
            means, variances = list(
                zip(*[(valid_adv.mean(), valid_adv.var())
                      for valid_adv in filter_valids(advantages, valids)]))
            advantages = F.batch_norm(advantages.t(),
                                      torch.Tensor(means),
                                      torch.Tensor(variances),
                                      eps=self._eps).t()

        #print(advantages)

        objective, log_likelihoods = self._compute_objective(advantages, valids, obs, actions,
                                            rewards,policy)

        print(len(paths))
        #return valid_objectives.sum()/len(paths)


        if current_lh is None:
            valid_objectives = filter_valids(objective, valids)
            valid_objectives = torch.cat(valid_objectives)
            return valid_objectives.mean(), log_likelihoods.detach()
        else:
            old_lh = log_likelihoods.detach()
            correct_w = compute_weights(old_lh, current_lh, th=self.th)
            # if correct_w.max() > 2.0:
            #     correct_w = torch.zeros_like(correct_w)

            objective = correct_w*objective
            valid_objectives = filter_valids(objective, valids)
            valid_objectives = torch.cat(valid_objectives)


            return valid_objectives.mean()

    def _compute_kl_constraint(self, obs):
        flat_obs = flatten_batch(obs)
        with torch.no_grad():
            old_dist = self._old_policy.forward(flat_obs)

        new_dist = self._policy.forward(flat_obs)

        kl_constraint = torch.distributions.kl.kl_divergence(
            old_dist, new_dist)

        return kl_constraint.mean()

    def _compute_policy_entropy(self, obs):
        policy_entropy = self._policy.entropy(obs)

        if self._stop_entropy_gradient:
            with torch.no_grad():
                policy_entropy = self._policy.entropy(obs)
        else:
            policy_entropy = self._policy.entropy(obs)

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _compute_objective(self, advantages, valids, obs, actions, rewards, policy):
        log_likelihoods = policy.log_likelihood(obs, actions)
        return log_likelihoods * advantages, log_likelihoods.detach()

    def _get_baselines(self, path):
        if hasattr(self.baseline, 'predict_n'):
            return torch.Tensor(self.baseline.predict_n(path))
        return torch.Tensor(self.baseline.predict(path))

    def query_loglikelihood(self, obs, actions, policy):

        log_likelihoods = policy.log_likelihood(obs, actions)
        return log_likelihoods

    def storm_optimization(self, itr, sample_data, paths):
        valids, obs, actions, rewards = sample_data
        loss, current_likelihoods = self._compute_loss(paths, valids, obs, actions, rewards, self._policy)
        loss.backward()
        grad = self._policy.get_grads()

        old_loss = self._compute_loss(paths, valids, obs, actions, rewards, self._old_policy, current_likelihoods)

        old_loss.backward()
        old_grad = self._old_policy.get_grads()

        self._old_policy.set_param_values(self._policy.get_param_values())

        self.storm_step(grad, old_grad, self.lr)

