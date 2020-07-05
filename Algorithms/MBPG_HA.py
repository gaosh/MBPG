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
    o_lh_sum = o_lh.sum(dim=1)
    c_lh_sum = c_lh.sum(dim=1)
    weigt = o_lh_sum - c_lh_sum
    weigt = torch.exp(weigt)
    weigt[weigt>=th] = th

    lth = 1- (th - 1)
    weigt[weigt<lth]=lth

    #print(weigt.max())
    return weigt.unsqueeze(1).expand_as(c_lh)

class MBPG_HA(BatchPolopt):

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

            batch_size=50000,
            im_flag=True,
            grad_factor = 3,

            max_path_length=500,
            num_train_per_epoch=1,
            discount=0.99,
            gae_lambda=1,
            center_adv=True,
            positive_adv=False,
            policy_ent_coeff=0.0,
            use_softplus_entropy=False,
            stop_entropy_gradient=False,
            decay_learning_rate=True,
            th = 1.8,
            g_max=0.06,
            delta = 1e-7,
            entropy_method='no_entropy',
            log_dir='./log'
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

        self.delta =delta

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)
        self._episode_reward_mean = collections.deque(maxlen=100)

        self.first_flag = True

        self.lr = policy_lr
        self.batch_size = batch_size
        self.n_timestep = n_timestep

        self.w = w
        self.c = c
        self.th = th
        self.storm_dict = {}
        self.im_flag = im_flag
        self.grad_factor = grad_factor
        super().__init__(policy=policy,
                         baseline=baseline,
                         discount=discount,
                         max_path_length=max_path_length,
                         n_samples=num_train_per_epoch)

        self._policy = copy.deepcopy(self.policy)
        self._old_policy = copy.deepcopy(self.policy)
        self._neg_policy = copy.deepcopy(self.policy)
        self._pos_policy = copy.deepcopy(self.policy)
        self._mix_policy = copy.deepcopy(self.policy)
        #self._backup_policy = copy.deepcopy(self.policy)

        #self._optimizer = STORM(self._policy.parameters())
        self.g_max = g_max

        self.decay_learning_rate = decay_learning_rate

        im_string = ''
        if im_flag:
            im_string = 'IM'
        self.log_dir = log_dir + '/MBPG_HA%s_%s_bs_%d_lr_%f_delta_%s' % (im_string, env_name,batch_size, self.lr,str(self.delta))
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

    def generate_mix_policy(self):
        a = np.random.uniform(0.0, 1.0)
        mix = a * self._policy.get_param_values() + (1 - a) * self._old_policy.get_param_values()
        self._mix_policy.set_param_values(mix)

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

            #d_p = grads + (1 - self.a.item()) * (d_p_buffer - o_g)
            #grads = torch.clamp(grads, -g_max, g_max)
            d_p = self.a.item()*grads + (1-self.a.item()) * (d_p_buffer+o_g)
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
        eta_t = lr / ((self.w + self.grad_factor*sum_gradnorm) ** (1 / 3))
        # print(eta_t)
        # if d_p.norm() > self.g_max:
        # #     g_max = G_p.item()
        # # else:
        # #     g_max = self.g_max
        #     d_p = d_p*(self.g_max/d_p.norm())

        d_p = torch.clamp(d_p, -g_max, g_max)

        params = self._policy.get_param_values()

        params = params + lr_mutipler*eta_t.item() * d_p

        self._policy.set_param_values(params)

        self.a = torch.min(torch.Tensor([1.0]), self.c * (eta_t ** 2))
        self.a = torch.max(self.a, torch.Tensor([0.3]))
        #print(self.a.item())
        # print(G_p.item())
        # print(self.c * (eta_t ** 2))
        # print(eta_t.item())
        #print(d_p.norm().item())
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
            if j ==0:
                self.policy = self._policy
                paths = runner.obtain_samples(j)
                sample_data = self.process_samples(j, paths)
            else:
                self.generate_mix_policy()

                self.policy = self._mix_policy

                paths = runner.obtain_samples(j)
                sample_data = self.process_samples(j, paths)
            j += sum([len(path["rewards"]) for path in paths])
            self.storm_optimization(j, sample_data, paths)
            # if j!=0:
            #     self.policy = self._policy
            #
            #     test_paths = runner.obtain_samples(j)
            #     #test_data = self.process_samples(j, paths)
            #     avg_returns = np.mean([sum(p["rewards"]) for p in test_paths])
            # else:
            avg_returns = np.mean([sum(p["rewards"]) for p in paths])
            print("timesteps: " + str(j) + " average return: ", avg_returns)
            file.write("%d %.4f\n" % (j, avg_returns))


        file.close()
        runner.sampler.shutdown_worker()
        return last_return
    def train_once(self, itr, paths):
        pass

    def get_advantages(self, paths, rewards, valids):

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


    def _compute_loss(self, paths, valids, obs, actions, rewards, policy):
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

        objective, lh = self._compute_objective(advantages, valids, obs, actions,
                                            rewards,policy)

        if self.im_flag:
            with torch.no_grad():
                _, mix_lh = self._compute_objective(advantages, valids, obs, actions,
                                                rewards,self._mix_policy)
            correct_w = compute_weights(lh.detach(), mix_lh.detach(), th=self.th)
            objective = correct_w*objective
        #print(len(paths))
        #return valid_objectives.sum()/len(paths)

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

        # This prevents entropy from becoming negativ7e for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _compute_objective(self, advantages, valids, obs, actions, rewards, policy):
        log_likelihoods = policy.log_likelihood(obs, actions)
        return log_likelihoods * advantages, log_likelihoods

    def _get_baselines(self, path):
        if hasattr(self.baseline, 'predict_n'):
            return torch.Tensor(self.baseline.predict_n(path))
        return torch.Tensor(self.baseline.predict(path))

    def storm_optimization(self, itr, sample_data, paths):
        valids, obs, actions, rewards = sample_data
        loss = self._compute_loss(paths, valids, obs, actions, rewards, self._policy)
        loss.backward()
        grad = self._policy.get_grads()
        # print(len(obs))
        # grad = grad/len(obs)
        if itr == 0:
            old_grad = torch.zeros_like(grad)
        else:
            advantages = self.get_advantages(paths, rewards, valids)
            eps = self.delta
            d_vector = self._policy.get_param_values() - self._old_policy.get_param_values()
            pos_params = self._mix_policy.get_param_values() + d_vector * eps
            neg_params = self._mix_policy.get_param_values() - d_vector * eps
            self._pos_policy.set_param_values(pos_params)
            self._neg_policy.set_param_values(neg_params)
            # first component: dot(likelihood, theta_t - theta_t-1) * policy gradient

            g_mix = self.get_gradinets(obs, actions, self._mix_policy, valids,advantages)
            g_lh = self.get_gradinets(obs, actions, self._mix_policy, valids,None)
            inner_product = torch.dot(g_lh, d_vector)
            fst = inner_product * g_mix

            # second component: dot(Hessian, theta_t - theta_t-1)

            g_pos = self.get_gradinets(obs, actions, self._pos_policy,valids, advantages)
            g_neg = self.get_gradinets(obs, actions, self._neg_policy,valids, advantages)

            hv = (g_pos - g_neg) / (2 * eps)
            old_grad = fst + hv

        self._old_policy.set_param_values(self._policy.get_param_values())

        self.storm_step(grad, old_grad, self.lr)

    #def outer_optimization

    def get_gradinets(self, sub_obs, sub_actions, policy,vailds, sub_adv):
        loss= self._compact_objective(sub_obs, sub_actions, policy,vailds ,sub_adv)
        loss = loss.mean()
        loss.backward()
        grad = policy.get_grads()
        return grad

    def _compact_objective(self, obs, actions, policy, valids, advantages=None):
        #print(advantages)
        # obs = torch.from_numpy(obs).float()
        # actions =  torch.from_numpy(actions).float()
        #print(actions.size())
        log_likelihoods = policy.log_likelihood(obs, actions)


        # valid_objectives = filter_valids(objective, valids)
        # valid_objectives = torch.cat(valid_objectives)

        if advantages is None:
            valid_ll = filter_valids(log_likelihoods, valids)
            valid_ll = torch.cat(valid_ll)
            return valid_ll
        else:
            obj = log_likelihoods * advantages
            valid_objectives = filter_valids(obj, valids)
            valid_objectives = torch.cat(valid_objectives)

            return valid_objectives



    def sample_paths(self, traj_num, sample_policy):
        paths = []
        # Sample Trajectories
        for _ in range(traj_num):
            observations = []
            actions = []
            rewards = []

            observation = self.env.reset()

            for _ in range(self.max_path_length):
                # policy.get_action() returns a pair of values. The second
                # one returns a dictionary, whose values contains
                # sufficient statistics for the action distribution. It
                # should at least contain entries that would be returned
                # by calling policy.dist_info(), which is the non-symbolic
                # analog of policy.dist_info_sym(). Storing these
                # statistics is useful, e.g., when forming importance
                # sampling ratios. In our case it is not needed.
                action, _ = sample_policy.get_action(torch.Tensor(observation))
                # Recall that the last entry of the tuple stores diagnostic
                # information about the environment. In our case it is not needed.
                next_observation, reward, terminal, _ = self.env.step(action)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                observation = next_observation
                if terminal:
                    # Finish rollout if terminal state reached
                    break

            # We need to compute the empirical return for each time step along the
            # trajectory
            path = dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
            )
            path_baseline = np.append(self.baseline.predict(path), 0)

            deltas = path["rewards"] + \
                     self.discount * path_baseline[1:] - \
                     path_baseline[:-1]
            advantages = discount_cumsum(
                deltas, self.discount * self._gae_lambda
            )
            # added for correction
            discount_array = self.discount ** np.arange(len(path["rewards"]))
            advantages = advantages * discount_array
            #print(path['rewards'])
            path["returns"] = discount_cumsum(
                path["rewards"], self.discount
            )

            '''
            returns = special.discount_cumsum(
                path["rewards"], self.discount)
            path['returns'] = returns * discount_array
            '''
            if self._center_adv:
                advantages = (advantages - np.mean(advantages)) / (
                    np.std(advantages) + 1e-8)

            path["advantages"] = torch.from_numpy(advantages)

            paths.append(path)
        return paths