import abc

import torch
from torch import nn
#from torch.distributions import Categorical
from .Categorical_Distribuation import Categorical
import torch.nn.functional as F
from garage.torch.modules.mlp_module import MLPModule

class CategoricalMLPPolicy(nn.Module):
    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,

                 layer_normalization=False):
        super().__init__()

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        self._input_dim = self._obs_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = self._action_dim
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

        self.module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization)

    @abc.abstractmethod
    def _get_mean_and_log_std(self, inputs):
        pass

    def forward(self, input):
        #print(input)
        input = self.module(torch.Tensor(input))
        input = F.softmax(input, dim=1)

        return Categorical(input)

    def get_action(self, observation):
        """Get a single action given an observation.
        Args:
            observation (torch.Tensor): Observation from the environment.
        Returns:
            tuple:
                * torch.Tensor: Predicted action.
                * dict:
                    * list[float]: Mean of the distribution
                    * list[float]: Standard deviation of logarithmic values of
                        the distribution
        """
        with torch.no_grad():
            observation = observation.unsqueeze(0)
            dist = self.forward(observation)
            return (dist.sample().squeeze(0).numpy(),
                    dict(mean=dist.mean.squeeze(0).numpy(),
                         log_std=(dist.variance**.5).log().squeeze(0).numpy()))

    def get_actions(self, observations):
        """Get actions given observations.
        Args:
            observations (torch.Tensor): Observations from the environment.
        Returns:
            tuple:
                * torch.Tensor: Predicted actions.
                * dict:
                    * list[float]: Mean of the distribution
                    * list[float]: Standard deviation of logarithmic values of
                        the distribution
        """
        with torch.no_grad():
            dist = self.forward(observations)
            return (dist.sample().numpy(),
                    dict(mean=dist.mean.numpy(),
                         log_std=(dist.variance**.5).log().numpy()))
    def log_likelihood(self, observation, action):
        """Compute log likelihood given observations and action.
        Args:
            observation (torch.Tensor): Observation from the environment.
            action (torch.Tensor): Predicted action.
        Returns:
            torch.Tensor: Calculated log likelihood value of the action given
                observation
        """
        dist = self.forward(observation)
        return dist.log_prob(action)

    def get_entropy(self, observation):
        """Get entropy given observations.
        Args:
            observation (torch.Tensor): Observation from the environment.
        Returns:
             torch.Tensor: Calculated entropy values given observation
        """
        dist = self.forward(observation)
        return dist.entropy()

    def reset(self, dones=None):
        """Reset the environment.
        Args:
            dones (numpy.ndarray): Reset values
        """

    @property
    def vectorized(self):
        """Vectorized or not.
        Returns:
            bool: flag for vectorized
        """
        return True

    def set_param_values(self, given_parameters):
        torch.nn.utils.vector_to_parameters(given_parameters, super().parameters())

        # for param_cur, given_param in zip(super().parameters(), given_parameters):
        #     param_cur.data = given_param.data

    def get_param_values(self):

        params = torch.nn.utils.parameters_to_vector(super().parameters())

        return params
    def get_grads(self):
        grads = []
        for param in super().parameters():
            grads.append(param.grad.data.view(-1))

        grads = torch.cat(grads)
        for param in super().parameters():
            param.grad.detach_()
            param.grad.zero_()
        # print(grads)
        return grads