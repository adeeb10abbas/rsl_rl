import torch
import torch.nn as nn
from torch.distributions import Categorical  # For discrete actions
from rsl_rl.utils import resolve_nn_activation

class ActorCriticDiscrete(nn.Module):
    """ActorCritic modified for a discrete action (token) space."""
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_tokens,  
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        **kwargs,
    ):
        super().__init__()
        activation = resolve_nn_activation(activation)
        
        # Actor network
        actor_layers = []
        mlp_input_dim_a = num_actor_obs
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            out_dim = actor_hidden_dims[layer_index]
            if layer_index < len(actor_hidden_dims) - 1:
                # Hidden layer
                actor_layers.append(nn.Linear(out_dim, actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
            else:
                actor_layers.append(nn.Linear(out_dim, num_tokens))  

        self.actor = nn.Sequential(*actor_layers)

        # Critic network
        critic_layers = []
        mlp_input_dim_c = num_critic_obs
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            out_dim = critic_hidden_dims[layer_index]
            if layer_index < len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(out_dim, critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
            else:
                # Output single value (critic)
                critic_layers.append(nn.Linear(out_dim, 1))

        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Instead of storing noise parameters, we keep a placeholder for discrete distribution
        self.distribution = None

    @property
    def entropy(self):
        return self.distribution.entropy()

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def update_distribution(self, observations):
        """Compute logits for discrete distribution."""
        logits = self.actor(observations)
        # Create Categorical distribution
        self.distribution = Categorical(logits=logits)

    def act(self, observations, **kwargs):
        """Sample a discrete token from the Categorical distribution."""
        self.update_distribution(observations)
        return self.distribution.sample()  # shape: (batch,)

    def act_inference(self, observations):
        """Use argmax (greedy) for inference if desired."""
        logits = self.actor(observations)
        return torch.argmax(logits, dim=-1)  # discrete argmax token

    def get_actions_log_prob(self, actions):
        """Compute log(prob) of the chosen discrete actions."""
        # For discrete, actions is typically shape [batch], each entry is a token index
        return self.distribution.log_prob(actions)

    def evaluate(self, critic_observations, **kwargs):
        """Evaluate critic to get value function estimates."""
        value = self.critic(critic_observations)
        return value
