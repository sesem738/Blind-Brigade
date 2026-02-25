"""ActorCriticRecurrentRayCast — MLP raycaster encoder + GRU memory + MLP head.

Architecture (per actor/critic):

    raycaster obs groups  →  MLP encoder  ──────────────────────────┐
    other 1D obs groups   →  normalizer   →  cat( enc | other_obs ) →  Memory (GRU/LSTM)  →  MLP
                                                                                          ↓
                                                                                  actions / value

Key design choices
------------------
* Raycaster observations (e.g. ray_caster_lidar distances) are expected to already
  be normalised to [0, 1] by the observation function (divides by max_distance).
  A dedicated MLP encoder maps them to a compact embedding each step.
* Other 1D observations (pose command, velocities, last action) have heterogeneous
  units; optional EmpiricalNormalization is available via actor/critic_obs_normalization.
* The embedding and normalised other-obs are concatenated and fed to the RNN.
* The RNN captures temporal context over the combined feature vector.
* `is_recurrent = True` → PPO uses recurrent_mini_batch_generator.

Shape contract
--------------
Inference (no masks):
    raycaster obs:  (B, R)    →  MLP encoder  →  (B, E)
    other obs:      (B, F)    →  normaliser   →  (B, F)
    cat:            (B, E+F)  →  Memory.unsqueeze(0) → squeeze(0) → (B, H)
    MLP:            (B, H)    →  (B, A) or (B, 1)

Training (masks provided, recurrent_mini_batch_generator):
    raycaster obs:  (T, N, R)    →  MLP encoder  →  (T, N, E)
    other obs:      (T, N, F)    →  normaliser   →  (T, N, F)
    cat:            (T, N, E+F)  →  Memory (batch mode)
                                 →  unpad_trajectories  →  (T, N, H)
    MLP:            (T, N, H)    →  (T, N, A) or (T, N, 1)

Note: Unlike CNN, the MLP encoder is shape-agnostic (operates on the last
dimension), so no reshape is needed for the training path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, Memory
from rsl_rl.networks.memory import HiddenState


class ActorCriticRecurrentRayCast(nn.Module):
    """Recurrent actor-critic with MLP encoding of raycaster observations."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_raycast_groups: list[str],
        critic_raycast_groups: list[str],
        raycast_embed_dim: int = 64,
        actor_raycast_encoder_hidden_dims: list[int] = [128, 64],
        critic_raycast_encoder_hidden_dims: list[int] = [128, 64],
        actor_raycast_normalization: bool = False,
        critic_raycast_normalization: bool = False,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "ActorCriticRecurrentRayCast.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.obs_groups = obs_groups

        # ── Classify actor observation groups ──────────────────────────────────
        actor_raycast_set = set(actor_raycast_groups)
        self.actor_raycast_groups: list[str] = []
        self.actor_other_groups: list[str] = []
        num_actor_raycast_obs = 0
        num_actor_other_obs = 0

        for group in obs_groups["policy"]:
            shape = obs[group].shape
            assert len(shape) == 2, (
                f"ActorCriticRecurrentRayCast only supports 1D (B, F) observations. "
                f"Group '{group}' has shape {shape}."
            )
            if group in actor_raycast_set:
                self.actor_raycast_groups.append(group)
                num_actor_raycast_obs += shape[-1]
            else:
                self.actor_other_groups.append(group)
                num_actor_other_obs += shape[-1]

        missing = actor_raycast_set - set(self.actor_raycast_groups)
        assert not missing, (
            f"actor_raycast_groups {missing} not found in obs_groups['policy']."
        )
        assert num_actor_raycast_obs > 0, "actor_raycast_groups must not be empty."

        # ── Classify critic observation groups ─────────────────────────────────
        critic_raycast_set = set(critic_raycast_groups)
        self.critic_raycast_groups: list[str] = []
        self.critic_other_groups: list[str] = []
        num_critic_raycast_obs = 0
        num_critic_other_obs = 0

        for group in obs_groups["critic"]:
            shape = obs[group].shape
            assert len(shape) == 2, (
                f"ActorCriticRecurrentRayCast only supports 1D (B, F) observations. "
                f"Group '{group}' has shape {shape}."
            )
            if group in critic_raycast_set:
                self.critic_raycast_groups.append(group)
                num_critic_raycast_obs += shape[-1]
            else:
                self.critic_other_groups.append(group)
                num_critic_other_obs += shape[-1]

        missing = critic_raycast_set - set(self.critic_raycast_groups)
        assert not missing, (
            f"critic_raycast_groups {missing} not found in obs_groups['critic']."
        )
        assert num_critic_raycast_obs > 0, "critic_raycast_groups must not be empty."

        # ── Actor raycaster normaliser (optional) ───────────────────────────────
        self.actor_raycast_normalization = actor_raycast_normalization
        self.actor_raycast_normalizer = (
            EmpiricalNormalization(num_actor_raycast_obs) if actor_raycast_normalization else nn.Identity()
        )

        # ── Actor raycaster MLP encoder ─────────────────────────────────────────
        self.actor_raycast_encoder = MLP(
            num_actor_raycast_obs,
            raycast_embed_dim,
            actor_raycast_encoder_hidden_dims,
            activation,
        )
        print(f"Actor raycaster encoder: {self.actor_raycast_encoder}")

        # ── Actor other-obs normaliser ──────────────────────────────────────────
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_other_obs) if actor_obs_normalization else nn.Identity()
        ) if num_actor_other_obs > 0 else None

        # ── Actor Memory (GRU/LSTM) ─────────────────────────────────────────────
        actor_rnn_input_dim = raycast_embed_dim + num_actor_other_obs
        self.memory_a = Memory(actor_rnn_input_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
        print(f"Actor RNN: {self.memory_a.rnn}")

        # ── Actor MLP head ──────────────────────────────────────────────────────
        self.state_dependent_std = state_dependent_std
        if state_dependent_std:
            self.actor = MLP(rnn_hidden_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(rnn_hidden_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # ── Critic raycaster normaliser (optional) ──────────────────────────────
        self.critic_raycast_normalization = critic_raycast_normalization
        self.critic_raycast_normalizer = (
            EmpiricalNormalization(num_critic_raycast_obs) if critic_raycast_normalization else nn.Identity()
        )

        # ── Critic raycaster MLP encoder ────────────────────────────────────────
        self.critic_raycast_encoder = MLP(
            num_critic_raycast_obs,
            raycast_embed_dim,
            critic_raycast_encoder_hidden_dims,
            activation,
        )
        print(f"Critic raycaster encoder: {self.critic_raycast_encoder}")

        # ── Critic other-obs normaliser ─────────────────────────────────────────
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_other_obs) if critic_obs_normalization else nn.Identity()
        ) if num_critic_other_obs > 0 else None

        # ── Critic Memory (GRU/LSTM) ────────────────────────────────────────────
        critic_rnn_input_dim = raycast_embed_dim + num_critic_other_obs
        self.memory_c = Memory(critic_rnn_input_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
        print(f"Critic RNN: {self.memory_c.rnn}")

        # ── Critic MLP head ─────────────────────────────────────────────────────
        self.critic = MLP(rnn_hidden_dim, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # ── Action noise ────────────────────────────────────────────────────────
        self.noise_std_type = noise_std_type
        if state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:],
                    torch.log(torch.tensor(init_noise_std + 1e-7)),
                )
            else:
                raise ValueError(f"Unknown noise_std_type '{noise_std_type}'. Use 'scalar' or 'log'.")
        else:
            if noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown noise_std_type '{noise_std_type}'. Use 'scalar' or 'log'.")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # ── Properties ──────────────────────────────────────────────────────────────

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    # ── Encoding helpers ─────────────────────────────────────────────────────────

    def _encode_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return the GRU input vector for the actor.

        Concatenates MLP-encoded raycaster obs with (optionally normalised)
        other 1D obs. Works for both inference (B, F) and training (T, N, F)
        shapes without any reshaping — MLP linear layers operate on the last dim.
        """
        raycast = torch.cat([obs[g] for g in self.actor_raycast_groups], dim=-1)
        raycast = self.actor_raycast_normalizer(raycast)
        raycast_enc = self.actor_raycast_encoder(raycast)

        if not self.actor_other_groups:
            return raycast_enc

        other = torch.cat([obs[g] for g in self.actor_other_groups], dim=-1)
        if self.actor_obs_normalizer is not None:
            other = self.actor_obs_normalizer(other)
        return torch.cat([raycast_enc, other], dim=-1)

    def _encode_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return the GRU input vector for the critic."""
        raycast = torch.cat([obs[g] for g in self.critic_raycast_groups], dim=-1)
        raycast = self.critic_raycast_normalizer(raycast)
        raycast_enc = self.critic_raycast_encoder(raycast)

        if not self.critic_other_groups:
            return raycast_enc

        other = torch.cat([obs[g] for g in self.critic_other_groups], dim=-1)
        if self.critic_obs_normalizer is not None:
            other = self.critic_obs_normalizer(other)
        return torch.cat([raycast_enc, other], dim=-1)

    # ── Distribution ─────────────────────────────────────────────────────────────

    def _update_distribution(self, rnn_out: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(rnn_out)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            else:  # log
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
        else:
            mean = self.actor(rnn_out)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            else:  # log
                std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    # ── Core interface (matches ActorCriticRecurrent contract) ───────────────────

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.memory_a.hidden_state, self.memory_c.hidden_state

    def act(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        features = self._encode_actor_obs(obs)
        rnn_out = self.memory_a(features, masks, hidden_state).squeeze(0)
        self._update_distribution(rnn_out)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        features = self._encode_actor_obs(obs)
        rnn_out = self.memory_a(features).squeeze(0)
        if self.state_dependent_std:
            return self.actor(rnn_out)[..., 0, :]
        return self.actor(rnn_out)

    def evaluate(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        features = self._encode_critic_obs(obs)
        rnn_out = self.memory_c(features, masks, hidden_state).squeeze(0)
        return self.critic(rnn_out)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update running statistics for all normalisers."""
        if self.actor_raycast_normalization and self.actor_raycast_groups:
            raycast = torch.cat([obs[g] for g in self.actor_raycast_groups], dim=-1)
            self.actor_raycast_normalizer.update(raycast)
        if self.actor_obs_normalization and self.actor_other_groups:
            other = torch.cat([obs[g] for g in self.actor_other_groups], dim=-1)
            self.actor_obs_normalizer.update(other)
        if self.critic_raycast_normalization and self.critic_raycast_groups:
            raycast = torch.cat([obs[g] for g in self.critic_raycast_groups], dim=-1)
            self.critic_raycast_normalizer.update(raycast)
        if self.critic_obs_normalization and self.critic_other_groups:
            other = torch.cat([obs[g] for g in self.critic_other_groups], dim=-1)
            self.critic_obs_normalizer.update(other)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
