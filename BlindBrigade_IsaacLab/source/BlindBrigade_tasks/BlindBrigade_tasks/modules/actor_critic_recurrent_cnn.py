"""ActorCriticRecurrentCNN — CNN encoder + RNN memory + MLP head.

Architecture (per actor/critic):

    4D obs groups  →  CNN  ──────────────────────────────┐
    1D obs groups  →  normalizer  →  cat( 1D | CNN enc ) →  Memory (LSTM/GRU)  →  MLP
                                                                              ↓
                                                                          actions / value

Key design choices
------------------
* CNN encodes each visual group independently each step (no temporal conv).
* CNN output + flat proprioceptive obs are concatenated → fed to the RNN.
* The RNN captures temporal context over the combined feature vector.
* Observation normalisation applies only to the 1D branch; CNN inputs are
  assumed to already be normalised (e.g. depth values in [0, 1]).
* `is_recurrent = True` → PPO uses recurrent_mini_batch_generator and saves
  hidden states every step; hidden states are reset on episode boundaries.

Shape contract
--------------
Inference (no masks):
    1D obs:  (B, F)        →  normaliser  →  (B, F)
    4D obs:  (B, C, H, W)  →  CNN         →  (B, E)
    cat:     (B, F + E)    →  Memory.unsqueeze(0) → squeeze(0) → (B, H)
    MLP:     (B, H)        →  (B, A) or (B, 1)

Training (masks provided, recurrent_mini_batch_generator):
    1D obs:  (T, N, F)           →  normaliser  →  (T, N, F)
    4D obs:  (T, N, C, H, W)     →  reshape (T*N, C, H, W)  →  CNN
                                 →  reshape (T, N, E)
    cat:     (T, N, F + E)       →  Memory (batch mode)
                                 →  unpad_trajectories  →  (T, N, H)
    MLP:     (T, N, H)           →  (T, N, A) or (T, N, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import CNN, MLP, EmpiricalNormalization, Memory
from rsl_rl.networks.memory import HiddenState


class ActorCriticRecurrentCNN(nn.Module):
    """Recurrent actor-critic with per-step CNN encoding of visual observations."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        actor_cnn_cfg: dict[str, dict] | dict | None = None,
        critic_cnn_cfg: dict[str, dict] | dict | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(
                "ActorCriticRecurrentCNN.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.obs_groups = obs_groups

        # ── Classify actor observation groups ──────────────────────────────
        num_actor_obs_1d = 0
        self.actor_obs_groups_1d: list[str] = []
        actor_in_dims_2d: list[tuple[int, int]] = []
        actor_in_channels_2d: list[int] = []
        self.actor_obs_groups_2d: list[str] = []

        for group in obs_groups["policy"]:
            shape = obs[group].shape
            if len(shape) == 4:  # (B, C, H, W)
                self.actor_obs_groups_2d.append(group)
                actor_in_dims_2d.append(shape[2:4])
                actor_in_channels_2d.append(shape[1])
            elif len(shape) == 2:  # (B, F)
                self.actor_obs_groups_1d.append(group)
                num_actor_obs_1d += shape[-1]
            else:
                raise ValueError(
                    f"Observation group '{group}' has unsupported shape {shape}. "
                    "Only 2D (B, F) and 4D (B, C, H, W) are supported."
                )

        # ── Classify critic observation groups ─────────────────────────────
        num_critic_obs_1d = 0
        self.critic_obs_groups_1d: list[str] = []
        critic_in_dims_2d: list[tuple[int, int]] = []
        critic_in_channels_2d: list[int] = []
        self.critic_obs_groups_2d: list[str] = []

        for group in obs_groups["critic"]:
            shape = obs[group].shape
            if len(shape) == 4:
                self.critic_obs_groups_2d.append(group)
                critic_in_dims_2d.append(shape[2:4])
                critic_in_channels_2d.append(shape[1])
            elif len(shape) == 2:
                self.critic_obs_groups_1d.append(group)
                num_critic_obs_1d += shape[-1]
            else:
                raise ValueError(
                    f"Observation group '{group}' has unsupported shape {shape}. "
                    "Only 2D (B, F) and 4D (B, C, H, W) are supported."
                )

        # ── Actor CNN ───────────────────────────────────────────────────────
        actor_cnn_encoding_dim = 0
        if self.actor_obs_groups_2d:
            assert actor_cnn_cfg is not None, (
                "actor_cnn_cfg is required when policy obs_groups contain 4D groups."
            )
            # If a single flat config dict is given, broadcast to all 2D groups
            if not all(isinstance(v, dict) for v in actor_cnn_cfg.values()):
                actor_cnn_cfg = {g: actor_cnn_cfg for g in self.actor_obs_groups_2d}
            assert len(actor_cnn_cfg) == len(self.actor_obs_groups_2d), (
                "Number of CNN configs must match number of 4D actor observation groups."
            )
            self.actor_cnns = nn.ModuleDict()
            for idx, group in enumerate(self.actor_obs_groups_2d):
                cnn = CNN(
                    input_dim=actor_in_dims_2d[idx],
                    input_channels=actor_in_channels_2d[idx],
                    **actor_cnn_cfg[group],
                )
                assert cnn.output_channels is None, (
                    f"Actor CNN for '{group}' must produce a flattened output. "
                    "Set global_pool='avg'/'max' or flatten=True in the CNN config."
                )
                self.actor_cnns[group] = cnn
                actor_cnn_encoding_dim += int(cnn.output_dim)
                print(f"Actor CNN for {group}: {cnn}")
        else:
            self.actor_cnns = None

        # ── Actor obs normaliser (1D only) ──────────────────────────────────
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs_1d) if actor_obs_normalization else nn.Identity()
        )

        # ── Actor Memory (RNN) ──────────────────────────────────────────────
        actor_rnn_input_dim = num_actor_obs_1d + actor_cnn_encoding_dim
        self.memory_a = Memory(actor_rnn_input_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
        print(f"Actor RNN: {self.memory_a.rnn}")

        # ── Actor MLP ───────────────────────────────────────────────────────
        self.state_dependent_std = state_dependent_std
        if state_dependent_std:
            self.actor = MLP(rnn_hidden_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(rnn_hidden_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # ── Critic CNN ──────────────────────────────────────────────────────
        critic_cnn_encoding_dim = 0
        if self.critic_obs_groups_2d:
            assert critic_cnn_cfg is not None, (
                "critic_cnn_cfg is required when critic obs_groups contain 4D groups."
            )
            if not all(isinstance(v, dict) for v in critic_cnn_cfg.values()):
                critic_cnn_cfg = {g: critic_cnn_cfg for g in self.critic_obs_groups_2d}
            assert len(critic_cnn_cfg) == len(self.critic_obs_groups_2d), (
                "Number of CNN configs must match number of 4D critic observation groups."
            )
            self.critic_cnns = nn.ModuleDict()
            for idx, group in enumerate(self.critic_obs_groups_2d):
                cnn = CNN(
                    input_dim=critic_in_dims_2d[idx],
                    input_channels=critic_in_channels_2d[idx],
                    **critic_cnn_cfg[group],
                )
                assert cnn.output_channels is None, (
                    f"Critic CNN for '{group}' must produce a flattened output. "
                    "Set global_pool='avg'/'max' or flatten=True in the CNN config."
                )
                self.critic_cnns[group] = cnn
                critic_cnn_encoding_dim += int(cnn.output_dim)
                print(f"Critic CNN for {group}: {cnn}")
        else:
            self.critic_cnns = None

        # ── Critic obs normaliser (1D only) ─────────────────────────────────
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs_1d) if critic_obs_normalization else nn.Identity()
        )

        # ── Critic Memory (RNN) ──────────────────────────────────────────────
        critic_rnn_input_dim = num_critic_obs_1d + critic_cnn_encoding_dim
        self.memory_c = Memory(critic_rnn_input_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
        print(f"Critic RNN: {self.memory_c.rnn}")

        # ── Critic MLP ───────────────────────────────────────────────────────
        self.critic = MLP(rnn_hidden_dim, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # ── Action noise ─────────────────────────────────────────────────────
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

    # ── Properties ────────────────────────────────────────────────────────────

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

    # ── Encoding helpers ──────────────────────────────────────────────────────

    def _apply_cnn(self, cnn: CNN, x: torch.Tensor) -> torch.Tensor:
        """Apply CNN to either (B, C, H, W) or (T, N, C, H, W) input."""
        if x.dim() == 4:  # inference: (B, C, H, W)
            return cnn(x)
        # training batch: (T, N, C, H, W)
        T, N = x.shape[:2]
        enc = cnn(x.reshape(T * N, *x.shape[2:]))
        return enc.reshape(T, N, -1)

    def _encode_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return the RNN input vector for the actor.

        Concatenates normalised 1D obs with CNN encodings of visual obs.
        Works for both inference (B, ...) and training (T, N, ...) shapes.
        """
        obs_1d = torch.cat([obs[g] for g in self.actor_obs_groups_1d], dim=-1)
        obs_1d = self.actor_obs_normalizer(obs_1d)
        if self.actor_cnns is None:
            return obs_1d
        cnn_encs = [self._apply_cnn(self.actor_cnns[g], obs[g]) for g in self.actor_obs_groups_2d]
        return torch.cat([obs_1d, *cnn_encs], dim=-1)

    def _encode_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return the RNN input vector for the critic."""
        obs_1d = torch.cat([obs[g] for g in self.critic_obs_groups_1d], dim=-1)
        obs_1d = self.critic_obs_normalizer(obs_1d)
        if self.critic_cnns is None:
            return obs_1d
        cnn_encs = [self._apply_cnn(self.critic_cnns[g], obs[g]) for g in self.critic_obs_groups_2d]
        return torch.cat([obs_1d, *cnn_encs], dim=-1)

    # ── Distribution ──────────────────────────────────────────────────────────

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

    # ── Core interface (matches ActorCriticRecurrent contract) ─────────────────

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
        """Update running statistics for 1D observation normalisers only."""
        if self.actor_obs_normalization and self.actor_obs_groups_1d:
            actor_1d = torch.cat([obs[g] for g in self.actor_obs_groups_1d], dim=-1)
            self.actor_obs_normalizer.update(actor_1d)
        if self.critic_obs_normalization and self.critic_obs_groups_1d:
            critic_1d = torch.cat([obs[g] for g in self.critic_obs_groups_1d], dim=-1)
            self.critic_obs_normalizer.update(critic_1d)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
