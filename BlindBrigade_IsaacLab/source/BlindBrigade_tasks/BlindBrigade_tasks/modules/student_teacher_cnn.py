"""StudentTeacherCNN — CNN/MLP image encoder + MLP student-teacher distillation.

Architecture:

    Student:
      CNN mode:  image_groups (4D)  →  CNN encoder  →  (B, E)  ─┐
      MLP mode:  image_groups (1D)  →  MLP encoder  →  (B, E)  ─┤
      other 1D obs groups           →  normalizer   →  (B, F)  ─┼→ cat → student MLP → actions
                                                                 ┘
    Teacher:
      teacher obs (1D)  →  normalizer  →  teacher MLP  →  actions

Key design choices
------------------
* The student image encoder can be either a CNN (for 4D image obs) or an MLP
  (for pre-flattened 1D image obs), selected via ``student_encoder_type``.
* Observation normalisation applies only to 1D obs branches; CNN inputs are
  assumed to already be normalised (e.g. depth values in [0, 1]).
* Teacher is loaded from a pre-trained ActorCritic checkpoint and kept frozen.
* ``is_recurrent = False`` — compatible with the ``Distillation`` algorithm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import CNN, MLP, EmpiricalNormalization
from rsl_rl.networks.memory import HiddenState


class StudentTeacherCNN(nn.Module):
    """Student-teacher distillation with CNN/MLP image encoder for the student."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_image_groups: list[str] | None = None,
        student_encoder_type: str = "cnn",
        student_cnn_cfg: dict | None = None,
        student_encoder_hidden_dims: list[int] = [256, 128],
        student_encoder_output_dim: int = 64,
        student_hidden_dims: list[int] = [256, 256, 256],
        teacher_hidden_dims: list[int] = [256, 256, 256],
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "StudentTeacherCNN.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.student_encoder_type = student_encoder_type

        # ── Split student obs groups into image and other ──────────────────
        if student_image_groups is None:
            student_image_groups = []
        self.student_image_groups: list[str] = []
        self.student_other_groups: list[str] = []

        for group in obs_groups["policy"]:
            if group in student_image_groups:
                self.student_image_groups.append(group)
            else:
                self.student_other_groups.append(group)

        # ── Build image encoder ────────────────────────────────────────────
        encoder_output_dim = 0

        if student_encoder_type == "cnn":
            # Image groups must be 4D (B, C, H, W)
            image_in_dims: list[tuple[int, int]] = []
            image_in_channels: list[int] = []
            for group in self.student_image_groups:
                shape = obs[group].shape
                assert len(shape) == 4, (
                    f"CNN encoder expects 4D image obs (B, C, H, W), "
                    f"but group '{group}' has shape {shape}."
                )
                image_in_dims.append(shape[2:4])
                image_in_channels.append(shape[1])

            if self.student_image_groups:
                assert student_cnn_cfg is not None, (
                    "student_cnn_cfg is required when student_image_groups is non-empty "
                    "and student_encoder_type='cnn'."
                )
                # If a single flat config dict is given, broadcast to all image groups
                if not all(isinstance(v, dict) for v in student_cnn_cfg.values()):
                    student_cnn_cfg = {g: student_cnn_cfg for g in self.student_image_groups}
                assert len(student_cnn_cfg) == len(self.student_image_groups), (
                    "Number of CNN configs must match number of student image groups."
                )
                self.student_cnns = nn.ModuleDict()
                for idx, group in enumerate(self.student_image_groups):
                    cnn = CNN(
                        input_dim=image_in_dims[idx],
                        input_channels=image_in_channels[idx],
                        **student_cnn_cfg[group],
                    )
                    assert cnn.output_channels is None, (
                        f"Student CNN for '{group}' must produce a flattened output. "
                        "Set global_pool='avg'/'max' or flatten=True in the CNN config."
                    )
                    self.student_cnns[group] = cnn
                    encoder_output_dim += int(cnn.output_dim)
                    print(f"Student CNN for {group}: {cnn}")
            else:
                self.student_cnns = None

        elif student_encoder_type == "mlp":
            # Image groups must be 1D (B, N) — pre-flattened
            total_image_obs_dim = 0
            for group in self.student_image_groups:
                shape = obs[group].shape
                assert len(shape) == 2, (
                    f"MLP encoder expects 1D image obs (B, N), "
                    f"but group '{group}' has shape {shape}."
                )
                total_image_obs_dim += shape[-1]

            if self.student_image_groups:
                self.student_encoder_mlp = MLP(
                    total_image_obs_dim,
                    student_encoder_output_dim,
                    student_encoder_hidden_dims,
                    activation,
                )
                encoder_output_dim = student_encoder_output_dim
                print(f"Student MLP encoder: {self.student_encoder_mlp}")
            else:
                self.student_encoder_mlp = None

            self.student_cnns = None  # Not used in MLP mode

        else:
            raise ValueError(
                f"Unknown student_encoder_type: {student_encoder_type}. Use 'cnn' or 'mlp'."
            )

        # ── Student 1D obs dimension ───────────────────────────────────────
        num_student_other_obs = 0
        for group in self.student_other_groups:
            shape = obs[group].shape
            assert len(shape) == 2, (
                f"Non-image observation group '{group}' must be 1D (B, F), "
                f"but has shape {shape}."
            )
            num_student_other_obs += shape[-1]

        # ── Student 1D obs normaliser ──────────────────────────────────────
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization and num_student_other_obs > 0:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_other_obs)
        else:
            self.student_obs_normalizer = nn.Identity()

        # ── Student MLP ────────────────────────────────────────────────────
        student_mlp_input_dim = encoder_output_dim + num_student_other_obs
        self.student = MLP(student_mlp_input_dim, num_actions, student_hidden_dims, activation)
        print(f"Student MLP: {self.student}")

        # ── Teacher ────────────────────────────────────────────────────────
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, (
                "The teacher only supports 1D observations."
            )
            num_teacher_obs += obs[obs_group].shape[-1]

        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        print(f"Teacher MLP: {self.teacher}")

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = nn.Identity()

        # ── Action noise ───────────────────────────────────────────────────
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(
                f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'"
            )

        self.distribution = None
        Normal.set_default_validate_args(False)

    # ── Properties ─────────────────────────────────────────────────────────

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

    # ── Encoding helpers ───────────────────────────────────────────────────

    def _apply_cnn(self, cnn: CNN, x: torch.Tensor) -> torch.Tensor:
        """Apply CNN to (B, C, H, W) input."""
        return cnn(x)

    def _encode_student_obs(self, obs: TensorDict) -> torch.Tensor:
        """Encode student observations: image encoder output + normalised 1D obs."""
        parts = []

        # Image encoder
        if self.student_image_groups:
            if self.student_encoder_type == "cnn":
                for group in self.student_image_groups:
                    parts.append(self._apply_cnn(self.student_cnns[group], obs[group]))
            elif self.student_encoder_type == "mlp":
                image_obs = torch.cat([obs[g] for g in self.student_image_groups], dim=-1)
                parts.append(self.student_encoder_mlp(image_obs))

        # 1D obs (normalised)
        if self.student_other_groups:
            obs_1d = torch.cat([obs[g] for g in self.student_other_groups], dim=-1)
            obs_1d = self.student_obs_normalizer(obs_1d)
            parts.append(obs_1d)

        return torch.cat(parts, dim=-1)

    # ── Distribution ───────────────────────────────────────────────────────

    def _update_distribution(self, encoded_obs: torch.Tensor) -> None:
        mean = self.student(encoded_obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )
        self.distribution = Normal(mean, std)

    # ── Core interface (compatible with Distillation algorithm) ────────────

    def act(self, obs: TensorDict) -> torch.Tensor:
        encoded = self._encode_student_obs(obs)
        self._update_distribution(encoded)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        encoded = self._encode_student_obs(obs)
        return self.student(encoded)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        obs_cat = self.get_teacher_obs(obs)
        obs_cat = self.teacher_obs_normalizer(obs_cat)
        with torch.no_grad():
            return self.teacher(obs_cat)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        """Return encoded student observation (image encoder + 1D obs)."""
        return self._encode_student_obs(obs)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization and self.student_other_groups:
            student_1d = torch.cat([obs[g] for g in self.student_other_groups], dim=-1)
            self.student_obs_normalizer.update(student_1d)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in ``state_dict`` match.

        Returns:
            Whether this training resumes a previous distillation training.
        """
        # Load from RL training checkpoint (ActorCritic) → map actor.* to teacher.*
        if any("actor" in key for key in state_dict):
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False  # Training does not resume
        # Resume previous distillation training
        elif any("student" in key for key in state_dict):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True  # Training resumes
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
