# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class PPORunnerBoxDistillationCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 300
    save_interval = 50
    experiment_name = "box_reward_trials"
    obs_groups = {"policy": ["policy"], "teacher": ["policy"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[256, 128, 128],
        teacher_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )


@configclass
class RslRlDistillationStudentTeacherCNNCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for student-teacher distillation with CNN/MLP image encoder."""

    class_name: str = "BlindBrigade_tasks.modules.student_teacher_cnn:StudentTeacherCNN"
    """The policy class name. Resolved by ``resolve_callable``."""

    student_image_groups: list = None
    """Which obs groups within ``obs_groups["policy"]`` are image observations."""

    student_encoder_type: str = "cnn"
    """Image encoder type: ``"cnn"`` for 4D obs or ``"mlp"`` for pre-flattened 1D obs."""

    student_cnn_cfg: dict = None
    """CNN config dict (when ``student_encoder_type="cnn"``). Same format as ``ActorCriticRecurrentCNN``."""

    student_encoder_hidden_dims: list = None
    """MLP encoder hidden dims (when ``student_encoder_type="mlp"``)."""

    student_encoder_output_dim: int = 64
    """Output dim of MLP encoder (when ``student_encoder_type="mlp"``)."""


@configclass
class PPORunnerBoxDistillationCNNCfg(RslRlDistillationRunnerCfg):
    """Distillation runner using StudentTeacherCNN with a CNN image encoder."""

    num_steps_per_env = 16
    max_iterations = 300
    save_interval = 50
    experiment_name = "box_reward_trials"
    obs_groups = {"policy": ["policy", "exteroceptive"], "teacher": ["policy"]}
    policy = RslRlDistillationStudentTeacherCNNCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[256, 128, 128],
        teacher_hidden_dims=[256, 128, 128],
        activation="elu",
        student_image_groups=["exteroceptive"],
        student_encoder_type="cnn",
        student_cnn_cfg={
            "output_channels": [32, 64],
            "kernel_size": 3,
            "stride": 2,
            "activation": "elu",
            "global_pool": "avg",
            "flatten": True,
        },
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )
