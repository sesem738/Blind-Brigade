# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

########################################################
#               Regular Actor Critic
########################################################

@configclass
class PPORunnerBoxCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 2000
    save_interval = 50
    experiment_name = "box_reward_trials"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class PPORunnerFlatCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 2000
    save_interval = 50
    experiment_name = "rosbot_p2p"
    obs_groups = {
        "policy": ["proprioceptive"],
        "critic": ["proprioceptive"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

########################################################
#          Image embedding + Actor Critic
########################################################

@configclass
class RslRlActorCriticCNNCfg(RslRlPpoActorCriticCfg):
    """Policy config for ActorCriticCNN — compatible with rsl_rl.modules.ActorCriticCNN.

    When agent_cfg.to_dict() is called, nested @configclass instances are recursively
    converted to dicts. ActorCriticCNN.__init__ receives actor_cnn_cfg as a plain dict
    and applies it as CNN kwargs.
    """

    @configclass
    class CNNCfg:
        """Maps 1-to-1 to rsl_rl.networks.CNN kwargs (except input_dim/input_channels, auto-filled)."""

        output_channels: list = MISSING
        kernel_size: list | int = MISSING
        stride: list | int = 1
        padding: str = "none"
        activation: str = "elu"
        max_pool: list | bool = False
        global_pool: str = "none"

    class_name: str = "BlindBrigade_tasks.modules.actor_critic_recurrent_raycast:ActorCriticCNN"
    actor_cnn_cfg: CNNCfg = MISSING
    critic_cnn_cfg: CNNCfg = MISSING

@configclass
class PPORunnerBoxCnnCfg(RslRlOnPolicyRunnerCfg):
    """Runner config for ActorCriticCNN with downward height-map input.

    obs_groups routes:
      proprioceptive (B, 9)        → MLP branch
      exteroceptive  (B, 1, 64, 64) → CNN branch
    """

    num_steps_per_env = 64
    max_iterations = 2000
    save_interval = 50
    experiment_name = "rosbot_cnn_mlp"
    obs_groups = {
        "policy": ["proprioceptive", "exteroceptive"],
        "critic": ["proprioceptive", "exteroceptive"],
    }
    policy = RslRlActorCriticCNNCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        actor_cnn_cfg=RslRlActorCriticCNNCfg.CNNCfg(
            output_channels=[32, 64],
            kernel_size=[7, 5],  # 64×64 → max_pool → ~28×28; kernel 5 → ~24×24; global_pool avg → 64-dim
            activation="elu",
            max_pool=[True, False],
            global_pool="avg",  # flatten CNN output to 64-dim vector
        ),
        critic_cnn_cfg=RslRlActorCriticCNNCfg.CNNCfg(
            output_channels=[32, 64],
            kernel_size=[7, 5],
            activation="elu",
            max_pool=[True, False],
            global_pool="avg",
        ),
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

########################################################
#      Height scan embed + Recurrent Actor Critic
########################################################

@configclass
class RslRlActorCriticRecurrentRayCastCfg(RslRlPpoActorCriticCfg):
    """Policy config for ActorCriticRecurrentRayCast.

    raycaster obs groups  →  MLP encoder  →  embedding
    other 1D obs groups   →  normaliser   →  flat features
    cat(embedding, flat)  →  GRU          →  MLP  →  actions / value
    """

    class_name: str = "BlindBrigade_tasks.modules.actor_critic_recurrent_raycast:ActorCriticRecurrentRayCast"

    # Which obs groups (within policy/critic) are routed to the raycaster MLP encoder
    actor_raycast_groups: list = MISSING
    critic_raycast_groups: list = MISSING

    # Raycaster MLP encoder architecture
    raycast_embed_dim: int = 64
    actor_raycast_encoder_hidden_dims: list = MISSING
    critic_raycast_encoder_hidden_dims: list = MISSING

    # Optional EmpiricalNormalization on raycaster branch (default off — values
    # are already normalised to [0, 1] by ray_caster_lidar observation function)
    actor_raycast_normalization: bool = False
    critic_raycast_normalization: bool = False

    # GRU / LSTM
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1


@configclass
class PPORunnerBoxRecurrentRayCastCfg(RslRlOnPolicyRunnerCfg):
    """Runner config for ActorCriticRecurrentRayCast with 1D raycaster input.

    obs_groups routes:
      raycaster     (B, N_rays)  →  MLP encoder branch
      proprioceptive (B, 9)      →  normaliser branch
    Both are concatenated into the GRU input.
    """

    num_steps_per_env = 64
    max_iterations = 2000
    save_interval = 50
    experiment_name = "rosbot_heightscan_gru_mlp"
    obs_groups = {
        "policy": ["heightscan", "proprioceptive"],
        "critic": ["heightscan", "proprioceptive"],
    }
    policy = RslRlActorCriticRecurrentRayCastCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        actor_raycast_groups=["heightscan"],
        critic_raycast_groups=["heightscan"],
        raycast_embed_dim=64,
        actor_raycast_encoder_hidden_dims=[128, 128],
        critic_raycast_encoder_hidden_dims=[128, 128],
        rnn_type="gru",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

########################################################
#      Image embedding + Recurrent Actor Critic
########################################################

@configclass
class RslRlActorCriticRecurrentCNNCfg(RslRlPpoActorCriticCfg):
    """Policy config for ActorCriticRecurrentCNN.

    class_name uses a colon-qualified path so resolve_callable can import it
    from the BlindBrigade_tasks package without modifying rsl_rl.
    """

    @configclass
    class CNNCfg:
        """Maps 1-to-1 to rsl_rl.networks.CNN kwargs (except input_dim/input_channels, auto-filled)."""

        output_channels: list = MISSING
        kernel_size: list | int = MISSING
        stride: list | int = 1
        padding: str = "none"
        activation: str = "elu"
        max_pool: list | bool = False
        global_pool: str = "none"

    class_name: str = "BlindBrigade_tasks.modules.actor_critic_recurrent_cnn:ActorCriticRecurrentCNN"
    actor_cnn_cfg: CNNCfg = MISSING
    critic_cnn_cfg: CNNCfg = MISSING
    rnn_type: str = "lstm"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1


@configclass
class PPORunnerBoxRecurrentCnnCfg(RslRlOnPolicyRunnerCfg):
    """Runner config for ActorCriticRecurrentCNN with downward height-map input.

    obs_groups routes:
      proprioceptive (B, 9)         → MLP branch
      exteroceptive  (B, 1, 64, 64) → CNN branch → LSTM
    """

    num_steps_per_env = 64
    max_iterations = 2000
    save_interval = 50
    experiment_name = "rosbot_recurrent_cnn"
    obs_groups = {
        "policy": ["proprioceptive", "exteroceptive"],
        "critic": ["proprioceptive", "exteroceptive"],
    }
    policy = RslRlActorCriticRecurrentCNNCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        actor_cnn_cfg=RslRlActorCriticRecurrentCNNCfg.CNNCfg(
            output_channels=[32, 64],
            kernel_size=[7, 5],
            activation="elu",
            max_pool=[True, False],
            global_pool="avg",
        ),
        critic_cnn_cfg=RslRlActorCriticRecurrentCNNCfg.CNNCfg(
            output_channels=[32, 64],
            kernel_size=[7, 5],
            activation="elu",
            max_pool=[True, False],
            global_pool="avg",
        ),
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
