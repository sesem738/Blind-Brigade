import gymnasium as gym
from . import agents

##
# Register Gym environments.
##


gym.register(
    id="BB-rosbot-flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rosbot_env_cfg:RosbotNavFlatTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFlatCfg",
    },
)

gym.register(
    id="BB-rosbot-flat-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rosbot_env_cfg:RosbotNavFlatTerrainEnvPLAYCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFlatCfg",
    },
)

gym.register(
    id="BB-rosbot-box-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rosbot_env_cfg:RosbotNavBoxTerrainEnvCfg",
        "rsl_rl_cfg_entry_point":               f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerBoxCfg",
        "rsl_rl_gru_cfg_entry_point":           f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerRecurrentRayCastCfg",
        "rsl_rl_cnn_cfg_entry_point":           f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerBoxCnnCfg",
        "rsl_rl_recurrent_cnn_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerBoxRecurrentCnnCfg",
    },
)

gym.register(
    id="BB-rosbot-box-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rosbot_env_cfg:RosbotNavBoxTerrainEnvPLAYCfg",
        "rsl_rl_gru_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerRecurrentRayCastCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerBoxCfg",
    },
)
