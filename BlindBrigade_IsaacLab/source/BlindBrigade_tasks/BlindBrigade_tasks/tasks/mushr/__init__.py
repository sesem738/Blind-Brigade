import gymnasium as gym
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="BB-mushr-flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mushr_env_cfg:MushrNavFlatTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFlatCfg",
    },
)

gym.register(
    id="BB-mushr-flat-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mushr_env_cfg:MushrNavFlatTerrainEnvPLAYCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFlatCfg",
    },
)

gym.register(
    id="BB-mushr-box-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mushr_env_cfg:MushrNavBoxTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerBoxCfg",
    },
)

gym.register(
    id="BB-mushr-box-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mushr_env_cfg:MushrNavBoxTerrainEnvPLAYCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerBoxCfg",
    },
)
