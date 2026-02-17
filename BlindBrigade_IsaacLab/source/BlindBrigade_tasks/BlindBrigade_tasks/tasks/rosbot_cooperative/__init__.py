import gymnasium as gym
from . import agents

##
# Register Gym environments.
##


gym.register(
    id="BB-rosbot-coop-flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rosbot_env_cfg:RosbotCoopNavFlatTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFlatCfg",
    },
)

gym.register(
    id="BB-rosbot-coop-flat-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rosbot_env_cfg:RosbotCoopNavFlatTerrainEnvPLAYCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerFlatCfg",
    },
)

gym.register(
    id="BB-rosbot-coop-box-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rosbot_env_cfg:RosbotCoopNavBoxTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerBoxCfg",
    },
)

gym.register(
    id="BB-rosbot-coop-box-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rosbot_env_cfg:RosbotCoopNavBoxTerrainEnvPLAYCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerBoxCfg",
    },
)
