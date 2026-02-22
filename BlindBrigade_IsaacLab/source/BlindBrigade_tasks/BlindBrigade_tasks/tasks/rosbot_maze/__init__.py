import gymnasium as gym
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="BB-rosbot-maze-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rosbot_maze_env_cfg:RosbotNavMazeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerMazeCfg",
    },
)

gym.register(
    id="BB-rosbot-maze-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rosbot_maze_env_cfg:RosbotNavMazeEnvPLAYCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerMazeCfg",
    },
)
