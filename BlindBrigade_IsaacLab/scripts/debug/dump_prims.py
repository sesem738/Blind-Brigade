"""Dump the USD prim tree for a task environment to find mesh prim paths."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump prim tree.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from pxr import Usd, UsdGeom

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import BlindBrigade_tasks.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=False)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Get the USD stage
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    # Print prim tree for env_0 only
    root = stage.GetPrimAtPath("/World/envs/env_0")
    if not root.IsValid():
        print("[ERROR] /World/envs/env_0 not found")
        env.close()
        return

    print("\n" + "=" * 80)
    print("PRIM TREE for /World/envs/env_0")
    print("=" * 80)

    for prim in Usd.PrimRange(root):
        depth = len(str(prim.GetPath()).split("/")) - 4  # indent relative to env_0
        indent = "  " * depth
        prim_type = prim.GetTypeName()
        has_mesh = prim_type == "Mesh"
        marker = " <-- MESH" if has_mesh else ""
        print(f"{indent}{prim.GetName()} [{prim_type}]{marker}")

    print("=" * 80)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()