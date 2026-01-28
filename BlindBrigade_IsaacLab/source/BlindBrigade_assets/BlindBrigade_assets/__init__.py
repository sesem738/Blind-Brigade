import os
import toml

# Conveniences to other module directories via relative paths
BLINDBRIGADE_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
"""Path to the extension source directory."""

BLINDBRIGADE_ASSETS_DATA_DIR = os.path.join(BLINDBRIGADE_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

BLINDBRIGADE_ASSETS_METADATA = toml.load(os.path.join(BLINDBRIGADE_ASSETS_DATA_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = BLINDBRIGADE_ASSETS_METADATA["package"]["version"]

from .rosorin import *