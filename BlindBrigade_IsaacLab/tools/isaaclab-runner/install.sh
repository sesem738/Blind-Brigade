#!/usr/bin/env bash
# Install the Isaac Lab Runner VS Code extension by symlinking it into ~/.vscode/extensions/
# and registering it in VS Code's extension registry.
# Run once per machine. After that, edits to tools/isaaclab-runner/ are live immediately.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_NAME="blindbrigade.isaaclab-runner-1.0.0"
EXT_DIR="$HOME/.vscode/extensions/$EXT_NAME"
REGISTRY="$HOME/.vscode/extensions/extensions.json"

# ── 1. Create symlink ──────────────────────────────────────────────────────────
if [ -L "$EXT_DIR" ]; then
    echo "Symlink already exists: $EXT_DIR -> $(readlink "$EXT_DIR")"
elif [ -d "$EXT_DIR" ]; then
    echo "Removing old install at $EXT_DIR"
    rm -rf "$EXT_DIR"
fi

if [ ! -L "$EXT_DIR" ]; then
    ln -s "$REPO_DIR" "$EXT_DIR"
    echo "Created: $EXT_DIR -> $REPO_DIR"
fi

# ── 2. Register in extensions.json ────────────────────────────────────────────
python3 - <<PYEOF
import json, time, sys

registry = "$REGISTRY"
ext_dir  = "$EXT_DIR"

try:
    with open(registry) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = []

if any(e.get('identifier', {}).get('id') == 'blindbrigade.isaaclab-runner' for e in data):
    print("Already registered in extensions.json")
    sys.exit(0)

data.append({
    "identifier": { "id": "blindbrigade.isaaclab-runner" },
    "version": "1.0.0",
    "location": {
        "\$mid": 1,
        "fsPath": ext_dir,
        "external": f"file://{ext_dir}",
        "path": ext_dir,
        "scheme": "file"
    },
    "relativeLocation": "$EXT_NAME",
    "metadata": {
        "isApplicationScoped": False,
        "isMachineScoped":     False,
        "isBuiltin":           False,
        "installedTimestamp":  int(time.time() * 1000),
        "pinned":              False,
        "source":              "local",
        "targetPlatform":      "undefined",
        "updated":             False,
        "private":             False,
        "isPreReleaseVersion": False,
        "hasPreReleaseVersion": False
    }
})

with open(registry, 'w') as f:
    json.dump(data, f, separators=(',', ':'))

print("Registered in extensions.json")
PYEOF

echo ""
echo "Done. Restart VS Code (File → Exit, then reopen) to activate the extension."
