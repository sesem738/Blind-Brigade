# Isaac Lab Runner – VS Code Extension

A sidebar panel for launching Isaac Lab train/play sessions without manually typing long CLI commands.

---

## Problem

Running Isaac Lab requires commands like:

```bash
python scripts/rsl_rl/train.py --task BB-rosbot-box-v0 --agent rsl_rl_cnn_cfg_entry_point \
    --num_envs 2048 --experiment_name rosbot_cnn_mlp --headless
```

or for playback:

```bash
python scripts/rsl_rl/play.py --task BB-rosbot-box-PLAY-v0 --agent rsl_rl_cfg_entry_point \
    --num_envs 1 --load_run 2026-02-26_01-54-24 --checkpoint model_1999.pt
```

Remembering argument names, task IDs, agent config keys, and checkpoint paths is error-prone and slow. This extension replaces all of that with dropdowns and a single click.

---

## Design

The extension is a **VS Code Webview View** — a sidebar panel that renders an HTML page inside the editor. The architecture has two layers that communicate via `postMessage`:

```
VS Code Extension Host (Node.js)          Webview (sandboxed browser)
────────────────────────────────          ───────────────────────────
 - reads the filesystem (logs/)    ←→     - renders the UI (HTML/CSS/JS)
 - builds and runs CLI commands            - sends user actions as messages
 - knows the repo root location            - receives data to populate dropdowns
```

The webview cannot access the filesystem or run processes directly — that's why the two sides communicate through messages. The extension host does all I/O and the webview is purely UI.

**Why no npm / build step?**
VS Code extensions normally use TypeScript and a bundler. This extension is plain JavaScript loaded directly by VS Code's Node.js runtime, so there are no dependencies, no `node_modules`, and no compilation. The tradeoff is no type safety, which is acceptable for a small internal tool.

**Why a symlink install?**
VS Code loads extensions from `~/.vscode/extensions/`. Rather than maintaining a separate copy there, `install.sh` creates a symlink pointing back into the repo. This means:
- The repo is the single source of truth
- Edits take effect on VS Code reload with no reinstall
- Git tracks only the real files, not the symlink

---

## File Structure

```
tools/isaaclab-runner/
├── package.json      — VS Code manifest: declares the sidebar view, icon, activation events
├── extension.js      — All extension logic (see sections below)
├── install.sh        — One-time setup script for each machine
└── media/
    └── icon.svg      — Activity Bar icon (must use currentColor for VS Code theming)
```

### `extension.js` sections

| Lines | Section | Purpose |
|-------|---------|---------|
| 10–47 | Data tables | `TASKS`, `TASK_AGENTS`, `EXPERIMENT_NAMES` — maps task IDs to valid agent configs and their default experiment folder names (sourced from the Python agent config classes) |
| 51–93 | View provider + message router | Registers the webview and routes incoming messages from the HTML panel to the right handler |
| 99–118 | `_getRepoRoot()` | Finds the repo by scanning workspace folders for `scripts/rsl_rl/train.py`. Handles the case where VS Code is opened at the parent `Repo/` directory |
| 120–163 | `_scan*()` | Reads `logs/rsl_rl/` to populate experiment, run, and checkpoint dropdowns dynamically |
| 165–218 | `_runCommand()` | Assembles the CLI argument list from the config object and runs it in a new VS Code terminal. For debug mode, prepends `python -m debugpy --listen 5678 --wait-for-client` |
| 222–715 | `_buildHtml()` | The entire panel UI as a template string — HTML structure, CSS using VS Code theme variables, and the JS event handlers |

---

## Maintenance

### Adding a new task

1. Add an entry to `TASKS` in `extension.js`:
   ```js
   { id: 'BB-rosbot-newenv-v0', label: 'Rosbot – New Env' },
   ```
2. Add its valid agent configs to `TASK_AGENTS`:
   ```js
   'BB-rosbot-newenv-v0': ['rsl_rl_cfg_entry_point'],
   ```
3. Add its default experiment name to `EXPERIMENT_NAMES`:
   ```js
   'BB-rosbot-newenv-v0_rsl_rl_cfg_entry_point': 'rosbot_newenv',
   ```
4. Reload VS Code (`Ctrl+Shift+P` → `Developer: Reload Window`).

### Adding a new agent config

Add it to the relevant task's array in `TASK_AGENTS` and add the corresponding `EXPERIMENT_NAMES` entry using the key pattern `{task_id}_{agent_entry_point}`.

### Changing the UI

Edit `_buildHtml()` in `extension.js`. The HTML, CSS, and JS are all inline in that method. Use `var(--vscode-*)` CSS variables to stay consistent with the user's theme.

### Debugging the extension itself

Open **Help → Toggle Developer Tools** in VS Code and check the Console tab for errors from the extension host. Webview JS errors appear there too.

---

## Installation (per machine)

```bash
bash tools/isaaclab-runner/install.sh
# then: File → Exit and reopen VS Code
```

The script:
1. Creates `~/.vscode/extensions/blindbrigade.isaaclab-runner-1.0.0` as a symlink to this directory
2. Registers the extension in `~/.vscode/extensions/extensions.json` (required by VS Code 1.74+)

Safe to re-run — it skips steps that are already done.

---

## Debug Mode

Clicking **🐛 Debug** runs the script with `debugpy` listening on port 5678. To attach:

1. Open the Run & Debug panel (`Ctrl+Shift+D`)
2. Select **"Isaac Lab: Attach Debugger"** from the dropdown
3. Press F5

This config is defined in `.vscode/launch.json` in the repo root.
