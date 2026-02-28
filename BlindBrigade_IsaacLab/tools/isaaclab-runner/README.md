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
    --num_envs 1 --load_run 2026-02-26_01-54-24 \
    --checkpoint /abs/path/to/logs/.../model_1999.pt
```

Remembering argument names, task IDs, agent config keys, and checkpoint paths is error-prone and slow. This extension replaces all of that with dropdowns and a single click.

---

## Design

The extension is a **VS Code Webview View** — a sidebar panel that renders an HTML page inside the editor. The architecture has two layers that communicate via `postMessage`:

```
VS Code Extension Host (Node.js)          Webview (sandboxed browser)
────────────────────────────────          ───────────────────────────
 - reads the filesystem (logs/)    ←→     - renders the UI (HTML/CSS/JS)
 - parses Python source files              - sends user actions as messages
 - builds and runs CLI commands            - receives data to populate dropdowns
 - knows the repo root location
```

The webview cannot access the filesystem or run processes directly — that's why the two sides communicate through messages. The extension host does all I/O and the webview is purely UI.

**Why no npm / build step?**
VS Code extensions normally use TypeScript and a bundler. This extension is plain JavaScript loaded directly by VS Code's Node.js runtime, so there are no dependencies, no `node_modules`, and no compilation. The tradeoff is no type safety, which is acceptable for a small internal tool.

**Why a symlink install?**
VS Code loads extensions from `~/.vscode/extensions/`. Rather than maintaining a separate copy there, `install.sh` creates a symlink pointing back into the repo. This means:
- The repo is the single source of truth
- Edits take effect on VS Code reload with no reinstall
- Git tracks only the real files, not the symlink

**Why Python as the terminal shell?**
The extension spawns Python directly as the terminal process (`shellPath: pythonExe`) rather than launching a bash shell and sending a command string. It means no shell initialisation, no `.bashrc`, and no conda init running mid-command. The absolute path to the conda env's Python binary is resolved from `~/miniconda3/envs/<env>/bin/python`, so no `conda activate` is ever needed. Ctrl+C still works because the terminal sends SIGINT directly to the Python process.

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

| Section | Purpose |
|---------|---------|
| `_parseTasksFromRepo()` | Scans `source/**/tasks/**/__init__.py` for `gym.register()` calls and extracts task IDs and agent entry point keys automatically |
| `_parseInitFile()` / `_parseRegisterBlock()` | Regex parsers that read `gym.register()` blocks and extract task ID, agent config keys, and entry point module/class references |
| `_parseExperimentName()` | Reads the `experiment_name = "..."` field from the agent config Python class body |
| `_scanExperiments/Runs/Checkpoints()` | Reads `logs/rsl_rl/` to populate experiment, run, and checkpoint dropdowns; run scan also reads `run_label.txt` to show human-readable labels alongside timestamps |
| `_setRunLabel()` | Writes or deletes `run_label.txt` in a run directory |
| `_scanCondaEnvs()` | Lists available conda environments from `~/miniconda3/envs/` to populate the Python environment dropdown |
| `_resolveCondaPython()` | Finds the absolute path to the Python binary for a conda environment without running `conda activate` |
| `_runCommand()` | Assembles the args array and creates a terminal with Python as the shell process |
| `resolveWebviewView()` | Sets up the webview, pushes initial data, and registers a file watcher on task and agent config files to auto-refresh on save |
| `_buildHtml()` | The entire panel UI as a template string — HTML structure, CSS using VS Code theme variables, and the JS event handlers |

---

## Auto-Detection

Tasks, agent configs, and experiment names are parsed live from the Python source files. No hardcoded tables.

- **Tasks**: discovered from `gym.register(id=...)` calls in `source/**/tasks/**/__init__.py`
- **Agent configs**: discovered from `rsl_rl_*_entry_point` keys in the same register blocks
- **Experiment names**: read from `experiment_name = "..."` in the agent config class body
- **Auto-refresh**: a file watcher on `source/**/tasks/**/{__init__.py,agents/*.py}` re-parses on every save, so the dropdowns update immediately when you add or rename a config

---

## Run Labelling

Training runs are saved as timestamped directories (e.g. `2026-02-27_09-07-42`). The extension supports attaching a human-readable label to any run:

- Select an experiment and run in the Play panel
- Type a label in the text field and click **Save**
- The label is written to `run_label.txt` inside the run directory
- Future scans show the label alongside the timestamp in the dropdown

Labels are stored as plain text files inside `logs/` which is gitignored, so they are local to each machine.

---

## Checkpoint Paths

Isaac Lab's `retrieve_file_path()` requires an **absolute path** to the checkpoint file. The extension constructs the full path from the known repo root, experiment name, run folder, and checkpoint filename before passing it to `--checkpoint`. Passing just a filename like `model_1999.pt` causes a `FileNotFoundError`.

---

## Maintenance

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

> **Known issue:** Debug mode does not currently work. The process launches and waits, but attaching the VS Code debugger does not connect successfully. Left here for future fixing.

Clicking **🐛 Debug** runs the script with `debugpy` listening on port 5678 and waiting for a client before starting execution.

**Prerequisites:**
- `debugpy` must be installed in the selected conda environment:
  ```bash
  pip install debugpy
  ```

**To attach:**
1. Click **🐛 Debug** — the terminal opens and freezes waiting for a client
2. Open the Run & Debug panel (`Ctrl+Shift+D`)
3. Select **"Isaac Lab: Attach Debugger"** from the dropdown
4. Press **F5**

The attach config is defined in `.vscode/launch.json` at the repo root.
