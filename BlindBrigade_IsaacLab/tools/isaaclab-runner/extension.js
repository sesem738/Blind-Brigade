// Isaac Lab Runner – VS Code Extension
// Provides a sidebar panel for running train/play sessions without typing long CLI commands.

const vscode = require('vscode');
const fs = require('fs');
const path = require('path');

// Tasks, agent configs, and experiment names are parsed dynamically from the repo.
// See _parseTasksFromRepo() below — no hardcoded tables needed.

// ── Webview provider ───────────────────────────────────────────────────────────

class IsaacLabRunnerViewProvider {
    constructor(context) {
        this._context = context;
        this._view = null;
        this._watcher = null;
        this._taskData = null;
    }

    resolveWebviewView(webviewView) {
        this._view = webviewView;
        webviewView.webview.options = { enableScripts: true };
        webviewView.webview.html = this._buildHtml(webviewView.webview);
        webviewView.webview.onDidReceiveMessage(msg => this._handleMessage(msg));

        // Push task data immediately when the panel opens
        this._refreshAndPush();

        // Watch task __init__.py and agent config files; re-parse on save
        if (!this._watcher) {
            const root = this._getRepoRoot();
            if (root) {
                const pattern = new vscode.RelativePattern(root,
                    'source/**/tasks/**/{__init__.py,agents/*.py}');
                this._watcher = vscode.workspace.createFileSystemWatcher(pattern);
                const refresh = () => this._refreshAndPush();
                this._watcher.onDidChange(refresh);
                this._watcher.onDidCreate(refresh);
                this._watcher.onDidDelete(refresh);
                this._context.subscriptions.push(this._watcher);
            }
        }
    }

    _refreshAndPush() {
        const data = this._parseTasksFromRepo();
        this._taskData = data;
        this._send({ type: 'tasks', ...data });
    }

    _handleMessage(msg) {
        switch (msg.type) {
            case 'scanExperiments':
                this._scanExperiments(msg.target);
                break;

            case 'scanRuns':
                this._scanRuns(msg.experimentName, msg.target);
                break;

            case 'scanCheckpoints':
                this._scanCheckpoints(msg.experimentName, msg.run, msg.target);
                break;

            case 'setRunLabel':
                this._setRunLabel(msg.experimentName, msg.run, msg.label);
                break;

            case 'scanCondaEnvs':
                this._scanCondaEnvs();
                break;

            case 'run':
                this._runCommand(msg.config);
                break;
        }
    }

    _send(data) {
        if (this._view) this._view.webview.postMessage(data);
    }

    _scanCondaEnvs() {
        const home = require('os').homedir();
        // Check common conda install locations
        const roots = ['miniconda3', 'anaconda3', 'mambaforge', 'miniforge3']
            .map(d => path.join(home, d))
            .filter(d => fs.existsSync(d));

        if (roots.length === 0) { this._send({ type: 'condaEnvs', envs: [] }); return; }

        const condaRoot = roots[0];
        const envsDir = path.join(condaRoot, 'envs');
        try {
            const envs = ['base', ...fs.readdirSync(envsDir)
                .filter(f => fs.statSync(path.join(envsDir, f)).isDirectory())
                .sort()];
            this._send({ type: 'condaEnvs', envs });
        } catch (_) {
            this._send({ type: 'condaEnvs', envs: [] });
        }
    }

    _resolveCondaPython(condaEnv) {
        if (!condaEnv) return 'python';
        const home = require('os').homedir();
        const condaRoot = ['miniconda3', 'anaconda3', 'mambaforge', 'miniforge3']
            .map(d => path.join(home, d))
            .find(d => fs.existsSync(d));
        if (!condaRoot) return 'python';
        const exe = condaEnv === 'base'
            ? path.join(condaRoot, 'bin', 'python')
            : path.join(condaRoot, 'envs', condaEnv, 'bin', 'python');
        return fs.existsSync(exe) ? exe : 'python';
    }

    // ── Task parsing ──────────────────────────────────────────────────────────

    _parseTasksFromRepo() {
        const root = this._getRepoRoot();
        const empty = { tasks: [], agentMap: {}, expNameMap: {} };
        if (!root) return empty;

        const tasksDir = path.join(root, 'source', 'BlindBrigade_tasks', 'BlindBrigade_tasks', 'tasks');
        if (!fs.existsSync(tasksDir)) return empty;

        const tasks = [], agentMap = {}, expNameMap = {};

        try {
            const taskDirs = fs.readdirSync(tasksDir)
                .filter(f => fs.statSync(path.join(tasksDir, f)).isDirectory());

            for (const dir of taskDirs) {
                const initFile = path.join(tasksDir, dir, '__init__.py');
                if (!fs.existsSync(initFile)) continue;

                const parsed = this._parseInitFile(fs.readFileSync(initFile, 'utf8'));

                for (const reg of parsed) {
                    if (reg.id.includes('-PLAY-')) continue;   // skip play variants

                    tasks.push({ id: reg.id, label: this._makeLabel(reg.id) });
                    agentMap[reg.id] = reg.agentKeys;

                    // Resolve experiment_name for each entry point from the agent config files
                    for (const { key, module, className } of reg.entryPoints) {
                        const agentFile = path.join(tasksDir, dir, 'agents', `${module}.py`);
                        if (fs.existsSync(agentFile)) {
                            const expName = this._parseExperimentName(agentFile, className);
                            if (expName) expNameMap[`${reg.id}_${key}`] = expName;
                        }
                    }
                }
            }
        } catch (e) {
            console.error('Isaac Lab Runner: parse error', e);
        }

        return { tasks, agentMap, expNameMap };
    }

    // Parse a single __init__.py and return one object per gym.register() call
    _parseInitFile(content) {
        const results = [];
        const lines = content.split('\n');
        let inBlock = false, blockLines = [];

        for (const line of lines) {
            if (line.trimStart().startsWith('gym.register(')) {
                inBlock = true;
                blockLines = [line];
            } else if (inBlock) {
                blockLines.push(line);
                if (line.trim() === ')') {
                    const parsed = this._parseRegisterBlock(blockLines.join('\n'));
                    if (parsed) results.push(parsed);
                    inBlock = false;
                    blockLines = [];
                }
            }
        }
        return results;
    }

    _parseRegisterBlock(block) {
        const idMatch = block.match(/id\s*=\s*["']([^"']+)["']/);
        if (!idMatch) return null;

        const agentKeys = [], entryPoints = [];

        for (const line of block.split('\n')) {
            // Match lines like: "rsl_rl_cfg_entry_point": f"...module:ClassName"
            const keyMatch = line.match(/"(rsl_rl[^"]*_entry_point)"\s*:/);
            if (!keyMatch) continue;

            const key = keyMatch[1];
            agentKeys.push(key);

            // Extract module and ClassName from the f-string value: .module_name:ClassName
            const mcMatch = line.match(/\.([\w_]+):([\w]+)/);
            if (mcMatch) entryPoints.push({ key, module: mcMatch[1], className: mcMatch[2] });
        }

        return { id: idMatch[1], agentKeys, entryPoints };
    }

    // Find experiment_name = "..." for a given class in a Python file
    _parseExperimentName(filePath, className) {
        try {
            const content = fs.readFileSync(filePath, 'utf8');
            const classStart = content.indexOf(`class ${className}`);
            if (classStart === -1) return null;
            const nextClass = content.indexOf('\nclass ', classStart + 1);
            const body = nextClass === -1 ? content.slice(classStart) : content.slice(classStart, nextClass);
            const m = body.match(/experiment_name\s*=\s*["']([^"']+)["']/);
            return m ? m[1] : null;
        } catch (_) { return null; }
    }

    _makeLabel(taskId) {
        // "BB-rosbot-coop-box-v0" → "Rosbot Coop – Box"
        const parts = taskId.replace(/^BB-rosbot-/, '').replace(/-v\d+$/, '')
            .split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1));
        if (parts.length === 1) return `Rosbot – ${parts[0]}`;
        return `Rosbot ${parts.slice(0, -1).join(' ')} – ${parts[parts.length - 1]}`;
    }

    // Find the workspace folder that contains scripts/rsl_rl/train.py
    _getRepoRoot() {
        const folders = vscode.workspace.workspaceFolders;
        if (!folders) return null;
        for (const folder of folders) {
            const candidate = path.join(folder.uri.fsPath, 'scripts', 'rsl_rl', 'train.py');
            if (fs.existsSync(candidate)) return folder.uri.fsPath;
        }
        // Fallback: look one level deeper (workspace root is parent of repo)
        for (const folder of folders) {
            try {
                const entries = fs.readdirSync(folder.uri.fsPath);
                for (const entry of entries) {
                    const candidate = path.join(folder.uri.fsPath, entry, 'scripts', 'rsl_rl', 'train.py');
                    if (fs.existsSync(candidate)) return path.join(folder.uri.fsPath, entry);
                }
            } catch (_) {}
        }
        return null;
    }

    _scanExperiments(target) {
        const root = this._getRepoRoot();
        if (!root) { this._send({ type: 'experiments', experiments: [], target }); return; }
        const logsDir = path.join(root, 'logs', 'rsl_rl');
        try {
            const entries = fs.readdirSync(logsDir)
                .filter(f => fs.statSync(path.join(logsDir, f)).isDirectory())
                .sort();
            this._send({ type: 'experiments', experiments: entries, target });
        } catch (_) {
            this._send({ type: 'experiments', experiments: [], target });
        }
    }

    _scanRuns(experimentName, target) {
        const root = this._getRepoRoot();
        if (!root || !experimentName) { this._send({ type: 'runs', runs: [], target }); return; }
        const expDir = path.join(root, 'logs', 'rsl_rl', experimentName);
        try {
            const runs = fs.readdirSync(expDir)
                .filter(f => fs.statSync(path.join(expDir, f)).isDirectory())
                .sort().reverse()   // newest first
                .map(name => {
                    const labelFile = path.join(expDir, name, 'run_label.txt');
                    let label = '';
                    try { label = fs.readFileSync(labelFile, 'utf8').trim(); } catch (_) {}
                    return { name, label };
                });
            this._send({ type: 'runs', runs, target });
        } catch (_) {
            this._send({ type: 'runs', runs: [], target });
        }
    }

    _setRunLabel(experimentName, run, label) {
        const root = this._getRepoRoot();
        if (!root || !experimentName || !run) return;
        const labelFile = path.join(root, 'logs', 'rsl_rl', experimentName, run, 'run_label.txt');
        try {
            if (label) {
                fs.writeFileSync(labelFile, label.trim(), 'utf8');
            } else {
                if (fs.existsSync(labelFile)) fs.unlinkSync(labelFile);
            }
            this._send({ type: 'labelSaved' });
        } catch (e) {
            vscode.window.showErrorMessage(`Isaac Lab Runner: Could not save label — ${e.message}`);
        }
    }

    _scanCheckpoints(experimentName, run, target) {
        const root = this._getRepoRoot();
        if (!root || !experimentName || !run) { this._send({ type: 'checkpoints', checkpoints: [], target }); return; }
        const runDir = path.join(root, 'logs', 'rsl_rl', experimentName, run);
        try {
            const checkpoints = fs.readdirSync(runDir)
                .filter(f => f.endsWith('.pt'))
                .sort((a, b) => {
                    const n = s => parseInt((s.match(/\d+/) || ['0'])[0]);
                    return n(b) - n(a);   // highest iteration first
                });
            this._send({ type: 'checkpoints', checkpoints, target });
        } catch (_) {
            this._send({ type: 'checkpoints', checkpoints: [], target });
        }
    }

    _runCommand(cfg) {
        const root = this._getRepoRoot();
        if (!root) { vscode.window.showErrorMessage('Isaac Lab Runner: Could not find repo root (scripts/rsl_rl/train.py not found in any workspace folder).'); return; }

        const script = cfg.mode === 'train'
            ? path.join(root, 'scripts', 'rsl_rl', 'train.py')
            : path.join(root, 'scripts', 'rsl_rl', 'play.py');

        // Build arg list
        const actualTask = cfg.mode === 'play'
            ? cfg.task.replace('-v0', '-PLAY-v0')
            : cfg.task;

        const args = ['--task', actualTask, '--agent', cfg.agent];

        if (cfg.numEnvs)       args.push('--num_envs', cfg.numEnvs);
        if (cfg.seed)          args.push('--seed', cfg.seed);

        if (cfg.mode === 'train') {
            if (cfg.experimentName) args.push('--experiment_name', cfg.experimentName);
            if (cfg.runName)        args.push('--run_name', cfg.runName);
            if (cfg.maxIterations)  args.push('--max_iterations', cfg.maxIterations);
            if (cfg.resume) {
                args.push('--resume');
                if (cfg.resumeRun)        args.push('--load_run', cfg.resumeRun);
                if (cfg.resumeCheckpoint) args.push('--checkpoint', cfg.resumeCheckpoint);
            }
        } else {
            if (cfg.loadRun)    args.push('--load_run', cfg.loadRun);
            if (cfg.checkpoint) {
                // Isaac Lab's retrieve_file_path() requires an absolute path — passing just the
                // filename causes FileNotFoundError. Build the full path from what we know.
                const fullCkpt = path.join(root, 'logs', 'rsl_rl', cfg.playExpName, cfg.loadRun, cfg.checkpoint);
                args.push('--checkpoint', fullCkpt);
            }
        }

        if (cfg.extraArgs) args.push(...cfg.extraArgs.split(/\s+/).filter(Boolean));

        // Use Python itself as the terminal shell process — same approach as the VS Code Python
        // debugger. Python is spawned directly (no bash, no .bashrc, no conda init) so nothing
        // can send stray characters before or during the run. Ctrl+C still works because the
        // terminal sends SIGINT directly to the Python process.
        const pythonExe = this._resolveCondaPython(cfg.condaEnv);
        const shellArgs = cfg.debug
            ? ['-m', 'debugpy', '--listen', '5678', '--wait-for-client', script, ...args]
            : [script, ...args];

        const terminal = vscode.window.createTerminal({
            name: `IsaacLab ${cfg.mode}`,
            cwd: root,
            shellPath: pythonExe,
            shellArgs,
        });
        terminal.show();

        if (cfg.debug) {
            vscode.window.showInformationMessage(
                'Isaac Lab is waiting for a debugger on port 5678. Attach with the "Python: Attach" launch config.',
                'Open Debug Panel'
            ).then(action => {
                if (action === 'Open Debug Panel') {
                    vscode.commands.executeCommand('workbench.view.debug');
                }
            });
        }
    }

    // ── HTML ─────────────────────────────────────────────────────────────────

    _buildHtml(webview) {
        return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline';">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    padding: 8px 10px 16px;
    font-family: var(--vscode-font-family);
    font-size: var(--vscode-font-size);
    color: var(--vscode-foreground);
    background: transparent;
  }

  /* ── Section ── */
  .section { margin-bottom: 14px; }

  .section-title {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--vscode-sideBarSectionHeader-foreground, var(--vscode-descriptionForeground));
    margin-bottom: 6px;
    padding-bottom: 3px;
    border-bottom: 1px solid var(--vscode-sideBarSectionHeader-border, var(--vscode-widget-border, #444));
  }

  /* ── Form elements ── */
  label {
    display: block;
    font-size: 11px;
    margin-bottom: 2px;
    color: var(--vscode-descriptionForeground);
  }

  select, input[type="text"], input[type="number"] {
    width: 100%;
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, transparent);
    padding: 3px 6px;
    font-size: 12px;
    font-family: var(--vscode-font-family);
    border-radius: 2px;
    margin-bottom: 7px;
    outline: none;
  }
  select:focus, input:focus {
    border-color: var(--vscode-focusBorder);
  }

  /* ── Mode toggle ── */
  .mode-row { display: flex; gap: 4px; margin-bottom: 10px; }
  .mode-btn {
    flex: 1;
    padding: 5px 0;
    font-size: 12px;
    font-weight: 600;
    font-family: var(--vscode-font-family);
    cursor: pointer;
    border: 1px solid var(--vscode-button-border, var(--vscode-contrastBorder, transparent));
    border-radius: 2px;
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
    transition: background 0.1s;
  }
  .mode-btn.active {
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
  }
  .mode-btn:hover:not(.active) { opacity: 0.8; }

  /* ── Two-column row ── */
  .row2 { display: flex; gap: 6px; }
  .row2 > .field { flex: 1; min-width: 0; }

  /* ── Scan row ── */
  .scan-row { display: flex; gap: 4px; margin-bottom: 4px; }
  .scan-row select { flex: 1; margin-bottom: 0; }
  .scan-btn {
    padding: 3px 7px;
    font-size: 11px;
    font-family: var(--vscode-font-family);
    cursor: pointer;
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
    border: 1px solid var(--vscode-button-border, transparent);
    border-radius: 2px;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .scan-btn:hover { opacity: 0.85; }

  /* ── Checkbox ── */
  .check-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
    cursor: pointer;
  }
  .check-row input[type="checkbox"] { width: auto; margin: 0; cursor: pointer; }
  .check-row label { margin: 0; cursor: pointer; color: var(--vscode-foreground); font-size: 12px; }

  /* ── Collapsible resume block ── */
  #resume-block { padding-left: 12px; border-left: 2px solid var(--vscode-focusBorder, #007acc); margin-bottom: 8px; }

  /* ── Command preview ── */
  .cmd-preview {
    margin-top: 6px;
    padding: 6px 8px;
    background: var(--vscode-editor-background, #1e1e1e);
    border: 1px solid var(--vscode-widget-border, #444);
    border-radius: 3px;
    font-family: var(--vscode-editor-font-family, monospace);
    font-size: 10.5px;
    color: var(--vscode-descriptionForeground);
    word-break: break-all;
    white-space: pre-wrap;
    line-height: 1.5;
    min-height: 36px;
  }

  /* ── Run buttons ── */
  .run-row { display: flex; gap: 5px; margin-top: 10px; }
  .run-btn {
    flex: 1;
    padding: 7px 0;
    font-size: 12px;
    font-weight: 700;
    font-family: var(--vscode-font-family);
    cursor: pointer;
    border: none;
    border-radius: 2px;
    transition: opacity 0.1s;
  }
  .run-btn:hover { opacity: 0.85; }
  .btn-run {
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
  }
  .btn-debug {
    background: var(--vscode-button-secondaryBackground);
    color: var(--vscode-button-secondaryForeground);
    border: 1px solid var(--vscode-focusBorder, #007acc);
  }

  .hidden { display: none !important; }

  /* ── Separator ── */
  hr { border: none; border-top: 1px solid var(--vscode-widget-border, #444); margin: 10px 0; }
</style>
</head>
<body>

<!-- ── Mode ── -->
<div class="section">
  <div class="section-title">Mode</div>
  <div class="mode-row">
    <button class="mode-btn active" id="btn-train" onclick="setMode('train')">Train</button>
    <button class="mode-btn"        id="btn-play"  onclick="setMode('play')">Play</button>
  </div>
</div>

<!-- ── Conda Env ── -->
<div class="section">
  <div class="section-title">Python Environment</div>
  <div class="scan-row">
    <select id="condaEnv" onchange="updatePreview()">
      <option value="">-- scan first --</option>
    </select>
    <button class="scan-btn" onclick="vscode.postMessage({type:'scanCondaEnvs'})">Scan</button>
  </div>
</div>

<!-- ── Environment ── -->
<div class="section">
  <div class="section-title">Environment</div>
  <label>Task</label>
  <select id="task" onchange="onTaskChange()">
    <option value="">-- loading... --</option>
  </select>
  <label>Agent Config</label>
  <select id="agent" onchange="onAgentChange()">
    <option value="">-- select task first --</option>
  </select>
</div>

<!-- ── Parameters ── -->
<div class="section">
  <div class="section-title">Parameters</div>
  <div class="row2">
    <div class="field">
      <label>Num Envs</label>
      <input type="number" id="numEnvs" value="2048" min="1" oninput="updatePreview()">
    </div>
    <div class="field">
      <label>Seed</label>
      <input type="number" id="seed" placeholder="random" oninput="updatePreview()">
    </div>
  </div>
</div>

<!-- ── Train Options ── -->
<div class="section" id="train-section">
  <div class="section-title">Train Options</div>
  <label>Experiment Name</label>
  <input type="text" id="experimentName" placeholder="auto (from agent cfg)" oninput="updatePreview()">
  <div class="row2">
    <div class="field">
      <label>Run Tag <span style="font-weight:400;opacity:0.7">(appended to folder name)</span></label>
      <input type="text" id="runName" placeholder="e.g. baseline, cnn_attempt_2" oninput="updatePreview()">
    </div>
    <div class="field">
      <label>Max Iterations</label>
      <input type="number" id="maxIterations" placeholder="default" oninput="updatePreview()">
    </div>
  </div>

  <!-- Resume -->
  <label class="check-row" onclick="toggleResume()">
    <input type="checkbox" id="resumeCheck"> Resume from checkpoint
  </label>
  <div id="resume-block" class="hidden">
    <label>Experiment (for resume)</label>
    <div class="scan-row">
      <select id="resumeExp" onchange="onResumeExpChange()">
        <option value="">-- scan first --</option>
      </select>
      <button class="scan-btn" onclick="scanExperiments('resume')">Scan</button>
    </div>
    <label>Run</label>
    <div class="scan-row" style="margin-top:4px">
      <select id="resumeRun" onchange="onResumeRunChange()">
        <option value="">-- select experiment --</option>
      </select>
    </div>
    <label style="margin-top:4px">Checkpoint</label>
    <select id="resumeCheckpoint">
      <option value="">-- select run --</option>
    </select>
  </div>
</div>

<!-- ── Play Options ── -->
<div class="section hidden" id="play-section">
  <div class="section-title">Play Options</div>
  <label>Experiment Name</label>
  <div class="scan-row">
    <select id="playExp" onchange="onPlayExpChange()">
      <option value="">-- scan first --</option>
    </select>
    <button class="scan-btn" onclick="scanExperiments('play')">Scan</button>
  </div>
  <label style="margin-top:4px">Run</label>
  <div class="scan-row" style="margin-top:4px">
    <select id="loadRun" onchange="onLoadRunChange()">
      <option value="">-- select experiment --</option>
    </select>
  </div>
  <div class="scan-row" style="margin-top:4px">
    <input type="text" id="runLabelInput" placeholder="Label this run…" style="margin-bottom:0">
    <button class="scan-btn" onclick="saveRunLabel()">Save</button>
  </div>
  <label style="margin-top:4px">Checkpoint</label>
  <select id="checkpoint" onchange="updatePreview()">
    <option value="">-- select run --</option>
  </select>
</div>

<!-- ── Extra ── -->
<div class="section">
  <div class="section-title">Extra Args</div>
  <input type="text" id="extraArgs" placeholder="e.g. --headless --video" oninput="updatePreview()">
</div>

<!-- ── Preview ── -->
<div class="section-title">Command Preview</div>
<div class="cmd-preview" id="cmdPreview">Select options above…</div>

<div class="run-row">
  <button class="run-btn btn-run"   onclick="doRun(false)">▶  Run</button>
  <button class="run-btn btn-debug" onclick="doRun(true)"> 🐛 Debug</button>
</div>

<script>
const vscode = acquireVsCodeApi();
let mode = 'train';
let taskData = { tasks: [], agentMap: {}, expNameMap: {} };

// ── Mode ────────────────────────────────────────────────────────────────────
function setMode(m) {
    mode = m;
    document.getElementById('btn-train').classList.toggle('active', m === 'train');
    document.getElementById('btn-play').classList.toggle('active', m === 'play');
    document.getElementById('train-section').classList.toggle('hidden', m !== 'train');
    document.getElementById('play-section').classList.toggle('hidden', m !== 'play');

    // Sensible num_envs default
    const ne = document.getElementById('numEnvs');
    if (m === 'play' && ne.value === '2048') ne.value = '1';
    else if (m === 'train' && ne.value === '1') ne.value = '2048';

    updatePreview();
}

// ── Task / Agent ─────────────────────────────────────────────────────────────
function onTaskChange() {
    const task = document.getElementById('task').value;
    const agents = taskData.agentMap[task] || [];
    populateSelect('agent', agents, '-- none --');
    onAgentChange();
}

function onAgentChange() {
    const task  = document.getElementById('task').value;
    const agent = document.getElementById('agent').value;
    const expName = taskData.expNameMap[\`\${task}_\${agent}\`] || '';
    if (expName) document.getElementById('experimentName').value = expName;
    updatePreview();
}

// ── Scanning ─────────────────────────────────────────────────────────────────
function scanExperiments(target) {
    // target: 'play' or 'resume'
    vscode.postMessage({ type: 'scanExperiments', target });
}

function onPlayExpChange() {
    const exp = document.getElementById('playExp').value;
    if (exp) vscode.postMessage({ type: 'scanRuns', experimentName: exp, target: 'play' });
    updatePreview();
}

function onLoadRunChange() {
    const exp = document.getElementById('playExp').value;
    const run = document.getElementById('loadRun').value;
    if (exp && run) vscode.postMessage({ type: 'scanCheckpoints', experimentName: exp, run, target: 'play' });
    // Populate label input with existing label for selected run
    const sel = document.getElementById('loadRun');
    const opt = sel.options[sel.selectedIndex];
    document.getElementById('runLabelInput').value = opt ? (opt.dataset.label || '') : '';
    updatePreview();
}

function saveRunLabel() {
    const exp  = document.getElementById('playExp').value;
    const run  = document.getElementById('loadRun').value;
    const label = document.getElementById('runLabelInput').value.trim();
    if (!exp || !run) return;
    vscode.postMessage({ type: 'setRunLabel', experimentName: exp, run, label });
}

function onResumeExpChange() {
    const exp = document.getElementById('resumeExp').value;
    if (exp) vscode.postMessage({ type: 'scanRuns', experimentName: exp, target: 'resume' });
    updatePreview();
}

function onResumeRunChange() {
    const exp = document.getElementById('resumeExp').value;
    const run = document.getElementById('resumeRun').value;
    if (exp && run) vscode.postMessage({ type: 'scanCheckpoints', experimentName: exp, run, target: 'resume' });
    updatePreview();
}

// ── Resume toggle ────────────────────────────────────────────────────────────
function toggleResume() {
    // The click on label already toggles the checkbox before this fires from onclick
    // so we read the checkbox state
    setTimeout(() => {
        const checked = document.getElementById('resumeCheck').checked;
        document.getElementById('resume-block').classList.toggle('hidden', !checked);
        updatePreview();
    }, 0);
}

// ── Command builder ──────────────────────────────────────────────────────────
function buildCmd(debug) {
    const task      = document.getElementById('task').value;
    const agent     = document.getElementById('agent').value;
    const ne        = document.getElementById('numEnvs').value;
    const seed      = document.getElementById('seed').value;
    const extra     = document.getElementById('extraArgs').value.trim();
    const condaEnv  = document.getElementById('condaEnv').value;

    const actualTask = mode === 'play' ? task.replace('-v0', '-PLAY-v0') : task;
    const script     = mode === 'train' ? 'scripts/rsl_rl/train.py' : 'scripts/rsl_rl/play.py';

    const a = ['--task', actualTask, '--agent', agent];
    if (ne)   a.push('--num_envs', ne);
    if (seed) a.push('--seed', seed);

    if (mode === 'train') {
        const expName = document.getElementById('experimentName').value.trim();
        const runName = document.getElementById('runName').value.trim();
        const maxIter = document.getElementById('maxIterations').value.trim();
        if (expName) a.push('--experiment_name', expName);
        if (runName) a.push('--run_name', runName);
        if (maxIter) a.push('--max_iterations', maxIter);
        if (document.getElementById('resumeCheck').checked) {
            a.push('--resume');
            const rRun  = document.getElementById('resumeRun').value;
            const rCkpt = document.getElementById('resumeCheckpoint').value;
            if (rRun  && rRun  !== '-- select experiment --') a.push('--load_run', rRun);
            if (rCkpt && rCkpt !== '-- select run --')        a.push('--checkpoint', rCkpt);
        }
    } else {
        const run  = document.getElementById('loadRun').value;
        const ckpt = document.getElementById('checkpoint').value;
        if (run  && run  !== '-- select experiment --') a.push('--load_run', run);
        if (ckpt && ckpt !== '-- select run --')        a.push('--checkpoint', ckpt);
    }

    if (extra) a.push(extra);

    const py = debug
        ? 'python -m debugpy --listen 5678 --wait-for-client'
        : 'python';

    const envLabel = condaEnv ? \`# env: \${condaEnv}\n\` : '';
    return envLabel + py + ' ' + script + ' ' + a.join(' ');
}

function updatePreview() {
    document.getElementById('cmdPreview').textContent = buildCmd(false);
}

function doRun(debug) {
    const task  = document.getElementById('task').value;
    const agent = document.getElementById('agent').value;

    const cfg = {
        mode,
        task,
        agent,
        condaEnv:   document.getElementById('condaEnv').value,
        numEnvs:    document.getElementById('numEnvs').value,
        seed:       document.getElementById('seed').value,
        extraArgs:  document.getElementById('extraArgs').value.trim(),
        debug,
    };

    if (mode === 'train') {
        cfg.experimentName = document.getElementById('experimentName').value.trim();
        cfg.runName        = document.getElementById('runName').value.trim();
        cfg.maxIterations  = document.getElementById('maxIterations').value.trim();
        cfg.resume         = document.getElementById('resumeCheck').checked;
        cfg.resumeRun      = document.getElementById('resumeRun').value;
        cfg.resumeCheckpoint = document.getElementById('resumeCheckpoint').value;
    } else {
        cfg.playExpName = document.getElementById('playExp').value;
        cfg.loadRun     = document.getElementById('loadRun').value;
        cfg.checkpoint  = document.getElementById('checkpoint').value;
    }

    vscode.postMessage({ type: 'run', config: cfg });
}

// ── Messages from extension ──────────────────────────────────────────────────
function populateSelect(id, items, emptyLabel) {
    const sel = document.getElementById(id);
    if (!items || items.length === 0) {
        sel.innerHTML = \`<option value="">\${emptyLabel}</option>\`;
    } else {
        sel.innerHTML = items.map(v => \`<option value="\${v}">\${v}</option>\`).join('');
    }
    updatePreview();
}

window.addEventListener('message', event => {
    const msg = event.data;
    switch (msg.type) {
        case 'tasks': {
            taskData = { tasks: msg.tasks, agentMap: msg.agentMap, expNameMap: msg.expNameMap };
            const sel = document.getElementById('task');
            const prev = sel.value;
            sel.innerHTML = msg.tasks.length
                ? msg.tasks.map(t => \`<option value="\${t.id}">\${t.label} (\${t.id})</option>\`).join('')
                : '<option value="">-- no tasks found --</option>';
            // Restore previous selection if still valid, otherwise trigger fresh update
            if (prev && msg.tasks.some(t => t.id === prev)) sel.value = prev;
            onTaskChange();
            break;
        }

        case 'experiments':
            if (msg.target === 'play') {
                populateSelect('playExp', msg.experiments, '-- not found --');
                onPlayExpChange();
            } else {
                populateSelect('resumeExp', msg.experiments, '-- not found --');
                onResumeExpChange();
            }
            break;

        case 'runs': {
            // runs is now [{name, label}] — build options with label shown if present
            const buildRunOptions = runs => runs.map(r => {
                const display = r.label ? \`\${r.label}  ·  \${r.name}\` : r.name;
                return \`<option value="\${r.name}" data-label="\${r.label || ''}">\${display}</option>\`;
            }).join('');
            if (msg.target === 'play') {
                const sel = document.getElementById('loadRun');
                sel.innerHTML = msg.runs.length ? buildRunOptions(msg.runs) : '<option value="">-- no runs found --</option>';
                if (msg.runs.length > 0) onLoadRunChange();
            } else {
                // Resume uses plain names (no label UI there)
                populateSelect('resumeRun', msg.runs.map(r => r.name), '-- no runs found --');
                if (msg.runs.length > 0) onResumeRunChange();
            }
            break;
        }

        case 'labelSaved':
            // Re-scan runs to reflect the updated label in the dropdown
            onPlayExpChange();
            break;

        case 'checkpoints':
            if (msg.target === 'play') {
                populateSelect('checkpoint', msg.checkpoints, '-- no checkpoints --');
            } else {
                populateSelect('resumeCheckpoint', msg.checkpoints, '-- no checkpoints --');
            }
            break;

        case 'condaEnvs': {
            const sel = document.getElementById('condaEnv');
            if (msg.envs.length === 0) {
                sel.innerHTML = '<option value="">-- none found --</option>';
            } else {
                sel.innerHTML = msg.envs.map(e => \`<option value="\${e}">\${e}</option>\`).join('');
                // Default to blindbrigade if present, otherwise first entry
                const preferred = msg.envs.find(e => e === 'blindbrigade') || msg.envs[0];
                sel.value = preferred;
            }
            updatePreview();
            break;
        }
    }
});

// Init — conda envs and tasks are both pushed from the extension on open
vscode.postMessage({ type: 'scanCondaEnvs' });
updatePreview();
</script>
</body>
</html>`;
    }
}

// ── Activation ─────────────────────────────────────────────────────────────────

function activate(context) {
    const provider = new IsaacLabRunnerViewProvider(context);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('isaacLabRunner', provider, {
            webviewOptions: { retainContextWhenHidden: true }
        })
    );
}

function deactivate() {}

module.exports = { activate, deactivate };
