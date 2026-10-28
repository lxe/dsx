#!/usr/bin/env bash
# Fresh isolated Pi run; keep generated work and transcripts outside this repo.
set -euo pipefail
run_dir="$(realpath -m "${1:?Usage: launch.sh RUN_DIR TASK_FILE [BASE_URL]}")"
task_file="$(realpath "${2:?Task file required}")"
base_url="${3:-http://localhost:8090/v1}"
config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Refuse to reuse a directory: no inherited code, transcript, or stale model config.
mkdir "$run_dir"
mkdir "$run_dir/agent" "$run_dir/game" "$run_dir/sessions"
cp "$config_dir/settings.json" "$run_dir/agent/settings.json"
cp "$task_file" "$run_dir/game/TASK.md"
python3 - "$config_dir/models.json" "$run_dir" "$base_url" <<'PYCONFIG'
import datetime, json, pathlib, secrets, sys
config = json.loads(pathlib.Path(sys.argv[1]).read_text())
run = pathlib.Path(sys.argv[2])
provider = config['providers']['dsx']
provider['baseUrl'] = sys.argv[3]
seed = secrets.randbelow(2**31)
provider['models'][0]['samplingParams']['seed'] = seed
(run / 'agent/models.json').write_text(json.dumps(config, indent=2) + '\n')
(run / 'manifest.json').write_text(json.dumps({
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'thinking': 'off', 'input': ['text', 'image'], 'seed': seed,
    'temperature': 1, 'top_p': 1, 'context': 262144,
    'pi_provider': 'dsx', 'base_url': sys.argv[3],
    'render_decision': 'Pi decides when to preview and finish',
    'code_carried_forward': False,
}, indent=2) + '\n')
PYCONFIG
export PI_CODING_AGENT_DIR="$run_dir/agent"
unset NODE_OPTIONS
cd "$run_dir/game"
exec pi --offline --provider dsx --model deepseek-v4.1-flash:off \
  --thinking off --tools read,bash,edit,write --no-extensions --no-mcp \
  --no-skills --no-prompt-templates --no-context-files \
  --session-dir "$run_dir/sessions" "$(cat "$task_file")"
