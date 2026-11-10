#!/usr/bin/env bash
# Start NS-P5's second runner on another node that shares this home directory and GPFS.
#
#   On the OTHER node:   bash ~/codes/active_repos/NeedleScan/research/TimeLens/nsp5_node_b.sh
#
# It runs the plan in logs/nsp5/runner_b/plan.json (Study H's 72B arms), disjoint from the first
# node's plan, so the two runners never write the same files. Everything it writes is in the shared
# home: logs/nsp5/runner_b/{node.txt,runner.log,status.json}, and each shard's .jsonl and .log in
# logs/nsp5/. To stop it, run on that node:
#
#   kill "$(python3 -c "import json; print(json.load(open('logs/nsp5/runner_b/status.json'))['pid'])")"
#
# Running evals survive a stop and are picked up again when the runner restarts.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
R=logs/nsp5/runner_b
mkdir -p "$R"

{
  echo "host $(hostname)   $(date -u '+%F %T UTC')"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
  echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
  echo "memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"
  echo "/dev/shm: $(df -h /dev/shm | tail -1)"
  .venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
} > "$R/node.txt" 2>&1
cat "$R/node.txt"

if pgrep -f "nsp5_runner.py" > /dev/null; then
  echo "An NS-P5 runner already runs on this node. Start this script on the other node." | tee -a "$R/node.txt"
  exit 1
fi
min_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
if [ "$min_mib" -lt 170000 ]; then
  echo "These GPUs have ${min_mib} MiB; a 72B answer model needs about 170 GB on one GPU. Not starting." | tee -a "$R/node.txt"
  exit 1
fi

echo "$(date -u '+%F %T') --- started on $(hostname)" >> "$R/runner.log"
NSP5_RUNNER_DIR=$R setsid nohup python3 -u nsp5_runner.py >> "$R/runner.log" 2>&1 < /dev/null &
sleep 5
tail -3 "$R/runner.log"
echo "Runner started on $(hostname). Log: research/TimeLens/$R/runner.log"
