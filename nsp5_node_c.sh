#!/usr/bin/env bash
# Start NS-P5's third runner on a new node that shares this home directory and GPFS (Study Z:
# CG-Bench's full release, the plan in logs/nsp5/runner_c/plan.json, built by nsp5_full_cg.py).
#
#   On the NEW node:   bash ~/codes/active_repos/NeedleScan/research/TimeLens/nsp5_node_c.sh
#
# CG-Bench needs host memory: on node A a single CG-Bench eval has reached 400-590 GiB when it meets
# one of the videos the reader handles badly, so this runner starts only with about 800 GB of RAM. It
# uses node A's settings: one eval per GPU, launches pause above 55% of memory and the largest eval
# is stopped above 75% (it resumes later). Everything it writes is in the shared home:
# logs/nsp5/runner_c/{node.txt,runner.log,status.json} and each shard's .jsonl and .log in logs/nsp5/.
# To stop it, run on that node:
#
#   kill "$(python3 -c "import json; print(json.load(open('logs/nsp5/runner_c/status.json'))['pid'])")"
#
# Running evals survive a stop and are picked up again when the runner restarts.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
R=logs/nsp5/runner_c
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
  echo "An NS-P5 runner already runs on this node. Start this script on the new node." | tee -a "$R/node.txt"
  exit 1
fi
if [ -f "$R/status.json" ] && [ $(( $(date +%s) - $(stat -c %Y "$R/status.json") )) -lt 120 ]; then
  echo "runner_c wrote its status in the last two minutes: it already runs on some node. Not starting." | tee -a "$R/node.txt"
  exit 1
fi
mem=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
if [ "$mem" != "max" ] && [ "$mem" -lt $((700 * 1024 * 1024 * 1024)) ]; then
  echo "This node allows $((mem / 1024 / 1024 / 1024)) GiB of memory; CG-Bench needs about 800 GB. Not starting." | tee -a "$R/node.txt"
  exit 1
fi
min_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
if [ "$min_mib" -lt 170000 ]; then
  echo "These GPUs have ${min_mib} MiB; a 72B answer model needs about 170 GB on one GPU. Not starting." | tee -a "$R/node.txt"
  exit 1
fi

echo "$(date -u '+%F %T') --- started on $(hostname)" >> "$R/runner.log"
NSP5_RUNNER_DIR=$R NSP5_MAX_MINE=4 NSP5_MAX_EXCLUSIVE=4 NSP5_MEM_LAUNCH=0.55 NSP5_MEM_SHED=0.75 \
  NSP5_SHED_MAX=20 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  setsid nohup python3 -u nsp5_runner.py >> "$R/runner.log" 2>&1 < /dev/null &
sleep 5
tail -3 "$R/runner.log"
echo "Runner started on $(hostname). Log: research/TimeLens/$R/runner.log"
