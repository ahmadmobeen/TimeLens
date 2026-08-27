#!/usr/bin/env bash
# Pause the running router shards, profile ONE clip on idle GPUs, then resume.
#
# Why SIGSTOP rather than kill: it halts compute while leaving each shard's
# process state and GPU memory intact, so SIGCONT resumes exactly where it was
# and no clip is lost. Killing would forfeit the in-flight clips of 8 shards.
#
# The resume is wired to an EXIT trap so the router comes back even if this
# script is killed. An earlier attempt ran this inline in a foreground shell
# call which the harness terminated mid-flight; nothing was paused that time,
# but the trap is the real safety net -- a router left paused forever is far
# worse than a profile we never get.
set -uo pipefail
cd "$(dirname "$0")"

# Model under test. Pass 7B to compare against 8B on an identical code path --
# same crop, same fps, same token budget -- which is the only way to tell an
# 8B-specific vision-config problem from a problem with the zoom path itself.
export PROFILE_MODEL=${1:-TencentARC/TimeLens-8B}
echo "PROFILING $PROFILE_MODEL"

PIDFILE=logs/lora_eval/tl8b_zoom_router.pids
[ -f "$PIDFILE" ] || { echo "no pidfile: $PIDFILE"; exit 1; }
PIDS=$(cat "$PIDFILE")
states() { for p in $PIDS; do awk '{printf "%s", $3}' "/proc/$p/stat" 2>/dev/null || printf 'x'; done; }

resume() { for p in $PIDS; do kill -CONT "$p" 2>/dev/null; done; echo "ROUTER RESUMED: $(states)"; }
trap resume EXIT INT TERM

for p in $PIDS; do kill -STOP "$p" 2>/dev/null; done
sleep 20   # let in-flight CUDA kernels drain before measuring
echo "ROUTER PAUSED: $(states)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python - <<'PYEOF'
import os, time, torch
from types import SimpleNamespace
from transformers import AutoProcessor, AutoModelForImageTextToText
from c_head import config as C
from timelens.dataset.timelens_data import DATASET_DICT
from evaluation.utils import GroundingDataset
from eval_zoom import _span

def p(*a): print(*a, flush=True)

M = os.environ.get("PROFILE_MODEL", "TencentARC/TimeLens-8B")
annos = DATASET_DICT["charades-timelens"].load_annos(split="test")
a = dict([x for x in annos if (_span(x)[1] - _span(x)[0]) < 2.0][0])
s, e = _span(a); c0 = (s + e) / 2
a["video_start"], a["video_end"] = max(0.0, c0 - 4.0), c0 + 4.0
proc = AutoProcessor.from_pretrained(M, padding_side="left", do_resize=False, trust_remote_code=True)
ds = GroundingDataset([a], proc, SimpleNamespace(
    model_path=M, min_tokens=C.MIN_TOKENS, total_tokens=14336, fps=8, split="test"))
model = AutoModelForImageTextToText.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda:0").eval()
inp = ds[0]["inputs"].to("cuda")
plen = int(inp["input_ids"].shape[-1])
p(f"CLEAN prefix={plen} tokens on an idle GPU")

# max_new=1 is prefill plus a single decode step; 16 adds fifteen more. If the
# two are close the cost is prefill; if it scales with steps the cost is decode.
for cap in (1, 1, 16, 512):
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        out = model.generate(**inp, do_sample=False, temperature=None, top_p=None,
                             top_k=None, max_new_tokens=cap)
    torch.cuda.synchronize()
    p(f"CLEAN max_new={cap:3d}: {time.time()-t:7.2f}s  generated={int(out.shape[-1])-plen}tok")
PYEOF
