#!/bin/bash
# MI325X droplet setup: scratch disk, ROCm dev packages, Rust, Python, model downloads. Logs in /root/*.log.
set -x
cloud-init status --wait > /dev/null 2>&1
while pgrep -x apt-get > /dev/null || pgrep -x dpkg > /dev/null; do sleep 3; done
mkdir -p /data && (mount | grep -q /data || mount /dev/vdc1 /data || (mkfs.ext4 -F /dev/vdc && mount /dev/vdc /data))
mkdir -p /data/models /data/hf
export DEBIAN_FRONTEND=noninteractive
for i in 1 2 3; do apt-get update -qq && apt-get install -y -qq amdrocm-core-dev7.14-gfx942 amdrocm-blas-dev7.14 build-essential pkg-config git curl python3.12-venv bc psmisc tmux htop nvme-cli fio > /root/apt.log 2>&1 && break; sleep 10; done
ls /opt/rocm/bin/hipcc && touch /root/hipcc.ready || echo APT-FAILED
python3 -m venv /root/venv && /root/venv/bin/pip install -q -U "huggingface_hub[hf_xet]" hf_transfer numpy safetensors > /root/pip.log 2>&1
echo PY-DONE
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /root/rustup.log 2>&1
echo SETUP-DONE
