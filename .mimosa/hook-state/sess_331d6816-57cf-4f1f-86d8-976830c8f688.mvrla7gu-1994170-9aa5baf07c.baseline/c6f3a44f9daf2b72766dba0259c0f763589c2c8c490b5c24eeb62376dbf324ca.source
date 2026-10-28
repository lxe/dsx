#!/bin/bash
# kbuild.sh NAME [hipcc defines...]: build the kernels from /root/dsx/amd into /root/NAME.hsaco
export PATH=$PATH:/opt/rocm/bin
cd /root/dsx/amd && hipcc --genco --offload-arch=gfx942 -O3 -std=c++17 "${@:2}" kernels/dsxa.hip -o /root/$1.hsaco 2>&1 | grep -E "error" -A3 | head -20
ls -la /root/$1.hsaco | awk '{print $5, $9}'
