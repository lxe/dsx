#!/bin/bash
# kbuild.sh NAME [hipcc defines...]: build the kernels from /root/dsx/amd into /root/NAME.hsaco
export PATH=$PATH:/opt/rocm/bin
arch=${DSX_AMD_ARCH-gfx942}
case "$arch" in
  gfx942|gfx950) ;;
  *) echo "DSX_AMD_ARCH must be gfx942 or gfx950, got $arch" >&2; exit 1 ;;
esac
# a failed compile must not leave an older object looking like this build's result
rm -f /root/$1.hsaco
cd /root/dsx/amd && hipcc --genco --offload-arch=$arch -O3 -std=c++17 "${@:2}" kernels/dsxa.hip -o /root/$1.hsaco 2>&1 | grep -E "error" -A3 | head -20
ls -la /root/$1.hsaco | awk '{print $5, $9}'
