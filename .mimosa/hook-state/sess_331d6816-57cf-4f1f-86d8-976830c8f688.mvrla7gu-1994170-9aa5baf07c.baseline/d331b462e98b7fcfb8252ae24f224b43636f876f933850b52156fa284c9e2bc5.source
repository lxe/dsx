#!/bin/bash
# run_guarded NAME MEM_GB CMD... : run CMD in a user cgroup scope with a hard RAM cap and no swap, so a runaway job
# is OOM-killed inside its scope instead of freezing the machine. The scope is named ds41-NAME (watchdog.sh stops ds41-* scopes).
# No MemoryHigh: the soft cap throttled the engine (page cache from Engram reads kept the scope at the cap: 43K
# throttle events per run, decode stalls). At MemoryMax the kernel reclaims page cache; anon past it is OOM-killed.
name=$1; mem=$2; shift 2
exec systemd-run --user --scope --unit="ds41-$name" -p MemoryMax=${mem}G -p MemorySwapMax=0 "$@"
