#!/bin/bash
# selftest, then decode speed at 4K / 128K / 256K with a per-kernel profile
cd /root/dsx/amd
./target/release/dsx-amd selftest > /root/selftest.log 2>&1
DSXA_PROF=1 ./target/release/dsx-amd bench --model /data/models/fp8 --rank /data/models/dsx-2bit/expert-rank.json --ctx 4096,131072,262144 --steps 24 > /root/bench1.log 2>&1
echo BENCH-DONE >> /root/bench1.log
