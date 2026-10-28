#!/bin/bash
# devcmd.sh "cmd1;cmd2": queue commands for a running `dsx-amd dev`, wait for them, print the new output
n=$(wc -l < /root/dev.log)
echo "$1" | tr ';' '\n' > /root/dev.cmd.tmp && mv /root/dev.cmd.tmp /root/dev.cmd
while [ -e /root/dev.cmd ]; do sleep 0.5; done
last=$(echo "$1" | tr ';' '\n' | tail -1)
# wait until the output after the last command header is followed by the idle marker
until grep -q "^== idle" <(tail -n +$((n+1)) /root/dev.log); do sleep 1; done
tail -n +$((n+1)) /root/dev.log | grep -v "^== idle"
