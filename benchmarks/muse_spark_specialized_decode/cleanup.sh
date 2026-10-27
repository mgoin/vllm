#!/usr/bin/env bash
# Kill this user's leftover microbench workers (torchrun workers get their own
# process group, so killing the chg process group does not reach them).
pkill -KILL -u "$USER" -f "[m]use_bench.py bench forward"; true
