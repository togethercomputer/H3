#!/bin/bash
set -e
cd "${0%/*}"
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

/bin/bash -c 'python examples/serving_h3.py --model_name H3-125M --ckpt_path /home/user/.together/models/H3-125M/model.pt' &

wait -n
exit $?