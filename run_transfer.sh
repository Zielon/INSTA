#!/bin/bash

# This script runs the transfer.py script for all combinations of target and source
# It will automatically generate the necessary meshes
set -e  # Stop the script if any command fails

targets=("justin" "malte_1")
sources=("biden" "obama" "bala")
exp_names=("insta")

for exp_name in "${exp_names[@]}"; do
  for target in "${targets[@]}"; do
    for source in "${sources[@]}"; do
      if [ "$target" == "$source" ]; then
        echo "Skipping: target and source are the same ($target)"
        continue
      fi

      echo "Processing target: $target, source: $source"
      
      python transfer.py --target "$target" --source "$source"
      
      ./build/rta --config transfer.json --scene data/"$target"/transforms_transfer.json --snapshot data/"$target"/experiments/$exp_name/debug/snapshot.msgpack --no-gui --width 512 --height 512
      
      dst=data/"$target"/experiments/$exp_name/debug/transfer_"$source"_to_"$target"
      
      rm -rf "$dst"
      mv data/"$target"/experiments/transfer "$dst"

    done
  done
done
