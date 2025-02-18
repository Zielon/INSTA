#!/bin/bash

# Simple script showing how to run the RTA for multiple actors without GUI.

actors=("duda" "obama")

for actor in "${actors[@]}"; do
  echo "Running for actor = $actor"
  ./build/rta --config insta.json --scene "data/$actor" --height 512 --width 512 --no-gui
done
