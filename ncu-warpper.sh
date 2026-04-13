#!/bin/bash
set -ex

# 1st arg as exec
exec="$1"

# name of exec as output report name
output_name=$(basename "$exec")

ncu --set full \
  -f -o ${output_name} \
  $exec
