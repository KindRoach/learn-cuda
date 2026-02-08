#!/bin/bash

set -ex

nvcc -ptx -O0 -lineinfo -I ../../cpp-bench-utils/include ./*.cu
