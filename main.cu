#include "common/deviceInfo.cuh"
#include "task/nestHelloWorld.cuh"
#include "task/algorithm/vectorAdd.cuh"
#include "task/algorithm/vectorSum.cuh"
#include "task/algorithm/matrixTranspose.cuh"
#include "task/memory/manualMemory.cuh"
#include "task/memory/pinnedMemory.cuh"
#include "task/memory/zeroCopyMemory.cuh"
#include "task/memory/unifiedMemory.cuh"
#include "task/memory/misalignedRead.cuh"
#include "task/sharedMemory/sharedMemoryVectorSum.cuh"
#include "task/stream/multiKernelConcurrent.cuh"


#include <cstdio>

int main() {
    multiKernelConcurrent(1 << 28, 32);
}