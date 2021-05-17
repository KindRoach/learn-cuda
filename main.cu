#include "common/deviceInfo.cuh"
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
#include "task/stream/syncStreamWithEvent.cuh"
#include "task/stream/vectorAddMultiStream.cuh"
#include "task/stream/graphConcurrent.cuh"
#include "task/instruction/floatPrecision.cuh"


#include <cstdio>

int main() {
    floatPrecision();
}