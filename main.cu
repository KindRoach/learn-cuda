#include "common/deviceInfo.cuh"
#include "task/vectorAdd.cuh"
#include "task/vectorSum.cuh"
#include "task/nestHelloWorld.cuh"
#include "task/manualMemory.cuh"
#include "task/pinnedMemory.cuh"
#include "task/zeroCopyMemory.cuh"
#include "task/unifiedMemory.cuh"


int main() {
    manualMemory(1 << 28, 1024);
}
