#include "common/deviceInfo.cuh"
#include "task/vectorAdd.cuh"
#include "task/vectorSum.cuh"
#include "task/nestHelloWorld.cuh"
#include "task/memoryManage.cuh"
#include "task/pinnedMemory.cuh"
#include "task/zeroCopyMemory.cuh"


int main() {
    zeroCopyMemory(1 << 30, 1024);
}
