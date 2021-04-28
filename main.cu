#include "common/deviceInfo.cuh"
#include "task/vectorAdd.cuh"
#include "task/vectorSum.cuh"
#include "task/nestHelloWorld.cuh"
#include "task/memoryManage.cuh"
#include "task/pinnedMemory.cuh"

int main() {
    pinnedMemory(1 << 28);
}
