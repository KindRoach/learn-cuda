#include "task/vectorAdd.cuh"
#include "task/vectorSum.cuh"
#include "common/deviceInfo.cuh"


int main() {
    // checkDeviceInfo();
    // performVectorAdd(1 << 28, 1024);
    performVectorSum(1 << 28, 1024);
}
