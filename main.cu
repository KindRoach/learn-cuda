#include "task/vectorAdd.cuh"
#include "task/vectorSum.cuh"


int main() {
    performVectorAdd(1 << 28, 1024);
    performVectorSum(1 << 28, 1024);
}
