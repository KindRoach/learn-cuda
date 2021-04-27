#include "task/vectorAdd.cuh"
#include "task/vectorSum.cuh"


int main() {
     performVectorAdd(1 << 26, 1024);
    performVectorSum(1 << 26, 1024);
}
