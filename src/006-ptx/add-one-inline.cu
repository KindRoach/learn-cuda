__global__ void add_one_ptx_kernel(int *A, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        int *ptr = A + idx;
        int val = 0;

        // A[ptr] += 1 using PTX inline assembly
        asm volatile(
            "ld.global.u32 %1, [%0];\n\t"
            "add.u32 %1, %1, 1;\n\t"
            "st.global.u32 [%0], %1;\n\t"
            :
            : "l"(ptr), "r"(val)
            : "memory");
    }
}

int main() {}
