#include <cstdio>

__global__ void kernel(int id) {
    printf("Kernel %d running on thread %d\n", id, threadIdx.x);
}

int main() {
    // create streams
    cudaStream_t stream, stream2, stream3;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // create events
    cudaEvent_t evt1, evt2, evt3;
    cudaEventCreate(&evt1);
    cudaEventCreate(&evt2);
    cudaEventCreate(&evt3);

    // create graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // start stream capture
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // --- Stage 1 ---
    kernel<<<1,1,0,stream>>>(1);
    cudaEventRecord(evt1, stream);

    // --- Stage 2a ---
    cudaStreamWaitEvent(stream2, evt1, 0);
    kernel<<<1,1,0,stream2>>>(2);
    cudaEventRecord(evt2, stream2);

    // --- Stage 2b ---
    cudaStreamWaitEvent(stream3, evt1, 0);
    kernel<<<1,1,0,stream3>>>(3);
    cudaEventRecord(evt3, stream3);

    // --- Stage 3 ---
    cudaStreamWaitEvent(stream, evt2, 0);
    cudaStreamWaitEvent(stream, evt3, 0);
    kernel<<<1,1,0,stream>>>(4);

    // stop capture
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // launch graph
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    // cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);

    cudaEventDestroy(evt1);
    cudaEventDestroy(evt2);
    cudaEventDestroy(evt3);

    cudaStreamDestroy(stream);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
}
