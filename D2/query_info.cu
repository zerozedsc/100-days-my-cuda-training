#include "share.h"

using namespace std;

void querySystemInfo()
{
    cudaDeviceProp prop;

    int count;

    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        printf("------General info for device %d------\n", i);
        printf("Device Name: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock Rate: %d\n", prop.clockRate);
        printf("Device copy overlap: ");
        if (prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Kernel Exectuion Timeout: ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("------Memory info for device %d------\n", i);
        printf("Total global mem:  %zu\n", prop.totalGlobalMem);
        printf("Total constant Mem:  %zu\n", prop.totalConstMem);
        printf("Max mem pitch:  %zu\n", prop.memPitch);
        printf("Texture Alignment:  %zu\n", prop.textureAlignment);
        printf("   ------ MP Information for device %d ------\n", i);
        printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp:  %zu\n", prop.sharedMemPerBlock);
        printf("Registers per mp:  %d\n", prop.regsPerBlock);
        printf("Threads in warp:  %d\n", prop.warpSize);
        printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }
}
