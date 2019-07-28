#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "test_header.h"


#define GRID 512

float get_rand()
{
    return (float)(rand() % 10) / 10;
}

int main(void)
{
    float *data_h, *data_d, *data_h_from_d;
    int irow = 28, icol = 28, ich = 3;
    int iN = ich * irow * icol;
    int iNBytes = iN * sizeof(float);

    // Memory allocate
    data_h = (float *)malloc(iNBytes);
    data_h_from_d = (float *)malloc(iNBytes);

    // Initialize
    for (int i = 0; i < iN; i++) {
        data_h[i] = get_rand();
    }

    // CUDA memory allocate
    cudaMalloc((void **)&data_d, iNBytes);

    cudaMemcpy(data_d, data_h, iNBytes, cudaMemcpyHostToDevice);

    // Execute
    relu(data_h, irow, ich);
    relu <<< GRID, iN / GRID + 1 >>> (data_d, irow, ich, iN);
    cudaMemcpy(data_h_from_d, data_d, iN * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < iN; i++) {
        printf("i: %d, CPU: %3.3f, GPU: %3.3f\n", i, data_h[i], data_h_from_d[i]);
        assert(abs(data_h[i] - data_h_from_d[i]) < 0.001);
    }
    printf("%d\n", iN);

    // Free
    free(data_h); free(data_h_from_d);
    // CUDA free
    cudaFree(data_d);

    printf("Success Test\n");
    return 0;
}
