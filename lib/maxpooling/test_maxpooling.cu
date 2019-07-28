#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "test_header.h"


#define GRID 512

float get_rand()
{
    return (float)(rand() % 10) / 100;
}

int main(void)
{
    float *data_h, *data_d, *out_h, *out_d, *out_h_from_d;
    int stride = 2;
    int irow = 28, icol = 28, ich = 3;
    int ksize = 2;
    int orow = 14, ocol = 14, och = 3;
    int iN = ich * irow * icol;
    int oN = och * orow * ocol;
    int iNBytes = iN * sizeof(float), oNBytes = oN * sizeof(float);

    // Memory allocate
    data_h = (float *)malloc(iNBytes);
    out_h = (float *)malloc(oNBytes);
    out_h_from_d = (float *)malloc(oNBytes);

    // Initialize
    for (int i = 0; i < iN; i++) {
        data_h[i] = get_rand();
    }

    // CUDA memory allocate
    cudaMalloc((void **)&data_d, iNBytes);
    cudaMalloc((void **)&out_d, oNBytes);

    cudaMemcpy(data_d, data_h, iNBytes, cudaMemcpyHostToDevice);

    // Execute
    maxpooling(data_h, irow, ich, out_h, orow, ksize, stride)
    maxpooling <<< GRID, oN / GRID + 1 >>> (data_d, irow, ich, out_d, orow, ksize, oN)
    cudaMemcpy(out_h_from_d, out_d, oN * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < oN; i++) {
        printf("i: %d, CPU: %3.3f, GPU: %3.3f\n", i, out_h[i], out_h_from_d[i]);
        assert(abs(out_h[i] - out_h_from_d[i]) < 0.001);
    }
    printf("%d\n", oN);

    // Free
    free(data_h); free(out_h); free(out_h_from_d); free(w_h); free(b_h);
    // CUDA free
    cudaFree(data_d), cudaFree(out_d), cudaFree(w_d), cudaFree(b_d);

    printf("Success Test\n");
    return 0;
}
