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
    float *data_h, *data_d, *out_h, *out_d, *out_h_from_d;
    int iN = 128;
    int oN = 64;
    int iNBytes = iN * sizeof(float), oNBytes = oN * sizeof(float);

    float *w_h, *w_d, *b_h, *b_d;

    // Memory allocate
    data_h = (float *)malloc(iNBytes);
    out_h = (float *)malloc(oNBytes);
    out_h_from_d = (float *)malloc(oNBytes);
    w_h = (float*)malloc(iN*oN*sizeof(float));
    b_h = (float*)malloc(oN*sizeof(float));

    // Initialize
    for (int i = 0; i < iN; i++) {
        data_h[i] = get_rand(); 
    }

    for (int i = 0; i < iN*oN; i++) {
        w_h[i] = get_rand();
    }


    for (int i = 0; i < oN; i++) {
        b_h[i] = get_rand();
    }

    // CUDA memory allocate
    cudaMalloc((void **)&data_d, iNBytes);
    cudaMalloc((void **)&out_d, oNBytes);
    cudaMalloc((void **)&w_d, iN*oN*sizeof(float));
    cudaMalloc((void **)&b_d, oN*sizeof(float));

    cudaMemcpy(data_d, data_h, iNBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(w_d, w_h, iN*oN*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, oN*sizeof(float), cudaMemcpyHostToDevice);

    // Execute
    classifier(data_h, iN, data_h, oN, w_h, b_h);
    classifier <<< GRID, oN / GRID + 1 >>> (data_d, iN, data_d, oN, w_d, b_d, oN);

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
