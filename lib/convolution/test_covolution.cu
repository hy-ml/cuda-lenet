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
    int stride = 1;
    int irow = 28, icol = 28, ich = 1;
    int ksize = 5;
    int orow = 24, ocol = 24, och = 5;
    int iN = ich * irow * icol;
    int oN = och * orow * ocol;
    int iNBytes = iN * sizeof(float), oNBytes = oN * sizeof(float);

    float *w_h, *w_d, *b_h, *b_d;

    // Memory allocate
    data_h = (float *)malloc(iNBytes);
    out_h = (float *)malloc(oNBytes);
    out_h_from_d = (float *)malloc(oNBytes);
    w_h = (float*)malloc(ksize*ksize*och*sizeof(float));
    b_h = (float*)malloc(och * sizeof(float));

    // Initialize
    for (int i = 0; i < iN; i++) {
        data_h[i] = get_rand(); 
    }

    for (int i = 0; i < och*ksize*ksize; i++) {
        w_h[i] = get_rand();
    }


    for (int i = 0; i < och; i++) {
        b_h[i] = get_rand();
    }

    // CUDA memory allocate
    cudaMalloc((void **)&data_d, iNBytes);
    cudaMalloc((void **)&out_d, oNBytes);
    cudaMalloc((void **)&w_d, ksize*ksize*och*sizeof(float));
    cudaMalloc((void **)&b_d, och*sizeof(float));

    cudaMemcpy(data_d, data_h, iNBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(w_d, w_h, och*ksize*ksize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, och*sizeof(float), cudaMemcpyHostToDevice);

    // Execute
    convolution(data_h, irow, ich, out_h, orow, och, w_h, b_h, ksize, stride);
    convolution <<< GRID, oN / GRID + 1 >>> (data_d, irow, ich, out_d, orow, och, w_d, b_d, ksize, stride, oN);

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
