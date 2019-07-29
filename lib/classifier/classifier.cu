#include <stdio.h>


__host__ void classifier(float *input, int isize, float *output, int osize,
        float *weight, float *bias) {
    int i, j;

    printf("Classifier:\n");
    printf("  isize=%d, osize=%d\n", isize, osize);

    for (i = 0; i < osize; i++) {
        *(output+i) = 0.0;

        for(j = 0; j < isize; j++) {
            *(output+i) += *(weight+i*isize+j) * *(input+j);
        }
        *(output+i) += *(bias+i);
    }
    printf("\n");fflush(stdout);
}



__global__ void classifier(float *input, int isize, float *output, int osize,
        float *weight, float *bias, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j;

    if (idx < N) {
        *(output + idx) = 0.0;
        for (j = 0; j < isize; j++) {
            *(output + idx) += *(weight + idx * isize + j) * *(input + j);
        }
        *(output + idx) += *(bias + idx);
    }
}
