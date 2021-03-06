#include <stdio.h>


__host__ void relu(float *input, int isize, int ichan) {
    int ocol, orow, och;

    printf("ReLu:\n");
    printf("  isize=%d, ichan=%d\n", isize, ichan);

    for (och= 0; och < ichan; och++) {
        for (orow = 0; orow < isize; orow++) {
            for (ocol = 0; ocol < isize; ocol++) {
                if (*(input+och*isize*isize+orow*isize+ocol) < 0.0) *(input+och*isize*isize+orow*isize+ocol) = 0.0;
            }
        }
    }
    printf("\n");fflush(stdout);
}


__global__ void relu(float *input, int isize, int ichan, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ocol, orow, och;

    if (idx < N) {
        // output size == input size
        och = idx / (isize * isize);
        orow = (idx % (isize * isize)) / isize;
        ocol = (idx % (isize * isize)) % isize;

        if (*(input + och * isize * isize + orow * isize + ocol) < 0.0)
            *(input + och * isize * isize + orow * isize + ocol) = 0.0;
    }
}
